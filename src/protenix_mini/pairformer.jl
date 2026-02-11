module Pairformer

using Random

import ..Primitives: Linear, LinearNoBias, LayerNorm, transition
import ..OpenFoldBlocks:
    PairAttentionNoS,
    TriangleMultiplication,
    TriangleAttention,
    OuterProductMean
import ..Utils: softmax_dim2, sample_msa_indices, one_hot_int

export TransitionBlock,
    PairformerBlock,
    PairformerStack,
    MSAPairWeightedAveraging,
    MSAStack,
    MSABlock,
    MSAModule,
    TemplateEmbedder,
    NoisyStructureEmbedder

_as_f32_array(x::AbstractArray{<:Real}) = x isa AbstractArray{Float32} ? x : Float32.(x)
_as_f32_copy(x::AbstractArray{<:Real}) = x isa AbstractArray{Float32} ? copy(x) : Float32.(x)

struct TransitionBlock
    c_in::Int
    n::Int
    layernorm1::LayerNorm
    linear_no_bias_a::LinearNoBias
    linear_no_bias_b::LinearNoBias
    linear_no_bias::LinearNoBias
end

function TransitionBlock(c_in::Int; n::Int = 4, rng::AbstractRNG = Random.default_rng())
    return TransitionBlock(
        c_in,
        n,
        LayerNorm(c_in),
        LinearNoBias(n * c_in, c_in; rng = rng),
        LinearNoBias(n * c_in, c_in; rng = rng),
        LinearNoBias(c_in, n * c_in; rng = rng),
    )
end

function (m::TransitionBlock)(x::AbstractArray{<:Real})
    return transition(x, m.layernorm1, m.linear_no_bias_a, m.linear_no_bias_b, m.linear_no_bias)
end

struct PairformerBlock
    c_s::Int
    tri_mul_out::TriangleMultiplication
    tri_mul_in::TriangleMultiplication
    tri_att_start::TriangleAttention
    tri_att_end::TriangleAttention
    pair_transition::TransitionBlock
    attention_pair_bias::Union{PairAttentionNoS, Nothing}
    single_transition::Union{TransitionBlock, Nothing}
end

function PairformerBlock(
    c_z::Int,
    c_s::Int;
    n_heads::Int = 16,
    c_hidden_mul::Int = 128,
    c_hidden_pair_att::Int = 32,
    no_heads_pair::Int = 4,
    rng::AbstractRNG = Random.default_rng(),
)
    apb = c_s > 0 ? PairAttentionNoS(c_s, c_z; n_heads = n_heads, rng = rng) : nothing
    st = c_s > 0 ? TransitionBlock(c_s; n = 4, rng = rng) : nothing
    return PairformerBlock(
        c_s,
        TriangleMultiplication(c_z, c_hidden_mul; outgoing = true, rng = rng),
        TriangleMultiplication(c_z, c_hidden_mul; outgoing = false, rng = rng),
        TriangleAttention(c_z, c_hidden_pair_att * no_heads_pair, no_heads_pair; starting = true, rng = rng),
        TriangleAttention(c_z, c_hidden_pair_att * no_heads_pair, no_heads_pair; starting = false, rng = rng),
        TransitionBlock(c_z; n = 4, rng = rng),
        apb,
        st,
    )
end

function (m::PairformerBlock)(
    s::Union{Nothing, AbstractMatrix{<:Real}},
    z::AbstractArray{<:Real,3};
    pair_mask::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
)
    z_f = _as_f32_copy(z)

    z_f .+= m.tri_mul_out(z_f; mask = pair_mask)
    z_f .+= m.tri_mul_in(z_f; mask = pair_mask)
    z_f .+= m.tri_att_start(z_f; mask = pair_mask)
    z_f .+= m.tri_att_end(z_f; mask = pair_mask)
    z_f .+= m.pair_transition(z_f)

    if m.c_s > 0
        s === nothing && error("PairformerBlock with c_s>0 requires s")
        s_f = _as_f32_copy(s)
        m.attention_pair_bias === nothing && error("Missing attention_pair_bias")
        m.single_transition === nothing && error("Missing single_transition")
        s_f .+= m.attention_pair_bias(s_f, z_f, pair_mask)
        s_f .+= m.single_transition(s_f)
        return s_f, z_f
    end

    return nothing, z_f
end

struct PairformerStack
    blocks::Vector{PairformerBlock}
end

function PairformerStack(
    c_z::Int,
    c_s::Int;
    n_blocks::Int,
    n_heads::Int = 16,
    rng::AbstractRNG = Random.default_rng(),
)
    n_blocks >= 0 || error("n_blocks must be >= 0")
    blocks = PairformerBlock[]
    for _ in 1:n_blocks
        push!(blocks, PairformerBlock(c_z, c_s; n_heads = n_heads, rng = rng))
    end
    return PairformerStack(blocks)
end

function (m::PairformerStack)(
    s::Union{Nothing, AbstractMatrix{<:Real}},
    z::AbstractArray{<:Real,3};
    pair_mask::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
)
    s_cur = s === nothing ? nothing : _as_f32_array(s)
    z_cur = _as_f32_array(z)
    for blk in m.blocks
        s_cur, z_cur = blk(s_cur, z_cur; pair_mask = pair_mask)
    end
    return s_cur, z_cur
end

struct MSAPairWeightedAveraging
    c_m::Int
    c::Int
    n_heads::Int
    c_z::Int
    layernorm_m::LayerNorm
    linear_no_bias_mv::LinearNoBias
    layernorm_z::LayerNorm
    linear_no_bias_z::LinearNoBias
    linear_no_bias_mg::LinearNoBias
    linear_no_bias_out::LinearNoBias
end

function MSAPairWeightedAveraging(
    c_m::Int,
    c_z::Int;
    c::Int = 32,
    n_heads::Int = 8,
    rng::AbstractRNG = Random.default_rng(),
)
    return MSAPairWeightedAveraging(
        c_m,
        c,
        n_heads,
        c_z,
        LayerNorm(c_m),
        LinearNoBias(c * n_heads, c_m; rng = rng),
        LayerNorm(c_z),
        LinearNoBias(n_heads, c_z; rng = rng),
        LinearNoBias(c * n_heads, c_m; rng = rng),
        LinearNoBias(c_m, c * n_heads; rng = rng),
    )
end

function (m::MSAPairWeightedAveraging)(
    msa::AbstractArray{<:Real,3},
    z::AbstractArray{<:Real,3},
)
    n_msa, n_tok, c_m = size(msa)
    c_m == m.c_m || error("MSAPairWeightedAveraging c_m mismatch")
    size(z) == (n_tok, n_tok, m.c_z) || error("MSAPairWeightedAveraging z shape mismatch")

    msa_ln = m.layernorm_m(msa)
    v_lin = m.linear_no_bias_mv(msa_ln) # [N_msa, N_tok, H*C]
    b = m.linear_no_bias_z(m.layernorm_z(z)) # [N, N, H]
    g_lin = 1f0 ./ (1f0 .+ exp.(-m.linear_no_bias_mg(msa_ln))) # [N_msa, N_tok, H*C]

    w = softmax_dim2(b) # softmax over second token dimension

    o_lin = zeros(Float32, n_msa, n_tok, m.n_heads * m.c)
    @inbounds for m_ix in 1:n_msa, i in 1:n_tok, h in 1:m.n_heads, c_ix in 1:m.c
        idx = (h - 1) * m.c + c_ix
        acc = 0f0
        for j in 1:n_tok
            acc += w[i, j, h] * v_lin[m_ix, j, idx]
        end
        o_lin[m_ix, i, idx] = g_lin[m_ix, i, idx] * acc
    end

    return m.linear_no_bias_out(o_lin)
end

struct MSAStack
    msa_pair_weighted_averaging::MSAPairWeightedAveraging
    transition_m::TransitionBlock
end

function MSAStack(
    c_m::Int,
    c_z::Int;
    rng::AbstractRNG = Random.default_rng(),
)
    return MSAStack(
        MSAPairWeightedAveraging(c_m, c_z; c = 8, n_heads = 8, rng = rng),
        TransitionBlock(c_m; n = 4, rng = rng),
    )
end

function (m::MSAStack)(msa::AbstractArray{<:Real,3}, z::AbstractArray{<:Real,3})
    msa_f = _as_f32_copy(msa)
    msa_f .+= m.msa_pair_weighted_averaging(msa_f, z)
    msa_f .+= m.transition_m(msa_f)
    return msa_f
end

struct MSABlock
    is_last_block::Bool
    outer_product_mean_msa::OuterProductMean
    msa_stack::Union{MSAStack, Nothing}
    pair_stack::PairformerBlock
end

function MSABlock(
    c_m::Int,
    c_z::Int;
    is_last_block::Bool,
    rng::AbstractRNG = Random.default_rng(),
)
    msa_stack = is_last_block ? nothing : MSAStack(c_m, c_z; rng = rng)
    return MSABlock(
        is_last_block,
        OuterProductMean(c_m, c_z, 32; rng = rng),
        msa_stack,
        PairformerBlock(c_z, 0; n_heads = 16, rng = rng),
    )
end

function (m::MSABlock)(
    msa::AbstractArray{<:Real,3},
    z::AbstractArray{<:Real,3};
    pair_mask::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
)
    msa_f = _as_f32_copy(msa)
    z_f = _as_f32_copy(z)
    z_f .+= m.outer_product_mean_msa(msa_f)

    if !m.is_last_block
        m.msa_stack === nothing && error("MSA stack missing on non-last block")
        msa_f = m.msa_stack(msa_f, z_f)
    end

    _, z_f = m.pair_stack(nothing, z_f; pair_mask = pair_mask)
    if m.is_last_block
        return nothing, z_f
    end
    return msa_f, z_f
end

struct MSAModule
    n_blocks::Int
    c_m::Int
    c_s_inputs::Int
    sample_cutoff::Int
    sample_lower_bound::Int
    sample_strategy::String
    linear_no_bias_m::LinearNoBias
    linear_no_bias_s::LinearNoBias
    blocks::Vector{MSABlock}
end

function MSAModule(
    c_z::Int,
    c_s_inputs::Int;
    n_blocks::Int,
    c_m::Int = 64,
    sample_cutoff::Int = 16384,
    sample_lower_bound::Int = 1,
    sample_strategy::String = "random",
    rng::AbstractRNG = Random.default_rng(),
)
    blocks = MSABlock[]
    for i in 1:n_blocks
        push!(blocks, MSABlock(c_m, c_z; is_last_block = (i == n_blocks), rng = rng))
    end
    return MSAModule(
        n_blocks,
        c_m,
        c_s_inputs,
        sample_cutoff,
        sample_lower_bound,
        sample_strategy,
        LinearNoBias(c_m, 34; rng = rng),
        LinearNoBias(c_m, c_s_inputs; rng = rng),
        blocks,
    )
end

function _select_rows(x::AbstractArray, idx::Vector{Int}, dim::Int)
    if dim == 1
        return x[idx, :, :]
    elseif dim == 2
        return x[:, idx, :]
    else
        error("_select_rows only supports dim=1 or dim=2")
    end
end

function (m::MSAModule)(
    input_feature_dict::AbstractDict{<:AbstractString, <:Any},
    z::AbstractArray{<:Real,3},
    s_inputs::AbstractMatrix{<:Real};
    pair_mask::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    rng::AbstractRNG = Random.default_rng(),
)
    m.n_blocks < 1 && return _as_f32_array(z)
    haskey(input_feature_dict, "msa") || return _as_f32_array(z)

    msa_raw = input_feature_dict["msa"]
    msa_raw isa AbstractMatrix || return _as_f32_array(z)
    size(msa_raw, 2) > 0 || return _as_f32_array(z)

    haskey(input_feature_dict, "has_deletion") || error("Missing 'has_deletion' for MSAModule")
    haskey(input_feature_dict, "deletion_value") || error("Missing 'deletion_value' for MSAModule")

    n_msa = size(msa_raw, 1)
    idx = sample_msa_indices(
        n_msa;
        cutoff = m.sample_cutoff,
        lower_bound = m.sample_lower_bound,
        strategy = m.sample_strategy,
        rng = rng,
    )
    isempty(idx) && return _as_f32_array(z)

    msa_sel = Int.(msa_raw[idx, :])
    del_mask = _as_f32_array(input_feature_dict["has_deletion"][idx, :])
    del_val = _as_f32_array(input_feature_dict["deletion_value"][idx, :])

    msa_onehot = one_hot_int(msa_sel, 32)
    msa_feat = cat(msa_onehot, reshape(del_mask, size(del_mask)..., 1), reshape(del_val, size(del_val)..., 1); dims = 3)

    msa_emb = m.linear_no_bias_m(msa_feat)
    s_proj = m.linear_no_bias_s(_as_f32_array(s_inputs))
    msa_emb .+= reshape(s_proj, 1, size(s_proj, 1), size(s_proj, 2))

    z_cur = _as_f32_copy(z)
    msa_cur = msa_emb
    for blk in m.blocks
        msa_cur, z_cur = blk(msa_cur, z_cur; pair_mask = pair_mask)
    end
    return z_cur
end

struct TemplateEmbedder
    n_blocks::Int
    c::Int
    c_z::Int
    layernorm_z::LayerNorm
    linear_no_bias_z::LinearNoBias
    linear_no_bias_a::LinearNoBias
    pairformer_stack::PairformerStack
    layernorm_v::LayerNorm
    linear_no_bias_u::LinearNoBias
end

function TemplateEmbedder(
    c_z::Int;
    n_blocks::Int,
    c::Int = 64,
    rng::AbstractRNG = Random.default_rng(),
)
    return TemplateEmbedder(
        n_blocks,
        c,
        c_z,
        LayerNorm(c_z),
        LinearNoBias(c, c_z; rng = rng),
        LinearNoBias(c, 39 + 1 + 3 + 1 + 32 + 32; rng = rng),
        PairformerStack(c, 0; n_blocks = n_blocks, n_heads = 4, rng = rng),
        LayerNorm(c),
        LinearNoBias(c_z, c; rng = rng),
    )
end

function (m::TemplateEmbedder)(
    input_feature_dict::AbstractDict{<:AbstractString, <:Any},
    z::AbstractArray{<:Real,3};
    pair_mask::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
)
    if m.n_blocks < 1 || !haskey(input_feature_dict, "template_restype")
        return zeros(Float32, size(z))
    end
    # Mirrors Python implementation: template branch currently disabled.
    return zeros(Float32, size(z))
end

struct NoisyStructureEmbedder
    c_z::Int
    bins::Vector{Float32}
    upper_bins::Vector{Float32}
    linear_struct::LinearNoBias
    layernorm_z::LayerNorm
    linear_z::LinearNoBias
    transition_out::TransitionBlock
end

function NoisyStructureEmbedder(
    c_z::Int;
    min_bin::Real = 3.25,
    max_bin::Real = 50.75,
    no_bins::Int = 39,
    rng::AbstractRNG = Random.default_rng(),
)
    c = div(c_z, 2)
    c * 2 == c_z || error("NoisyStructureEmbedder requires even c_z")
    bins = collect(Float32, range(Float32(min_bin), Float32(max_bin), length = no_bins))
    upper = vcat(bins[2:end], Float32[1f6])
    return NoisyStructureEmbedder(
        c_z,
        bins,
        upper,
        LinearNoBias(c, no_bins + 1; rng = rng),
        LayerNorm(c_z),
        LinearNoBias(c, c_z; rng = rng),
        TransitionBlock(c_z; n = 2, rng = rng),
    )
end

function _one_hot_binned_sqdist(
    x::AbstractMatrix{<:Real},
    bins::Vector{Float32},
    upper_bins::Vector{Float32},
)
    n = size(x, 1)
    out = zeros(Float32, n, n, length(bins))
    @inbounds for i in 1:n, j in 1:n
        dx = Float32(x[i, 1]) - Float32(x[j, 1])
        dy = Float32(x[i, 2]) - Float32(x[j, 2])
        dz = Float32(x[i, 3]) - Float32(x[j, 3])
        d = sqrt(dx * dx + dy * dy + dz * dz)
        for b in 1:length(bins)
            out[i, j, b] = (d > bins[b] && d < upper_bins[b]) ? 1f0 : 0f0
        end
    end
    return out
end

function (m::NoisyStructureEmbedder)(
    input_feature_dict::AbstractDict{<:AbstractString, <:Any},
    z::AbstractArray{<:Real,3},
)
    n = size(z, 1)
    x = if haskey(input_feature_dict, "struct_cb_coords")
        _as_f32_array(input_feature_dict["struct_cb_coords"])
    else
        zeros(Float32, n, 3)
    end
    mask = if haskey(input_feature_dict, "struct_cb_mask")
        Bool.(input_feature_dict["struct_cb_mask"])
    else
        falses(n)
    end

    pair_mask = zeros(Float32, n, n, 1)
    @inbounds for i in 1:n, j in 1:n
        pair_mask[i, j, 1] = (mask[i] && mask[j]) ? 1f0 : 0f0
    end

    d = _one_hot_binned_sqdist(x, m.bins, m.upper_bins) .* pair_mask
    d = cat(d, pair_mask; dims = 3)
    d = m.linear_struct(d)

    z_proj = m.linear_z(m.layernorm_z(z))
    z_cat = cat(z_proj, d; dims = 3)
    out = m.transition_out(z_cat)

    any(mask) || (out .= 0f0)
    return out
end

end
