module Pairformer

using Random
using ConcreteStructs
using Flux: @layer

import ..Primitives: Linear, LinearNoBias, LayerNorm, transition
import ..Features: ProtenixFeatures
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

@concrete struct TransitionBlock
    c_in
    n
    layernorm1
    linear_no_bias_a
    linear_no_bias_b
    linear_no_bias
end
@layer TransitionBlock

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

@concrete struct PairformerBlock
    c_s
    tri_mul_out
    tri_mul_in
    tri_att_start
    tri_att_end
    pair_transition
    attention_pair_bias
    single_transition
end
@layer PairformerBlock

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

@concrete struct PairformerStack
    blocks
end
@layer PairformerStack

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

@concrete struct MSAPairWeightedAveraging
    c_m
    c
    n_heads
    c_z
    layernorm_m
    linear_no_bias_mv
    layernorm_z
    linear_no_bias_z
    linear_no_bias_mg
    linear_no_bias_out
end
@layer MSAPairWeightedAveraging

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

    w = softmax_dim2(b) # softmax over second token dimension [N, N, H]

    # Vectorized weighted sum: for each head h, weighted_v[:, i, hc] = sum_j w[i,j,h] * v[:, j, hc]
    # Process per-head to stay GPU friendly
    o_lin = fill!(similar(v_lin, Float32, n_msa, n_tok, m.n_heads * m.c), 0f0)
    for h in 1:m.n_heads
        r = ((h - 1) * m.c + 1):(h * m.c)
        wh = w[:, :, h] # [n_tok, n_tok]
        for s in 1:n_msa
            # v_slice: [n_tok, c], wh: [n_tok, n_tok] → wh * v_slice = [n_tok, c]
            o_lin[s, :, r] .= wh * v_lin[s, :, r]
        end
    end
    o_lin = o_lin .* g_lin

    return m.linear_no_bias_out(o_lin)
end

@concrete struct MSAStack
    msa_pair_weighted_averaging
    transition_m
end
@layer MSAStack

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

@concrete struct MSABlock
    is_last_block
    outer_product_mean_msa
    msa_stack
    pair_stack
end
@layer MSABlock

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

@concrete struct MSAModule
    n_blocks
    c_m
    c_s_inputs
    sample_cutoff
    sample_lower_bound
    sample_strategy
    linear_no_bias_m
    linear_no_bias_s
    blocks
end
@layer MSAModule

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
    feat::ProtenixFeatures,
    z::AbstractArray{<:Real,3},
    s_inputs::AbstractMatrix{<:Real};
    pair_mask::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    rng::AbstractRNG = Random.default_rng(),
)
    m.n_blocks < 1 && return _as_f32_array(z)
    feat.msa === nothing && return _as_f32_array(z)
    feat.has_deletion === nothing && error("Missing 'has_deletion' for MSAModule")
    feat.deletion_value === nothing && error("Missing 'deletion_value' for MSAModule")

    msa_raw = feat.msa
    size(msa_raw, 2) > 0 || return _as_f32_array(z)

    n_msa = size(msa_raw, 1)
    idx = sample_msa_indices(
        n_msa;
        cutoff = m.sample_cutoff,
        lower_bound = m.sample_lower_bound,
        strategy = m.sample_strategy,
        rng = rng,
    )
    isempty(idx) && return _as_f32_array(z)

    # one_hot_int uses scalar loops on CPU; pull Int MSA to CPU.
    msa_sel = Int.(Array(msa_raw[idx, :]))
    del_mask = _as_f32_array(feat.has_deletion[idx, :])
    del_val = _as_f32_array(feat.deletion_value[idx, :])

    msa_onehot_cpu = one_hot_int(msa_sel, 32)
    # Transfer one-hot to same device as z (GPU if model on GPU).
    msa_onehot = z isa Array ? msa_onehot_cpu : copyto!(similar(z, Float32, size(msa_onehot_cpu)...), msa_onehot_cpu)
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

    msa_sel = Int.(Array(msa_raw[idx, :]))
    del_mask = _as_f32_array(input_feature_dict["has_deletion"][idx, :])
    del_val = _as_f32_array(input_feature_dict["deletion_value"][idx, :])

    msa_onehot_cpu = one_hot_int(msa_sel, 32)
    msa_onehot = z isa Array ? msa_onehot_cpu : copyto!(similar(z, Float32, size(msa_onehot_cpu)...), msa_onehot_cpu)
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

@concrete struct TemplateEmbedder
    n_blocks
    c
    c_z
    layernorm_z
    linear_no_bias_z
    linear_no_bias_a
    pairformer_stack
    layernorm_v
    linear_no_bias_u
end
@layer TemplateEmbedder

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
    feat::ProtenixFeatures,
    z::AbstractArray{<:Real,3};
    pair_mask::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
)
    if m.n_blocks < 1 || feat.template_restype === nothing
        return fill!(similar(z, Float32, size(z)), 0f0)
    end
    # Mirrors Python implementation: template branch currently disabled.
    return fill!(similar(z, Float32, size(z)), 0f0)
end

function (m::TemplateEmbedder)(
    input_feature_dict::AbstractDict{<:AbstractString, <:Any},
    z::AbstractArray{<:Real,3};
    pair_mask::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
)
    if m.n_blocks < 1 || !haskey(input_feature_dict, "template_restype")
        return fill!(similar(z, Float32, size(z)), 0f0)
    end
    # Mirrors Python implementation: template branch currently disabled.
    return fill!(similar(z, Float32, size(z)), 0f0)
end

@concrete struct NoisyStructureEmbedder
    c_z
    bins
    upper_bins
    linear_struct
    layernorm_z
    linear_z
    transition_out
end
@layer NoisyStructureEmbedder

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
    bins::AbstractVector{Float32},
    upper_bins::AbstractVector{Float32},
)
    n = size(x, 1)
    xf = Float32.(x)
    # Vectorized pairwise distance
    diff = reshape(xf, n, 1, 3) .- reshape(xf, 1, n, 3)
    d = dropdims(sqrt.(sum(diff .^ 2; dims=3)); dims=3) # [n, n]
    # Vectorized interval one-hot
    nb = length(bins)
    d3 = reshape(d, n, n, 1)
    lo = reshape(bins, 1, 1, nb)
    hi = reshape(upper_bins, 1, 1, nb)
    return Float32.((d3 .> lo) .& (d3 .< hi))
end

function _noisy_structure_forward(
    m::NoisyStructureEmbedder,
    z::AbstractArray{<:Real,3},
    x::AbstractMatrix{<:Real},
    mask::AbstractVector{Bool},
)
    # Early exit: if no atoms are masked, the entire contribution is zero.
    any(mask) || return fill!(similar(z, Float32, size(z)), 0f0)

    n = size(z, 1)
    # Vectorized pair mask: mask[i] & mask[j]
    mf_cpu = Float32.(mask)
    pair_mask_cpu = reshape(mf_cpu, n, 1, 1) .* reshape(mf_cpu, 1, n, 1) # [n, n, 1]
    # Transfer to device of z for broadcasting with GPU tensors.
    pair_mask = z isa Array ? pair_mask_cpu : copyto!(similar(z, Float32, size(pair_mask_cpu)...), pair_mask_cpu)

    d = _one_hot_binned_sqdist(x, m.bins, m.upper_bins) .* pair_mask
    d = cat(d, pair_mask; dims = 3)
    d = m.linear_struct(d)

    z_proj = m.linear_z(m.layernorm_z(z))
    z_cat = cat(z_proj, d; dims = 3)
    out = m.transition_out(z_cat)

    return out
end

function (m::NoisyStructureEmbedder)(
    feat::ProtenixFeatures,
    z::AbstractArray{<:Real,3},
)
    n = size(z, 1)
    x = feat.struct_cb_coords === nothing ? fill!(similar(z, Float32, n, 3), 0f0) : Float32.(feat.struct_cb_coords)
    # Create mask on CPU (used in scalar-compatible Bool operations).
    mask = feat.struct_cb_mask === nothing ? falses(n) : Bool.(Array(feat.struct_cb_mask))
    return _noisy_structure_forward(m, z, x, mask)
end

function (m::NoisyStructureEmbedder)(
    input_feature_dict::AbstractDict{<:AbstractString, <:Any},
    z::AbstractArray{<:Real,3},
)
    n = size(z, 1)
    x = if haskey(input_feature_dict, "struct_cb_coords")
        _as_f32_array(input_feature_dict["struct_cb_coords"])
    else
        fill!(similar(z, Float32, n, 3), 0f0)
    end
    mask = if haskey(input_feature_dict, "struct_cb_mask")
        Bool.(Array(input_feature_dict["struct_cb_mask"]))
    else
        falses(n)
    end
    return _noisy_structure_forward(m, z, x, mask)
end

end
