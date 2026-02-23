module Pairformer

using Random
using ConcreteStructs
using Flux: @layer
using NNlib

import ..Primitives: Linear, LinearNoBias, LayerNorm, transition
import ..Features: ProtenixFeatures
import ..OpenFoldBlocks:
    PairAttentionNoS,
    TriangleMultiplication,
    TriangleAttention,
    OuterProductMean
import ..Utils: softmax_lastdim, sample_msa_indices, one_hot_int

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

# ─── TransitionBlock ───────────────────────────────────────────────────────────
# Features-first: (c_in, ...) → (c_in, ...)

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
        LinearNoBias(c_in, n * c_in; rng = rng),   # c_in → n*c_in
        LinearNoBias(c_in, n * c_in; rng = rng),   # c_in → n*c_in
        LinearNoBias(n * c_in, c_in; rng = rng),   # n*c_in → c_in
    )
end

function (m::TransitionBlock)(x::AbstractArray{<:Real})
    return transition(x, m.layernorm1, m.linear_no_bias_a, m.linear_no_bias_b, m.linear_no_bias)
end

# ─── PairformerBlock ───────────────────────────────────────────────────────────
# Features-first: s (c_s, N), z (c_z, N, N)

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
    transition_n::Int = 4,
    rng::AbstractRNG = Random.default_rng(),
)
    apb = c_s > 0 ? PairAttentionNoS(c_s, c_z; n_heads = n_heads, rng = rng) : nothing
    st = c_s > 0 ? TransitionBlock(c_s; n = transition_n, rng = rng) : nothing
    return PairformerBlock(
        c_s,
        TriangleMultiplication(c_z, c_hidden_mul; outgoing = true, rng = rng),
        TriangleMultiplication(c_z, c_hidden_mul; outgoing = false, rng = rng),
        TriangleAttention(c_z, c_hidden_pair_att * no_heads_pair, no_heads_pair; starting = true, rng = rng),
        TriangleAttention(c_z, c_hidden_pair_att * no_heads_pair, no_heads_pair; starting = false, rng = rng),
        TransitionBlock(c_z; n = transition_n, rng = rng),
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

# ─── PairformerStack ───────────────────────────────────────────────────────────

@concrete struct PairformerStack
    blocks
end
@layer PairformerStack

function PairformerStack(
    c_z::Int,
    c_s::Int;
    n_blocks::Int,
    n_heads::Int = 16,
    transition_n::Int = 4,
    rng::AbstractRNG = Random.default_rng(),
)
    n_blocks >= 0 || error("n_blocks must be >= 0")
    blocks = PairformerBlock[]
    for _ in 1:n_blocks
        push!(blocks, PairformerBlock(c_z, c_s; n_heads = n_heads, transition_n = transition_n, rng = rng))
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

# ─── MSAPairWeightedAveraging ─────────────────────────────────────────────────
# Features-first: msa (c_m, n_tok, n_msa), z (c_z, n_tok, n_tok)

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
        LinearNoBias(c_m, c * n_heads; rng = rng),   # c_m → H*C
        LayerNorm(c_z),
        LinearNoBias(c_z, n_heads; rng = rng),        # c_z → n_heads
        LinearNoBias(c_m, c * n_heads; rng = rng),   # c_m → H*C
        LinearNoBias(c * n_heads, c_m; rng = rng),   # H*C → c_m
    )
end

function (m::MSAPairWeightedAveraging)(
    msa::AbstractArray{<:Real,3},
    z::AbstractArray{<:Real,3},
)
    # msa: (c_m, n_tok, n_msa), z: (c_z, n_tok, n_tok) features-first
    c_m, n_tok, n_msa = size(msa)
    c_m == m.c_m || error("MSAPairWeightedAveraging c_m mismatch")
    size(z) == (m.c_z, n_tok, n_tok) || error("MSAPairWeightedAveraging z shape mismatch")

    msa_ln = m.layernorm_m(msa)                # (c_m, n_tok, n_msa)
    v_lin = m.linear_no_bias_mv(msa_ln)        # (H*C, n_tok, n_msa)
    b = m.linear_no_bias_z(m.layernorm_z(z))   # (H, n_tok, n_tok) — pair weights
    g_lin = 1f0 ./ (1f0 .+ exp.(-m.linear_no_bias_mg(msa_ln)))  # (H*C, n_tok, n_msa)

    # Softmax over last dim (key token dimension) of (H, n_tok_q, n_tok_k)
    w = softmax_lastdim(b)  # (H, n_tok, n_tok)

    # Batched weighted sum over all heads simultaneously:
    # o[c, i, s] (for head h) = sum_j v_h[c, j, s] * w[h, i, j]
    # Reshape v_lin: (H*C, n_tok, n_msa) → (C, H, n_tok, n_msa) → (C*n_msa, n_tok, H)
    v4 = reshape(v_lin, m.c, m.n_heads, n_tok, n_msa)
    v_flat = reshape(permutedims(v4, (1, 4, 3, 2)), m.c * n_msa, n_tok, m.n_heads)
    # w transposed per head: (n_tok, n_tok, H)
    w_t = permutedims(w, (3, 2, 1))  # w_t[:,:,h] = w[h,:,:]^T
    # Single batched BLAS call
    o_flat = NNlib.batched_mul(v_flat, w_t)  # (C*n_msa, n_tok, H)
    # Reshape back: (C, n_msa, n_tok, H) → (C, H, n_tok, n_msa) → (H*C, n_tok, n_msa)
    o_lin = reshape(permutedims(reshape(o_flat, m.c, n_msa, n_tok, m.n_heads), (1, 4, 3, 2)),
                    m.n_heads * m.c, n_tok, n_msa)
    o_lin = o_lin .* g_lin

    return m.linear_no_bias_out(o_lin)  # (c_m, n_tok, n_msa)
end

# ─── MSAStack ─────────────────────────────────────────────────────────────────

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

# ─── MSABlock ─────────────────────────────────────────────────────────────────

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

# ─── MSAModule ────────────────────────────────────────────────────────────────
# Features-first: z (c_z, N, N), s_inputs (c_s_inputs, N)

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
        LinearNoBias(34, c_m; rng = rng),          # 34 → c_m
        LinearNoBias(c_s_inputs, c_m; rng = rng),  # c_s_inputs → c_m
        blocks,
    )
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

    # feat.msa: (N_tok, N_msa) features-first
    msa_raw = feat.msa
    size(msa_raw, 1) > 0 || return _as_f32_array(z)

    n_msa = size(msa_raw, 2)  # MSA sequences in dim 2
    idx = sample_msa_indices(
        n_msa;
        cutoff = m.sample_cutoff,
        lower_bound = m.sample_lower_bound,
        strategy = m.sample_strategy,
        rng = rng,
    )
    isempty(idx) && return _as_f32_array(z)

    # Select MSA columns: (N_tok, n_sel)
    msa_sel = Int.(Array(msa_raw[:, idx]))
    del_mask = _as_f32_array(feat.has_deletion[:, idx])   # (N_tok, n_sel)
    del_val = _as_f32_array(feat.deletion_value[:, idx])  # (N_tok, n_sel)

    # one_hot_int on (N_tok, n_sel) → (32, N_tok, n_sel) features-first
    msa_onehot_cpu = one_hot_int(msa_sel, 32)
    msa_onehot = z isa Array ? msa_onehot_cpu : copyto!(similar(z, Float32, size(msa_onehot_cpu)...), msa_onehot_cpu)

    # Concatenate along dim=1: (32, N_tok, n_sel) + (1, N_tok, n_sel) + (1, N_tok, n_sel) → (34, N_tok, n_sel)
    n_tok_msa, n_sel = size(del_mask)
    msa_feat = cat(
        msa_onehot,
        reshape(del_mask, 1, n_tok_msa, n_sel),
        reshape(del_val, 1, n_tok_msa, n_sel);
        dims = 1,
    )

    msa_emb = m.linear_no_bias_m(msa_feat)  # (c_m, N_tok, n_sel)
    s_proj = m.linear_no_bias_s(_as_f32_array(s_inputs))  # (c_m, N_tok)
    msa_emb .+= reshape(s_proj, size(s_proj, 1), size(s_proj, 2), 1)  # broadcast over MSA dim

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
    size(msa_raw, 1) > 0 || return _as_f32_array(z)

    haskey(input_feature_dict, "has_deletion") || error("Missing 'has_deletion' for MSAModule")
    haskey(input_feature_dict, "deletion_value") || error("Missing 'deletion_value' for MSAModule")

    # Dict values are features-last from Python: msa (N_msa, N_tok)
    # Transpose to features-first: (N_tok, N_msa)
    msa_ff = permutedims(Int.(msa_raw))
    n_msa = size(msa_ff, 2)
    idx = sample_msa_indices(
        n_msa;
        cutoff = m.sample_cutoff,
        lower_bound = m.sample_lower_bound,
        strategy = m.sample_strategy,
        rng = rng,
    )
    isempty(idx) && return _as_f32_array(z)

    msa_sel = Int.(Array(msa_ff[:, idx]))  # (N_tok, n_sel)
    del_mask_raw = _as_f32_array(input_feature_dict["has_deletion"])
    del_val_raw = _as_f32_array(input_feature_dict["deletion_value"])
    del_mask = permutedims(del_mask_raw)[:, idx]  # → (N_tok, n_sel)
    del_val = permutedims(del_val_raw)[:, idx]    # → (N_tok, n_sel)

    msa_onehot_cpu = one_hot_int(msa_sel, 32)  # (32, N_tok, n_sel)
    msa_onehot = z isa Array ? msa_onehot_cpu : copyto!(similar(z, Float32, size(msa_onehot_cpu)...), msa_onehot_cpu)

    n_tok_msa, n_sel = size(del_mask)
    msa_feat = cat(
        msa_onehot,
        reshape(del_mask, 1, n_tok_msa, n_sel),
        reshape(del_val, 1, n_tok_msa, n_sel);
        dims = 1,
    )

    msa_emb = m.linear_no_bias_m(msa_feat)
    s_proj = m.linear_no_bias_s(_as_f32_array(s_inputs))
    msa_emb .+= reshape(s_proj, size(s_proj, 1), size(s_proj, 2), 1)

    z_cur = _as_f32_copy(z)
    msa_cur = msa_emb
    for blk in m.blocks
        msa_cur, z_cur = blk(msa_cur, z_cur; pair_mask = pair_mask)
    end
    return z_cur
end

# ─── TemplateEmbedder ─────────────────────────────────────────────────────────
# Template branch is currently disabled in reference; structure kept for checkpoint compat.

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
        LinearNoBias(c_z, c; rng = rng),                          # c_z → c
        LinearNoBias(39 + 1 + 3 + 1 + 32 + 32, c; rng = rng),   # 108 → c
        PairformerStack(c, 0; n_blocks = n_blocks, n_heads = 4, transition_n = 2, rng = rng),
        LayerNorm(c),
        LinearNoBias(c, c_z; rng = rng),                          # c → c_z
    )
end

function _template_embedder_core(
    m::TemplateEmbedder,
    z::AbstractArray{<:Real,3},
    template_restype::AbstractMatrix{<:Integer},
    template_distogram::AbstractArray{<:Real},
    template_unit_vector::AbstractArray{<:Real},
    template_pseudo_beta_mask::AbstractArray{<:Real},
    template_backbone_frame_mask::AbstractArray{<:Real},
    asym_id::AbstractVector{<:Integer};
    pair_mask::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
)
    # Features-first: z (c_z, N, N)
    T = Float32
    n_token = size(z, 2)

    # Build multichain mask: same asymmetric unit → 1, different → 0
    multichain_mask = T.(reshape(asym_id, n_token, 1) .== reshape(asym_id, 1, n_token))
    pair_mask_plane = pair_mask === nothing ? ones(T, n_token, n_token) : T.(pair_mask)

    z_ln = m.layernorm_z(z)
    z_proj = m.linear_no_bias_z(z_ln)    # (c, N, N)

    # template_restype: (N_template, N_token)
    num_templates = size(template_restype, 1)
    u = fill!(similar(z, T, (m.c, n_token, n_token)), zero(T))

    for template_id in 1:num_templates
        # Distogram: expect features-first (39, N, N) per template
        dgram = T.(selectdim(template_distogram, 1, template_id))
        if ndims(dgram) == 3 && size(dgram, 1) != 39
            dgram = permutedims(dgram, (3, 1, 2))  # (N,N,39) → (39,N,N)
        end

        pb_mask_2d = T.(selectdim(template_pseudo_beta_mask, 1, template_id))
        aatype = Int.(selectdim(template_restype, 1, template_id))
        unit_vec = T.(selectdim(template_unit_vector, 1, template_id))
        if ndims(unit_vec) == 3 && size(unit_vec, 1) != 3
            unit_vec = permutedims(unit_vec, (3, 1, 2))  # (N,N,3) → (3,N,N)
        end
        bb_mask_2d = T.(selectdim(template_backbone_frame_mask, 1, template_id))

        # Apply masks
        mask_3d = reshape(multichain_mask .* pair_mask_plane, 1, n_token, n_token)
        dgram = dgram .* mask_3d
        pb_mask_2d = pb_mask_2d .* multichain_mask .* pair_mask_plane
        unit_vec = unit_vec .* mask_3d
        bb_mask_2d = bb_mask_2d .* multichain_mask .* pair_mask_plane

        # One-hot aatype → (32, N)
        aatype_oh = zeros(T, 32, n_token)
        for j in 1:n_token
            idx = aatype[j]
            if 0 <= idx < 32
                aatype_oh[idx + 1, j] = one(T)
            end
        end
        aatype_i = repeat(reshape(aatype_oh, 32, n_token, 1), 1, 1, n_token)
        aatype_j = repeat(reshape(aatype_oh, 32, 1, n_token), 1, n_token, 1)

        # Concatenate template features: (108, N, N) features-first
        at = cat(
            dgram,                                                    # (39, N, N)
            reshape(pb_mask_2d, 1, n_token, n_token),                # (1, N, N)
            aatype_i,                                                 # (32, N, N)
            aatype_j,                                                 # (32, N, N)
            unit_vec,                                                 # (3, N, N)
            reshape(bb_mask_2d, 1, n_token, n_token);                # (1, N, N)
            dims = 1,
        )

        v = z_proj .+ m.linear_no_bias_a(at)   # (c, N, N)
        _, v = m.pairformer_stack(nothing, v)
        u .+= m.layernorm_v(v)
    end

    u ./= (T(1e-7) + T(num_templates))
    u = m.linear_no_bias_u(max.(u, zero(T)))   # (c_z, N, N), ReLU before final projection
    return u
end

function (m::TemplateEmbedder)(
    feat::ProtenixFeatures,
    z::AbstractArray{<:Real,3};
    pair_mask::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
)
    if m.n_blocks < 1 || feat.template_restype === nothing
        return fill!(similar(z, Float32, size(z)), 0f0)
    end
    # Template features from ProtenixFeatures not yet wired for full template embedding.
    # Return zeros when template features haven't been fully prepared.
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
    # Check for v1-style template features needed by the actual embedder
    required_keys = ("template_distogram", "template_unit_vector",
                     "template_pseudo_beta_mask", "template_backbone_frame_mask")
    has_all = all(k -> haskey(input_feature_dict, k), required_keys)
    if !has_all
        return fill!(similar(z, Float32, size(z)), 0f0)
    end
    asym_id = haskey(input_feature_dict, "asym_id") ? Int.(input_feature_dict["asym_id"]) :
              ones(Int, size(z, 2))
    return _template_embedder_core(
        m, z,
        Int.(input_feature_dict["template_restype"]),
        input_feature_dict["template_distogram"],
        input_feature_dict["template_unit_vector"],
        input_feature_dict["template_pseudo_beta_mask"],
        input_feature_dict["template_backbone_frame_mask"],
        asym_id;
        pair_mask = pair_mask,
    )
end

# ─── NoisyStructureEmbedder ──────────────────────────────────────────────────
# Features-first: z (c_z, N, N), coords (3, N)

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
        LinearNoBias(no_bins + 1, c; rng = rng),  # no_bins+1 → c
        LayerNorm(c_z),
        LinearNoBias(c_z, c; rng = rng),           # c_z → c
        TransitionBlock(c_z; n = 2, rng = rng),
    )
end

"""
Pairwise binned distance one-hot, features-first.
Input: (3, N), output: (no_bins, N, N).
"""
function _one_hot_binned_sqdist(
    x::AbstractMatrix{<:Real},
    bins::AbstractVector{Float32},
    upper_bins::AbstractVector{Float32},
)
    size(x, 1) == 3 || error("Expected (3, N) input for _one_hot_binned_sqdist")
    n = size(x, 2)
    xf = Float32.(x)
    # (3, N, 1) - (3, 1, N) → (3, N, N)
    diff = reshape(xf, 3, n, 1) .- reshape(xf, 3, 1, n)
    d = dropdims(sqrt.(sum(diff .^ 2; dims=1)); dims=1)  # (N, N)
    nb = length(bins)
    d3 = reshape(d, 1, n, n)  # (1, N, N)
    lo = reshape(bins, nb, 1, 1)
    hi = reshape(upper_bins, nb, 1, 1)
    return Float32.((d3 .> lo) .& (d3 .< hi))  # (nb, N, N) features-first
end

function _noisy_structure_forward(
    m::NoisyStructureEmbedder,
    z::AbstractArray{<:Real,3},
    x::AbstractMatrix{<:Real},
    mask::AbstractVector{Bool},
)
    any(mask) || return fill!(similar(z, Float32, size(z)), 0f0)

    _, n, _ = size(z)
    mf_cpu = Float32.(mask)
    # pair_mask: (1, N, N) for broadcasting with features-first tensors
    pair_mask_cpu = reshape(mf_cpu, 1, n, 1) .* reshape(mf_cpu, 1, 1, n)
    pair_mask = z isa Array ? pair_mask_cpu : copyto!(similar(z, Float32, size(pair_mask_cpu)...), pair_mask_cpu)

    d = _one_hot_binned_sqdist(x, m.bins, m.upper_bins) .* pair_mask  # (nb, N, N)
    d = cat(d, pair_mask; dims = 1)  # (nb+1, N, N)
    d = m.linear_struct(d)  # (c/2, N, N)

    z_proj = m.linear_z(m.layernorm_z(z))  # (c/2, N, N)
    z_cat = cat(z_proj, d; dims = 1)  # (c_z, N, N)
    out = m.transition_out(z_cat)  # (c_z, N, N)

    return out
end

function (m::NoisyStructureEmbedder)(
    feat::ProtenixFeatures,
    z::AbstractArray{<:Real,3},
)
    _, n, _ = size(z)
    # struct_cb_coords is (3, N) features-first in ProtenixFeatures
    x = feat.struct_cb_coords === nothing ? fill!(similar(z, Float32, 3, n), 0f0) : Float32.(feat.struct_cb_coords)
    mask = feat.struct_cb_mask === nothing ? falses(n) : Bool.(Array(feat.struct_cb_mask))
    return _noisy_structure_forward(m, z, x, mask)
end

function (m::NoisyStructureEmbedder)(
    input_feature_dict::AbstractDict{<:AbstractString, <:Any},
    z::AbstractArray{<:Real,3},
)
    _, n, _ = size(z)
    x = if haskey(input_feature_dict, "struct_cb_coords")
        # Dict values from Python are features-last (N, 3); transpose to (3, N)
        raw = _as_f32_array(input_feature_dict["struct_cb_coords"])
        ndims(raw) == 2 && size(raw, 2) == 3 ? permutedims(raw) : raw
    else
        fill!(similar(z, Float32, 3, n), 0f0)
    end
    mask = if haskey(input_feature_dict, "struct_cb_mask")
        Bool.(Array(input_feature_dict["struct_cb_mask"]))
    else
        falses(n)
    end
    return _noisy_structure_forward(m, z, x, mask)
end

end
