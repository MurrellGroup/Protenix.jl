module TransformerBlocks

using Random
using ConcreteStructs
using Flux: @layer
using NNlib
import Onion

import ..Primitives: AdaptiveLayerNorm, LayerNormFirst, BGLinear, LinearNoBias, silu

export ConditionedTransitionBlock, AttentionPairBias, DiffusionTransformerBlock, DiffusionTransformer

# ---------------------------------------------------------------------------
# ConditionedTransitionBlock — features-first
# ---------------------------------------------------------------------------
@concrete struct ConditionedTransitionBlock
    c_a
    c_s
    n
    adaln
    linear_a1
    linear_a2
    linear_b
    linear_s     # gate projection: c_s → c_a, with bias
end
@layer ConditionedTransitionBlock

function ConditionedTransitionBlock(
    c_a::Int,
    c_s::Int;
    n::Int = 2,
    biasinit::Real = -2.0,
    rng::AbstractRNG = Random.default_rng(),
)
    c_a > 0 || error("c_a must be positive.")
    c_s > 0 || error("c_s must be positive.")
    n > 0 || error("n must be positive.")
    gate_proj = BGLinear(c_s, c_a; bias = true)
    gate_proj.bias .= Float32(biasinit)
    return ConditionedTransitionBlock(
        c_a,
        c_s,
        n,
        AdaptiveLayerNorm(c_a, c_s; rng = rng),
        LinearNoBias(c_a, n * c_a),
        LinearNoBias(c_a, n * c_a),
        LinearNoBias(n * c_a, c_a),
        gate_proj,
    )
end

function (blk::ConditionedTransitionBlock)(a::AbstractArray{<:Real}, s::AbstractArray{<:Real})
    # a: (c_a, N, ...), s: (c_s, N, ...)
    a0 = blk.adaln(a, s)
    b = silu(blk.linear_a1(a0)) .* blk.linear_a2(a0)
    proj = blk.linear_b(b)
    gate = NNlib.sigmoid.(blk.linear_s(s))
    return gate .* proj
end

# ---------------------------------------------------------------------------
# AttentionPairBias — wrapper around Onion.AttentionPairBias
# ---------------------------------------------------------------------------
@concrete struct AttentionPairBias
    n_heads
    c_a
    c_s
    c_z
    cross_attention_mode
    adaln_a
    adaln_kv           # nothing when cross_attention_mode=false
    pair_bias_norm     # LayerNormFirst for z projection
    pair_bias_proj     # BGLinear(c_z, n_heads; bias=false)
    attn               # Onion.AttentionPairBias (compute_pair_bias=false)
    output_gate        # BGLinear(c_s, c_a; bias=true) — sigmoid gate on output
end
@layer AttentionPairBias

function AttentionPairBias(
    c_a::Int,
    c_s::Int,
    c_z::Int;
    n_heads::Int = 16,
    biasinit::Real = -2.0,
    cross_attention_mode::Bool = false,
    rng::AbstractRNG = Random.default_rng(),
)
    c_a % n_heads == 0 || error("c_a must be divisible by n_heads.")
    adaln_kv = cross_attention_mode ? AdaptiveLayerNorm(c_a, c_s; rng = rng) : nothing
    gate_proj = BGLinear(c_s, c_a; bias = true)
    gate_proj.bias .= Float32(biasinit)
    return AttentionPairBias(
        n_heads,
        c_a,
        c_s,
        c_z,
        cross_attention_mode,
        AdaptiveLayerNorm(c_a, c_s; rng = rng),
        adaln_kv,
        LayerNormFirst(c_z),
        LinearNoBias(c_z, n_heads),
        Onion.AttentionPairBias(c_a, c_z, n_heads; compute_pair_bias = false),
        gate_proj,
    )
end

# Windowing helpers for local-trunk atom attention
function _window_queries(a::AbstractArray{Float32}, n_queries::Int)
    # a: (c, N) — window into (c, n_queries, n_trunks)
    c = size(a, 1)
    N = size(a, 2)
    n_trunks = cld(N, n_queries)
    padded_N = n_trunks * n_queries
    if N < padded_N
        pad = fill!(similar(a, Float32, c, padded_N - N), 0f0)
        a = cat(a, pad; dims = 2)
    end
    return reshape(a, c, n_queries, n_trunks)
end

function _window_keys(a::AbstractArray{Float32}, n_queries::Int, n_keys::Int)
    # a: (c, N) — extract overlapping key windows: (c, n_keys, n_trunks)
    c = size(a, 1)
    N = size(a, 2)
    n_trunks = cld(N, n_queries)
    left = div(n_keys - n_queries, 2)
    total_padded = (n_trunks - 1) * n_queries + n_keys
    right_pad = total_padded - N - left

    a_cpu = Array(a)
    a_padded = cat(
        zeros(Float32, c, left),
        a_cpu,
        zeros(Float32, c, max(0, right_pad));
        dims = 2,
    )

    out = zeros(Float32, c, n_keys, n_trunks)
    for b in 1:n_trunks
        start = (b - 1) * n_queries + 1
        out[:, :, b] = a_padded[:, start:(start + n_keys - 1)]
    end

    return copyto!(similar(a, Float32, size(out)...), out)
end

function _create_window_mask(n_atom::Int, n_queries::Int, n_keys::Int)
    # Returns (n_keys, n_trunks) mask: 1=valid, 0=padding
    n_trunks = cld(n_atom, n_queries)
    left = div(n_keys - n_queries, 2)
    mask = zeros(Float32, n_keys, n_trunks)
    for b in 1:n_trunks
        k_start = (b - 1) * n_queries + 1 - left
        for kl in 1:n_keys
            k_idx = k_start + kl - 1
            if 1 <= k_idx <= n_atom
                mask[kl, b] = 1f0
            end
        end
    end
    return mask
end

function _window_pair_bias(z_bias::AbstractArray{Float32}, n_queries::Int, n_keys::Int)
    # z_bias: (n_heads, N_q, N_k) or (n_heads, N_q, N_k, B)
    # Output: (n_heads, n_queries, n_keys, n_trunks) or (n_heads, n_queries, n_keys, n_trunks * B)
    n_heads = size(z_bias, 1)
    N = size(z_bias, 2)
    has_batch = ndims(z_bias) == 4
    B = has_batch ? size(z_bias, 4) : 1
    n_trunks = cld(N, n_queries)
    left = div(n_keys - n_queries, 2)

    padded_q = n_trunks * n_queries
    total_padded_k = (n_trunks - 1) * n_queries + n_keys
    right_pad_k = total_padded_k - N - left

    if has_batch
        out = zeros(Float32, n_heads, n_queries, n_keys, n_trunks * B)
        for bi in 1:B
            z_cpu = Array(z_bias[:, :, :, bi])
            if padded_q > N
                z_cpu = cat(z_cpu, zeros(Float32, n_heads, padded_q - N, size(z_cpu, 3)); dims = 2)
            end
            z_cpu = cat(
                zeros(Float32, n_heads, size(z_cpu, 2), left),
                z_cpu,
                zeros(Float32, n_heads, size(z_cpu, 2), max(0, right_pad_k));
                dims = 3,
            )
            for t in 1:n_trunks
                q_start = (t - 1) * n_queries + 1
                k_start = (t - 1) * n_queries + 1
                out[:, :, :, (bi - 1) * n_trunks + t] = z_cpu[:, q_start:(q_start + n_queries - 1), k_start:(k_start + n_keys - 1)]
            end
        end
        return copyto!(similar(z_bias, Float32, size(out)...), out)
    else
        z_cpu = Array(z_bias)
        if padded_q > N
            z_cpu = cat(z_cpu, zeros(Float32, n_heads, padded_q - N, size(z_cpu, 3)); dims = 2)
        end
        z_cpu = cat(
            zeros(Float32, n_heads, size(z_cpu, 2), left),
            z_cpu,
            zeros(Float32, n_heads, size(z_cpu, 2), max(0, right_pad_k));
            dims = 3,
        )
        out = zeros(Float32, n_heads, n_queries, n_keys, n_trunks)
        for t in 1:n_trunks
            q_start = (t - 1) * n_queries + 1
            k_start = (t - 1) * n_queries + 1
            out[:, :, :, t] = z_cpu[:, q_start:(q_start + n_queries - 1), k_start:(k_start + n_keys - 1)]
        end
        return copyto!(similar(z_bias, Float32, size(out)...), out)
    end
end

function _unwindow(out_windowed::AbstractArray{Float32,3}, n_atom::Int)
    # out_windowed: (c, n_queries, n_trunks) → (c, n_atom)
    c = size(out_windowed, 1)
    flat = reshape(out_windowed, c, :)
    return flat[:, 1:n_atom]
end

function (blk::AttentionPairBias)(
    a::AbstractArray{<:Real},
    s::AbstractArray{<:Real},
    z::AbstractArray{<:Real},
    mask::AbstractArray{<:Real};
    n_queries::Union{Nothing, Int} = nothing,
    n_keys::Union{Nothing, Int} = nothing,
)
    # Features-first convention:
    # Dense: a (c_a, N, B), s (c_s, N, B), z (c_z, N, N, B), mask (N, B)
    # Windowed: a (c_a, N), s (c_s, N), z (c_z, n_q, n_k, n_trunks), mask (N,)

    a_f = Float32.(a)
    s_f = Float32.(s)
    a_norm = blk.adaln_a(a_f, s_f)
    kv_norm = if blk.cross_attention_mode && blk.adaln_kv !== nothing
        blk.adaln_kv(a_norm, s_f)
    else
        a_norm
    end

    # Compute pair bias from z
    z_f = Float32.(z)
    z_bias = blk.pair_bias_proj(blk.pair_bias_norm(z_f))  # (n_heads, ...)

    if n_queries !== nothing && n_keys !== nothing
        # Windowed attention for atom transformer
        n_atom = size(a_f, 2)
        q_win = _window_queries(a_norm, n_queries)     # (c_a, n_queries, n_trunks)
        kv_win = _window_keys(kv_norm, n_queries, n_keys)  # (c_a, n_keys, n_trunks)
        # z_bias is already windowed if input z was pre-windowed (4D, matching n_queries x n_keys);
        # otherwise window it from the full (n_heads, N, N) pair bias.
        if ndims(z_bias) == 4 && size(z_bias, 2) == n_queries && size(z_bias, 3) == n_keys
            z_bias_win = z_bias
        else
            z_bias_win = _window_pair_bias(z_bias, n_queries, n_keys)
        end
        mask_win_cpu = _create_window_mask(n_atom, n_queries, n_keys) # (n_keys, n_trunks)
        mask_win = copyto!(similar(a_f, Float32, size(mask_win_cpu)...), mask_win_cpu)
        attn_out_win = blk.attn(q_win, z_bias_win, mask_win, kv_win)  # (c_a, n_queries, n_trunks)
        attn_out = _unwindow(attn_out_win, n_atom)  # (c_a, n_atom)
    else
        # Dense attention
        attn_out = blk.attn(a_norm, z_bias, mask, kv_norm)
    end

    gate = NNlib.sigmoid.(blk.output_gate(s_f))
    return gate .* attn_out
end

# Backward-compatible call without explicit mask (creates all-ones mask)
function (blk::AttentionPairBias)(
    a::AbstractArray{<:Real},
    s::AbstractArray{<:Real},
    z::AbstractArray{<:Real};
    n_queries::Union{Nothing, Int} = nothing,
    n_keys::Union{Nothing, Int} = nothing,
)
    # For dense attention: a (c_a, N, B), infer mask from a
    if n_queries !== nothing && n_keys !== nothing
        # Windowed — no explicit mask needed, windowing creates its own
        n_atom = size(a, 2)
        mask = ones(Float32, n_atom)
        mask_dev = copyto!(similar(Float32.(a), Float32, size(mask)...), mask)
        return blk(a, s, z, mask_dev; n_queries = n_queries, n_keys = n_keys)
    end
    # Dense attention — create all-ones mask
    input_was_2d = ndims(a) == 2
    if ndims(a) == 3
        N, B = size(a, 2), size(a, 3)
        mask = ones(Float32, N, B)
    else
        N = size(a, 2)
        mask = ones(Float32, N, 1)
    end
    mask_dev = copyto!(similar(Float32.(a), Float32, size(mask)...), mask)
    result = blk(a, s, z, mask_dev)
    # If input was 2D, Onion adds batch=1 — squeeze it back
    if input_was_2d && ndims(result) == 3 && size(result, 3) == 1
        return dropdims(result; dims=3)
    end
    return result
end

# ---------------------------------------------------------------------------
# DiffusionTransformerBlock
# ---------------------------------------------------------------------------
@concrete struct DiffusionTransformerBlock
    attention_pair_bias
    conditioned_transition_block
end
@layer DiffusionTransformerBlock

function DiffusionTransformerBlock(
    c_a::Int,
    c_s::Int,
    c_z::Int;
    n_heads::Int = 16,
    biasinit::Real = -2.0,
    cross_attention_mode::Bool = false,
    rng::AbstractRNG = Random.default_rng(),
)
    return DiffusionTransformerBlock(
        AttentionPairBias(
            c_a,
            c_s,
            c_z;
            n_heads = n_heads,
            biasinit = biasinit,
            cross_attention_mode = cross_attention_mode,
            rng = rng,
        ),
        ConditionedTransitionBlock(c_a, c_s; n = 2, biasinit = biasinit, rng = rng),
    )
end

function (blk::DiffusionTransformerBlock)(
    a::AbstractArray{<:Real},
    s::AbstractArray{<:Real},
    z::AbstractArray{<:Real},
    mask::AbstractArray{<:Real};
    n_queries::Union{Nothing, Int} = nothing,
    n_keys::Union{Nothing, Int} = nothing,
)
    a_f = Float32.(a)
    attn = blk.attention_pair_bias(a_f, s, z, mask; n_queries = n_queries, n_keys = n_keys)
    a_f = a_f .+ attn
    tr = blk.conditioned_transition_block(a_f, s)
    a_f = a_f .+ tr
    return a_f
end

# ---------------------------------------------------------------------------
# DiffusionTransformer
# ---------------------------------------------------------------------------
@concrete struct DiffusionTransformer
    blocks
end
@layer DiffusionTransformer

function DiffusionTransformer(
    c_a::Int,
    c_s::Int,
    c_z::Int;
    n_blocks::Int = 16,
    n_heads::Int = 16,
    cross_attention_mode::Bool = false,
    rng::AbstractRNG = Random.default_rng(),
)
    n_blocks > 0 || error("DiffusionTransformer n_blocks must be positive.")
    blocks = DiffusionTransformerBlock[]
    for _ in 1:n_blocks
        push!(
            blocks,
            DiffusionTransformerBlock(
                c_a,
                c_s,
                c_z;
                n_heads = n_heads,
                cross_attention_mode = cross_attention_mode,
                rng = rng,
            ),
        )
    end
    return DiffusionTransformer(blocks)
end

function (tr::DiffusionTransformer)(
    a::AbstractArray{<:Real},
    s::AbstractArray{<:Real},
    z::AbstractArray{<:Real},
    mask::AbstractArray{<:Real};
    n_queries::Union{Nothing, Int} = nothing,
    n_keys::Union{Nothing, Int} = nothing,
)
    a_f = Float32.(a)
    s_f = Float32.(s)
    z_f = Float32.(z)
    for blk in tr.blocks
        a_f = blk(a_f, s_f, z_f, mask; n_queries = n_queries, n_keys = n_keys)
    end
    return a_f
end

# Convenience: no explicit mask
function (tr::DiffusionTransformer)(
    a::AbstractArray{<:Real},
    s::AbstractArray{<:Real},
    z::AbstractArray{<:Real};
    n_queries::Union{Nothing, Int} = nothing,
    n_keys::Union{Nothing, Int} = nothing,
)
    input_was_2d = ndims(a) == 2
    if ndims(a) == 3
        N, B = size(a, 2), size(a, 3)
        mask = ones(Float32, N, B)
    elseif ndims(a) == 2
        N = size(a, 2)
        mask = ones(Float32, N, 1)
    else
        error("DiffusionTransformer: a must be rank 2 or 3")
    end
    mask_dev = copyto!(similar(Float32.(a), Float32, size(mask)...), mask)
    result = tr(a, s, z, mask_dev; n_queries = n_queries, n_keys = n_keys)
    if input_was_2d && ndims(result) == 3 && size(result, 3) == 1
        return dropdims(result; dims=3)
    end
    return result
end

end
