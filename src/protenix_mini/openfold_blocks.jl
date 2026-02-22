module OpenFoldBlocks

using Random
using ConcreteStructs
using Flux: @layer
using NNlib
import Onion: flash_attention_forward, flash_attention_bias_forward

import ..Primitives: Linear, LinearNoBias, LayerNorm

export TriangleMultiplication,
    TriangleAttention,
    OuterProductMean,
    PairAttentionNoS

"""
Pair attention used in Pairformer when `has_s=false`.
Features-first: a (c_a, N), z (c_z, N, N).
"""
@concrete struct PairAttentionNoS
    n_heads
    c_a
    c_z
    layernorm_a
    layernorm_z
    linear_nobias_z
    linear_q
    linear_k
    linear_v
    linear_o
    linear_g
end
@layer PairAttentionNoS

function PairAttentionNoS(
    c_a::Int,
    c_z::Int;
    n_heads::Int,
    rng::AbstractRNG = Random.default_rng(),
)
    c_a % n_heads == 0 || error("c_a must be divisible by n_heads")
    return PairAttentionNoS(
        n_heads,
        c_a,
        c_z,
        LayerNorm(c_a),
        LayerNorm(c_z),
        LinearNoBias(c_z, n_heads; rng = rng),
        Linear(c_a, c_a; bias = true),
        LinearNoBias(c_a, c_a; rng = rng),
        LinearNoBias(c_a, c_a; rng = rng),
        LinearNoBias(c_a, c_a; rng = rng),
        LinearNoBias(c_a, c_a; rng = rng),
    )
end

function (m::PairAttentionNoS)(
    a::AbstractMatrix{<:Real},
    z::AbstractArray{<:Real,3},
    pair_mask::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
)
    # a: (c_a, N), z: (c_z, N, N)
    c_a_in = size(a, 1)
    n = size(a, 2)
    c_a_in == m.c_a || error("PairAttentionNoS a channel mismatch")
    size(z) == (m.c_z, n, n) || error("PairAttentionNoS z shape mismatch")

    a_ln = m.layernorm_a(a)   # (c_a, N)
    z_ln = m.layernorm_z(z)   # (c_z, N, N)

    q = m.linear_q(a_ln)      # (c_a, N) = (d*H, N)
    k = m.linear_k(a_ln)
    v = m.linear_v(a_ln)
    g = 1f0 ./ (1f0 .+ exp.(-m.linear_g(a_ln)))  # (c_a, N)

    z_bias = m.linear_nobias_z(z_ln) # (n_heads, N, N)
    if pair_mask === nothing
        pair_bias = fill!(similar(q, Float32, n, n), 0f0)
    else
        size(pair_mask) == (n, n) || error("pair_mask shape mismatch")
        pair_bias = (Float32.(pair_mask) .- 1f0) .* 1f9
    end

    H = m.n_heads
    d = div(m.c_a, H)

    # Reshape (d*H, N) → (d, N, H, 1) for flash attention
    q4 = reshape(permutedims(reshape(q, d, H, n), (1, 3, 2)), d, n, H, 1)
    k4 = reshape(permutedims(reshape(k, d, H, n), (1, 3, 2)), d, n, H, 1)
    v4 = reshape(permutedims(reshape(v, d, H, n), (1, 3, 2)), d, n, H, 1)

    # Combined bias in (H, Q, K) → flash format (K, Q, H, B=1)
    combined_bias = z_bias .+ reshape(pair_bias, 1, n, n)  # (H, Q, K)
    flash_bias = reshape(permutedims(combined_bias, (3, 2, 1)), n, n, H, 1)  # (K, Q, H, 1)

    ctx4 = flash_attention_bias_forward(q4, k4, v4, flash_bias)  # (d, N, H, 1)

    # Reshape back: (d, N, H, 1) → (d, H, N) → (d*H, N) = (c_a, N)
    out = reshape(permutedims(reshape(ctx4, d, n, H), (1, 3, 2)), m.c_a, n)
    out = out .* g
    out = m.linear_o(out)
    return out
end

"""
OpenFold triangle multiplicative update, inference path only.
Features-first: z (c_z, n_i, n_j).
"""
@concrete struct TriangleMultiplication
    c_z
    c_hidden
    outgoing
    layer_norm_in
    layer_norm_out
    linear_a_p
    linear_a_g
    linear_b_p
    linear_b_g
    linear_z
    linear_g
end
@layer TriangleMultiplication

function TriangleMultiplication(
    c_z::Int,
    c_hidden::Int;
    outgoing::Bool,
    rng::AbstractRNG = Random.default_rng(),
)
    return TriangleMultiplication(
        c_z,
        c_hidden,
        outgoing,
        LayerNorm(c_z),
        LayerNorm(c_hidden),
        LinearNoBias(c_z, c_hidden; rng = rng),
        LinearNoBias(c_z, c_hidden; rng = rng),
        LinearNoBias(c_z, c_hidden; rng = rng),
        LinearNoBias(c_z, c_hidden; rng = rng),
        LinearNoBias(c_hidden, c_z; rng = rng),
        LinearNoBias(c_z, c_z; rng = rng),
    )
end

function (m::TriangleMultiplication)(
    z::AbstractArray{<:Real,3};
    mask::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
)
    # z: (c_z, n_i, n_j)
    c_z, n_i, n_j = size(z)
    n_i == n_j || error("TriangleMultiplication expects square pair tensor")
    c_z == m.c_z || error("TriangleMultiplication c_z mismatch")

    if mask === nothing
        mask_f = fill!(similar(z, Float32, n_i, n_j), 1f0)
    else
        size(mask) == (n_i, n_j) || error("TriangleMultiplication mask shape mismatch")
        mask_f = Float32.(mask)
    end

    z0 = m.layer_norm_in(z)                                        # (c_z, n_i, n_j)
    a = m.linear_a_p(z0) .* (1f0 ./ (1f0 .+ exp.(-m.linear_a_g(z0)))) # (c_hidden, n_i, n_j)
    b = m.linear_b_p(z0) .* (1f0 ./ (1f0 .+ exp.(-m.linear_b_g(z0)))) # (c_hidden, n_i, n_j)
    a = a .* reshape(mask_f, 1, n_i, n_j)
    b = b .* reshape(mask_f, 1, n_i, n_j)

    # Batched matmul over c_hidden: (c_hidden, n, n) → (n, n, c_hidden)
    a_perm = permutedims(a, (2, 3, 1))  # (n_i, n_j, c_hidden)
    b_perm = permutedims(b, (2, 3, 1))  # (n_i, n_j, c_hidden)
    if m.outgoing
        x = NNlib.batched_mul(a_perm, NNlib.batched_transpose(b_perm))
    else
        x = NNlib.batched_mul(NNlib.batched_transpose(a_perm), b_perm)
    end
    x = permutedims(x, (3, 1, 2))  # (c_hidden, n_i, n_j)

    x = m.layer_norm_out(x)    # (c_hidden, n_i, n_j)
    x = m.linear_z(x)          # (c_z, n_i, n_j)
    g = 1f0 ./ (1f0 .+ exp.(-m.linear_g(z0)))
    x = x .* g
    return x
end

"""
OpenFold triangle attention, inference path only.
Features-first: x (c_in, i_dim, j_dim).
"""
@concrete struct TriangleAttention
    c_in
    c_hidden
    n_heads
    starting
    inf
    layer_norm
    linear # c_in -> n_heads
    linear_q
    linear_k
    linear_v
    linear_o
    linear_g
end
@layer TriangleAttention

function TriangleAttention(
    c_in::Int,
    c_hidden::Int,
    n_heads::Int;
    starting::Bool,
    inf::Real = 1f9,
    rng::AbstractRNG = Random.default_rng(),
)
    c_hidden % n_heads == 0 || error("c_hidden must be divisible by n_heads")
    return TriangleAttention(
        c_in,
        c_hidden,
        n_heads,
        starting,
        Float32(inf),
        LayerNorm(c_in),
        LinearNoBias(c_in, n_heads; rng = rng),
        LinearNoBias(c_in, c_hidden; rng = rng),
        LinearNoBias(c_in, c_hidden; rng = rng),
        LinearNoBias(c_in, c_hidden; rng = rng),
        LinearNoBias(c_hidden, c_in; rng = rng),
        LinearNoBias(c_in, c_hidden; rng = rng),
    )
end

"""
Core triangle attention: batches the non-attention spatial dimension into the
flash attention batch dimension (like BoltzGen). For large N, processes in chunks
to avoid OOM from the (H, N, N, chunk) bias expansion in flash_attention_bias_forward.
"""
function (m::TriangleAttention)(
    x::AbstractArray{<:Real,3};
    mask::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
)
    # x: (c_in, N, N) — pair representation (always square)
    x_f = Float32.(x)
    c_in, n_i, n_j = size(x_f)
    n_i == n_j || error("TriangleAttention expects square pair tensor")
    c_in == m.c_in || error("TriangleAttention c_in mismatch")
    N = n_i

    # Treat as (C, I, J, B=1). Batch the non-attention dim into flash batch.
    # Starting (row): attend along J, batch over I → x_att = (C, J, 1, I)
    # Ending (col): attend along I, batch over J → x_att = (C, I, 1, J)
    x_4d = reshape(x_f, c_in, N, N, 1)
    if m.starting
        x_att = permutedims(x_4d, (1, 3, 4, 2))  # (C, J, 1, I)
    else
        x_att = permutedims(x_4d, (1, 2, 4, 3))  # (C, I, 1, J)
    end
    # x_att: (C, seq=N, 1, batch=N)

    x_att = m.layer_norm(x_att)
    tri_bias = m.linear(x_att)    # (H, N, 1, N)
    q = m.linear_q(x_att)         # (c_hidden, N, 1, N)
    k = m.linear_k(x_att)
    v = m.linear_v(x_att)
    g = 1f0 ./ (1f0 .+ exp.(-m.linear_g(x_att)))  # (c_hidden, N, 1, N)

    H = m.n_heads
    d = div(m.c_hidden, H)

    # Reshape Q/K/V: (d*H, N, 1, N) → (d, H, N, N) → (d, N, H, N) for flash
    q4 = permutedims(reshape(q, d, H, N, N), (1, 3, 2, 4))  # (d, N, H, N)
    k4 = permutedims(reshape(k, d, H, N, N), (1, 3, 2, 4))
    v4 = permutedims(reshape(v, d, H, N, N), (1, 3, 2, 4))

    # Bias: tri_bias is (H, seq=N, 1, batch=N)
    # bias_flash = permutedims(tri_bias, (2, 4, 1, 3)) = (N, N, H, 1)
    # In Onion (K, Q, H, B) format
    bias_flash = permutedims(tri_bias, (2, 4, 1, 3))  # (N, N, H, 1)

    ctx4 = flash_attention_bias_forward(q4, k4, v4, bias_flash)  # (d, N, H, N)

    # Reshape back: (d, N, H, N) → (d, H, N, N) → (d*H, N, 1, N)
    out_4d = reshape(permutedims(ctx4, (1, 3, 2, 4)), m.c_hidden, N, 1, N)
    out_4d = out_4d .* g
    out_4d = m.linear_o(out_4d)  # (c_in, N, 1, N)

    # Reverse the permutation to get back to (c_in, N, N)
    if m.starting
        out = reshape(permutedims(out_4d, (1, 4, 2, 3)), c_in, N, N)  # reverse (1,3,4,2)
    else
        out = reshape(permutedims(out_4d, (1, 2, 4, 3)), c_in, N, N)  # reverse (1,2,4,3)
    end

    return out
end

"""
OpenFold outer product mean, inference path only.
Features-first: msa (c_m, n_seq, n_res), output (c_z, n_res, n_res).
"""
@concrete struct OuterProductMean
    c_m
    c_z
    c_hidden
    eps
    layer_norm
    linear_1
    linear_2
    linear_out
end
@layer OuterProductMean

function OuterProductMean(
    c_m::Int,
    c_z::Int,
    c_hidden::Int;
    eps::Real = 1f-3,
    rng::AbstractRNG = Random.default_rng(),
)
    return OuterProductMean(
        c_m,
        c_z,
        c_hidden,
        Float32(eps),
        LayerNorm(c_m),
        LinearNoBias(c_m, c_hidden; rng = rng),
        LinearNoBias(c_m, c_hidden; rng = rng),
        Linear(c_hidden * c_hidden, c_z; bias = true),
    )
end

function (m::OuterProductMean)(
    msa::AbstractArray{<:Real,3};
    mask::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
)
    # msa: (c_m, n_tok, n_msa) features-first
    # Sum over n_msa (dim 3), pair over n_tok (dim 2) → output (c_z, n_tok, n_tok)
    c_m, n_tok, n_msa = size(msa)
    c_m == m.c_m || error("OuterProductMean c_m mismatch")

    if mask === nothing
        mask_f = fill!(similar(msa, Float32, n_tok, n_msa), 1f0)
    else
        size(mask) == (n_tok, n_msa) || error("OuterProductMean mask shape mismatch")
        mask_f = Float32.(mask)
    end

    ln = m.layer_norm(msa)    # (c_m, n_tok, n_msa)
    a = m.linear_1(ln) .* reshape(mask_f, 1, n_tok, n_msa)   # (c_hidden, n_tok, n_msa)
    b = m.linear_2(ln) .* reshape(mask_f, 1, n_tok, n_msa)   # (c_hidden, n_tok, n_msa)
    c_h = m.c_hidden

    # Outer product via single matmul contracting over n_msa:
    # z[c1, i, c2, j] = Σ_s a[c1, i, s] * b[c2, j, s]
    a_flat = reshape(a, c_h * n_tok, n_msa)   # (c_h*N, S)
    b_flat = reshape(b, c_h * n_tok, n_msa)   # (c_h*N, S)
    z_flat = a_flat * transpose(b_flat)         # (c_h*N, c_h*N)

    # Reshape to (c_h, N, c_h, N) → permute to (c_h², N, N)
    z_4d = reshape(z_flat, c_h, n_tok, c_h, n_tok)
    # vec(outer) in original had c2 varying fastest → permute (c2, c1, i, j)
    z_3d = reshape(permutedims(z_4d, (3, 1, 2, 4)), c_h * c_h, n_tok, n_tok)

    norm = mask_f * transpose(mask_f)  # (n_tok, n_tok)
    z_3d = z_3d ./ reshape(norm .+ m.eps, 1, n_tok, n_tok)

    out = m.linear_out(z_3d)  # (c_z, n_tok, n_tok)
    return out
end

end
