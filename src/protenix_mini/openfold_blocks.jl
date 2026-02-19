module OpenFoldBlocks

using Random
using ConcreteStructs
using Flux: @layer

import ..Primitives: Linear, LinearNoBias, LayerNorm

export TriangleMultiplication,
    TriangleAttention,
    OuterProductMean,
    PairAttentionNoS

function _row_softmax(scores::AbstractMatrix{<:Real})
    m = maximum(scores; dims=2)
    ex = exp.(scores .- m)
    return ex ./ sum(ex; dims=2)
end

"""
Pair attention used in Pairformer when `has_s=false`.
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
        LinearNoBias(n_heads, c_z; rng = rng),
        Linear(c_a, c_a; bias = true, rng = rng),
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
    n = size(a, 1)
    size(a, 2) == m.c_a || error("PairAttentionNoS a channel mismatch")
    size(z) == (n, n, m.c_z) || error("PairAttentionNoS z shape mismatch")

    a_ln = m.layernorm_a(a)
    z_ln = m.layernorm_z(z)

    q = m.linear_q(a_ln)
    k = m.linear_k(a_ln)
    v = m.linear_v(a_ln)
    g = 1f0 ./ (1f0 .+ exp.(-m.linear_g(a_ln)))

    z_bias = m.linear_nobias_z(z_ln) # [N, N, H]
    if pair_mask === nothing
        pair_bias = fill!(similar(q, Float32, n, n), 0f0)
    else
        size(pair_mask) == (n, n) || error("pair_mask shape mismatch")
        pair_bias = (Float32.(pair_mask) .- 1f0) .* 1f9
    end

    d = div(m.c_a, m.n_heads)
    scale = inv(sqrt(Float32(d)))
    out = fill!(similar(q, Float32, n, m.c_a), 0f0)

    for h in 1:m.n_heads
        r = ((h - 1) * d + 1):(h * d)
        qh = q[:, r]
        kh = k[:, r]
        vh = v[:, r]

        scores = (qh * transpose(kh)) .* scale
        scores = scores .+ z_bias[:, :, h] .+ pair_bias
        scores = _row_softmax(scores)

        ctx = scores * vh
        out[:, r] .= ctx
    end

    out = out .* g
    out = m.linear_o(out)
    return out
end

"""
OpenFold triangle multiplicative update, inference path only.
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
        LinearNoBias(c_hidden, c_z; rng = rng),
        LinearNoBias(c_hidden, c_z; rng = rng),
        LinearNoBias(c_hidden, c_z; rng = rng),
        LinearNoBias(c_hidden, c_z; rng = rng),
        LinearNoBias(c_z, c_hidden; rng = rng),
        LinearNoBias(c_z, c_z; rng = rng),
    )
end

function (m::TriangleMultiplication)(
    z::AbstractArray{<:Real,3};
    mask::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
)
    n_i, n_j, c_z = size(z)
    n_i == n_j || error("TriangleMultiplication expects square pair tensor")
    c_z == m.c_z || error("TriangleMultiplication c_z mismatch")

    if mask === nothing
        mask_f = fill!(similar(z, Float32, n_i, n_j), 1f0)
    else
        size(mask) == (n_i, n_j) || error("TriangleMultiplication mask shape mismatch")
        mask_f = Float32.(mask)
    end

    z0 = m.layer_norm_in(z)
    a = m.linear_a_p(z0) .* (1f0 ./ (1f0 .+ exp.(-m.linear_a_g(z0))))
    b = m.linear_b_p(z0) .* (1f0 ./ (1f0 .+ exp.(-m.linear_b_g(z0))))
    a = a .* reshape(mask_f, n_i, n_j, 1)
    b = b .* reshape(mask_f, n_i, n_j, 1)

    x = fill!(similar(a, Float32, n_i, n_j, m.c_hidden), 0f0)
    for c in 1:m.c_hidden
        ac = a[:, :, c]
        bc = b[:, :, c]
        if m.outgoing
            # x[:,:,c] = a[:,:,c] * b[:,:,c]'
            x[:, :, c] .= ac * transpose(bc)
        else
            # x[:,:,c] = a[:,:,c]' * b[:,:,c]
            x[:, :, c] .= transpose(ac) * bc
        end
    end

    x = m.layer_norm_out(x)
    x = m.linear_z(x)
    g = 1f0 ./ (1f0 .+ exp.(-m.linear_g(z0)))
    x = x .* g
    return x
end

"""
OpenFold triangle attention, inference path only.
Implements row/column variants controlled by `starting`.
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
        LinearNoBias(n_heads, c_in; rng = rng),
        LinearNoBias(c_hidden, c_in; rng = rng),
        LinearNoBias(c_hidden, c_in; rng = rng),
        LinearNoBias(c_hidden, c_in; rng = rng),
        LinearNoBias(c_in, c_hidden; rng = rng),
        LinearNoBias(c_hidden, c_in; rng = rng),
    )
end

function _triangle_attention_row(
    m::TriangleAttention,
    x::AbstractArray{<:Real,3},
    mask::AbstractMatrix{<:Real},
)
    i_dim, j_dim, c_in = size(x)
    c_in == m.c_in || error("TriangleAttention c_in mismatch")

    x_ln = m.layer_norm(x)
    tri_bias = m.linear(x_ln) # [I, J, H]

    q = m.linear_q(x_ln)
    k = m.linear_k(x_ln)
    v = m.linear_v(x_ln)
    g = 1f0 ./ (1f0 .+ exp.(-m.linear_g(x_ln)))

    d = div(m.c_hidden, m.n_heads)
    scale = inv(sqrt(Float32(d)))
    out = fill!(similar(q, Float32, i_dim, j_dim, m.c_hidden), 0f0)

    for b in 1:i_dim
        mask_bias_key = reshape(m.inf .* (mask[b:b, :] .- 1f0), 1, j_dim)
        for h in 1:m.n_heads
            r = ((h - 1) * d + 1):(h * d)
            qh = q[b, :, r] # [Q, d]
            kh = k[b, :, r] # [K, d]
            vh = v[b, :, r] # [K, d]

            scores = (qh * transpose(kh)) .* scale
            scores = scores .+ tri_bias[:, :, h:h][:, :, 1] .+ mask_bias_key
            scores = _row_softmax(scores)

            ctx = scores * vh
            out[b, :, r] .= ctx
        end
    end

    out = out .* g
    out = m.linear_o(out)
    return out
end

function _triangle_attention_col(
    m::TriangleAttention,
    x::AbstractArray{<:Real,3},
    mask::AbstractMatrix{<:Real},
)
    i_dim, j_dim, c_in = size(x)
    c_in == m.c_in || error("TriangleAttention c_in mismatch")

    x_ln = m.layer_norm(x)
    tri_bias = m.linear(x_ln) # [I, J, H]

    q = m.linear_q(x_ln)
    k = m.linear_k(x_ln)
    v = m.linear_v(x_ln)
    g = 1f0 ./ (1f0 .+ exp.(-m.linear_g(x_ln)))

    d = div(m.c_hidden, m.n_heads)
    scale = inv(sqrt(Float32(d)))
    out = fill!(similar(q, Float32, i_dim, j_dim, m.c_hidden), 0f0)

    for b in 1:j_dim
        mask_bias_key = reshape(m.inf .* (mask[:, b:b] .- 1f0), i_dim, 1)
        for h in 1:m.n_heads
            r = ((h - 1) * d + 1):(h * d)
            qh = q[:, b, r] # [Q, d]
            kh = k[:, b, r] # [K, d]
            vh = v[:, b, r] # [K, d]

            scores = (qh * transpose(kh)) .* scale
            # tri_bias[:, b, h] broadcasts across key dimension
            scores = scores .+ reshape(tri_bias[:, b, h:h], i_dim, 1) .+ mask_bias_key
            scores = _row_softmax(scores)
            ctx = scores * vh
            out[:, b, r] .= ctx
        end
    end

    out = out .* g
    out = m.linear_o(out)
    return out
end

function (m::TriangleAttention)(
    x::AbstractArray{<:Real,3};
    mask::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
)
    x_f = Float32.(x)
    n_i, n_j, _ = size(x_f)
    if mask === nothing
        mask_f = fill!(similar(x_f, Float32, n_i, n_j), 1f0)
    else
        size(mask) == (n_i, n_j) || error("TriangleAttention mask shape mismatch")
        mask_f = Float32.(mask)
    end

    if m.starting
        return _triangle_attention_row(m, x_f, mask_f)
    end
    return _triangle_attention_col(m, x_f, mask_f)
end

"""
OpenFold outer product mean, inference path only.
`m` shape: `[N_seq, N_res, C_m]`
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
        LinearNoBias(c_hidden, c_m; rng = rng),
        LinearNoBias(c_hidden, c_m; rng = rng),
        Linear(c_z, c_hidden * c_hidden; bias = true, rng = rng),
    )
end

function (m::OuterProductMean)(
    msa::AbstractArray{<:Real,3};
    mask::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
)
    n_seq, n_res, c_m = size(msa)
    c_m == m.c_m || error("OuterProductMean c_m mismatch")

    if mask === nothing
        mask_f = fill!(similar(msa, Float32, n_seq, n_res), 1f0)
    else
        size(mask) == (n_seq, n_res) || error("OuterProductMean mask shape mismatch")
        mask_f = Float32.(mask)
    end

    ln = m.layer_norm(msa)
    a = m.linear_1(ln) .* reshape(mask_f, n_seq, n_res, 1)
    b = m.linear_2(ln) .* reshape(mask_f, n_seq, n_res, 1)

    # a: [n_seq, n_res, c_hidden], b: [n_seq, n_res, c_hidden]
    # For each (i,j): tmp = a[:,i,:]' * b[:,j,:] → [c_hidden, c_hidden]
    # Then flatten row-major and apply linear_out
    out = fill!(similar(a, Float32, n_res, n_res, m.c_z), 0f0)

    for i in 1:n_res
        # a_i: [n_seq, c_hidden]
        a_i = a[:, i, :]
        for j in 1:n_res
            b_j = b[:, j, :]
            # tmp[c1, c2] = sum_s a_i[s, c1] * b_j[s, c2]
            tmp = transpose(a_i) * b_j  # [c_hidden, c_hidden]
            # Flatten in row-major order (c1 varies slowest, c2 varies fastest)
            flat = reshape(vec(tmp), 1, :)
            out[i, j, :] .= vec(m.linear_out(flat))
        end
    end

    norm = transpose(mask_f) * mask_f
    out = out ./ reshape(norm .+ m.eps, n_res, n_res, 1)
    return out
end

end
