module OpenFoldBlocks

using Random

import ..Primitives: Linear, LinearNoBias, LayerNorm

export TriangleMultiplication,
    TriangleAttention,
    OuterProductMean,
    PairAttentionNoS

function _row_softmax!(scores::Matrix{Float32})
    @inbounds for i in 1:size(scores, 1)
        row = @view scores[i, :]
        m = maximum(row)
        row .= exp.(row .- m)
        s = sum(row)
        row ./= s
    end
    return scores
end

"""
Pair attention used in Pairformer when `has_s=false`.
"""
struct PairAttentionNoS
    n_heads::Int
    c_a::Int
    c_z::Int
    layernorm_a::LayerNorm
    layernorm_z::LayerNorm
    linear_nobias_z::LinearNoBias
    linear_q::Linear
    linear_k::LinearNoBias
    linear_v::LinearNoBias
    linear_o::LinearNoBias
    linear_g::LinearNoBias
end

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
        pair_bias = zeros(Float32, n, n)
    else
        size(pair_mask) == (n, n) || error("pair_mask shape mismatch")
        pair_bias = (Float32.(pair_mask) .- 1f0) .* 1f9
    end

    d = div(m.c_a, m.n_heads)
    scale = inv(sqrt(Float32(d)))
    out = zeros(Float32, n, m.c_a)

    @inbounds for h in 1:m.n_heads
        r = ((h - 1) * d + 1):(h * d)
        qh = @view q[:, r]
        kh = @view k[:, r]
        vh = @view v[:, r]

        scores = (qh * transpose(kh)) .* scale
        scores .+= @view z_bias[:, :, h]
        scores .+= pair_bias
        _row_softmax!(scores)

        ctx = scores * vh
        @view(out[:, r]) .= ctx
    end

    out .*= g
    out = m.linear_o(out)
    return out
end

"""
OpenFold triangle multiplicative update, inference path only.
"""
struct TriangleMultiplication
    c_z::Int
    c_hidden::Int
    outgoing::Bool
    layer_norm_in::LayerNorm
    layer_norm_out::LayerNorm
    linear_a_p::LinearNoBias
    linear_a_g::LinearNoBias
    linear_b_p::LinearNoBias
    linear_b_g::LinearNoBias
    linear_z::LinearNoBias
    linear_g::LinearNoBias
end

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
        mask_f = ones(Float32, n_i, n_j)
    else
        size(mask) == (n_i, n_j) || error("TriangleMultiplication mask shape mismatch")
        mask_f = Float32.(mask)
    end

    z0 = m.layer_norm_in(z)
    a = m.linear_a_p(z0) .* (1f0 ./ (1f0 .+ exp.(-m.linear_a_g(z0))))
    b = m.linear_b_p(z0) .* (1f0 ./ (1f0 .+ exp.(-m.linear_b_g(z0))))
    a .*= reshape(mask_f, n_i, n_j, 1)
    b .*= reshape(mask_f, n_i, n_j, 1)

    x = zeros(Float32, n_i, n_j, m.c_hidden)
    @inbounds for c in 1:m.c_hidden
        if m.outgoing
            # out[i,j,c] = sum_k a[i,k,c] * b[j,k,c]
            for i in 1:n_i, j in 1:n_j
                s = 0f0
                for k_idx in 1:n_i
                    s += a[i, k_idx, c] * b[j, k_idx, c]
                end
                x[i, j, c] = s
            end
        else
            # out[i,j,c] = sum_k a[k,i,c] * b[k,j,c]
            for i in 1:n_i, j in 1:n_j
                s = 0f0
                for k_idx in 1:n_i
                    s += a[k_idx, i, c] * b[k_idx, j, c]
                end
                x[i, j, c] = s
            end
        end
    end

    x = m.layer_norm_out(x)
    x = m.linear_z(x)
    g = 1f0 ./ (1f0 .+ exp.(-m.linear_g(z0)))
    x .*= g
    return x
end

"""
OpenFold triangle attention, inference path only.
Implements row/column variants controlled by `starting`.
"""
struct TriangleAttention
    c_in::Int
    c_hidden::Int
    n_heads::Int
    starting::Bool
    inf::Float32
    layer_norm::LayerNorm
    linear::LinearNoBias # c_in -> n_heads
    linear_q::LinearNoBias
    linear_k::LinearNoBias
    linear_v::LinearNoBias
    linear_o::LinearNoBias
    linear_g::LinearNoBias
end

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
    x::Array{Float32,3},
    mask::Matrix{Float32},
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
    out = zeros(Float32, i_dim, j_dim, m.c_hidden)

    @inbounds for b in 1:i_dim
        mask_bias_key = m.inf .* (mask[b, :] .- 1f0)
        for h in 1:m.n_heads
            r = ((h - 1) * d + 1):(h * d)
            qh = @view q[b, :, r] # [Q, d]
            kh = @view k[b, :, r] # [K, d]
            vh = @view v[b, :, r] # [K, d]

            scores = (qh * transpose(kh)) .* scale
            for qi in 1:j_dim
                # OpenFold triangle bias is non-batched: depends on (query, key, head),
                # not on the batch/row index after reshaping.
                @views scores[qi, :] .+= tri_bias[qi, :, h]
                @views scores[qi, :] .+= mask_bias_key
            end
            _row_softmax!(scores)
            ctx = scores * vh
            @view(out[b, :, r]) .= ctx
        end
    end

    out .*= g
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
        mask_f = ones(Float32, n_i, n_j)
    else
        size(mask) == (n_i, n_j) || error("TriangleAttention mask shape mismatch")
        mask_f = Float32.(mask)
    end

    if m.starting
        return _triangle_attention_row(m, x_f, mask_f)
    else
        x_t = permutedims(x_f, (2, 1, 3))
        m_t = permutedims(mask_f, (2, 1))
        out_t = _triangle_attention_row(m, x_t, m_t)
        return permutedims(out_t, (2, 1, 3))
    end
end

"""
OpenFold outer product mean, inference path only.
`m` shape: `[N_seq, N_res, C_m]`
"""
struct OuterProductMean
    c_m::Int
    c_z::Int
    c_hidden::Int
    eps::Float32
    layer_norm::LayerNorm
    linear_1::LinearNoBias
    linear_2::LinearNoBias
    linear_out::Linear
end

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
        mask_f = ones(Float32, n_seq, n_res)
    else
        size(mask) == (n_seq, n_res) || error("OuterProductMean mask shape mismatch")
        mask_f = Float32.(mask)
    end

    ln = m.layer_norm(msa)
    a = m.linear_1(ln) .* reshape(mask_f, n_seq, n_res, 1)
    b = m.linear_2(ln) .* reshape(mask_f, n_seq, n_res, 1)

    out = zeros(Float32, n_res, n_res, m.c_z)
    tmp = zeros(Float32, m.c_hidden, m.c_hidden)

    @inbounds for i in 1:n_res, j in 1:n_res
        fill!(tmp, 0f0)
        for s in 1:n_seq
            for c1 in 1:m.c_hidden
                ai = a[s, i, c1]
                for c2 in 1:m.c_hidden
                    tmp[c1, c2] += ai * b[s, j, c2]
                end
            end
        end
        # Python flattens `[c, e]` with `e` as the fastest axis.
        # Julia is column-major, so transpose before flattening to match.
        flat = reshape(permutedims(tmp, (2, 1)), 1, :)
        out[i, j, :] .= vec(m.linear_out(flat))
    end

    norm = transpose(mask_f) * mask_f
    @inbounds for i in 1:n_res, j in 1:n_res
        denom = norm[i, j] + m.eps
        out[i, j, :] ./= denom
    end
    return out
end

end
