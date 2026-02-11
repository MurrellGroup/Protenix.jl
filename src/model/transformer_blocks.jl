module TransformerBlocks

using Random

import ..Primitives: AdaptiveLayerNorm, LayerNormNoOffset, LinearNoBias, silu

export ConditionedTransitionBlock, AttentionPairBias, DiffusionTransformerBlock, DiffusionTransformer

struct ConditionedTransitionBlock
    c_a::Int
    c_s::Int
    n::Int
    adaln::AdaptiveLayerNorm
    linear_a1::LinearNoBias
    linear_a2::LinearNoBias
    linear_b::LinearNoBias
    linear_s::LinearNoBias
    bias_s::Vector{Float32}
end

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
    return ConditionedTransitionBlock(
        c_a,
        c_s,
        n,
        AdaptiveLayerNorm(c_a, c_s; rng = rng),
        LinearNoBias(n * c_a, c_a; rng = rng),
        LinearNoBias(n * c_a, c_a; rng = rng),
        LinearNoBias(c_a, n * c_a; rng = rng),
        LinearNoBias(c_a, c_s; rng = rng),
        fill(Float32(biasinit), c_a),
    )
end

function (blk::ConditionedTransitionBlock)(a::AbstractArray{<:Real}, s::AbstractArray{<:Real})
    size(a, ndims(a)) == blk.c_a || error("ConditionedTransitionBlock: a last dim must be c_a=$(blk.c_a)")
    size(s, ndims(s)) == blk.c_s || error("ConditionedTransitionBlock: s last dim must be c_s=$(blk.c_s)")
    size(a)[1:(ndims(a)-1)] == size(s)[1:(ndims(s)-1)] ||
        error("ConditionedTransitionBlock: a and s batch/token dims must match.")

    a0 = blk.adaln(a, s)
    b = silu(blk.linear_a1(a0)) .* blk.linear_a2(a0)
    proj = blk.linear_b(b)
    gate = blk.linear_s(s) .+ reshape(blk.bias_s, ntuple(_ -> 1, ndims(s)-1)..., :)
    return (1f0 ./ (1f0 .+ exp.(-gate))) .* proj
end

struct AttentionPairBias
    n_heads::Int
    c_a::Int
    c_s::Int
    c_z::Int
    cross_attention_mode::Bool
    adaln_a::AdaptiveLayerNorm
    adaln_kv::Union{AdaptiveLayerNorm, Nothing}
    layernorm_z::LayerNormNoOffset
    linear_bias_z::LinearNoBias # c_z -> n_heads
    linear_q::LinearNoBias
    bias_q::Vector{Float32}
    linear_k::LinearNoBias
    linear_v::LinearNoBias
    linear_o::LinearNoBias
    linear_g::LinearNoBias
    linear_a_last::LinearNoBias
    bias_a_last::Vector{Float32}
end

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
    return AttentionPairBias(
        n_heads,
        c_a,
        c_s,
        c_z,
        cross_attention_mode,
        AdaptiveLayerNorm(c_a, c_s; rng = rng),
        adaln_kv,
        LayerNormNoOffset(c_z),
        LinearNoBias(n_heads, c_z; rng = rng),
        LinearNoBias(c_a, c_a; rng = rng),
        zeros(Float32, c_a),
        LinearNoBias(c_a, c_a; rng = rng),
        LinearNoBias(c_a, c_a; rng = rng),
        LinearNoBias(c_a, c_a; rng = rng),
        LinearNoBias(c_a, c_a; rng = rng),
        LinearNoBias(c_a, c_s; rng = rng),
        fill(Float32(biasinit), c_a),
    )
end

function _col_softmax!(scores::Matrix{Float32})
    for j in 1:size(scores, 2)
        col = @view(scores[:, j])
        m = maximum(col)
        col .= exp.(col .- m)
        s = sum(col)
        col ./= s
    end
    return scores
end

function _create_local_attn_bias(n::Int, n_queries::Int, n_keys::Int)
    n > 0 || error("n must be positive.")
    n_queries > 0 || error("n_queries must be positive.")
    n_keys >= n_queries || error("n_keys must be >= n_queries.")
    n_queries % 2 == 0 || error("n_queries must be even.")
    n_keys % 2 == 0 || error("n_keys must be even.")

    n_trunks = cld(n, n_queries)
    padded_n = n_trunks * n_queries
    mask = zeros(Float32, padded_n, padded_n)
    left = div(n_keys - n_queries, 2)
    right = div(n_queries + n_keys, 2)
    for block in 0:(n_trunks - 1)
        i1 = block * n_queries + 1
        i2 = i1 + n_queries - 1
        j1 = max(1, block * n_queries + 1 - left)
        j2 = min(padded_n, block * n_queries + right)
        @inbounds mask[i1:i2, j1:j2] .= 1f0
    end
    attn_bias = (1f0 .- mask) .* (-1f10)
    return @view(attn_bias[1:n, 1:n])
end

function _attention_with_pair_bias(
    blk::AttentionPairBias,
    q_in::Matrix{Float32},
    kv_in::Matrix{Float32},
    z_bias::Array{Float32,3},
    local_attn_bias::Union{Nothing, AbstractMatrix{Float32}} = nothing,
)
    n_token = size(q_in, 1)
    c_a = blk.c_a
    n_heads = blk.n_heads
    d = div(c_a, n_heads)
    scale = inv(sqrt(Float32(d)))

    q = blk.linear_q(q_in) .+ reshape(blk.bias_q, 1, :)
    k = blk.linear_k(kv_in)
    v = blk.linear_v(kv_in)
    g = 1f0 ./ (1f0 .+ exp.(-blk.linear_g(q_in)))

    out = zeros(Float32, n_token, c_a)
    for h in 1:n_heads
        r = ((h - 1) * d + 1):(h * d)
        qh = @view(q[:, r])
        kh = @view(k[:, r])
        vh = @view(v[:, r])
        scores = Array{Float32}(undef, n_token, n_token) # [key, query]
        scores .= (kh * transpose(qh)) .* scale
        scores .+= transpose(@view(z_bias[:, :, h]))
        if local_attn_bias !== nothing
            scores .+= transpose(local_attn_bias)
        end
        _col_softmax!(scores)
        ctx = transpose(scores) * vh
        @view(out[:, r]) .= ctx
    end

    out .*= g
    out = blk.linear_o(out)
    return out
end

function _attention_with_pair_bias_local_trunks(
    blk::AttentionPairBias,
    q_in::Matrix{Float32},
    kv_in::Matrix{Float32},
    z_bias_trunk::Array{Float32,4}, # [n_trunks, n_queries, n_keys, n_heads]
    n_queries::Int,
    n_keys::Int,
)
    n_token = size(q_in, 1)
    c_a = blk.c_a
    n_heads = blk.n_heads
    d = div(c_a, n_heads)
    scale = inv(sqrt(Float32(d)))
    n_trunks = size(z_bias_trunk, 1)
    size(z_bias_trunk, 2) == n_queries || error("z trunk query dim mismatch.")
    size(z_bias_trunk, 3) == n_keys || error("z trunk key dim mismatch.")
    size(z_bias_trunk, 4) == n_heads || error("z trunk head dim mismatch.")

    q = blk.linear_q(q_in) .+ reshape(blk.bias_q, 1, :)
    k = blk.linear_k(kv_in)
    v = blk.linear_v(kv_in)
    g = 1f0 ./ (1f0 .+ exp.(-blk.linear_g(q_in)))

    out = zeros(Float32, n_token, c_a)
    left = div(n_keys - n_queries, 2)

    for h in 1:n_heads
        r = ((h - 1) * d + 1):(h * d)
        qh = @view(q[:, r])
        kh = @view(k[:, r])
        vh = @view(v[:, r])
        for b in 1:n_trunks
            q_start = (b - 1) * n_queries + 1
            q_start > n_token && break
            q_end = min(q_start + n_queries - 1, n_token)
            q_len = q_end - q_start + 1
            k_start = q_start - left

            k_block = zeros(Float32, n_keys, d)
            v_block = zeros(Float32, n_keys, d)
            valid_key = falses(n_keys)
            for kk in 1:n_keys
                k_idx = k_start + kk - 1
                if 1 <= k_idx <= n_token
                    valid_key[kk] = true
                    @inbounds k_block[kk, :] .= kh[k_idx, :]
                    @inbounds v_block[kk, :] .= vh[k_idx, :]
                end
            end

            scores = (k_block * transpose(qh[q_start:q_end, :])) .* scale # [key, query]
            scores .+= transpose(@view(z_bias_trunk[b, 1:q_len, :, h]))
            for kk in 1:n_keys
                if !valid_key[kk]
                    @inbounds scores[kk, :] .= -1f10
                end
            end
            _col_softmax!(scores)
            ctx = transpose(scores) * v_block
            @view(out[q_start:q_end, r]) .= ctx
        end
    end

    out .*= g
    out = blk.linear_o(out)
    return out
end

function (blk::AttentionPairBias)(
    a::AbstractArray{<:Real,2},
    s::AbstractArray{<:Real,2},
    z::AbstractArray{<:Real},
    n_queries::Union{Nothing, Int} = nothing,
    n_keys::Union{Nothing, Int} = nothing,
)
    size(a, 2) == blk.c_a || error("AttentionPairBias: a last dim must be c_a=$(blk.c_a)")
    size(s, 2) == blk.c_s || error("AttentionPairBias: s last dim must be c_s=$(blk.c_s)")
    size(a, 1) == size(s, 1) || error("AttentionPairBias: a/s token axis mismatch.")
    ndims(z) in (3, 4) || error("AttentionPairBias: z must be rank-3 dense or rank-4 local trunk tensor.")
    size(z, ndims(z)) == blk.c_z || error("AttentionPairBias: z last dim must be c_z=$(blk.c_z)")

    a_f = Float32.(a)
    s_f = Float32.(s)
    a_norm = blk.adaln_a(a_f, s_f)
    kv_norm = if blk.cross_attention_mode
        blk.adaln_kv === nothing && error("cross_attention_mode=true but adaln_kv is missing.")
        blk.adaln_kv(a_norm, s_f)
    else
        a_norm
    end
    out = if ndims(z) == 3
        z_f = Float32.(z)
        size(z_f, 1) == size(a, 1) || error("AttentionPairBias: dense z token axis mismatch.")
        size(z_f, 2) == size(a, 1) || error("AttentionPairBias: dense z token axis mismatch.")
        local_bias = if n_queries === nothing || n_keys === nothing
            nothing
        else
            _create_local_attn_bias(size(a_f, 1), n_queries, n_keys)
        end
        z_bias = blk.linear_bias_z(blk.layernorm_z(z_f)) # [N, N, n_heads]
        _attention_with_pair_bias(blk, a_norm, kv_norm, z_bias, local_bias)
    else
        n_queries === nothing && error("AttentionPairBias: rank-4 z requires n_queries.")
        n_keys === nothing && error("AttentionPairBias: rank-4 z requires n_keys.")
        z_f = Float32.(z)
        z_bias_trunk = blk.linear_bias_z(blk.layernorm_z(z_f)) # [n_trunks, n_queries, n_keys, n_heads]
        _attention_with_pair_bias_local_trunks(blk, a_norm, kv_norm, z_bias_trunk, n_queries, n_keys)
    end
    gate = blk.linear_a_last(s_f) .+ reshape(blk.bias_a_last, 1, :)
    return (1f0 ./ (1f0 .+ exp.(-gate))) .* out
end

struct DiffusionTransformerBlock
    attention_pair_bias::AttentionPairBias
    conditioned_transition_block::ConditionedTransitionBlock
end

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
    a::AbstractArray{<:Real,2},
    s::AbstractArray{<:Real,2},
    z::AbstractArray{<:Real},
    n_queries::Union{Nothing, Int} = nothing,
    n_keys::Union{Nothing, Int} = nothing,
)
    a_f = Float32.(a)
    attn = blk.attention_pair_bias(a_f, s, z, n_queries, n_keys)
    a_f = a_f .+ attn
    tr = blk.conditioned_transition_block(a_f, s)
    a_f = a_f .+ tr
    return a_f, Float32.(s), Float32.(z)
end

struct DiffusionTransformer
    blocks::Vector{DiffusionTransformerBlock}
end

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
    a::AbstractArray{<:Real,2},
    s::AbstractArray{<:Real,2},
    z::AbstractArray{<:Real},
    n_queries::Union{Nothing, Int} = nothing,
    n_keys::Union{Nothing, Int} = nothing,
)
    a_f = Float32.(a)
    s_f = Float32.(s)
    z_f = Float32.(z)
    for blk in tr.blocks
        a_f, s_f, z_f = blk(a_f, s_f, z_f, n_queries, n_keys)
    end
    return a_f
end

end
