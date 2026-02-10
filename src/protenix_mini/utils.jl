module Utils

using Random

export softmax_lastdim,
    softmax_dim2,
    one_hot_interval,
    one_hot_int,
    clamp01,
    broadcast_token_to_atom,
    aggregate_atom_to_token_mean,
    pairwise_distances,
    sample_msa_indices

function clamp01(x::AbstractArray{<:Real})
    out = Float32.(x)
    @inbounds for i in eachindex(out)
        if out[i] < 0f0
            out[i] = 0f0
        elseif out[i] > 1f0
            out[i] = 1f0
        end
    end
    return out
end

function softmax_lastdim(x::AbstractArray{<:Real})
    y = Float32.(x)
    n = ndims(y)
    n >= 1 || error("softmax_lastdim requires rank >= 1")
    c = size(y, n)
    flat = reshape(y, :, c)
    out = similar(flat)
    @inbounds for i in 1:size(flat, 1)
        row = @view flat[i, :]
        m = maximum(row)
        ex = exp.(row .- m)
        s = sum(ex)
        out[i, :] .= ex ./ s
    end
    return reshape(out, size(y))
end

"""
Softmax along the second dimension of a rank-3 tensor `[I, J, H]`.
"""
function softmax_dim2(x::AbstractArray{<:Real,3})
    y = Float32.(x)
    i_dim, j_dim, h_dim = size(y)
    out = similar(y)
    @inbounds for i in 1:i_dim, h in 1:h_dim
        m = -Inf32
        for j in 1:j_dim
            v = y[i, j, h]
            if v > m
                m = v
            end
        end
        s = 0f0
        for j in 1:j_dim
            e = exp(y[i, j, h] - m)
            out[i, j, h] = e
            s += e
        end
        invs = inv(s)
        for j in 1:j_dim
            out[i, j, h] *= invs
        end
    end
    return out
end

function one_hot_int(x::AbstractVector{<:Integer}, num_classes::Int)
    num_classes > 0 || error("num_classes must be positive")
    out = zeros(Float32, length(x), num_classes)
    @inbounds for i in eachindex(x)
        idx = Int(x[i]) + 1
        1 <= idx <= num_classes || error("one_hot_int index $(x[i]) out of range [0, $(num_classes-1)]")
        out[i, idx] = 1f0
    end
    return out
end

function one_hot_int(x::AbstractMatrix{<:Integer}, num_classes::Int)
    num_classes > 0 || error("num_classes must be positive")
    n1, n2 = size(x)
    out = zeros(Float32, n1, n2, num_classes)
    @inbounds for i in 1:n1, j in 1:n2
        idx = Int(x[i, j]) + 1
        1 <= idx <= num_classes || error("one_hot_int index $(x[i, j]) out of range [0, $(num_classes-1)]")
        out[i, j, idx] = 1f0
    end
    return out
end

"""
Interval one-hot for scalar distances.
Returns `[N, M, B]` mask where `lower[b] < x < upper[b]`.
"""
function one_hot_interval(
    x::AbstractMatrix{<:Real},
    lower_bins::AbstractVector{<:Real},
    upper_bins::AbstractVector{<:Real},
)
    length(lower_bins) == length(upper_bins) || error("lower_bins/upper_bins length mismatch")
    n, m = size(x)
    b = length(lower_bins)
    out = zeros(Float32, n, m, b)
    @inbounds for i in 1:n, j in 1:m
        v = Float32(x[i, j])
        for k in 1:b
            lo = Float32(lower_bins[k])
            hi = Float32(upper_bins[k])
            out[i, j, k] = (v > lo && v < hi) ? 1f0 : 0f0
        end
    end
    return out
end

function broadcast_token_to_atom(
    x_token::AbstractMatrix{<:Real},
    atom_to_token_idx::AbstractVector{<:Integer},
)
    n_atom = length(atom_to_token_idx)
    c = size(x_token, 2)
    out = zeros(Float32, n_atom, c)
    @inbounds for a in 1:n_atom
        t = Int(atom_to_token_idx[a]) + 1
        1 <= t <= size(x_token, 1) || error("atom_to_token_idx[$a] out of range")
        out[a, :] .= Float32.(x_token[t, :])
    end
    return out
end

function aggregate_atom_to_token_mean(
    x_atom::AbstractMatrix{<:Real},
    atom_to_token_idx::AbstractVector{<:Integer},
    n_token::Int,
)
    n_token > 0 || error("n_token must be positive")
    size(x_atom, 1) == length(atom_to_token_idx) || error("x_atom/atom_to_token_idx length mismatch")
    c = size(x_atom, 2)
    out = zeros(Float32, n_token, c)
    counts = zeros(Int, n_token)
    @inbounds for a in 1:size(x_atom, 1)
        t = Int(atom_to_token_idx[a]) + 1
        1 <= t <= n_token || error("atom_to_token_idx[$a] out of range")
        out[t, :] .+= Float32.(x_atom[a, :])
        counts[t] += 1
    end
    @inbounds for t in 1:n_token
        if counts[t] > 0
            out[t, :] ./= counts[t]
        end
    end
    return out
end

function pairwise_distances(x::AbstractMatrix{<:Real})
    n = size(x, 1)
    size(x, 2) == 3 || error("pairwise_distances expects [N,3]")
    out = zeros(Float32, n, n)
    @inbounds for i in 1:n
        xi = Float32(x[i, 1])
        yi = Float32(x[i, 2])
        zi = Float32(x[i, 3])
        for j in 1:n
            dx = xi - Float32(x[j, 1])
            dy = yi - Float32(x[j, 2])
            dz = zi - Float32(x[j, 3])
            out[i, j] = sqrt(dx * dx + dy * dy + dz * dz)
        end
    end
    return out
end

"""
Sample MSA row indices without replacement, matching Python behavior at inference:
- sample size is random in `[lower_bound, n]`
- `strategy="random"` uses random permutation
- `strategy="topk"` uses first rows
- optional `cutoff` truncates sampled indices
"""
function sample_msa_indices(
    n::Int;
    cutoff::Int = 512,
    lower_bound::Int = 1,
    strategy::String = "random",
    rng::AbstractRNG = Random.default_rng(),
)
    n > 0 || return Int[]
    strategy in ("random", "topk") || error("Unsupported MSA sampling strategy: $strategy")
    lb = min(lower_bound, n)
    sample_size = rand(rng, lb:n)
    idx = if strategy == "random"
        randperm(rng, n)[1:sample_size]
    else
        collect(1:sample_size)
    end
    if cutoff > 0 && length(idx) > cutoff
        idx = idx[1:cutoff]
    end
    return Int.(idx)
end

end
