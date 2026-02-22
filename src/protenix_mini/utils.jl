module Utils

using Random

export softmax_firstdim,
    softmax_lastdim,
    softmax_dim2,
    one_hot_interval,
    one_hot_int,
    clamp01,
    broadcast_token_to_atom,
    aggregate_atom_to_token_mean,
    pairwise_distances,
    sample_msa_indices

function clamp01(x::AbstractArray{<:Real})
    return clamp.(Float32.(x), 0f0, 1f0)
end

"""
Softmax along dim=1 (features dimension in features-first convention).
"""
function softmax_firstdim(x::AbstractArray{<:Real})
    y = Float32.(x)
    c = size(y, 1)
    rest = Base.tail(size(y))
    flat = reshape(y, c, :)
    m = maximum(flat; dims=1)
    ex = exp.(flat .- m)
    s = sum(ex; dims=1)
    return reshape(ex ./ s, size(y))
end

"""
Softmax along the last dimension of an array (e.g., for attention over keys).
"""
function softmax_lastdim(x::AbstractArray{<:Real})
    y = Float32.(x)
    n = ndims(y)
    n >= 1 || error("softmax_lastdim requires rank >= 1")
    c = size(y, n)
    flat = reshape(y, :, c)
    m = maximum(flat; dims=2)
    ex = exp.(flat .- m)
    s = sum(ex; dims=2)
    return reshape(ex ./ s, size(y))
end

"""
Softmax along the second dimension of a rank-3 tensor `[H, I, J]`.
Used for attention weights where softmax is over the I dimension.
"""
function softmax_dim2(x::AbstractArray{<:Real,3})
    y = Float32.(x)
    m = maximum(y; dims=2)
    ex = exp.(y .- m)
    s = sum(ex; dims=2)
    return ex ./ s
end

"""
One-hot encoding for a 1D integer vector.
Features-first: returns `(num_classes, N)`.
"""
function one_hot_int(x::AbstractVector{<:Integer}, num_classes::Int)
    num_classes > 0 || error("num_classes must be positive")
    out = zeros(Float32, num_classes, length(x))
    @inbounds for i in eachindex(x)
        idx = Int(x[i]) + 1
        1 <= idx <= num_classes || error("one_hot_int index $(x[i]) out of range [0, $(num_classes-1)]")
        out[idx, i] = 1f0
    end
    return out
end

"""
One-hot encoding for a 2D integer matrix.
Features-first: returns `(num_classes, n1, n2)` from input `(n1, n2)`.
"""
function one_hot_int(x::AbstractMatrix{<:Integer}, num_classes::Int)
    num_classes > 0 || error("num_classes must be positive")
    n1, n2 = size(x)
    out = zeros(Float32, num_classes, n1, n2)
    @inbounds for i in 1:n1, j in 1:n2
        idx = Int(x[i, j]) + 1
        1 <= idx <= num_classes || error("one_hot_int index $(x[i, j]) out of range [0, $(num_classes-1)]")
        out[idx, i, j] = 1f0
    end
    return out
end

"""
Interval one-hot for scalar distances.
Features-first: returns `(B, N, M)` mask where `lower[b] < x < upper[b]`.
"""
function one_hot_interval(
    x::AbstractMatrix{<:Real},
    lower_bins::AbstractVector{<:Real},
    upper_bins::AbstractVector{<:Real},
)
    length(lower_bins) == length(upper_bins) || error("lower_bins/upper_bins length mismatch")
    n, m = size(x)
    b = length(lower_bins)
    xf = Float32.(reshape(x, 1, n, m))
    lo = reshape(Float32.(lower_bins), b, 1, 1)
    hi = reshape(Float32.(upper_bins), b, 1, 1)
    return Float32.((xf .> lo) .& (xf .< hi))
end

"""
Broadcast token features to atoms.
Features-first: `(c, N_token)` → `(c, N_atom)`.
"""
function broadcast_token_to_atom(
    x_token::AbstractMatrix{<:Real},
    atom_to_token_idx::AbstractVector{<:Integer},
)
    idx_cpu = Array(Int.(atom_to_token_idx)) .+ 1
    all(i -> 1 <= i <= size(x_token, 2), idx_cpu) || error("atom_to_token_idx out of range")
    return Float32.(x_token[:, idx_cpu])
end

"""
Aggregate atom features to token features by mean.
Features-first: `(c, N_atom)` → `(c, N_token)`.
"""
function aggregate_atom_to_token_mean(
    x_atom::AbstractMatrix{<:Real},
    atom_to_token_idx::AbstractVector{<:Integer},
    n_token::Int,
)
    n_token > 0 || error("n_token must be positive")
    c = size(x_atom, 1)
    n_atom = size(x_atom, 2)
    n_atom == length(atom_to_token_idx) || error("x_atom/atom_to_token_idx length mismatch")
    x_f = Float32.(x_atom)
    idx_cpu = Array(Int.(atom_to_token_idx)) .+ 1
    # Build one-hot mapping on CPU: (N_token, N_atom)
    oh_cpu = zeros(Float32, n_token, n_atom)
    @inbounds for i in 1:n_atom
        1 <= idx_cpu[i] <= n_token || error("atom_to_token_idx[$i] out of range")
        oh_cpu[idx_cpu[i], i] = 1f0
    end
    oh = copyto!(similar(x_f, Float32, n_token, n_atom), oh_cpu)
    # (c, N_atom) × (N_atom, N_token)' → (c, N_token)
    sums = x_f * transpose(oh)
    counts = max.(sum(oh; dims=2)', 1f0)  # (1, N_token)
    return sums ./ counts
end

"""
Pairwise Euclidean distances.
Features-first: input `(3, N)`, output `(N, N)`.
"""
function pairwise_distances(x::AbstractMatrix{<:Real})
    size(x, 1) == 3 || error("pairwise_distances expects (3, N)")
    n = size(x, 2)
    xf = Float32.(x)
    # (3, N, 1) - (3, 1, N) → (3, N, N), then sum squared over dim 1
    diff = reshape(xf, 3, n, 1) .- reshape(xf, 3, 1, n)
    return dropdims(sqrt.(sum(diff .^ 2; dims=1)); dims=1)
end

"""
Sample MSA row indices without replacement, matching Python behavior at inference.
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
