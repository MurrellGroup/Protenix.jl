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
    return clamp.(Float32.(x), 0f0, 1f0)
end

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
Softmax along the second dimension of a rank-3 tensor `[I, J, H]`.
"""
function softmax_dim2(x::AbstractArray{<:Real,3})
    y = Float32.(x)
    m = maximum(y; dims=2)
    ex = exp.(y .- m)
    s = sum(ex; dims=2)
    return ex ./ s
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
    xf = Float32.(reshape(x, n, m, 1))
    lo = reshape(Float32.(lower_bins), 1, 1, b)
    hi = reshape(Float32.(upper_bins), 1, 1, b)
    return Float32.((xf .> lo) .& (xf .< hi))
end

function broadcast_token_to_atom(
    x_token::AbstractMatrix{<:Real},
    atom_to_token_idx::AbstractVector{<:Integer},
)
    idx_cpu = Array(Int.(atom_to_token_idx)) .+ 1  # ensure CPU for range check
    all(i -> 1 <= i <= size(x_token, 1), idx_cpu) || error("atom_to_token_idx out of range")
    return Float32.(x_token[idx_cpu, :])
end

function aggregate_atom_to_token_mean(
    x_atom::AbstractMatrix{<:Real},
    atom_to_token_idx::AbstractVector{<:Integer},
    n_token::Int,
)
    n_token > 0 || error("n_token must be positive")
    n_atom = size(x_atom, 1)
    n_atom == length(atom_to_token_idx) || error("x_atom/atom_to_token_idx length mismatch")
    c = size(x_atom, 2)
    x_f = Float32.(x_atom)
    idx_cpu = Array(Int.(atom_to_token_idx)) .+ 1  # ensure CPU, 0→1 indexed
    # Build one-hot mapping on CPU: (N_token, N_atom)
    oh_cpu = zeros(Float32, n_token, n_atom)
    @inbounds for i in 1:n_atom
        1 <= idx_cpu[i] <= n_token || error("atom_to_token_idx[$i] out of range")
        oh_cpu[idx_cpu[i], i] = 1f0
    end
    # Transfer to same device as x_atom
    oh = copyto!(similar(x_f, Float32, n_token, n_atom), oh_cpu)
    # Matrix multiply: (N_token, N_atom) × (N_atom, c) = (N_token, c)
    sums = oh * x_f
    # Counts per token: (N_token, 1)
    counts = max.(sum(oh; dims=2), 1f0)
    return sums ./ counts
end

function pairwise_distances(x::AbstractMatrix{<:Real})
    n = size(x, 1)
    size(x, 2) == 3 || error("pairwise_distances expects [N,3]")
    xf = Float32.(x)
    # [n,1,3] - [1,n,3] → [n,n,3], then sum squared over dim 3
    diff = reshape(xf, n, 1, 3) .- reshape(xf, 1, n, 3)
    return dropdims(sqrt.(sum(diff .^ 2; dims=3)); dims=3)
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
