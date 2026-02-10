module Primitives

using Random
using Statistics

export Linear,
    LinearNoBias,
    LayerNorm,
    transition,
    silu

silu(x::Real) = x / (1 + exp(-x))
silu(x::AbstractArray) = x ./ (1 .+ exp.(-x))

struct Linear
    weight::Matrix{Float32} # [out, in]
    bias::Union{Vector{Float32}, Nothing}
end

function Linear(
    out_features::Int,
    in_features::Int;
    bias::Bool = true,
    rng::AbstractRNG = Random.default_rng(),
)
    out_features > 0 || error("out_features must be positive")
    in_features > 0 || error("in_features must be positive")
    w = 0.02f0 .* randn(rng, Float32, out_features, in_features)
    b = bias ? zeros(Float32, out_features) : nothing
    return Linear(w, b)
end

function (lin::Linear)(x::AbstractArray{<:Real})
    in_features = size(x, ndims(x))
    size(lin.weight, 2) == in_features ||
        error("Linear input mismatch: expected $(size(lin.weight, 2)), got $in_features")
    x_f = Float32.(x)
    flat = reshape(x_f, :, in_features)
    y = flat * transpose(lin.weight)
    if lin.bias !== nothing
        y .+= reshape(lin.bias, 1, :)
    end
    return reshape(y, (size(x)[1:(ndims(x)-1)]..., size(lin.weight, 1)))
end

struct LinearNoBias
    weight::Matrix{Float32} # [out, in]
end

function LinearNoBias(out_features::Int, in_features::Int; rng::AbstractRNG = Random.default_rng())
    out_features > 0 || error("out_features must be positive")
    in_features > 0 || error("in_features must be positive")
    w = 0.02f0 .* randn(rng, Float32, out_features, in_features)
    return LinearNoBias(w)
end

function (lin::LinearNoBias)(x::AbstractArray{<:Real})
    in_features = size(x, ndims(x))
    size(lin.weight, 2) == in_features ||
        error("LinearNoBias input mismatch: expected $(size(lin.weight, 2)), got $in_features")
    x_f = Float32.(x)
    flat = reshape(x_f, :, in_features)
    y = flat * transpose(lin.weight)
    return reshape(y, (size(x)[1:(ndims(x)-1)]..., size(lin.weight, 1)))
end

struct LayerNorm
    weight::Vector{Float32}
    bias::Vector{Float32}
    eps::Float32
end

function LayerNorm(c::Int; eps::Real = 1f-5)
    c > 0 || error("LayerNorm channel must be positive")
    return LayerNorm(ones(Float32, c), zeros(Float32, c), Float32(eps))
end

function (ln::LayerNorm)(x::AbstractArray{<:Real})
    c = size(x, ndims(x))
    length(ln.weight) == c || error("LayerNorm weight length mismatch")
    length(ln.bias) == c || error("LayerNorm bias length mismatch")
    x_f = Float32.(x)
    flat = reshape(x_f, :, c)
    out = similar(flat)
    @inbounds for i in 1:size(flat, 1)
        row = @view flat[i, :]
        μ = mean(row)
        v = mean((row .- μ) .^ 2)
        inv_std = inv(sqrt(v + ln.eps))
        out[i, :] .= (row .- μ) .* inv_std .* ln.weight .+ ln.bias
    end
    return reshape(out, size(x_f))
end

"""
Inference transition used throughout Protenix modules.
"""
function transition(
    x::AbstractArray{<:Real},
    layernorm::LayerNorm,
    linear_a::LinearNoBias,
    linear_b::LinearNoBias,
    linear_out::LinearNoBias,
)
    x0 = layernorm(x)
    return linear_out(silu(linear_a(x0)) .* linear_b(x0))
end

end
