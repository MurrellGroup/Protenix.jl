module Primitives

using Random
using Statistics
using ConcreteStructs
using Flux: @layer

export Linear,
    LinearNoBias,
    LayerNorm,
    transition,
    silu

silu(x::Real) = x / (1 + exp(-x))
silu(x::AbstractArray) = x ./ (1 .+ exp.(-x))

_as_f32_array(x::AbstractArray{<:Real}) = x isa AbstractArray{Float32} ? x : Float32.(x)

# Move input to same device as weights when they differ (CPU→GPU).
function _to_weight_device(x::AbstractArray, w::AbstractArray)
    x isa Array && !(w isa Array) && return copyto!(similar(w, eltype(x), size(x)), x)
    return x
end

@concrete struct Linear
    weight # [out, in]
    bias   # Union{Vector, Nothing}
end
@layer Linear

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
    x_f = _to_weight_device(_as_f32_array(x), lin.weight)
    flat = reshape(x_f, :, in_features)
    y = flat * transpose(lin.weight)
    if lin.bias !== nothing
        y = y .+ reshape(lin.bias, 1, :)
    end
    return reshape(y, (size(x)[1:(ndims(x)-1)]..., size(lin.weight, 1)))
end

@concrete struct LinearNoBias
    weight # [out, in]
end
@layer LinearNoBias

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
    x_f = _to_weight_device(_as_f32_array(x), lin.weight)
    flat = reshape(x_f, :, in_features)
    y = flat * transpose(lin.weight)
    return reshape(y, (size(x)[1:(ndims(x)-1)]..., size(lin.weight, 1)))
end

@concrete struct LayerNorm
    weight
    bias
    eps
end
@layer LayerNorm

function LayerNorm(c::Int; eps::Real = 1f-5)
    c > 0 || error("LayerNorm channel must be positive")
    return LayerNorm(ones(Float32, c), zeros(Float32, c), Float32(eps))
end

function (ln::LayerNorm)(x::AbstractArray{<:Real})
    c = size(x, ndims(x))
    length(ln.weight) == c || error("LayerNorm weight length mismatch")
    length(ln.bias) == c || error("LayerNorm bias length mismatch")
    x_f = _to_weight_device(_as_f32_array(x), ln.weight)
    flat = reshape(x_f, :, c)
    μ = mean(flat; dims=2)
    diff = flat .- μ
    v = mean(diff .^ 2; dims=2)
    inv_std = 1f0 ./ sqrt.(v .+ ln.eps)
    out = diff .* inv_std .* reshape(ln.weight, 1, :) .+ reshape(ln.bias, 1, :)
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
