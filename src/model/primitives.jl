module Primitives

using Random
using Statistics
using ConcreteStructs
using Flux: @layer

export LinearNoBias,
    LayerNormNoOffset,
    AdaptiveLayerNorm,
    Transition,
    silu

silu(x::Real) = x / (1 + exp(-x))
silu(x::AbstractArray) = x ./ (1 .+ exp.(-x))

_as_f32_array(x::AbstractArray{<:Real}) = x isa AbstractArray{Float32} ? x : Float32.(x)

@concrete struct LinearNoBias
    weight # [out_features, in_features]
end
@layer LinearNoBias

function LinearNoBias(out_features::Int, in_features::Int; rng::AbstractRNG = Random.default_rng())
    out_features > 0 || error("out_features must be positive.")
    in_features > 0 || error("in_features must be positive.")
    # Lightweight init for scaffold modules; checkpoint load will overwrite.
    w = 0.02f0 .* randn(rng, Float32, out_features, in_features)
    return LinearNoBias(w)
end

function (linear::LinearNoBias)(x::AbstractArray{<:Real})
    in_features = size(x, ndims(x))
    size(linear.weight, 2) == in_features ||
        error("LinearNoBias shape mismatch: expected input dim $(size(linear.weight, 2)), got $in_features")

    flat = reshape(_as_f32_array(x), :, in_features)
    y = flat * transpose(linear.weight) # [*, in] x [in, out]
    out_shape = (size(x)[1:(ndims(x)-1)]..., size(linear.weight, 1))
    return reshape(y, out_shape)
end

@concrete struct LayerNormNoOffset
    weight
    bias
    eps
end
@layer LayerNormNoOffset

function LayerNormNoOffset(c::Int; eps::Real = 1f-5, with_bias::Bool = false)
    c > 0 || error("LayerNorm channel dim must be positive.")
    b = with_bias ? zeros(Float32, c) : zeros(Float32, c)
    return LayerNormNoOffset(ones(Float32, c), b, Float32(eps))
end

function (ln::LayerNormNoOffset)(x::AbstractArray{<:Real})
    c = size(x, ndims(x))
    length(ln.weight) == c || error("LayerNorm weight length mismatch: $(length(ln.weight)) vs $c")
    length(ln.bias) == c || error("LayerNorm bias length mismatch: $(length(ln.bias)) vs $c")

    flat = reshape(_as_f32_array(x), :, c)
    μ = mean(flat; dims=2)
    diff = flat .- μ
    v = mean(diff .^ 2; dims=2)
    inv_std = 1f0 ./ sqrt.(v .+ ln.eps)
    out = diff .* inv_std .* reshape(ln.weight, 1, :) .+ reshape(ln.bias, 1, :)
    return reshape(out, size(x))
end

@concrete struct AdaptiveLayerNorm
    layernorm_a
    layernorm_s
    linear_s
    linear_nobias_s
    bias_s # explicit bias for linear_s
end
@layer AdaptiveLayerNorm

function AdaptiveLayerNorm(c_a::Int, c_s::Int; rng::AbstractRNG = Random.default_rng())
    return AdaptiveLayerNorm(
        LayerNormNoOffset(c_a),
        LayerNormNoOffset(c_s),
        LinearNoBias(c_a, c_s; rng = rng),
        LinearNoBias(c_a, c_s; rng = rng),
        zeros(Float32, c_a),
    )
end

function (ada::AdaptiveLayerNorm)(a::AbstractArray{<:Real}, s::AbstractArray{<:Real})
    size(a, ndims(a)) == length(ada.bias_s) || error("AdaptiveLayerNorm a-channel mismatch.")
    size(s, ndims(s)) == size(ada.linear_s.weight, 2) || error("AdaptiveLayerNorm s-channel mismatch.")
    size(a)[1:(ndims(a)-1)] == size(s)[1:(ndims(s)-1)] || error("AdaptiveLayerNorm batch/token dims must align.")

    a0 = ada.layernorm_a(a)
    s0 = ada.layernorm_s(s)

    g = ada.linear_s(s0)
    g = g .+ reshape(ada.bias_s, ntuple(_ -> 1, ndims(g)-1)..., :)
    shift = ada.linear_nobias_s(s0)
    return (1f0 ./ (1f0 .+ exp.(-g))) .* a0 .+ shift
end

@concrete struct Transition
    n
    c_in
    layernorm
    linear_a
    linear_b
    linear_out
end
@layer Transition

function Transition(c_in::Int, n::Int; rng::AbstractRNG = Random.default_rng())
    c_in > 0 || error("Transition c_in must be positive.")
    n > 0 || error("Transition n must be positive.")
    return Transition(
        n,
        c_in,
        LayerNormNoOffset(c_in),
        LinearNoBias(n * c_in, c_in; rng = rng),
        LinearNoBias(n * c_in, c_in; rng = rng),
        LinearNoBias(c_in, n * c_in; rng = rng),
    )
end

function (tr::Transition)(x::AbstractArray{<:Real})
    size(x, ndims(x)) == tr.c_in || error("Transition input channel mismatch.")
    x0 = tr.layernorm(x)
    a = tr.linear_a(x0)
    b = tr.linear_b(x0)
    return tr.linear_out(silu(a) .* b)
end

end
