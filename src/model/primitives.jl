module Primitives

using Random
using Statistics
using ConcreteStructs
using Flux: @layer
using NNlib
import Onion

export silu,
    AdaptiveLayerNorm,
    LayerNormFirst,
    BGLinear,
    LinearNoBias,
    Transition

# Re-export Onion types used throughout PXDesign
const LayerNormFirst = Onion.LayerNormFirst
const BGLinear = Onion.BGLinear

# Convenience alias: LinearNoBias(in, out) returns a BGLinear with no bias.
# NOTE: argument order is (in_features, out_features) matching Onion convention.
# This differs from the OLD PXDesign convention which was (out_features, in_features).
LinearNoBias(in_dim::Int, out_dim::Int) = BGLinear(in_dim, out_dim; bias = false)

# Re-export Onion.Transition directly.
# Constructor: Transition(dim, hidden) where hidden = expansion_factor * dim.
# Old PXDesign: Transition(c_in, n) where hidden = n * c_in.
const Transition = Onion.Transition

silu(x::Real) = x / (1 + exp(-x))
silu(x::AbstractArray) = x ./ (1 .+ exp.(-x))

@concrete struct AdaptiveLayerNorm
    layernorm_a   # LayerNormFirst(c_a) — not loaded from checkpoint, stays identity-affine
    layernorm_s   # LayerNormFirst(c_s) — loaded from checkpoint
    linear_s      # BGLinear(c_s, c_a; bias=true) — sigmoid gate
    linear_nobias_s # BGLinear(c_s, c_a; bias=false) — shift
end
@layer AdaptiveLayerNorm

function AdaptiveLayerNorm(c_a::Int, c_s::Int; rng::AbstractRNG = Random.default_rng())
    c_a > 0 || error("AdaptiveLayerNorm c_a must be positive.")
    c_s > 0 || error("AdaptiveLayerNorm c_s must be positive.")
    return AdaptiveLayerNorm(
        LayerNormFirst(c_a),
        LayerNormFirst(c_s),
        BGLinear(c_s, c_a; bias = true),
        BGLinear(c_s, c_a; bias = false),
    )
end

function (ada::AdaptiveLayerNorm)(a::AbstractArray{<:Real}, s::AbstractArray{<:Real})
    # a: (c_a, N, ...), s: (c_s, N, ...)
    # Features-first: normalization along dim=1
    a0 = ada.layernorm_a(a)
    s0 = ada.layernorm_s(s)
    g = ada.linear_s(s0)            # (c_a, N, ...) — gate with bias
    shift = ada.linear_nobias_s(s0) # (c_a, N, ...) — shift
    return NNlib.sigmoid.(g) .* a0 .+ shift
end

end
