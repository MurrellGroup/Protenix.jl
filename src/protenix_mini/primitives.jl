module Primitives

using Random
import Onion

# Features-first convention: all layers operate on dim=1 (features), compatible with
# Julia's column-major memory layout.
#
# Re-export Onion types used throughout ProtenixMini.

export Linear,
    LinearNoBias,
    LayerNorm,
    transition,
    silu

# Linear with optional bias, features-first: (in_dim, *) → (out_dim, *)
const Linear = Onion.BGLinear

# Linear without bias, features-first: (in_dim, *) → (out_dim, *)
LinearNoBias(in_dim::Int, out_dim::Int; rng::AbstractRNG = Random.default_rng()) =
    Onion.BGLinear(in_dim, out_dim; bias = false)

# Layer normalization along dim=1 (features-first)
const LayerNorm = Onion.LayerNormFirst

silu(x::Real) = x / (1 + exp(-x))
silu(x::AbstractArray) = x ./ (1 .+ exp.(-x))

"""
Inference transition used throughout Protenix modules.
Features-first: (c_in, *) → (c_in, *)
"""
function transition(
    x::AbstractArray{<:Real},
    layernorm::LayerNorm,
    linear_a::Onion.BGLinear,
    linear_b::Onion.BGLinear,
    linear_out::Onion.BGLinear,
)
    x0 = layernorm(x)
    return linear_out(silu(linear_a(x0)) .* linear_b(x0))
end

end
