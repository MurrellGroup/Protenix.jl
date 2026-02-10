module DiffusionConditioningModule

using Random

import ..Embedders: RelativePositionEncoding, fourier_embedding
import ..Primitives: LayerNormNoOffset, LinearNoBias, Transition

export DiffusionConditioning, prepare_pair_cache

struct DiffusionConditioning
    sigma_data::Float32
    c_z::Int
    c_s::Int
    c_s_inputs::Int
    c_noise_embedding::Int
    relpe::RelativePositionEncoding
    layernorm_z::LayerNormNoOffset
    linear_no_bias_z::LinearNoBias
    transition_z1::Transition
    transition_z2::Transition
    layernorm_s::LayerNormNoOffset
    linear_no_bias_s::LinearNoBias
    w_noise::Vector{Float32}
    b_noise::Vector{Float32}
    layernorm_n::LayerNormNoOffset
    linear_no_bias_n::LinearNoBias
    transition_s1::Transition
    transition_s2::Transition
end

function DiffusionConditioning(
    sigma_data::Real = 16.0;
    c_z::Int = 128,
    c_s::Int = 384,
    c_s_inputs::Int = 449,
    c_noise_embedding::Int = 256,
    r_max::Int = 32,
    s_max::Int = 2,
    rng::AbstractRNG = Random.default_rng(),
)
    relpe = RelativePositionEncoding(r_max, s_max, c_z; rng = rng)
    return DiffusionConditioning(
        Float32(sigma_data),
        c_z,
        c_s,
        c_s_inputs,
        c_noise_embedding,
        relpe,
        LayerNormNoOffset(2 * c_z),
        LinearNoBias(c_z, 2 * c_z; rng = rng),
        Transition(c_z, 2; rng = rng),
        Transition(c_z, 2; rng = rng),
        LayerNormNoOffset(c_s + c_s_inputs),
        LinearNoBias(c_s, c_s + c_s_inputs; rng = rng),
        randn(rng, Float32, c_noise_embedding),
        randn(rng, Float32, c_noise_embedding),
        LayerNormNoOffset(c_noise_embedding),
        LinearNoBias(c_s, c_noise_embedding; rng = rng),
        Transition(c_s, 2; rng = rng),
        Transition(c_s, 2; rng = rng),
    )
end

function _cat_lastdim(a::Array{Float32,3}, b::Array{Float32,3})
    size(a, 1) == size(b, 1) || error("cat_lastdim axis-1 mismatch")
    size(a, 2) == size(b, 2) || error("cat_lastdim axis-2 mismatch")
    return cat(a, b; dims = 3)
end

function _cat_lastdim(a::Array{Float32,2}, b::Array{Float32,2})
    size(a, 1) == size(b, 1) || error("cat_lastdim axis-1 mismatch")
    return cat(a, b; dims = 2)
end

function prepare_pair_cache(
    cond::DiffusionConditioning,
    relpos_input::NamedTuple{(:asym_id, :residue_index, :entity_id, :sym_id, :token_index)},
    z_trunk::AbstractArray{<:Real,3},
)
    size(z_trunk, 3) == cond.c_z || error("z_trunk last dim must be c_z=$(cond.c_z)")
    zf = Float32.(z_trunk)
    rel = cond.relpe(relpos_input) # [N_token, N_token, c_z]
    pair_z = _cat_lastdim(zf, rel) # [N_token, N_token, 2*c_z]
    pair_z = cond.linear_no_bias_z(cond.layernorm_z(pair_z)) # [N_token, N_token, c_z]
    pair_z = pair_z + cond.transition_z1(pair_z)
    pair_z = pair_z + cond.transition_z2(pair_z)
    return pair_z
end

function _expand_noise_to_tokens(single_s::Array{Float32,2}, n_token::Int)
    n_sample, c_s = size(single_s)
    out = Array{Float32}(undef, n_sample, n_token, c_s)
    for i in 1:n_sample
        for t in 1:n_token
            @inbounds out[i, t, :] = single_s[i, :]
        end
    end
    return out
end

function (cond::DiffusionConditioning)(
    t_hat_noise_level::AbstractVector{<:Real},
    relpos_input::NamedTuple{(:asym_id, :residue_index, :entity_id, :sym_id, :token_index)},
    s_inputs::AbstractArray{<:Real,2},
    s_trunk::AbstractArray{<:Real,2},
    z_trunk::AbstractArray{<:Real,3};
    pair_z = nothing,
    use_conditioning::Bool = true,
)
    size(s_inputs, 1) == size(s_trunk, 1) || error("s_inputs/s_trunk token axis mismatch.")
    size(s_inputs, 2) == cond.c_s_inputs || error("s_inputs last dim must be c_s_inputs=$(cond.c_s_inputs)")
    size(s_trunk, 2) == cond.c_s || error("s_trunk last dim must be c_s=$(cond.c_s)")
    n_token = size(s_inputs, 1)

    s_trunk_f = Float32.(s_trunk)
    z_trunk_f = Float32.(z_trunk)
    if !use_conditioning
        s_trunk_f .= 0f0
        z_trunk_f .= 0f0
    end

    pair_cache = pair_z === nothing ? prepare_pair_cache(cond, relpos_input, z_trunk_f) : Float32.(pair_z)
    size(pair_cache, 3) == cond.c_z || error("pair_z last dim must be c_z=$(cond.c_z)")

    single_s = _cat_lastdim(s_trunk_f, Float32.(s_inputs))             # [N_token, c_s + c_s_inputs]
    single_s = cond.linear_no_bias_s(cond.layernorm_s(single_s))       # [N_token, c_s]

    noise_input = log.(Float32.(t_hat_noise_level) ./ cond.sigma_data) ./ 4f0
    noise_emb = fourier_embedding(noise_input, cond.w_noise, cond.b_noise) # [N_sample, c_noise_embedding]
    noise_proj = cond.linear_no_bias_n(cond.layernorm_n(noise_emb))         # [N_sample, c_s]

    base = reshape(single_s, 1, n_token, cond.c_s) .+ _expand_noise_to_tokens(noise_proj, n_token)
    base = base + cond.transition_s1(base)
    base = base + cond.transition_s2(base)
    return base, pair_cache
end

end
