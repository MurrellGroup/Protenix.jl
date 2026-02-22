module DiffusionConditioningModule

using Random
using ConcreteStructs
using Flux: @layer
import Onion

import ..Embedders: RelativePositionEncoding, fourier_embedding
import ..Primitives: LayerNormFirst, BGLinear, LinearNoBias, Transition

export DiffusionConditioning, prepare_pair_cache

@concrete struct DiffusionConditioning
    sigma_data
    c_z
    c_s
    c_s_inputs
    c_noise_embedding
    relpe
    layernorm_z
    linear_no_bias_z
    transition_z1
    transition_z2
    layernorm_s
    linear_no_bias_s
    w_noise
    b_noise
    layernorm_n
    linear_no_bias_n
    transition_s1
    transition_s2
end
@layer DiffusionConditioning

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
        LayerNormFirst(2 * c_z),
        LinearNoBias(2 * c_z, c_z),
        Transition(c_z, 2 * c_z),
        Transition(c_z, 2 * c_z),
        LayerNormFirst(c_s + c_s_inputs),
        LinearNoBias(c_s + c_s_inputs, c_s),
        randn(rng, Float32, c_noise_embedding),
        randn(rng, Float32, c_noise_embedding),
        LayerNormFirst(c_noise_embedding),
        LinearNoBias(c_noise_embedding, c_s),
        Transition(c_s, 2 * c_s),
        Transition(c_s, 2 * c_s),
    )
end

function prepare_pair_cache(
    cond::DiffusionConditioning,
    relpos_input::NamedTuple{(:asym_id, :residue_index, :entity_id, :sym_id, :token_index)},
    z_trunk::AbstractArray{<:Real,3},
)
    # z_trunk: (c_z, N_token, N_token) — features-first
    size(z_trunk, 1) == cond.c_z || error("z_trunk first dim must be c_z=$(cond.c_z)")
    zf = Float32.(z_trunk)
    rel = cond.relpe(relpos_input) # (c_z, N_token, N_token)
    pair_z = cat(zf, rel; dims = 1) # (2*c_z, N_token, N_token)
    pair_z = cond.linear_no_bias_z(cond.layernorm_z(pair_z)) # (c_z, N_token, N_token)
    pair_z = pair_z + cond.transition_z1(pair_z)
    pair_z = pair_z + cond.transition_z2(pair_z)
    return pair_z
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
    # Features-first convention:
    # s_inputs: (c_s_inputs, N_token), s_trunk: (c_s, N_token)
    # z_trunk: (c_z, N_token, N_token)
    size(s_inputs, 1) == cond.c_s_inputs || error("s_inputs first dim must be c_s_inputs=$(cond.c_s_inputs)")
    size(s_trunk, 1) == cond.c_s || error("s_trunk first dim must be c_s=$(cond.c_s)")
    size(s_inputs, 2) == size(s_trunk, 2) || error("s_inputs/s_trunk token axis mismatch.")
    n_token = size(s_inputs, 2)

    s_trunk_f = Float32.(s_trunk)
    z_trunk_f = Float32.(z_trunk)
    if !use_conditioning
        s_trunk_f .= 0f0
        z_trunk_f .= 0f0
    end

    pair_cache = pair_z === nothing ? prepare_pair_cache(cond, relpos_input, z_trunk_f) : Float32.(pair_z)
    size(pair_cache, 1) == cond.c_z || error("pair_z first dim must be c_z=$(cond.c_z)")

    # Concatenate single features along dim=1 (features-first)
    single_s = cat(s_trunk_f, Float32.(s_inputs); dims = 1) # (c_s + c_s_inputs, N_token)
    single_s = cond.linear_no_bias_s(cond.layernorm_s(single_s)) # (c_s, N_token)

    # Noise embedding — features-first
    noise_cpu = log.(Float32.(t_hat_noise_level) ./ cond.sigma_data) ./ 4f0
    noise_input = copyto!(similar(cond.w_noise, Float32, size(noise_cpu)...), noise_cpu)
    noise_emb = fourier_embedding(noise_input, cond.w_noise, cond.b_noise) # (c_noise_embedding, N_sample)
    noise_proj = cond.linear_no_bias_n(cond.layernorm_n(noise_emb))         # (c_s, N_sample)

    n_sample = length(t_hat_noise_level)
    # Broadcast: (c_s, N_token, 1) .+ (c_s, 1, N_sample) → (c_s, N_token, N_sample)
    base = reshape(single_s, cond.c_s, n_token, 1) .+ reshape(noise_proj, cond.c_s, 1, n_sample)
    base = base + cond.transition_s1(base)
    base = base + cond.transition_s2(base)
    return base, pair_cache
end

end
