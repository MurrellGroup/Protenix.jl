module DiffusionModuleModule

using Random
using ConcreteStructs
using Flux: @layer

import ..DiffusionConditioningModule: DiffusionConditioning
import ..AtomAttentionModule: AtomAttentionEncoder, AtomAttentionDecoder
import ..Primitives: LayerNormFirst, LinearNoBias
import ..TransformerBlocks: DiffusionTransformer

export DiffusionModule

@concrete struct DiffusionModule
    c_token
    c_s
    c_z
    c_atom
    c_atompair
    diffusion_conditioning
    atom_attention_encoder
    layernorm_s
    linear_no_bias_s
    diffusion_transformer
    layernorm_a
    atom_attention_decoder
    linear_no_bias_out # token -> xyz delta
end
@layer DiffusionModule

function DiffusionModule(
    c_token::Int = 384,
    c_s::Int = 384,
    c_z::Int = 128,
    c_s_inputs::Int = 449;
    c_atom::Int = 128,
    c_atompair::Int = 16,
    atom_encoder_blocks::Int = 3,
    atom_encoder_heads::Int = 4,
    n_blocks::Int = 16,
    n_heads::Int = 16,
    atom_decoder_blocks::Int = 3,
    atom_decoder_heads::Int = 4,
    rng::AbstractRNG = Random.default_rng(),
)
    return DiffusionModule(
        c_token,
        c_s,
        c_z,
        c_atom,
        c_atompair,
        DiffusionConditioning(16.0; c_z = c_z, c_s = c_s, c_s_inputs = c_s_inputs, rng = rng),
        AtomAttentionEncoder(
            c_token;
            has_coords = true,
            c_atom = c_atom,
            c_atompair = c_atompair,
            c_s = c_s,
            c_z = c_z,
            n_blocks = atom_encoder_blocks,
            n_heads = atom_encoder_heads,
            rng = rng,
        ),
        LayerNormFirst(c_s),
        LinearNoBias(c_s, c_token),
        DiffusionTransformer(c_token, c_s, c_z; n_blocks = n_blocks, n_heads = n_heads, rng = rng),
        LayerNormFirst(c_token),
        AtomAttentionDecoder(
            c_token;
            c_atom = c_atom,
            c_atompair = c_atompair,
            n_blocks = atom_decoder_blocks,
            n_heads = atom_decoder_heads,
            rng = rng,
        ),
        LinearNoBias(c_token, 3),
    )
end

function _as_t_hat_vector(t_hat, n_sample::Int)
    if t_hat isa Real
        return fill(Float32(t_hat), n_sample)
    elseif t_hat isa AbstractVector
        length(t_hat) == n_sample || error("t_hat vector length mismatch with n_sample")
        return Float32.(t_hat)
    end
    error("Unsupported t_hat type $(typeof(t_hat))")
end

# Features-first: token_delta (3, N_token, N_sample) → (3, N_atom, N_sample)
function _token_to_atom_deltas(
    token_delta::AbstractArray{Float32,3},
    atom_to_token_idx::AbstractVector{<:Integer},
    n_atom::Int,
)
    size(token_delta, 1) == 3 || error("token_delta first dim must be 3.")
    idx = Int.(atom_to_token_idx) .+ 1
    all(i -> 1 <= i <= size(token_delta, 2), idx) || error("atom_to_token_idx out of token range.")
    return token_delta[:, idx, :]  # (3, N_atom, N_sample)
end

"""
Scaffold forward for diffusion denoising with typed module topology.

Features-first convention:
- `x_noisy`: `(3, N_atom, N_sample)`
- `s_inputs`: `(c_s_inputs, N_token)`
- `s_trunk`: `(c_s, N_token)`
- `z_trunk`: `(c_z, N_token, N_token)`
"""
function (dm::DiffusionModule)(
    x_noisy::AbstractArray{<:Real,3},
    t_hat;
    relpos_input::NamedTuple{(:asym_id, :residue_index, :entity_id, :sym_id, :token_index)},
    s_inputs::AbstractArray{<:Real,2},
    s_trunk::AbstractArray{<:Real,2},
    z_trunk::AbstractArray{<:Real,3},
    atom_to_token_idx::AbstractVector{<:Integer},
    input_feature_dict = nothing,
)
    # x_noisy: (3, N_atom, N_sample)
    n_atom = size(x_noisy, 2)
    n_sample = size(x_noisy, 3)
    length(atom_to_token_idx) == n_atom || error("atom_to_token_idx length must equal N_atom.")

    t_vec = _as_t_hat_vector(t_hat, n_sample)
    single_s, pair_z = dm.diffusion_conditioning(t_vec, relpos_input, s_inputs, s_trunk, z_trunk)
    # single_s: (c_s, N_token, N_sample), pair_z: (c_z, N_token, N_token)
    size(single_s, 1) == dm.c_s || error("single_s first dim mismatch.")

    if input_feature_dict !== nothing
        feat = input_feature_dict
        sigma_data = dm.diffusion_conditioning.sigma_data
        t_scale_cpu = sqrt.(sigma_data^2 .+ t_vec .^ 2)
        x_f = Float32.(x_noisy)  # (3, N_atom, N_sample)
        # Move t_scale to same device as x_f
        t_scale = copyto!(similar(x_f, Float32, length(t_scale_cpu)), t_scale_cpu)
        r_noisy = x_f ./ reshape(t_scale, 1, 1, n_sample)

        n_token = size(single_s, 2)
        s_trunk_f = Float32.(s_trunk)  # (c_s, N_token)
        # Broadcast trunk features across samples: (c_s, N_token, N_sample)
        s_tok = repeat(reshape(s_trunk_f, size(s_trunk_f, 1), n_token, 1), 1, 1, n_sample)
        # (c_z, N_token, N_token, N_sample)
        z_tok = repeat(reshape(pair_z, size(pair_z)..., 1), 1, 1, 1, n_sample)

        a_token, q_skip, c_skip, p_skip = dm.atom_attention_encoder(
            feat;
            r_l = r_noisy,
            s = s_tok,
            z = z_tok,
        )
        a_token .+= dm.linear_no_bias_s(dm.layernorm_s(single_s))

        a_token_out = similar(a_token)
        for i in 1:n_sample
            a_i = Float32.(a_token[:, :, i])     # (c_token, N_token)
            s_i = Float32.(single_s[:, :, i])    # (c_s, N_token)
            a_i = dm.diffusion_transformer(a_i, s_i, pair_z)
            a_token_out[:, :, i] .= dm.layernorm_a(a_i)
        end

        r_update = dm.atom_attention_decoder(feat, a_token_out, q_skip, c_skip, p_skip)
        # Move t_vec to device for broadcast with GPU tensors
        t_dev = copyto!(similar(x_f, Float32, length(t_vec)), t_vec)
        s_ratio = reshape(t_dev ./ sigma_data, 1, 1, n_sample)
        t_expanded = reshape(t_dev, 1, 1, n_sample)
        x_denoised = (1f0 ./ (1f0 .+ s_ratio .^ 2)) .* x_f .+
                     (t_expanded ./ sqrt.(1f0 .+ s_ratio .^ 2)) .* r_update
        return x_denoised
    end

    # Legacy fallback path used by module-level unit tests without full atom features.
    n_token = size(single_s, 2)
    token_xyz = similar(single_s, Float32, 3, n_token, n_sample)
    for si in 1:n_sample
        s_sample = Float32.(single_s[:, :, si])         # (c_s, N_token)
        a = dm.linear_no_bias_s(dm.layernorm_s(s_sample)) # (c_token, N_token)
        a = dm.diffusion_transformer(a, s_sample, pair_z)
        a = dm.layernorm_a(a)
        token_xyz[:, :, si] .= dm.linear_no_bias_out(a)  # (3, N_token)
    end

    atom_delta = _token_to_atom_deltas(token_xyz, atom_to_token_idx, n_atom)
    x_denoised = Float32.(x_noisy) .- atom_delta
    return x_denoised
end

end
