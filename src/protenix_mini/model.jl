module Model

using Random

import ..Primitives: LinearNoBias, LayerNorm
import ..Features: ProtenixFeatures, as_protenix_features, relpos_input, atom_attention_input
import ..Embedders: InputFeatureEmbedder, RelativePositionEncoding
import ..Pairformer: TemplateEmbedder, NoisyStructureEmbedder, MSAModule, PairformerStack
import ..Heads: DistogramHead, ConfidenceHead
import ...Model: DiffusionModule, InferenceNoiseScheduler, sample_diffusion

export ProtenixMiniModel, get_pairformer_output, run_inference

_as_f32_array(x::AbstractArray{<:Real}) = x isa AbstractArray{Float32} ? x : Float32.(x)

struct ProtenixMiniModel
    n_cycle::Int
    input_embedder::InputFeatureEmbedder
    relative_position_encoding::RelativePositionEncoding
    template_embedder::TemplateEmbedder
    noisy_structure_embedder::Union{NoisyStructureEmbedder, Nothing}
    msa_module::MSAModule
    pairformer_stack::PairformerStack
    diffusion_module::DiffusionModule
    distogram_head::DistogramHead
    confidence_head::ConfidenceHead
    linear_no_bias_sinit::LinearNoBias
    linear_no_bias_zinit1::LinearNoBias
    linear_no_bias_zinit2::LinearNoBias
    linear_no_bias_token_bond::LinearNoBias
    linear_no_bias_z_cycle::LinearNoBias
    linear_no_bias_s::LinearNoBias
    layernorm_z_cycle::LayerNorm
    layernorm_s::LayerNorm
    diffusion_batch_size::Int
    sample_gamma0::Float32
    sample_gamma_min::Float32
    sample_noise_scale_lambda::Float32
    sample_step_scale_eta::Float32
    sample_n_step::Int
    sample_n_sample::Int
    scheduler::InferenceNoiseScheduler
end

function ProtenixMiniModel(
    c_token_diffusion::Int,
    c_token_input::Int,
    c_s::Int,
    c_z::Int,
    c_s_inputs::Int;
    c_atom::Int,
    c_atompair::Int,
    n_cycle::Int = 4,
    pairformer_blocks::Int = 16,
    msa_blocks::Int = 1,
    diffusion_transformer_blocks::Int = 8,
    diffusion_atom_encoder_blocks::Int = 1,
    diffusion_atom_decoder_blocks::Int = 1,
    confidence_max_atoms_per_token::Int = 20,
    sample_gamma0::Real = 0.0,
    sample_gamma_min::Real = 1.0,
    sample_noise_scale_lambda::Real = 1.003,
    sample_step_scale_eta::Real = 1.0,
    sample_n_step::Int = 5,
    sample_n_sample::Int = 1,
    input_esm_enable::Bool = false,
    input_esm_embedding_dim::Int = 2560,
    rng::AbstractRNG = Random.default_rng(),
)
    input_embedder = InputFeatureEmbedder(
        c_atom,
        c_atompair,
        c_token_input;
        esm_enable = input_esm_enable,
        esm_embedding_dim = input_esm_embedding_dim,
        rng = rng,
    )
    relpos = RelativePositionEncoding(32, 2, c_z; rng = rng)
    template_embedder = TemplateEmbedder(c_z; n_blocks = 0, c = 64, rng = rng)
    noisy = nothing
    msa = MSAModule(c_z, c_s_inputs; n_blocks = msa_blocks, rng = rng)
    pairformer = PairformerStack(c_z, c_s; n_blocks = pairformer_blocks, n_heads = 16, rng = rng)

    dm = DiffusionModule(
        c_token_diffusion,
        c_s,
        c_z,
        c_s_inputs;
        c_atom = c_atom,
        c_atompair = c_atompair,
        atom_encoder_blocks = diffusion_atom_encoder_blocks,
        atom_encoder_heads = 4,
        n_blocks = diffusion_transformer_blocks,
        n_heads = 16,
        atom_decoder_blocks = diffusion_atom_decoder_blocks,
        atom_decoder_heads = 4,
        rng = rng,
    )

    return ProtenixMiniModel(
        n_cycle,
        input_embedder,
        relpos,
        template_embedder,
        noisy,
        msa,
        pairformer,
        dm,
        DistogramHead(c_z; no_bins = 64, rng = rng),
        ConfidenceHead(c_s, c_z, c_s_inputs; n_blocks = 4, max_atoms_per_token = confidence_max_atoms_per_token, rng = rng),
        LinearNoBias(c_s, c_s_inputs; rng = rng),
        LinearNoBias(c_z, c_s; rng = rng),
        LinearNoBias(c_z, c_s; rng = rng),
        LinearNoBias(c_z, 1; rng = rng),
        LinearNoBias(c_z, c_z; rng = rng),
        LinearNoBias(c_s, c_s; rng = rng),
        LayerNorm(c_z),
        LayerNorm(c_s),
        48,
        Float32(sample_gamma0),
        Float32(sample_gamma_min),
        Float32(sample_noise_scale_lambda),
        Float32(sample_step_scale_eta),
        sample_n_step,
        sample_n_sample,
        InferenceNoiseScheduler(),
    )
end

function _pair_mask(feat::ProtenixFeatures)
    if feat.token_mask !== nothing
        m = feat.token_mask
        n = length(m)
        out = zeros(Float32, n, n)
        @inbounds for i in 1:n, j in 1:n
            out[i, j] = m[i] * m[j]
        end
        return out
    end
    n = length(feat.token_index)
    return ones(Float32, n, n)
end

function get_pairformer_output(
    model::ProtenixMiniModel,
    feat::ProtenixFeatures;
    n_cycle::Int = model.n_cycle,
    rng::AbstractRNG = Random.default_rng(),
)
    n_cycle > 0 || error("n_cycle must be positive")

    s_inputs = model.input_embedder(feat)
    n_tok = size(s_inputs, 1)

    s_init = model.linear_no_bias_sinit(s_inputs)
    z1 = model.linear_no_bias_zinit1(s_init)
    z2 = model.linear_no_bias_zinit2(s_init)
    z_init = reshape(z1, n_tok, 1, size(z1, 2)) .+ reshape(z2, 1, n_tok, size(z2, 2))

    z_init .+= model.relative_position_encoding(relpos_input(feat))

    token_bonds = feat.token_bonds
    size(token_bonds) == (n_tok, n_tok) || error("token_bonds shape mismatch")
    z_init .+= model.linear_no_bias_token_bond(reshape(token_bonds, n_tok, n_tok, 1))

    z = zeros(Float32, size(z_init))
    s = zeros(Float32, size(s_init))
    pair_mask = _pair_mask(feat)

    for _ in 1:n_cycle
        z = z_init .+ model.linear_no_bias_z_cycle(model.layernorm_z_cycle(z))
        if model.noisy_structure_embedder !== nothing
            z .+= model.noisy_structure_embedder(feat, z)
        end
        z .+= model.template_embedder(feat, z; pair_mask = pair_mask)
        z = model.msa_module(feat, z, s_inputs; pair_mask = pair_mask, rng = rng)
        s = s_init .+ model.linear_no_bias_s(model.layernorm_s(s))
        s_new, z = model.pairformer_stack(s, z; pair_mask = pair_mask)
        s_new === nothing && error("PairformerStack returned no single-state output")
        s = s_new
    end

    return (s_inputs = s_inputs, s = s, z = z)
end

function get_pairformer_output(
    model::ProtenixMiniModel,
    input_feature_dict::AbstractDict{<:AbstractString, <:Any};
    n_cycle::Int = model.n_cycle,
    rng::AbstractRNG = Random.default_rng(),
)
    return get_pairformer_output(model, as_protenix_features(input_feature_dict); n_cycle = n_cycle, rng = rng)
end

"""
Run full Protenix-mini inference loop (infer-only).
Expected input features are Protenix-style tensors already prepared in Julia.
"""
function run_inference(
    model::ProtenixMiniModel,
    feat::ProtenixFeatures;
    n_cycle::Int = model.n_cycle,
    n_step::Int = model.sample_n_step,
    n_sample::Int = model.sample_n_sample,
    rng::AbstractRNG = Random.default_rng(),
)
    trunk = get_pairformer_output(model, feat; n_cycle = n_cycle, rng = rng)

    relpos = relpos_input(feat)
    atom_input = atom_attention_input(feat)
    atom_to_token_idx = feat.atom_to_token_idx
    n_atom = length(atom_to_token_idx)

    noise_schedule = model.scheduler(n_step)

    denoise = (x_noisy, t_hat; kwargs...) -> model.diffusion_module(x_noisy, t_hat; kwargs...)
    coords = sample_diffusion(
        denoise;
        noise_schedule = noise_schedule,
        N_sample = n_sample,
        N_atom = n_atom,
        gamma0 = model.sample_gamma0,
        gamma_min = model.sample_gamma_min,
        noise_scale_lambda = model.sample_noise_scale_lambda,
        step_scale_eta = model.sample_step_scale_eta,
        rng = rng,
        relpos_input = relpos,
        s_inputs = trunk.s_inputs,
        s_trunk = trunk.s,
        z_trunk = trunk.z,
        atom_to_token_idx = atom_to_token_idx,
        input_feature_dict = atom_input,
    )

    distogram_logits = model.distogram_head(trunk.z)
    pair_mask = _pair_mask(feat)
    plddt, pae, pde, resolved = model.confidence_head(
        input_feature_dict = feat,
        s_inputs = trunk.s_inputs,
        s_trunk = trunk.s,
        z_trunk = trunk.z,
        pair_mask = pair_mask,
        x_pred_coords = coords,
    )

    return (
        coordinate = coords,
        s_inputs = trunk.s_inputs,
        s_trunk = trunk.s,
        z_trunk = trunk.z,
        distogram_logits = distogram_logits,
        plddt = plddt,
        pae = pae,
        pde = pde,
        resolved = resolved,
    )
end

function run_inference(
    model::ProtenixMiniModel,
    input_feature_dict::AbstractDict{<:AbstractString, <:Any};
    n_cycle::Int = model.n_cycle,
    n_step::Int = model.sample_n_step,
    n_sample::Int = model.sample_n_sample,
    rng::AbstractRNG = Random.default_rng(),
)
    return run_inference(
        model,
        as_protenix_features(input_feature_dict);
        n_cycle = n_cycle,
        n_step = n_step,
        n_sample = n_sample,
        rng = rng,
    )
end

end
