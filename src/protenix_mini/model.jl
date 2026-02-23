module Model

using Random
using ConcreteStructs
using Flux: @layer

import ..Primitives: LinearNoBias, LayerNorm
import ..Features: ProtenixFeatures, as_protenix_features, features_to_device, relpos_input, atom_attention_input
import ..Embedders: InputFeatureEmbedder, RelativePositionEncoding
import ..Pairformer: TemplateEmbedder, NoisyStructureEmbedder, MSAModule, PairformerStack
import ..Heads: DistogramHead, ConfidenceHead
import ..Constraint: ConstraintEmbedder
import ...Model: DiffusionModule, InferenceNoiseScheduler, sample_diffusion

export ProtenixMiniModel, get_pairformer_output, run_inference

_as_f32_array(x::AbstractArray{<:Real}) = x isa AbstractArray{Float32} ? x : Float32.(x)

@concrete struct ProtenixMiniModel
    n_cycle
    input_embedder
    relative_position_encoding
    template_embedder
    noisy_structure_embedder
    msa_module
    constraint_embedder
    pairformer_stack
    diffusion_module
    distogram_head
    confidence_head
    linear_no_bias_sinit
    linear_no_bias_zinit1
    linear_no_bias_zinit2
    linear_no_bias_token_bond
    linear_no_bias_z_cycle
    linear_no_bias_s
    layernorm_z_cycle
    layernorm_s
    diffusion_batch_size
    sample_gamma0
    sample_gamma_min
    sample_noise_scale_lambda
    sample_step_scale_eta
    sample_n_step
    sample_n_sample
    scheduler
end
@layer ProtenixMiniModel

# Detect a reference GPU array from model weights (returns nothing if CPU).
function _model_dev_ref(model::ProtenixMiniModel)
    dm = model.diffusion_module
    dm === nothing && return nothing
    w = dm.linear_no_bias_out.weight
    return w isa Array ? nothing : w
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
    constraint_enable::Bool = false,
    constraint_substructure_enable::Bool = false,
    constraint_substructure_architecture::Symbol = :linear,
    constraint_substructure_hidden_dim::Int = 128,
    constraint_substructure_n_layers::Int = 1,
    constraint_substructure_n_heads::Int = 4,
    template_blocks::Int = 0,
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
    template_embedder = TemplateEmbedder(c_z; n_blocks = template_blocks, c = 64, rng = rng)
    noisy = nothing
    msa = MSAModule(c_z, c_s_inputs; n_blocks = msa_blocks, rng = rng)
    constraint = constraint_enable ? ConstraintEmbedder(
        c_z;
        pocket_enable = true,
        contact_enable = true,
        contact_atom_enable = true,
        substructure_enable = constraint_substructure_enable,
        substructure_architecture = constraint_substructure_architecture,
        substructure_hidden_dim = constraint_substructure_hidden_dim,
        substructure_n_layers = constraint_substructure_n_layers,
        substructure_n_heads = constraint_substructure_n_heads,
        initialize_method = :zero,
        rng = rng,
    ) : nothing
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
        constraint,
        pairformer,
        dm,
        DistogramHead(c_z; no_bins = 64, rng = rng),
        ConfidenceHead(c_s, c_z, c_s_inputs; n_blocks = 4, max_atoms_per_token = confidence_max_atoms_per_token, rng = rng),
        LinearNoBias(c_s_inputs, c_s; rng = rng),   # c_s_inputs → c_s
        LinearNoBias(c_s, c_z; rng = rng),           # c_s → c_z
        LinearNoBias(c_s, c_z; rng = rng),           # c_s → c_z
        LinearNoBias(1, c_z; rng = rng),             # 1 → c_z
        LinearNoBias(c_z, c_z; rng = rng),           # c_z → c_z
        LinearNoBias(c_s, c_s; rng = rng),           # c_s → c_s
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
        return reshape(m, n, 1) .* reshape(m, 1, n)
    end
    n = length(feat.token_index)
    return fill!(similar(feat.restype, Float32, n, n), 1f0)
end

"""
Features-first pairformer trunk.
Returns (s_inputs, s, z) all in features-first layout:
  s_inputs: (c_s_inputs, N_tok)
  s:        (c_s, N_tok)
  z:        (c_z, N_tok, N_tok)
"""
function get_pairformer_output(
    model::ProtenixMiniModel,
    feat::ProtenixFeatures;
    n_cycle::Int = model.n_cycle,
    rng::AbstractRNG = Random.default_rng(),
)
    n_cycle > 0 || error("n_cycle must be positive")

    # InputFeatureEmbedder returns features-first (c_s_inputs, N_tok)
    s_inputs = model.input_embedder(feat)
    n_tok = size(s_inputs, 2)

    s_init = model.linear_no_bias_sinit(s_inputs)  # (c_s, N_tok)
    z1 = model.linear_no_bias_zinit1(s_init)       # (c_z, N_tok)
    z2 = model.linear_no_bias_zinit2(s_init)       # (c_z, N_tok)
    # Outer sum: (c_z, N, 1) + (c_z, 1, N) → (c_z, N, N)
    c_z = size(z1, 1)
    z_init = reshape(z1, c_z, n_tok, 1) .+ reshape(z2, c_z, 1, n_tok)

    # RelativePositionEncoding returns features-first (c_z, N, N) — no permutedims needed
    z_init .+= model.relative_position_encoding(relpos_input(feat))

    token_bonds = feat.token_bonds  # (N_tok, N_tok)
    size(token_bonds) == (n_tok, n_tok) || error("token_bonds shape mismatch")
    # Reshape to (1, N_tok, N_tok) for linear on dim=1: 1 → c_z
    z_init .+= model.linear_no_bias_token_bond(reshape(token_bonds, 1, n_tok, n_tok))

    if model.constraint_embedder !== nothing && feat.constraint_feature !== nothing
        z_constraint = model.constraint_embedder(feat.constraint_feature)
        if z_constraint !== nothing
            size(z_constraint) == size(z_init) ||
                error("constraint z shape mismatch: expected $(size(z_init)), got $(size(z_constraint))")
            z_init .+= z_constraint
        end
    end

    z = fill!(similar(z_init), 0f0)
    s = fill!(similar(s_init), 0f0)
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
All tensors are features-first throughout — no permutedims bridges.
"""
function run_inference(
    model::ProtenixMiniModel,
    feat::ProtenixFeatures;
    n_cycle::Int = model.n_cycle,
    n_step::Int = model.sample_n_step,
    n_sample::Int = model.sample_n_sample,
    rng::AbstractRNG = Random.default_rng(),
)
    # Transfer features to model device (GPU if model is on GPU).
    dev_ref = _model_dev_ref(model)
    if dev_ref !== nothing
        feat = features_to_device(feat, dev_ref)
    end

    trunk = get_pairformer_output(model, feat; n_cycle = n_cycle, rng = rng)

    relpos = relpos_input(feat)
    atom_input = atom_attention_input(feat)
    atom_to_token_idx = feat.atom_to_token_idx
    n_atom = length(atom_to_token_idx)

    noise_schedule = model.scheduler(n_step)

    denoise = (x_noisy, t_hat; kwargs...) -> model.diffusion_module(x_noisy, t_hat; kwargs...)
    # Everything is features-first — pass directly to DiffusionModule
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
        device_ref = dev_ref,
        relpos_input = relpos,
        s_inputs = trunk.s_inputs,       # (c_s_inputs, N_tok) features-first
        s_trunk = trunk.s,               # (c_s, N_tok) features-first
        z_trunk = trunk.z,               # (c_z, N_tok, N_tok) features-first
        atom_to_token_idx = atom_to_token_idx,
        input_feature_dict = atom_input,
    )
    # coords is (3, N_atom, N_sample) features-first

    distogram_logits = model.distogram_head(trunk.z)
    pair_mask = _pair_mask(feat)
    plddt, pae, pde, resolved = model.confidence_head(
        input_feature_dict = feat,
        s_inputs = trunk.s_inputs,
        s_trunk = trunk.s,
        z_trunk = trunk.z,
        pair_mask = pair_mask,
        x_pred_coords = coords,  # (3, N_atom, N_sample) features-first
    )

    return (
        coordinate = coords,             # (3, N_atom, N_sample) features-first
        s_inputs = trunk.s_inputs,       # (c_s_inputs, N_tok) features-first
        s_trunk = trunk.s,               # (c_s, N_tok) features-first
        z_trunk = trunk.z,               # (c_z, N_tok, N_tok) features-first
        distogram_logits = distogram_logits,  # (no_bins, N, N) features-first
        plddt = plddt,                   # (b_plddt, N_atom, N_sample)
        pae = pae,                       # (b_pae, N, N, N_sample)
        pde = pde,                       # (b_pde, N, N, N_sample)
        resolved = resolved,             # (b_resolved, N_atom, N_sample)
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
