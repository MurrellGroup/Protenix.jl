module Model

using Random

import ...ProtenixMini: ProtenixMiniModel, run_inference as run_inference_mini
import ...ProtenixMini:
    infer_protenix_mini_dims,
    load_protenix_mini_model!,
    build_sequence_atoms as build_sequence_atoms_mini,
    build_sequence_feature_bundle as build_sequence_feature_bundle_mini

export ProtenixBaseModel
export infer_protenix_base_dims, build_protenix_base_model, load_protenix_base_model!, run_inference
export build_sequence_atoms, build_sequence_feature_bundle, fold_sequence

const ProtenixBaseModel = ProtenixMiniModel

function infer_protenix_base_dims(weights::AbstractDict{<:AbstractString,<:Any})
    return infer_protenix_mini_dims(weights)
end

"""
Build Protenix-base model scaffold from checkpoint-derived dimensions.
Defaults follow `protenix_base_default_v0.5.0` recommendations.
"""
function build_protenix_base_model(
    weights::AbstractDict{<:AbstractString,<:Any};
    n_cycle::Int = 10,
    sample_n_step::Int = 200,
    sample_n_sample::Int = 5,
    sample_gamma0::Real = 0.8,
    sample_gamma_min::Real = 1.0,
    sample_noise_scale_lambda::Real = 1.003,
    sample_step_scale_eta::Real = 1.5,
    rng::AbstractRNG = MersenneTwister(0),
)
    d = infer_protenix_base_dims(weights)
    return ProtenixMiniModel(
        d.c_token_diffusion,
        d.c_token_input,
        d.c_s,
        d.c_z,
        d.c_s_inputs;
        c_atom = d.c_atom,
        c_atompair = d.c_atompair,
        n_cycle = n_cycle,
        pairformer_blocks = d.pairformer_blocks,
        msa_blocks = max(d.msa_blocks, 1),
        diffusion_transformer_blocks = d.diffusion_blocks,
        diffusion_atom_encoder_blocks = d.diffusion_atom_encoder_blocks,
        diffusion_atom_decoder_blocks = d.diffusion_atom_decoder_blocks,
        confidence_max_atoms_per_token = d.max_atoms_per_token,
        sample_gamma0 = sample_gamma0,
        sample_gamma_min = sample_gamma_min,
        sample_noise_scale_lambda = sample_noise_scale_lambda,
        sample_step_scale_eta = sample_step_scale_eta,
        sample_n_step = sample_n_step,
        sample_n_sample = sample_n_sample,
        rng = rng,
    )
end

function load_protenix_base_model!(
    model::ProtenixBaseModel,
    weights::AbstractDict{<:AbstractString,<:Any};
    strict::Bool = true,
)
    return load_protenix_mini_model!(model, weights; strict = strict)
end

function run_inference(
    model::ProtenixBaseModel,
    input_feature_dict;
    n_cycle::Int = model.n_cycle,
    n_step::Int = model.sample_n_step,
    n_sample::Int = model.sample_n_sample,
    rng::AbstractRNG = Random.default_rng(),
)
    return run_inference_mini(
        model,
        input_feature_dict;
        n_cycle = n_cycle,
        n_step = n_step,
        n_sample = n_sample,
        rng = rng,
    )
end

function build_sequence_atoms(sequence::AbstractString; chain_id::String = "A0")
    return build_sequence_atoms_mini(sequence; chain_id = chain_id)
end

function build_sequence_feature_bundle(
    sequence::AbstractString;
    chain_id::String = "A0",
    task_name::String = "protenix_base_sequence",
    rng::AbstractRNG = Random.default_rng(),
)
    return build_sequence_feature_bundle_mini(
        sequence;
        chain_id = chain_id,
        task_name = task_name,
        rng = rng,
    )
end

"""
Run infer-only Protenix-base folding from a one-letter sequence.
"""
function fold_sequence(
    model::ProtenixBaseModel,
    sequence::AbstractString;
    chain_id::String = "A0",
    n_cycle::Int = model.n_cycle,
    n_step::Int = model.sample_n_step,
    n_sample::Int = model.sample_n_sample,
    rng::AbstractRNG = Random.default_rng(),
)
    bundle = build_sequence_feature_bundle(
        sequence;
        chain_id = chain_id,
        task_name = "protenix_base_sequence",
        rng = rng,
    )
    pred = run_inference(
        model,
        bundle["input_feature_dict"];
        n_cycle = n_cycle,
        n_step = n_step,
        n_sample = n_sample,
        rng = rng,
    )
    return (
        atoms = bundle["atoms"],
        feature_bundle = bundle,
        prediction = pred,
    )
end

end
