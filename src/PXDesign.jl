module PXDesign

include("config.jl")
include("jsonlite.jl")
include("ranges.jl")
include("cache.jl")
include("inputs.jl")
include("schema.jl")
include("data.jl")
include("esm_provider.jl")
include("weights_hub.jl")
include("device.jl")
include("model.jl")
include("protenix_mini.jl")
include("protenix_base.jl")
include("output.jl")
include("protenix_api.jl")
include("infer.jl")
include("cli.jl")

using .Device: to_device, zeros_like, ones_like, device_ref, feats_to_device, feats_to_cpu
using .Config: default_config, default_urls
using .Infer: run_infer
using .Schema: parse_tasks
using .Ranges: parse_ranges, format_ranges
using .Model: InferenceNoiseScheduler, sample_diffusion
using .WeightsHub: resolve_weight_source, download_model_weights
using .ProtenixMini: ProtenixMiniModel, run_inference, build_sequence_atoms, build_sequence_feature_bundle, fold_sequence
using .ProtenixBase:
    ProtenixBaseModel,
    infer_protenix_base_dims,
    build_protenix_base_model,
    load_protenix_base_model!,
    run_inference as run_inference_protenix_base,
    build_sequence_atoms as build_sequence_atoms_protenix_base,
    build_sequence_feature_bundle as build_sequence_feature_bundle_protenix_base,
    fold_sequence as fold_sequence_protenix_base
using .ProtenixAPI:
    ProtenixModelSpec,
    ProtenixPredictOptions,
    ProtenixSequenceOptions,
    MODEL_SPECS,
    resolve_model_spec,
    recommended_params,
    list_supported_models,
    default_weights_path,
    predict_json,
    predict_sequence,
    convert_structure_to_infer_json,
    add_precomputed_msa_to_json
using .CLI: main

export ProtenixMiniModel, run_inference, build_sequence_atoms, build_sequence_feature_bundle, fold_sequence
export ProtenixBaseModel
export infer_protenix_base_dims, build_protenix_base_model, load_protenix_base_model!
export run_inference_protenix_base, build_sequence_atoms_protenix_base, build_sequence_feature_bundle_protenix_base, fold_sequence_protenix_base
export ProtenixModelSpec,
    ProtenixPredictOptions,
    ProtenixSequenceOptions,
    MODEL_SPECS,
    resolve_model_spec,
    recommended_params,
    list_supported_models,
    default_weights_path,
    predict_json,
    predict_sequence,
    convert_structure_to_infer_json,
    add_precomputed_msa_to_json
export resolve_weight_source, download_model_weights
export to_device, zeros_like, ones_like, device_ref, feats_to_device, feats_to_cpu

end # module PXDesign
