module ProtenixBase

include("protenix_base/model.jl")

using .Model:
    ProtenixBaseModel,
    infer_protenix_base_dims,
    build_protenix_base_model,
    load_protenix_base_model!,
    run_inference,
    build_sequence_atoms,
    build_sequence_feature_bundle,
    fold_sequence

export ProtenixBaseModel
export infer_protenix_base_dims, build_protenix_base_model, load_protenix_base_model!
export run_inference, build_sequence_atoms, build_sequence_feature_bundle, fold_sequence

end
