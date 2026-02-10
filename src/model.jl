module Model

include("model/checkpoint_map.jl")
include("model/embedders.jl")
include("model/feature_views.jl")
include("model/primitives.jl")
include("model/diffusion_conditioning.jl")
include("model/transformer_blocks.jl")
include("model/atom_attention.jl")
include("model/design_condition_embedder.jl")
include("model/diffusion_module.jl")
include("model/raw_weights.jl")
include("model/parity.jl")
include("model/state_load.jl")
include("model/scheduler.jl")
include("model/sampler.jl")

using .CheckpointMap: expected_checkpoint_prefixes, load_checkpoint_index, checkpoint_prefix_counts
using .Embedders:
    ConditionTemplateEmbedder,
    RelativePositionEncoding,
    condition_template_embedding,
    relative_position_features,
    fourier_embedding
using .DesignConditionEmbedderModule:
    DesignAtomAttentionEncoder,
    InputFeatureEmbedderDesign,
    DesignConditionEmbedder
using .FeatureViews: as_relpos_input, as_template_input, as_atom_attention_input
using .Primitives: LinearNoBias, LayerNormNoOffset, AdaptiveLayerNorm, Transition, silu
using .DiffusionConditioningModule: DiffusionConditioning, prepare_pair_cache
using .TransformerBlocks:
    ConditionedTransitionBlock,
    AttentionPairBias,
    DiffusionTransformerBlock,
    DiffusionTransformer
using .AtomAttentionModule: AtomAttentionEncoder, AtomAttentionDecoder
using .DiffusionModuleModule: DiffusionModule
using .RawWeights: RawWeightEntry, load_raw_manifest, load_raw_weights
using .ParityHarness:
    TensorParityStats,
    TensorParityReport,
    tensor_parity_report,
    compare_raw_weight_dirs
using .StateLoad:
    load_condition_template_embedder!,
    load_design_condition_embedder!,
    load_relative_position_encoding!,
    load_diffusion_conditioning!,
    load_diffusion_transformer!,
    load_diffusion_module!,
    infer_model_scaffold_dims,
    infer_design_condition_embedder_dims,
    expected_diffusion_module_keys,
    expected_design_condition_embedder_keys,
    checkpoint_coverage_report
using .Scheduler: InferenceNoiseScheduler
using .Sampler: sample_diffusion

export InferenceNoiseScheduler, sample_diffusion
export expected_checkpoint_prefixes, load_checkpoint_index, checkpoint_prefix_counts
export ConditionTemplateEmbedder, RelativePositionEncoding
export condition_template_embedding, relative_position_features, fourier_embedding
export DesignAtomAttentionEncoder, InputFeatureEmbedderDesign, DesignConditionEmbedder
export as_relpos_input, as_template_input, as_atom_attention_input
export LinearNoBias, LayerNormNoOffset, AdaptiveLayerNorm, Transition, silu
export DiffusionConditioning, prepare_pair_cache
export ConditionedTransitionBlock
export AttentionPairBias, DiffusionTransformerBlock, DiffusionTransformer
export AtomAttentionEncoder, AtomAttentionDecoder
export DiffusionModule
export RawWeightEntry, load_raw_manifest, load_raw_weights
export TensorParityStats, TensorParityReport, tensor_parity_report, compare_raw_weight_dirs
export load_condition_template_embedder!,
    load_design_condition_embedder!,
    load_relative_position_encoding!,
    load_diffusion_conditioning!,
    load_diffusion_transformer!,
    load_diffusion_module!,
    infer_model_scaffold_dims,
    infer_design_condition_embedder_dims,
    expected_diffusion_module_keys,
    expected_design_condition_embedder_keys,
    checkpoint_coverage_report

end
