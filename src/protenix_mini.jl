module ProtenixMini

include("protenix_mini/utils.jl")
include("protenix_mini/primitives.jl")
include("protenix_mini/features.jl")
include("protenix_mini/constraint.jl")
include("protenix_mini/openfold_blocks.jl")
include("protenix_mini/embedders.jl")
include("protenix_mini/pairformer.jl")
include("protenix_mini/heads.jl")
include("protenix_mini/model.jl")
include("protenix_mini/state_load.jl")
include("protenix_mini/sequence.jl")

using .Utils:
    softmax_lastdim,
    softmax_dim2,
    one_hot_interval,
    one_hot_int,
    clamp01,
    broadcast_token_to_atom,
    aggregate_atom_to_token_mean,
    pairwise_distances,
    sample_msa_indices

using .Primitives: Linear, LinearNoBias, LayerNorm, transition, silu
using .Features: ProtenixFeatures, as_protenix_features, relpos_input, atom_attention_input
using .Constraint: ConstraintEmbedder
using .OpenFoldBlocks: TriangleMultiplication, TriangleAttention, OuterProductMean, PairAttentionNoS
using .Embedders: InputFeatureEmbedder, RelativePositionEncoding
using .Pairformer:
    TransitionBlock,
    PairformerBlock,
    PairformerStack,
    MSAPairWeightedAveraging,
    MSAStack,
    MSABlock,
    MSAModule,
    TemplateEmbedder,
    NoisyStructureEmbedder
using .Heads: DistogramHead, ConfidenceHead
using .Model: ProtenixMiniModel, get_pairformer_output, run_inference
using .StateLoad: infer_protenix_mini_dims, build_protenix_mini_model, load_protenix_mini_model!
using .Sequence: build_sequence_atoms, build_sequence_feature_bundle, fold_sequence

export softmax_lastdim,
    softmax_dim2,
    one_hot_interval,
    one_hot_int,
    clamp01,
    broadcast_token_to_atom,
    aggregate_atom_to_token_mean,
    pairwise_distances,
    sample_msa_indices

export Linear, LinearNoBias, LayerNorm, transition, silu
export ProtenixFeatures, as_protenix_features, relpos_input, atom_attention_input
export ConstraintEmbedder
export TriangleMultiplication, TriangleAttention, OuterProductMean, PairAttentionNoS
export InputFeatureEmbedder, RelativePositionEncoding
export TransitionBlock,
    PairformerBlock,
    PairformerStack,
    MSAPairWeightedAveraging,
    MSAStack,
    MSABlock,
    MSAModule,
    TemplateEmbedder,
    NoisyStructureEmbedder
export DistogramHead, ConfidenceHead
export ProtenixMiniModel, get_pairformer_output, run_inference
export infer_protenix_mini_dims, build_protenix_mini_model, load_protenix_mini_model!
export build_sequence_atoms, build_sequence_feature_bundle, fold_sequence

end
