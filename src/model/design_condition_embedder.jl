module DesignConditionEmbedderModule

using Random
using ConcreteStructs
using Flux: @layer

import ..Embedders: ConditionTemplateEmbedder, condition_template_embedding
import ..AtomAttentionModule: AtomAttentionEncoder

export DesignAtomAttentionEncoder, InputFeatureEmbedderDesign, DesignConditionEmbedder

# Convert to features-first: (features, N)
function _matrix_f32(x, name::String)
    x isa AbstractMatrix || error("$name must be a matrix, got $(typeof(x))")
    return permutedims(Float32.(x))  # (features, N) — features-first
end

# Convert vector to features-first: (1, N)
function _column_f32(x, name::String, n::Int)
    if x isa AbstractVector
        length(x) == n || error("$name length mismatch: expected $n, got $(length(x))")
        return reshape(Float32.(x), 1, n)  # (1, N) — features-first
    elseif x isa AbstractMatrix && size(x, 1) == n && size(x, 2) == 1
        return permutedims(Float32.(x))  # (1, N) — features-first
    end
    error("$name must be a vector or [N,1], got $(typeof(x))")
end

DesignAtomAttentionEncoder(
    c_token::Int;
    c_atom::Int = 128,
    c_atompair::Int = 16,
    n_blocks::Int = 3,
    n_heads::Int = 4,
    rng::AbstractRNG = Random.default_rng(),
) = AtomAttentionEncoder(
    c_token;
    has_coords = false,
    c_atom = c_atom,
    c_atompair = c_atompair,
    n_blocks = n_blocks,
    n_heads = n_heads,
    rng = rng,
)

@concrete struct InputFeatureEmbedderDesign
    c_s_inputs
    atom_attention_encoder
    input_map_weight # (c_s_inputs, c_token + 46) — features-first: weight * x
    input_map_bias   # (c_s_inputs,)
end
@layer InputFeatureEmbedderDesign

function InputFeatureEmbedderDesign(
    c_token::Int;
    c_s_inputs::Int = 449,
    c_atom::Int = 128,
    c_atompair::Int = 16,
    n_blocks::Int = 3,
    n_heads::Int = 4,
    rng::AbstractRNG = Random.default_rng(),
)
    in_features = c_token + (36 + 1 + 1 + 4 + 4)
    return InputFeatureEmbedderDesign(
        c_s_inputs,
        DesignAtomAttentionEncoder(
            c_token;
            c_atom = c_atom,
            c_atompair = c_atompair,
            n_blocks = n_blocks,
            n_heads = n_heads,
            rng = rng,
        ),
        0.02f0 .* randn(rng, Float32, c_s_inputs, in_features),
        zeros(Float32, c_s_inputs),
    )
end

function (embedder::InputFeatureEmbedderDesign)(input_feature_dict::AbstractDict{<:AbstractString, <:Any})
    haskey(input_feature_dict, "restype") || error("Missing input feature 'restype' for InputFeatureEmbedderDesign.")
    restype = _matrix_f32(input_feature_dict["restype"], "restype")  # (36, N_token)
    n_token = size(restype, 2)
    size(restype, 1) == 36 || error("restype must have shape (36, N_token)")

    plddt = haskey(input_feature_dict, "plddt") ?
            _column_f32(input_feature_dict["plddt"], "plddt", n_token) :
            fill!(similar(restype, Float32, 1, n_token), 0f0)
    hotspot = haskey(input_feature_dict, "hotspot") ?
              _column_f32(input_feature_dict["hotspot"], "hotspot", n_token) :
              fill!(similar(restype, Float32, 1, n_token), 0f0)
    add_feat = fill!(similar(restype, Float32, 4, n_token), 0f0)
    add_feat[1, :] .= 1f0

    a_token, _, _, _ = embedder.atom_attention_encoder(input_feature_dict)  # (c_token, N_token)
    size(a_token, 2) == n_token || error("Atom/token embedding count mismatch: $(size(a_token, 2)) vs $n_token")

    # Concatenate along dim 1 (features): (c_token + 46, N_token)
    s_cat = cat(a_token, restype, plddt, hotspot, add_feat, add_feat; dims = 1)
    size(s_cat, 1) == size(embedder.input_map_weight, 2) || error(
        "InputFeatureEmbedderDesign input_map mismatch: expected $(size(embedder.input_map_weight, 2)), got $(size(s_cat, 1))",
    )
    # Features-first: weight * x + bias → (c_s_inputs, N_token)
    return embedder.input_map_weight * s_cat .+ reshape(embedder.input_map_bias, :, 1)
end

@concrete struct DesignConditionEmbedder
    input_embedder
    condition_template_embedder
end
@layer DesignConditionEmbedder

function DesignConditionEmbedder(
    c_token::Int;
    c_s_inputs::Int = 449,
    c_z::Int = 128,
    c_atom::Int = 128,
    c_atompair::Int = 16,
    n_blocks::Int = 3,
    n_heads::Int = 4,
    rng::AbstractRNG = Random.default_rng(),
)
    return DesignConditionEmbedder(
        InputFeatureEmbedderDesign(
            c_token;
            c_s_inputs = c_s_inputs,
            c_atom = c_atom,
            c_atompair = c_atompair,
            n_blocks = n_blocks,
            n_heads = n_heads,
            rng = rng,
        ),
        ConditionTemplateEmbedder(65, c_z; rng = rng),
    )
end

function (embedder::DesignConditionEmbedder)(input_feature_dict::AbstractDict{<:AbstractString, <:Any})
    haskey(input_feature_dict, "conditional_templ") ||
        error("Missing input feature 'conditional_templ' for DesignConditionEmbedder.")
    haskey(input_feature_dict, "conditional_templ_mask") ||
        error("Missing input feature 'conditional_templ_mask' for DesignConditionEmbedder.")

    s_inputs = embedder.input_embedder(input_feature_dict)  # (c_s_inputs, N_token)
    z = condition_template_embedding(
        embedder.condition_template_embedder,
        Int.(input_feature_dict["conditional_templ"]),
        input_feature_dict["conditional_templ_mask"],
    )  # (c_z, N_i, N_j)
    return s_inputs, z
end

end
