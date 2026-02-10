module Embedders

using Random

import ..Primitives: LinearNoBias
import ..Utils: one_hot_int
import ...Model: AtomAttentionEncoder, RelativePositionEncoding

export InputFeatureEmbedder, RelativePositionEncoding

struct InputFeatureEmbedder
    c_atom::Int
    c_atompair::Int
    c_token::Int
    atom_attention_encoder::AtomAttentionEncoder
    esm_enable::Bool
    linear_esm::Union{LinearNoBias, Nothing}
end

function InputFeatureEmbedder(
    c_atom::Int,
    c_atompair::Int,
    c_token::Int;
    esm_enable::Bool = false,
    esm_embedding_dim::Int = 2560,
    rng::AbstractRNG = Random.default_rng(),
)
    enc = AtomAttentionEncoder(
        c_token;
        has_coords = false,
        c_atom = c_atom,
        c_atompair = c_atompair,
        c_s = c_token,
        c_z = c_atompair,
        n_blocks = 3,
        n_heads = 4,
        rng = rng,
    )
    linear_esm = esm_enable ? LinearNoBias(c_token + 65, esm_embedding_dim; rng = rng) : nothing
    return InputFeatureEmbedder(c_atom, c_atompair, c_token, enc, esm_enable, linear_esm)
end

function _as_f32_matrix(x, name::String)
    x isa AbstractMatrix || error("$name must be matrix")
    return Float32.(x)
end

function _as_f32_vector(x, name::String)
    x isa AbstractVector || error("$name must be vector")
    return Float32.(x)
end

"""
Implements Protenix `InputFeatureEmbedder` forward.
Returns `s_inputs` with shape `[N_token, c_token + 32 + 32 + 1]`.
"""
function (m::InputFeatureEmbedder)(
    input_feature_dict::AbstractDict{<:AbstractString, <:Any};
    chunk_size::Union{Nothing, Int} = nothing,
)
    a, _, _, _ = m.atom_attention_encoder(input_feature_dict)
    n_tok = size(a, 1)

    restype = if haskey(input_feature_dict, "restype")
        _as_f32_matrix(input_feature_dict["restype"], "restype")
    elseif haskey(input_feature_dict, "restype_index")
        one_hot_int(Int.(input_feature_dict["restype_index"]), 32)
    else
        error("Missing restype/restype_index for InputFeatureEmbedder")
    end
    profile = haskey(input_feature_dict, "profile") ?
        _as_f32_matrix(input_feature_dict["profile"], "profile") : zeros(Float32, n_tok, 32)
    deletion_mean = haskey(input_feature_dict, "deletion_mean") ?
        _as_f32_vector(input_feature_dict["deletion_mean"], "deletion_mean") : zeros(Float32, n_tok)

    size(restype, 1) == n_tok || error("restype token dim mismatch")
    size(restype, 2) == 32 || error("restype feature dim must be 32")
    size(profile, 1) == n_tok || error("profile token dim mismatch")
    size(profile, 2) == 32 || error("profile feature dim must be 32")
    length(deletion_mean) == n_tok || error("deletion_mean token dim mismatch")

    s_inputs = hcat(a, restype, profile, reshape(deletion_mean, n_tok, 1))

    if m.esm_enable
        haskey(input_feature_dict, "esm_token_embedding") || error("Missing esm_token_embedding")
        m.linear_esm === nothing && error("linear_esm missing")
        esm_emb = m.linear_esm(_as_f32_matrix(input_feature_dict["esm_token_embedding"], "esm_token_embedding"))
        s_inputs .+= esm_emb
    end

    return s_inputs
end

end
