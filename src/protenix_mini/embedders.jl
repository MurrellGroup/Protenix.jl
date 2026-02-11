module Embedders

using Random

import ..Primitives: LinearNoBias
import ..Features: ProtenixFeatures, as_protenix_features, atom_attention_input
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

"""
Implements Protenix `InputFeatureEmbedder` forward.
Returns `s_inputs` with shape `[N_token, c_token + 32 + 32 + 1]`.
"""
function (m::InputFeatureEmbedder)(
    feat::ProtenixFeatures;
    chunk_size::Union{Nothing, Int} = nothing,
)
    a, _, _, _ = m.atom_attention_encoder(atom_attention_input(feat))
    n_tok = size(a, 1)

    restype = feat.restype
    profile = feat.profile
    deletion_mean = feat.deletion_mean

    size(restype, 1) == n_tok || error("restype token dim mismatch")
    size(restype, 2) == 32 || error("restype feature dim must be 32")
    size(profile, 1) == n_tok || error("profile token dim mismatch")
    size(profile, 2) == 32 || error("profile feature dim must be 32")
    length(deletion_mean) == n_tok || error("deletion_mean token dim mismatch")

    s_inputs = hcat(a, restype, profile, reshape(deletion_mean, n_tok, 1))

    if m.esm_enable
        feat.esm_token_embedding === nothing && error("Missing esm_token_embedding")
        m.linear_esm === nothing && error("linear_esm missing")
        esm_emb = m.linear_esm(feat.esm_token_embedding)
        s_inputs .+= esm_emb
    end

    return s_inputs
end

function (m::InputFeatureEmbedder)(
    input_feature_dict::AbstractDict{<:AbstractString, <:Any};
    chunk_size::Union{Nothing, Int} = nothing,
)
    return m(as_protenix_features(input_feature_dict); chunk_size = chunk_size)
end

end
