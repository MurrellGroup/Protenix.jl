module Embedders

using Random
using ConcreteStructs
using Flux: @layer

import ..Primitives: LinearNoBias
import ..Features: ProtenixFeatures, as_protenix_features, atom_attention_input
import ...Model: AtomAttentionEncoder, RelativePositionEncoding

export InputFeatureEmbedder, RelativePositionEncoding

@concrete struct InputFeatureEmbedder
    c_atom
    c_atompair
    c_token
    atom_attention_encoder
    esm_enable
    linear_esm
end
@layer InputFeatureEmbedder

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
    # LinearNoBias(in, out): projects esm_embedding_dim → c_token + 65
    linear_esm = esm_enable ? LinearNoBias(esm_embedding_dim, c_token + 65; rng = rng) : nothing
    return InputFeatureEmbedder(c_atom, c_atompair, c_token, enc, esm_enable, linear_esm)
end

"""
Implements Protenix `InputFeatureEmbedder` forward.
Features-first: returns `s_inputs` with shape `(c_token + 65, N_token)`.
"""
function (m::InputFeatureEmbedder)(
    feat::ProtenixFeatures;
    chunk_size::Union{Nothing, Int} = nothing,
)
    # AtomAttentionEncoder returns features-first (c_token, N_token)
    a_ff, _, _, _ = m.atom_attention_encoder(atom_attention_input(feat))
    n_tok = size(a_ff, 2)

    # Features are features-first: restype (32, N_tok), profile (32, N_tok), deletion_mean (N_tok,)
    restype = feat.restype
    profile = feat.profile
    deletion_mean = feat.deletion_mean

    size(restype, 2) == n_tok || error("restype token dim mismatch")
    size(restype, 1) == 32 || error("restype feature dim must be 32")
    size(profile, 2) == n_tok || error("profile token dim mismatch")
    size(profile, 1) == 32 || error("profile feature dim must be 32")
    length(deletion_mean) == n_tok || error("deletion_mean token dim mismatch")

    # Concatenate along dim=1 (features-first)
    s_inputs = vcat(a_ff, restype, profile, reshape(deletion_mean, 1, n_tok))

    if m.esm_enable
        feat.esm_token_embedding === nothing && error("Missing esm_token_embedding")
        m.linear_esm === nothing && error("linear_esm missing")
        esm_emb = m.linear_esm(feat.esm_token_embedding)  # (c_token+65, N_tok)
        s_inputs .+= esm_emb
    end

    return s_inputs  # (c_s_inputs, N_tok)
end

function (m::InputFeatureEmbedder)(
    input_feature_dict::AbstractDict{<:AbstractString, <:Any};
    chunk_size::Union{Nothing, Int} = nothing,
)
    return m(as_protenix_features(input_feature_dict); chunk_size = chunk_size)
end

end
