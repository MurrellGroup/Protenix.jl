module Embedders

using Random

export ConditionTemplateEmbedder,
    RelativePositionEncoding,
    condition_template_embedding,
    relative_position_features,
    fourier_embedding

struct ConditionTemplateEmbedder
    c_templ_in::Int
    c_z::Int
    weight::Matrix{Float32} # [c_templ_in, c_z]
end

function ConditionTemplateEmbedder(c_templ_in::Int = 65, c_z::Int = 128; rng::AbstractRNG = Random.default_rng())
    c_templ_in > 0 || error("c_templ_in must be positive.")
    c_z > 0 || error("c_z must be positive.")
    w = randn(rng, Float32, c_templ_in, c_z)
    return ConditionTemplateEmbedder(c_templ_in, c_z, w)
end

function condition_template_embedding(
    embedder::ConditionTemplateEmbedder,
    conditional_templ::AbstractMatrix{<:Integer},
    conditional_templ_mask::AbstractMatrix,
)
    size(conditional_templ) == size(conditional_templ_mask) ||
        error("conditional_templ and conditional_templ_mask shapes must match.")

    n_i, n_j = size(conditional_templ)
    out = Array{Float32}(undef, n_i, n_j, embedder.c_z)

    for i in 1:n_i
        for j in 1:n_j
            mask_ij = conditional_templ_mask[i, j] != 0
            idx0 = mask_ij ? (1 + Int(conditional_templ[i, j])) : 0
            idx = clamp(idx0 + 1, 1, embedder.c_templ_in)
            @inbounds out[i, j, :] = embedder.weight[idx, :]
        end
    end
    return out
end

function condition_template_embedding(
    embedder::ConditionTemplateEmbedder,
    template_input::NamedTuple{(:conditional_templ, :conditional_templ_mask)},
)
    return condition_template_embedding(
        embedder,
        template_input.conditional_templ,
        template_input.conditional_templ_mask,
    )
end

struct RelativePositionEncoding
    r_max::Int
    s_max::Int
    c_z::Int
    weight::Matrix{Float32} # [c_z, in_features] (LinearNoBias in PyTorch layout)
end

function RelativePositionEncoding(r_max::Int = 32, s_max::Int = 2, c_z::Int = 128; rng::AbstractRNG = Random.default_rng())
    r_max >= 0 || error("r_max must be non-negative.")
    s_max >= 0 || error("s_max must be non-negative.")
    c_z > 0 || error("c_z must be positive.")
    in_features = 4 * r_max + 2 * s_max + 7
    w = randn(rng, Float32, c_z, in_features)
    return RelativePositionEncoding(r_max, s_max, c_z, w)
end

function _onehot_index!(dst::Vector{Float32}, idx0::Int)
    fill!(dst, 0f0)
    idx = idx0 + 1
    (1 <= idx <= length(dst)) || error("One-hot index $idx0 out of range [0, $(length(dst)-1)]")
    dst[idx] = 1f0
    return dst
end

function relative_position_features(
    asym_id::AbstractVector{<:Integer},
    residue_index::AbstractVector{<:Integer},
    entity_id::AbstractVector{<:Integer},
    sym_id::AbstractVector{<:Integer},
    token_index::AbstractVector{<:Integer};
    r_max::Int = 32,
    s_max::Int = 2,
)
    n = length(asym_id)
    length(residue_index) == n || error("residue_index length mismatch.")
    length(entity_id) == n || error("entity_id length mismatch.")
    length(sym_id) == n || error("sym_id length mismatch.")
    length(token_index) == n || error("token_index length mismatch.")

    d_r = 2 * (r_max + 1)
    d_s = 2 * (s_max + 1)
    d_total = d_r + d_r + 1 + d_s
    out = zeros(Float32, n, n, d_total)

    tmp_r = zeros(Float32, d_r)
    tmp_t = zeros(Float32, d_r)
    tmp_c = zeros(Float32, d_s)

    for i in 1:n
        for j in 1:n
            same_chain = asym_id[i] == asym_id[j] ? 1 : 0
            same_residue = residue_index[i] == residue_index[j] ? 1 : 0
            same_entity = entity_id[i] == entity_id[j] ? 1 : 0

            rel_res = residue_index[i] - residue_index[j]
            d_res = clamp(rel_res + r_max, 0, 2 * r_max) * same_chain + (1 - same_chain) * (2 * r_max + 1)
            _onehot_index!(tmp_r, d_res)

            rel_tok = token_index[i] - token_index[j]
            same_chain_and_res = same_chain * same_residue
            d_tok = clamp(rel_tok + r_max, 0, 2 * r_max) * same_chain_and_res +
                    (1 - same_chain_and_res) * (2 * r_max + 1)
            _onehot_index!(tmp_t, d_tok)

            rel_chain = sym_id[i] - sym_id[j]
            d_chain = clamp(rel_chain + s_max, 0, 2 * s_max) * same_entity + (1 - same_entity) * (2 * s_max + 1)
            _onehot_index!(tmp_c, d_chain)

            pos = 1
            out[i, j, pos:(pos + d_r - 1)] .= tmp_r
            pos += d_r
            out[i, j, pos:(pos + d_r - 1)] .= tmp_t
            pos += d_r
            out[i, j, pos] = Float32(same_entity)
            pos += 1
            out[i, j, pos:(pos + d_s - 1)] .= tmp_c
        end
    end

    return out
end

function _linear_no_bias_lastdim(x::Array{Float32,3}, weight::Matrix{Float32})
    in_features = size(x, 3)
    size(weight, 2) == in_features || error("Linear weight/input mismatch: weight $(size(weight)) vs input $(size(x)).")
    flat = reshape(x, :, in_features)
    y = flat * transpose(weight) # [*, in] x [in, out] -> [*, out]
    return reshape(y, size(x, 1), size(x, 2), size(weight, 1))
end

function (relpe::RelativePositionEncoding)(input_feature_dict::AbstractDict{<:AbstractString, <:Any})
    keys = ("asym_id", "residue_index", "entity_id", "sym_id", "token_index")
    for k in keys
        haskey(input_feature_dict, k) || error("Missing input feature '$k' for RelativePositionEncoding.")
    end

    f = relative_position_features(
        input_feature_dict["asym_id"],
        input_feature_dict["residue_index"],
        input_feature_dict["entity_id"],
        input_feature_dict["sym_id"],
        input_feature_dict["token_index"];
        r_max = relpe.r_max,
        s_max = relpe.s_max,
    )
    return _linear_no_bias_lastdim(f, relpe.weight)
end

function (relpe::RelativePositionEncoding)(
    relpos_input::NamedTuple{(:asym_id, :residue_index, :entity_id, :sym_id, :token_index)},
)
    f = relative_position_features(
        relpos_input.asym_id,
        relpos_input.residue_index,
        relpos_input.entity_id,
        relpos_input.sym_id,
        relpos_input.token_index;
        r_max = relpe.r_max,
        s_max = relpe.s_max,
    )
    return _linear_no_bias_lastdim(f, relpe.weight)
end

function fourier_embedding(
    t_hat_noise_level::AbstractVector{<:Real},
    w::AbstractVector{<:Real},
    b::AbstractVector{<:Real},
)
    length(w) == length(b) || error("w and b must have same length.")
    n = length(t_hat_noise_level)
    c = length(w)
    out = Array{Float32}(undef, n, c)
    for i in 1:n
        ti = Float32(t_hat_noise_level[i])
        for j in 1:c
            out[i, j] = cos(2f0 * Float32(pi) * (ti * Float32(w[j]) + Float32(b[j])))
        end
    end
    return out
end

end
