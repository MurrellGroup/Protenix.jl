module Features

import ..Utils: one_hot_int

export ConstraintFeatures, ProtenixFeatures, as_protenix_features, relpos_input, atom_attention_input

struct ConstraintFeatures
    contact::Union{Nothing, Array{Float32,3}}
    pocket::Union{Nothing, Array{Float32,3}}
    contact_atom::Union{Nothing, Array{Float32,3}}
    substructure::Union{Nothing, Array{Float32,3}}
end

struct ProtenixFeatures
    asym_id::Vector{Int}
    residue_index::Vector{Int}
    entity_id::Vector{Int}
    sym_id::Vector{Int}
    token_index::Vector{Int}
    token_mask::Union{Nothing, Vector{Float32}}
    token_bonds::Matrix{Float32}
    restype::Matrix{Float32}
    profile::Matrix{Float32}
    deletion_mean::Vector{Float32}
    msa::Union{Nothing, Matrix{Int}}
    has_deletion::Union{Nothing, Matrix{Float32}}
    deletion_value::Union{Nothing, Matrix{Float32}}
    template_restype::Union{Nothing, Matrix{Int}}
    template_all_atom_mask::Union{Nothing, Array{Float32,3}}
    template_all_atom_positions::Union{Nothing, Array{Float32,4}}
    struct_cb_coords::Union{Nothing, Matrix{Float32}}
    struct_cb_mask::Union{Nothing, Vector{Bool}}
    esm_token_embedding::Union{Nothing, Matrix{Float32}}
    ref_pos::Matrix{Float32}
    ref_charge::Vector{Float32}
    ref_mask::Vector{Float32}
    ref_element::Matrix{Float32}
    ref_atom_name_chars::Matrix{Float32}
    ref_space_uid::Vector{Int}
    atom_to_token_idx::Vector{Int}
    atom_to_tokatom_idx::Vector{Int}
    distogram_rep_atom_mask::Vector{Bool}
    constraint_feature::Union{Nothing, ConstraintFeatures}
end

_as_i_vec(x, name::String) = x isa AbstractVector ? Int.(x) : error("$name must be a vector")
_as_f_vec(x, name::String) = x isa AbstractVector ? Float32.(x) : error("$name must be a vector")
_as_b_vec(x, name::String) = x isa AbstractVector ? Bool.(x) : error("$name must be a vector")
_as_i_mat(x, name::String) = x isa AbstractMatrix ? Int.(x) : error("$name must be a matrix")
_as_f_mat(x, name::String) = x isa AbstractMatrix ? Float32.(x) : error("$name must be a matrix")

function _optional_f_mat(feat::AbstractDict{<:AbstractString, <:Any}, key::String)
    return haskey(feat, key) ? _as_f_mat(feat[key], key) : nothing
end

function _optional_i_mat(feat::AbstractDict{<:AbstractString, <:Any}, key::String)
    return haskey(feat, key) ? _as_i_mat(feat[key], key) : nothing
end

function _optional_f_arr3(feat::AbstractDict{<:AbstractString, <:Any}, key::String)
    haskey(feat, key) || return nothing
    x = feat[key]
    x isa AbstractArray || error("$key must be array-like")
    ndims(x) == 3 || error("$key must have rank 3")
    return Float32.(x)
end

function _optional_f_arr4(feat::AbstractDict{<:AbstractString, <:Any}, key::String)
    haskey(feat, key) || return nothing
    x = feat[key]
    x isa AbstractArray || error("$key must be array-like")
    ndims(x) == 4 || error("$key must have rank 4")
    return Float32.(x)
end

function _to_f_arr3(x, key::String)
    x isa AbstractArray || error("$key must be array-like")
    ndims(x) == 3 || error("$key must have rank 3")
    return Float32.(x)
end

function _optional_constraint_features(feat::AbstractDict{<:AbstractString, <:Any})
    haskey(feat, "constraint_feature") || return nothing
    c_any = feat["constraint_feature"]
    c_any isa AbstractDict || error("constraint_feature must be an object")
    c = c_any
    contact = haskey(c, "contact") ? _to_f_arr3(c["contact"], "constraint_feature.contact") : nothing
    pocket = haskey(c, "pocket") ? _to_f_arr3(c["pocket"], "constraint_feature.pocket") : nothing
    contact_atom = haskey(c, "contact_atom") ? _to_f_arr3(c["contact_atom"], "constraint_feature.contact_atom") : nothing
    substructure = haskey(c, "substructure") ? _to_f_arr3(c["substructure"], "constraint_feature.substructure") : nothing
    return ConstraintFeatures(contact, pocket, contact_atom, substructure)
end

function _coalesce_restype(feat::AbstractDict{<:AbstractString, <:Any})
    if haskey(feat, "restype")
        restype = _as_f_mat(feat["restype"], "restype")
        size(restype, 2) == 32 || error("restype feature dim must be 32, got $(size(restype, 2))")
        return restype
    end
    if haskey(feat, "restype_index")
        idx = _as_i_vec(feat["restype_index"], "restype_index")
        return one_hot_int(idx, 32)
    end
    error("Missing restype/restype_index")
end

function as_protenix_features(feat::AbstractDict{<:AbstractString, <:Any})
    restype = _coalesce_restype(feat)
    n_tok = size(restype, 1)
    profile = haskey(feat, "profile") ? _as_f_mat(feat["profile"], "profile") : zeros(Float32, n_tok, 32)
    size(profile, 1) == n_tok || error("profile token dim mismatch")
    size(profile, 2) == 32 || error("profile feature dim must be 32")

    deletion_mean = haskey(feat, "deletion_mean") ?
        _as_f_vec(feat["deletion_mean"], "deletion_mean") : zeros(Float32, n_tok)
    length(deletion_mean) == n_tok || error("deletion_mean token dim mismatch")

    token_mask = haskey(feat, "token_mask") ? _as_f_vec(feat["token_mask"], "token_mask") : nothing

    return ProtenixFeatures(
        _as_i_vec(feat["asym_id"], "asym_id"),
        _as_i_vec(feat["residue_index"], "residue_index"),
        _as_i_vec(feat["entity_id"], "entity_id"),
        _as_i_vec(feat["sym_id"], "sym_id"),
        _as_i_vec(feat["token_index"], "token_index"),
        token_mask,
        _as_f_mat(feat["token_bonds"], "token_bonds"),
        restype,
        profile,
        deletion_mean,
        _optional_i_mat(feat, "msa"),
        _optional_f_mat(feat, "has_deletion"),
        _optional_f_mat(feat, "deletion_value"),
        _optional_i_mat(feat, "template_restype"),
        _optional_f_arr3(feat, "template_all_atom_mask"),
        _optional_f_arr4(feat, "template_all_atom_positions"),
        _optional_f_mat(feat, "struct_cb_coords"),
        haskey(feat, "struct_cb_mask") ? _as_b_vec(feat["struct_cb_mask"], "struct_cb_mask") : nothing,
        _optional_f_mat(feat, "esm_token_embedding"),
        _as_f_mat(feat["ref_pos"], "ref_pos"),
        _as_f_vec(feat["ref_charge"], "ref_charge"),
        _as_f_vec(feat["ref_mask"], "ref_mask"),
        _as_f_mat(feat["ref_element"], "ref_element"),
        _as_f_mat(feat["ref_atom_name_chars"], "ref_atom_name_chars"),
        _as_i_vec(feat["ref_space_uid"], "ref_space_uid"),
        _as_i_vec(feat["atom_to_token_idx"], "atom_to_token_idx"),
        _as_i_vec(feat["atom_to_tokatom_idx"], "atom_to_tokatom_idx"),
        _as_b_vec(feat["distogram_rep_atom_mask"], "distogram_rep_atom_mask"),
        _optional_constraint_features(feat),
    )
end

as_protenix_features(feat::ProtenixFeatures) = feat

function relpos_input(feat::ProtenixFeatures)
    return (
        asym_id = feat.asym_id,
        residue_index = feat.residue_index,
        entity_id = feat.entity_id,
        sym_id = feat.sym_id,
        token_index = feat.token_index,
    )
end

function atom_attention_input(feat::ProtenixFeatures)
    return (
        ref_pos = feat.ref_pos,
        ref_charge = feat.ref_charge,
        ref_mask = feat.ref_mask,
        ref_element = feat.ref_element,
        ref_atom_name_chars = feat.ref_atom_name_chars,
        ref_space_uid = feat.ref_space_uid,
        atom_to_token_idx = feat.atom_to_token_idx,
    )
end

end
