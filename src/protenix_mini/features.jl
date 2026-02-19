module Features

using ConcreteStructs
using Flux: @layer

import ..Utils: one_hot_int

export ConstraintFeatures, ProtenixFeatures, as_protenix_features, features_to_device, relpos_input, atom_attention_input

@concrete struct ConstraintFeatures
    contact   # Union{Nothing, AbstractArray{Float32,3}}
    pocket    # Union{Nothing, AbstractArray{Float32,3}}
    contact_atom  # Union{Nothing, AbstractArray{Float32,3}}
    substructure  # Union{Nothing, AbstractArray{Float32,3}}
end
@layer ConstraintFeatures

@concrete struct ProtenixFeatures
    asym_id               # AbstractVector{Int}
    residue_index         # AbstractVector{Int}
    entity_id             # AbstractVector{Int}
    sym_id                # AbstractVector{Int}
    token_index           # AbstractVector{Int}
    token_mask            # Union{Nothing, AbstractVector{Float32}}
    token_bonds           # AbstractMatrix{Float32}
    restype               # AbstractMatrix{Float32}
    profile               # AbstractMatrix{Float32}
    deletion_mean         # AbstractVector{Float32}
    msa                   # Union{Nothing, AbstractMatrix{Int}}
    has_deletion          # Union{Nothing, AbstractMatrix{Float32}}
    deletion_value        # Union{Nothing, AbstractMatrix{Float32}}
    template_restype      # Union{Nothing, AbstractMatrix{Int}}
    template_all_atom_mask      # Union{Nothing, AbstractArray{Float32,3}}
    template_all_atom_positions # Union{Nothing, AbstractArray{Float32,4}}
    struct_cb_coords      # Union{Nothing, AbstractMatrix{Float32}}
    struct_cb_mask        # Union{Nothing, AbstractVector{Bool}}
    esm_token_embedding   # Union{Nothing, AbstractMatrix{Float32}}
    ref_pos               # AbstractMatrix{Float32}
    ref_charge            # AbstractVector{Float32}
    ref_mask              # AbstractVector{Float32}
    ref_element           # AbstractMatrix{Float32}
    ref_atom_name_chars   # AbstractMatrix{Float32}
    ref_space_uid         # AbstractVector{Int}
    atom_to_token_idx     # AbstractVector{Int}
    atom_to_tokatom_idx   # AbstractVector{Int}
    distogram_rep_atom_mask # AbstractVector{Bool}
    constraint_feature    # Union{Nothing, ConstraintFeatures}
end
@layer ProtenixFeatures

_as_i_vec(x, name::String) = x isa AbstractVector ? Int.(x) : error("$name must be a vector")
_as_f_vec(x, name::String) = x isa AbstractVector ? Float32.(x) : error("$name must be a vector")
_as_b_vec(x, name::String) = x isa AbstractVector ? collect(Bool, x) : error("$name must be a vector")
_as_i_mat(x, name::String) = x isa AbstractMatrix ? Int.(x) : error("$name must be a matrix")
_as_f_mat(x, name::String) = x isa AbstractMatrix ? Float32.(x) : error("$name must be a matrix")
function _as_ref_atom_name_chars_mat(x)
    if x isa AbstractArray && ndims(x) == 3 && size(x, 2) == 4 && size(x, 3) == 64
        # Python flattens [N,4,64] with channel index (pos-1)*64 + bucket.
        # In Julia, preserve that order explicitly before reshape.
        xp = permutedims(Float32.(x), (1, 3, 2))
        return reshape(xp, size(x, 1), 256)
    end
    return _as_f_mat(x, "ref_atom_name_chars")
end

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
    c_any isa AbstractDict || c_any isa NamedTuple || error("constraint_feature must be an object/NamedTuple")
    c = c_any
    get_cf(k::String) = c isa AbstractDict ? get(c, k, nothing) : (hasproperty(c, Symbol(k)) ? getproperty(c, Symbol(k)) : nothing)
    contact_raw = get_cf("contact")
    pocket_raw = get_cf("pocket")
    contact_atom_raw = get_cf("contact_atom")
    substructure_raw = get_cf("substructure")
    contact = contact_raw === nothing ? nothing : _to_f_arr3(contact_raw, "constraint_feature.contact")
    pocket = pocket_raw === nothing ? nothing : _to_f_arr3(pocket_raw, "constraint_feature.pocket")
    contact_atom = contact_atom_raw === nothing ? nothing : _to_f_arr3(contact_atom_raw, "constraint_feature.contact_atom")
    substructure = substructure_raw === nothing ? nothing : _to_f_arr3(substructure_raw, "constraint_feature.substructure")
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
        _as_ref_atom_name_chars_mat(feat["ref_atom_name_chars"]),
        _as_i_vec(feat["ref_space_uid"], "ref_space_uid"),
        _as_i_vec(feat["atom_to_token_idx"], "atom_to_token_idx"),
        _as_i_vec(feat["atom_to_tokatom_idx"], "atom_to_tokatom_idx"),
        _as_b_vec(feat["distogram_rep_atom_mask"], "distogram_rep_atom_mask"),
        _optional_constraint_features(feat),
    )
end

as_protenix_features(feat::ProtenixFeatures) = feat

# Transfer float arrays to the device of `ref`, keep int/bool vectors on CPU.
function _float_to_dev(x::AbstractArray{<:AbstractFloat}, ref::AbstractArray)
    copyto!(similar(ref, Float32, size(x)...), Float32.(x))
end
_float_to_dev(::Nothing, ::AbstractArray) = nothing

function _constraint_to_device(c::ConstraintFeatures, ref::AbstractArray)
    mv = x -> _float_to_dev(x, ref)
    return ConstraintFeatures(mv(c.contact), mv(c.pocket), mv(c.contact_atom), mv(c.substructure))
end

function features_to_device(feat::ProtenixFeatures, ref::AbstractArray)
    mv = x -> _float_to_dev(x, ref)
    return ProtenixFeatures(
        feat.asym_id,
        feat.residue_index,
        feat.entity_id,
        feat.sym_id,
        feat.token_index,
        mv(feat.token_mask),
        mv(feat.token_bonds),
        mv(feat.restype),
        mv(feat.profile),
        mv(feat.deletion_mean),
        feat.msa,
        mv(feat.has_deletion),
        mv(feat.deletion_value),
        feat.template_restype,
        mv(feat.template_all_atom_mask),
        mv(feat.template_all_atom_positions),
        mv(feat.struct_cb_coords),
        feat.struct_cb_mask,
        mv(feat.esm_token_embedding),
        mv(feat.ref_pos),
        mv(feat.ref_charge),
        mv(feat.ref_mask),
        mv(feat.ref_element),
        mv(feat.ref_atom_name_chars),
        feat.ref_space_uid,
        feat.atom_to_token_idx,
        feat.atom_to_tokatom_idx,
        feat.distogram_rep_atom_mask,
        feat.constraint_feature === nothing ? nothing : _constraint_to_device(feat.constraint_feature, ref),
    )
end

# No-op when ref is a regular CPU Array
features_to_device(feat::ProtenixFeatures, ::Array) = feat

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
