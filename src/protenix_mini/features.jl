module Features

using ConcreteStructs
using Flux: @layer

import ..Utils: one_hot_int

export ConstraintFeatures, ProtenixFeatures, as_protenix_features, features_to_device, relpos_input, atom_attention_input

@concrete struct ConstraintFeatures
    contact   # Union{Nothing, AbstractArray{Float32,3}}  — (C, N, N) features-first
    pocket    # Union{Nothing, AbstractArray{Float32,3}}  — (C, N, N) features-first
    contact_atom  # Union{Nothing, AbstractArray{Float32,3}}  — (C, N, N) features-first
    substructure  # Union{Nothing, AbstractArray{Float32,3}}  — (C, N, N) features-first
end
@layer ConstraintFeatures

# All feature tensors are stored features-first (features dim = 1).
@concrete struct ProtenixFeatures
    asym_id               # AbstractVector{Int}
    residue_index         # AbstractVector{Int}
    entity_id             # AbstractVector{Int}
    sym_id                # AbstractVector{Int}
    token_index           # AbstractVector{Int}
    token_mask            # Union{Nothing, AbstractVector{Float32}}
    token_bonds           # AbstractMatrix{Float32}     — (N_tok, N_tok)
    restype               # AbstractMatrix{Float32}     — (32, N_tok) features-first
    profile               # AbstractMatrix{Float32}     — (32, N_tok) features-first
    deletion_mean         # AbstractVector{Float32}     — (N_tok,)
    msa                   # Union{Nothing, AbstractMatrix{Int}}  — (N_tok, N_msa) features-first
    has_deletion          # Union{Nothing, AbstractMatrix{Float32}}  — (N_tok, N_msa) features-first
    deletion_value        # Union{Nothing, AbstractMatrix{Float32}}  — (N_tok, N_msa) features-first
    template_restype      # Union{Nothing, AbstractMatrix{Int}}  — (N_tmpl, N_tok) NOT transposed
    template_all_atom_mask      # Union{Nothing, AbstractArray{Float32,3}}
    template_all_atom_positions # Union{Nothing, AbstractArray{Float32,4}}
    template_distogram          # Union{Nothing, AbstractArray{Float32}}    — (N_tmpl, N, N, 39)
    template_unit_vector        # Union{Nothing, AbstractArray{Float32}}    — (N_tmpl, N, N, 3)
    template_pseudo_beta_mask   # Union{Nothing, AbstractArray{Float32}}    — (N_tmpl, N, N)
    template_backbone_frame_mask # Union{Nothing, AbstractArray{Float32}}   — (N_tmpl, N, N)
    struct_cb_coords      # Union{Nothing, AbstractMatrix{Float32}}  — (3, N_tok) features-first
    struct_cb_mask        # Union{Nothing, AbstractVector{Bool}}
    esm_token_embedding   # Union{Nothing, AbstractMatrix{Float32}}  — (C_esm, N_tok) features-first
    ref_pos               # AbstractMatrix{Float32}     — (3, N_atom) features-first
    ref_charge            # AbstractVector{Float32}     — (N_atom,)
    ref_mask              # AbstractVector{Float32}     — (N_atom,)
    ref_element           # AbstractMatrix{Float32}     — (128, N_atom) features-first
    ref_atom_name_chars   # AbstractMatrix{Float32}     — (256, N_atom) features-first
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

"""
Convert ref_atom_name_chars to features-first (256, N_atom).
Input may be (N_atom, 4, 64) rank-3 or (N_atom, 256) rank-2 (both features-last from Python).
"""
function _as_ref_atom_name_chars_ff(x)
    if x isa AbstractArray && ndims(x) == 3 && size(x, 2) == 4 && size(x, 3) == 64
        # Python layout: [N_atom, 4, 64] → flatten to (N_atom, 256) then transpose
        xp = permutedims(Float32.(x), (1, 3, 2))
        flat = reshape(xp, size(x, 1), 256)  # (N_atom, 256)
        return permutedims(flat)  # (256, N_atom)
    end
    # 2D input: (N_atom, 256) → transpose to (256, N_atom)
    return permutedims(_as_f_mat(x, "ref_atom_name_chars"))
end

function _optional_f_mat_ff(feat::AbstractDict{<:AbstractString, <:Any}, key::String)
    haskey(feat, key) || return nothing
    return permutedims(_as_f_mat(feat[key], key))
end

function _optional_i_mat_ff(feat::AbstractDict{<:AbstractString, <:Any}, key::String)
    haskey(feat, key) || return nothing
    return permutedims(_as_i_mat(feat[key], key))
end

function _optional_f_arr(feat::AbstractDict{<:AbstractString, <:Any}, key::String)
    haskey(feat, key) || return nothing
    x = feat[key]
    x isa AbstractArray || error("$key must be array-like")
    return Float32.(x)
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

function _to_f_arr3_ff(x, key::String)
    x isa AbstractArray || error("$key must be array-like")
    ndims(x) == 3 || error("$key must have rank 3")
    # Input is features-last (N, N, C); transpose to features-first (C, N, N)
    return Float32.(permutedims(x, (3, 1, 2)))
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
    # Transpose from features-last (N,N,C) to features-first (C,N,N)
    contact = contact_raw === nothing ? nothing : _to_f_arr3_ff(contact_raw, "constraint_feature.contact")
    pocket = pocket_raw === nothing ? nothing : _to_f_arr3_ff(pocket_raw, "constraint_feature.pocket")
    contact_atom = contact_atom_raw === nothing ? nothing : _to_f_arr3_ff(contact_atom_raw, "constraint_feature.contact_atom")
    substructure = substructure_raw === nothing ? nothing : _to_f_arr3_ff(substructure_raw, "constraint_feature.substructure")
    return ConstraintFeatures(contact, pocket, contact_atom, substructure)
end

"""
Coalesce restype from dict. Returns features-first (32, N_tok).
Input dict values are features-last from Python data pipeline.
"""
function _coalesce_restype(feat::AbstractDict{<:AbstractString, <:Any})
    if haskey(feat, "restype")
        restype_raw = _as_f_mat(feat["restype"], "restype")
        # Input is features-last (N_tok, 32); transpose to features-first (32, N_tok)
        restype = permutedims(restype_raw)
        size(restype, 1) == 32 || error("restype feature dim must be 32, got $(size(restype, 1))")
        return restype
    end
    if haskey(feat, "restype_index")
        idx = _as_i_vec(feat["restype_index"], "restype_index")
        return one_hot_int(idx, 32)  # already (32, N_tok) features-first
    end
    error("Missing restype/restype_index")
end

"""
Convert a features-last input dict (from Python/data pipeline) to features-first ProtenixFeatures.
"""
function as_protenix_features(feat::AbstractDict{<:AbstractString, <:Any})
    restype = _coalesce_restype(feat)  # (32, N_tok)
    n_tok = size(restype, 2)

    profile = if haskey(feat, "profile")
        permutedims(_as_f_mat(feat["profile"], "profile"))  # (N_tok, 32) → (32, N_tok)
    else
        zeros(Float32, 32, n_tok)
    end
    size(profile, 2) == n_tok || error("profile token dim mismatch")
    size(profile, 1) == 32 || error("profile feature dim must be 32")

    deletion_mean = haskey(feat, "deletion_mean") ?
        _as_f_vec(feat["deletion_mean"], "deletion_mean") : zeros(Float32, n_tok)
    length(deletion_mean) == n_tok || error("deletion_mean token dim mismatch")

    token_mask = haskey(feat, "token_mask") ? _as_f_vec(feat["token_mask"], "token_mask") : nothing

    # MSA features: transpose from (N_msa, N_tok) to (N_tok, N_msa) features-first
    msa = _optional_i_mat_ff(feat, "msa")
    has_deletion = _optional_f_mat_ff(feat, "has_deletion")
    deletion_value = _optional_f_mat_ff(feat, "deletion_value")

    # Template features: keep template_restype in (N_template, N_token) layout
    # (NOT transposed — _template_embedder_core indexes dim 1 as template batch dim)
    template_restype = if haskey(feat, "template_restype")
        _as_i_mat(feat["template_restype"], "template_restype")
    elseif haskey(feat, "template_aatype")
        _as_i_mat(feat["template_aatype"], "template_aatype")
    else
        nothing
    end

    # Atom features: transpose to features-first
    ref_pos = permutedims(_as_f_mat(feat["ref_pos"], "ref_pos"))  # (N_atom, 3) → (3, N_atom)
    ref_element = permutedims(_as_f_mat(feat["ref_element"], "ref_element"))  # (N_atom, 128) → (128, N_atom)
    ref_atom_name_chars = _as_ref_atom_name_chars_ff(feat["ref_atom_name_chars"])  # (256, N_atom)

    struct_cb_coords = _optional_f_mat_ff(feat, "struct_cb_coords")  # (N_tok, 3) → (3, N_tok)

    esm_token_embedding = _optional_f_mat_ff(feat, "esm_token_embedding")

    return ProtenixFeatures(
        _as_i_vec(feat["asym_id"], "asym_id"),
        _as_i_vec(feat["residue_index"], "residue_index"),
        _as_i_vec(feat["entity_id"], "entity_id"),
        _as_i_vec(feat["sym_id"], "sym_id"),
        _as_i_vec(feat["token_index"], "token_index"),
        token_mask,
        _as_f_mat(feat["token_bonds"], "token_bonds"),  # (N_tok, N_tok) stays same
        restype,  # (32, N_tok)
        profile,  # (32, N_tok)
        deletion_mean,  # (N_tok,)
        msa,  # (N_tok, N_msa)
        has_deletion,  # (N_tok, N_msa)
        deletion_value,  # (N_tok, N_msa)
        template_restype,
        _optional_f_arr3(feat, "template_all_atom_mask"),
        _optional_f_arr4(feat, "template_all_atom_positions"),
        _optional_f_arr(feat, "template_distogram"),
        _optional_f_arr(feat, "template_unit_vector"),
        _optional_f_arr(feat, "template_pseudo_beta_mask"),
        _optional_f_arr(feat, "template_backbone_frame_mask"),
        struct_cb_coords,  # (3, N_tok)
        haskey(feat, "struct_cb_mask") ? _as_b_vec(feat["struct_cb_mask"], "struct_cb_mask") : nothing,
        esm_token_embedding,  # (C_esm, N_tok)
        ref_pos,  # (3, N_atom)
        _as_f_vec(feat["ref_charge"], "ref_charge"),
        _as_f_vec(feat["ref_mask"], "ref_mask"),
        ref_element,  # (128, N_atom)
        ref_atom_name_chars,  # (256, N_atom)
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
        mv(feat.template_distogram),
        mv(feat.template_unit_vector),
        mv(feat.template_pseudo_beta_mask),
        mv(feat.template_backbone_frame_mask),
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
        ref_pos = feat.ref_pos,           # (3, N_atom) features-first
        ref_charge = feat.ref_charge,     # (N_atom,)
        ref_mask = feat.ref_mask,         # (N_atom,)
        ref_element = feat.ref_element,   # (128, N_atom) features-first
        ref_atom_name_chars = feat.ref_atom_name_chars,  # (256, N_atom) features-first
        ref_space_uid = feat.ref_space_uid,
        atom_to_token_idx = feat.atom_to_token_idx,
    )
end

end
