module FeatureViews

export as_relpos_input, as_template_input, as_atom_attention_input

function _to_int_vector(x, key::AbstractString)
    x isa AbstractVector || error("Feature '$key' must be a vector.")
    return Int.(x)
end

function _to_int_matrix(x, key::AbstractString)
    x isa AbstractMatrix || error("Feature '$key' must be a matrix.")
    return Int.(x)
end

function _to_f32_vector(x, key::AbstractString)
    x isa AbstractVector || error("Feature '$key' must be a vector.")
    return Float32.(x)
end

function _to_f32_matrix_ff(x, key::AbstractString)
    if key == "ref_atom_name_chars" && x isa AbstractArray && ndims(x) == 3 && size(x, 2) == 4 && size(x, 3) == 64
        # Match Python flatten order: (pos-1)*64 + bucket.
        xp = permutedims(Float32.(x), (1, 3, 2))
        flat = reshape(xp, size(x, 1), 256)
        return permutedims(flat)  # (256, N_atom) features-first
    end
    x isa AbstractMatrix || error("Feature '$key' must be a matrix.")
    return permutedims(Float32.(x))  # features-last → features-first
end

"""
Typed adapter for relative-position input features.
"""
function as_relpos_input(input_feature_dict::AbstractDict{<:AbstractString, <:Any})
    required = ("asym_id", "residue_index", "entity_id", "sym_id", "token_index")
    for k in required
        haskey(input_feature_dict, k) || error("Missing feature '$k' for relative-position encoding.")
    end
    return (
        asym_id = _to_int_vector(input_feature_dict["asym_id"], "asym_id"),
        residue_index = _to_int_vector(input_feature_dict["residue_index"], "residue_index"),
        entity_id = _to_int_vector(input_feature_dict["entity_id"], "entity_id"),
        sym_id = _to_int_vector(input_feature_dict["sym_id"], "sym_id"),
        token_index = _to_int_vector(input_feature_dict["token_index"], "token_index"),
    )
end

"""
Typed adapter for condition-template embedding input features.
"""
function as_template_input(input_feature_dict::AbstractDict{<:AbstractString, <:Any})
    required = ("conditional_templ", "conditional_templ_mask")
    for k in required
        haskey(input_feature_dict, k) || error("Missing feature '$k' for condition-template embedding.")
    end
    return (
        conditional_templ = _to_int_matrix(input_feature_dict["conditional_templ"], "conditional_templ"),
        conditional_templ_mask = _to_int_matrix(input_feature_dict["conditional_templ_mask"], "conditional_templ_mask"),
    )
end

"""
Typed adapter for atom-attention input features.
"""
function as_atom_attention_input(input_feature_dict::AbstractDict{<:AbstractString, <:Any})
    required = (
        "ref_pos",
        "ref_charge",
        "ref_mask",
        "ref_element",
        "ref_atom_name_chars",
        "ref_space_uid",
        "atom_to_token_idx",
    )
    for k in required
        haskey(input_feature_dict, k) || error("Missing feature '$k' for atom attention.")
    end
    return (
        ref_pos = _to_f32_matrix_ff(input_feature_dict["ref_pos"], "ref_pos"),
        ref_charge = _to_f32_vector(input_feature_dict["ref_charge"], "ref_charge"),
        ref_mask = _to_f32_vector(input_feature_dict["ref_mask"], "ref_mask"),
        ref_element = _to_f32_matrix_ff(input_feature_dict["ref_element"], "ref_element"),
        ref_atom_name_chars = _to_f32_matrix_ff(input_feature_dict["ref_atom_name_chars"], "ref_atom_name_chars"),
        ref_space_uid = _to_int_vector(input_feature_dict["ref_space_uid"], "ref_space_uid"),
        atom_to_token_idx = _to_int_vector(input_feature_dict["atom_to_token_idx"], "atom_to_token_idx"),
    )
end

end
