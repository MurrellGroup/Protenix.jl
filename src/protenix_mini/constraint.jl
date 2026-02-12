module Constraint

using Random

import ..Primitives: LinearNoBias

export ConstraintEmbedder

_as_f32_array(x::AbstractArray{<:Real}) = x isa AbstractArray{Float32} ? x : Float32.(x)

struct ConstraintEmbedder
    pocket_z_embedder::Union{Nothing, LinearNoBias}
    contact_z_embedder::Union{Nothing, LinearNoBias}
    contact_atom_z_embedder::Union{Nothing, LinearNoBias}
    substructure_z_embedder::Union{Nothing, LinearNoBias}
end

function ConstraintEmbedder(
    c_constraint_z::Int;
    pocket_enable::Bool = false,
    pocket_c_z_input::Int = 1,
    contact_enable::Bool = false,
    contact_c_z_input::Int = 2,
    contact_atom_enable::Bool = false,
    contact_atom_c_z_input::Int = 2,
    substructure_enable::Bool = false,
    substructure_n_classes::Int = 4,
    initialize_method::Symbol = :zero,
    rng::AbstractRNG = Random.default_rng(),
)
    p = pocket_enable ? LinearNoBias(c_constraint_z, pocket_c_z_input; rng = rng) : nothing
    c = contact_enable ? LinearNoBias(c_constraint_z, contact_c_z_input; rng = rng) : nothing
    ca = contact_atom_enable ? LinearNoBias(c_constraint_z, contact_atom_c_z_input; rng = rng) : nothing
    s = substructure_enable ? LinearNoBias(c_constraint_z, substructure_n_classes; rng = rng) : nothing

    if initialize_method == :zero
        p !== nothing && fill!(p.weight, 0f0)
        c !== nothing && fill!(c.weight, 0f0)
        ca !== nothing && fill!(ca.weight, 0f0)
        s !== nothing && fill!(s.weight, 0f0)
    end
    return ConstraintEmbedder(p, c, ca, s)
end

function _constraint_arr3(x, key::String)
    x isa AbstractArray || error("constraint_feature.$key must be array-like")
    ndims(x) == 3 || error("constraint_feature.$key must be rank-3 [N_token,N_token,C]")
    return _as_f32_array(x)
end

function _constraint_get(constraint_feature, key::String)
    if constraint_feature isa AbstractDict
        haskey(constraint_feature, key) || return nothing
        return constraint_feature[key]
    elseif constraint_feature isa NamedTuple
        hasproperty(constraint_feature, Symbol(key)) || return nothing
        return getproperty(constraint_feature, Symbol(key))
    end
    error("constraint_feature must be a Dict/NamedTuple")
end

function (cemb::ConstraintEmbedder)(constraint_feature)
    z_constraint = nothing

    if cemb.pocket_z_embedder !== nothing
        pocket_raw = _constraint_get(constraint_feature, "pocket")
        pocket_raw === nothing || begin
            pocket = _constraint_arr3(pocket_raw, "pocket")
            size(pocket, 3) == size(cemb.pocket_z_embedder.weight, 2) ||
                error("constraint_feature.pocket channel mismatch")
            zp = cemb.pocket_z_embedder(pocket)
            z_constraint = z_constraint === nothing ? zp : z_constraint .+ zp
        end
    end

    if cemb.contact_z_embedder !== nothing
        contact_raw = _constraint_get(constraint_feature, "contact")
        contact_raw === nothing || begin
            contact = _constraint_arr3(contact_raw, "contact")
            size(contact, 3) == size(cemb.contact_z_embedder.weight, 2) ||
                error("constraint_feature.contact channel mismatch")
            zc = cemb.contact_z_embedder(contact)
            z_constraint = z_constraint === nothing ? zc : z_constraint .+ zc
        end
    end

    if cemb.contact_atom_z_embedder !== nothing
        contact_atom_raw = _constraint_get(constraint_feature, "contact_atom")
        contact_atom_raw === nothing || begin
            contact_atom = _constraint_arr3(contact_atom_raw, "contact_atom")
            size(contact_atom, 3) == size(cemb.contact_atom_z_embedder.weight, 2) ||
                error("constraint_feature.contact_atom channel mismatch")
            zca = cemb.contact_atom_z_embedder(contact_atom)
            z_constraint = z_constraint === nothing ? zca : z_constraint .+ zca
        end
    end

    if cemb.substructure_z_embedder !== nothing
        sub_raw = _constraint_get(constraint_feature, "substructure")
        sub_raw === nothing || begin
            sub = if sub_raw isa AbstractArray && ndims(sub_raw) == 2
                idx = Int.(sub_raw)
                k = size(cemb.substructure_z_embedder.weight, 2)
                out = zeros(Float32, size(idx, 1), size(idx, 2), k)
                for i in axes(idx, 1), j in axes(idx, 2)
                    cls = clamp(idx[i, j] + 1, 1, k)
                    out[i, j, cls] = 1f0
                end
                out
            else
                _constraint_arr3(sub_raw, "substructure")
            end
            size(sub, 3) == size(cemb.substructure_z_embedder.weight, 2) ||
                error("constraint_feature.substructure channel mismatch")
            zs = cemb.substructure_z_embedder(sub)
            z_constraint = z_constraint === nothing ? zs : z_constraint .+ zs
        end
    end

    return z_constraint
end

end

