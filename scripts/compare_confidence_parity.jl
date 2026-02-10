#!/usr/bin/env julia

using LinearAlgebra
using Statistics
using PXDesign

function _nested_dims(x)
    dims = Int[]
    cur = x
    while cur isa AbstractVector
        push!(dims, length(cur))
        isempty(cur) && break
        cur = cur[1]
    end
    return dims
end

function _to_array_f32(x)
    x isa AbstractVector || return Float32(x)
    dims = _nested_dims(x)
    out = Array{Float32}(undef, Tuple(dims)...)
    idx = Int[]
    function walk(y)
        if y isa AbstractVector
            for (i, z) in enumerate(y)
                push!(idx, i)
                walk(z)
                pop!(idx)
            end
        else
            out[Tuple(idx)...] = Float32(y)
        end
    end
    walk(x)
    return out
end

function _to_array_i(x)
    x isa AbstractVector || return Int(round(Float64(x)))
    dims = _nested_dims(x)
    out = Array{Int}(undef, Tuple(dims)...)
    idx = Int[]
    function walk(y)
        if y isa AbstractVector
            for (i, z) in enumerate(y)
                push!(idx, i)
                walk(z)
                pop!(idx)
            end
        else
            out[Tuple(idx)...] = Int(round(Float64(y)))
        end
    end
    walk(x)
    return out
end

function _to_bool_vec(x)
    x isa AbstractVector || error("expected vector")
    return [Bool(v) for v in x]
end

function _report(name::AbstractString, py, jl)
    a = Float32.(py)
    b = Float32.(jl)
    size(a) == size(b) || error("$name shape mismatch: py=$(size(a)) jl=$(size(b))")
    d = abs.(a .- b)
    rel = d ./ max.(abs.(a), 1f-6)
    println(name)
    println("  py_norm=", norm(a))
    println("  jl_norm=", norm(b))
    println("  max_abs=", maximum(d), " mean_abs=", mean(d), " max_rel=", maximum(rel))
end

function main()
    path = get(ENV, "CONF_DIAG", "/tmp/py_conf_diag.json")
    raw = PXDesign.JSONLite.parse_json(read(path, String))

    s_inputs = _to_array_f32(raw["s_inputs"])
    s_trunk = _to_array_f32(raw["s_trunk"])
    z_trunk = _to_array_f32(raw["z_trunk"])
    x_pred_coords = _to_array_f32(raw["x_pred_coords"])

    feat = Dict{String,Any}(
        "atom_to_token_idx" => vec(_to_array_i(raw["atom_to_token_idx"])),
        "atom_to_tokatom_idx" => vec(_to_array_i(raw["atom_to_tokatom_idx"])),
        "distogram_rep_atom_mask" => _to_bool_vec(raw["distogram_rep_atom_mask"]),
    )

    plddt_py = _to_array_f32(raw["plddt"])
    pae_py = _to_array_f32(raw["pae"])
    pde_py = _to_array_f32(raw["pde"])
    resolved_py = _to_array_f32(raw["resolved"])

    w = PXDesign.Model.load_safetensors_weights(joinpath(pwd(), "weights_safetensors_protenix_mini_default_v0.5.0"))
    m = PXDesign.ProtenixMini.build_protenix_mini_model(w)
    PXDesign.ProtenixMini.load_protenix_mini_model!(m, w; strict = true)

    plddt_jl, pae_jl, pde_jl, resolved_jl = m.confidence_head(
        input_feature_dict = feat,
        s_inputs = s_inputs,
        s_trunk = s_trunk,
        z_trunk = z_trunk,
        pair_mask = nothing,
        x_pred_coords = x_pred_coords,
        use_embedding = true,
    )

    _report("confidence.plddt", plddt_py, plddt_jl)
    _report("confidence.pae", pae_py, pae_jl)
    _report("confidence.pde", pde_py, pde_jl)
    _report("confidence.resolved", resolved_py, resolved_jl)
end

main()
