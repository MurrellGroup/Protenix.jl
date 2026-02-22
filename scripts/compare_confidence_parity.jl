#!/usr/bin/env julia

using LinearAlgebra
using Statistics
using PXDesign

const ROOT = normpath(joinpath(@__DIR__, ".."))

function _default_weights_dir(dirname::AbstractString)
    candidates = (
        joinpath(ROOT, "release_data", dirname),
        joinpath(ROOT, dirname),
    )
    for path in candidates
        isdir(path) && return path
    end
    return first(candidates)
end

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

    # Python reference data is features-last; permute to features-first for Julia
    s_inputs_py = _to_array_f32(raw["s_inputs"])             # (N_tok, c_s_inputs)
    s_trunk_py = _to_array_f32(raw["s_trunk"])               # (N_tok, c_s)
    z_trunk_py = _to_array_f32(raw["z_trunk"])               # (N_tok, N_tok, c_z)
    x_pred_coords_py = _to_array_f32(raw["x_pred_coords"])   # (N_sample, N_atom, 3) or (N_atom, 3)

    s_inputs = permutedims(s_inputs_py)                      # (c_s_inputs, N_tok)
    s_trunk = permutedims(s_trunk_py)                        # (c_s, N_tok)
    z_trunk = permutedims(z_trunk_py, (3, 1, 2))            # (c_z, N_tok, N_tok)
    x_pred_coords = if ndims(x_pred_coords_py) == 3
        permutedims(x_pred_coords_py, (3, 2, 1))            # (3, N_atom, N_sample)
    else
        permutedims(x_pred_coords_py)                        # (3, N_atom)
    end

    feat = Dict{String,Any}(
        "atom_to_token_idx" => vec(_to_array_i(raw["atom_to_token_idx"])),
        "atom_to_tokatom_idx" => vec(_to_array_i(raw["atom_to_tokatom_idx"])),
        "distogram_rep_atom_mask" => _to_bool_vec(raw["distogram_rep_atom_mask"]),
    )

    plddt_py = _to_array_f32(raw["plddt"])
    pae_py = _to_array_f32(raw["pae"])
    pde_py = _to_array_f32(raw["pde"])
    resolved_py = _to_array_f32(raw["resolved"])

    weights_dir = get(
        ENV,
        "CONF_WEIGHTS_DIR",
        _default_weights_dir("weights_safetensors_protenix_mini_default_v0.5.0"),
    )
    w = PXDesign.Model.load_safetensors_weights(weights_dir)
    m = PXDesign.ProtenixMini.build_protenix_mini_model(w)
    PXDesign.ProtenixMini.load_protenix_mini_model!(m, w; strict = true)

    plddt_jl, pae_jl, pde_jl, resolved_jl = m.confidence_head(
        input_feature_dict = feat,
        s_inputs = s_inputs,             # (c_s_inputs, N_tok) features-first
        s_trunk = s_trunk,               # (c_s, N_tok) features-first
        z_trunk = z_trunk,               # (c_z, N_tok, N_tok) features-first
        pair_mask = nothing,
        x_pred_coords = x_pred_coords,   # (3, N_atom, N_sample) features-first
        use_embedding = true,
    )

    # Compare with Python reference (permute Julia outputs to Python convention)
    _report("confidence.plddt", plddt_py, permutedims(plddt_jl, (2, 3, 1)))
    _report("confidence.pae", pae_py, permutedims(pae_jl, (2, 3, 4, 1)))
    _report("confidence.pde", pde_py, permutedims(pde_jl, (2, 3, 4, 1)))
    _report("confidence.resolved", resolved_py, permutedims(resolved_jl, (2, 3, 1)))
end

main()
