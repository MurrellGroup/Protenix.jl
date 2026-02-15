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
    path = get(ENV, "PAIRFORMER_DIAG", "/tmp/py_pairformer_diag.json")
    weights_dir = get(
        ENV,
        "PAIRFORMER_WEIGHTS_DIR",
        _default_weights_dir("weights_safetensors_protenix_mini_default_v0.5.0"),
    )
    raw = PXDesign.JSONLite.parse_json(read(path, String))
    s_in = _to_array_f32(raw["s_in"])
    z_in = _to_array_f32(raw["z_in"])
    s_out_py = _to_array_f32(raw["s_out"])
    z_out_py = _to_array_f32(raw["z_out"])

    w = PXDesign.Model.load_safetensors_weights(weights_dir)
    m = PXDesign.ProtenixMini.build_protenix_mini_model(w)
    PXDesign.ProtenixMini.load_protenix_mini_model!(m, w; strict = true)

    s_out_jl, z_out_jl = m.pairformer_stack(s_in, z_in; pair_mask = nothing)
    s_out_jl === nothing && error("pairformer_stack returned no single output")

    _report("pairformer.s_out", s_out_py, s_out_jl)
    _report("pairformer.z_out", z_out_py, z_out_jl)
end

main()
