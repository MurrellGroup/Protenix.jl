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
    println("  max_abs=", maximum(d), " mean_abs=", mean(d), " max_rel=", maximum(rel))
end

function main()
    path = get(ENV, "PAIRFORMER_BLOCK_DIAG", "/tmp/py_pairformer_block_diag.json")
    raw = PXDesign.JSONLite.parse_json(read(path, String))

    # Python reference is features-last; permute to features-first
    s_py = _to_array_f32(raw["s_in"])    # (N_tok, c_s)
    z_py = _to_array_f32(raw["z_in"])    # (N_tok, N_tok, c_z)
    s = permutedims(s_py)                # (c_s, N_tok) features-first
    z = permutedims(z_py, (3, 1, 2))    # (c_z, N_tok, N_tok) features-first

    weights_dir = get(
        ENV,
        "PAIRFORMER_BLOCK_WEIGHTS_DIR",
        _default_weights_dir("weights_safetensors_protenix_mini_default_v0.5.0"),
    )
    w = PXDesign.Model.load_safetensors_weights(weights_dir)
    m = PXDesign.ProtenixMini.build_protenix_mini_model(w)
    PXDesign.ProtenixMini.load_protenix_mini_model!(m, w; strict = true)
    blk = m.pairformer_stack.blocks[1]

    # Features-first: z is (c_z, N, N), triangle ops swap dims 2,3 for "end" attention
    tmu_out = blk.tri_mul_out(z; mask = nothing)
    z1 = z + tmu_out
    tmu_in = blk.tri_mul_in(z1; mask = nothing)
    z2 = z1 + tmu_in
    ta_start = blk.tri_att_start(z2; mask = nothing)
    z3 = z2 + ta_start
    z4 = permutedims(z3, (1, 3, 2))   # swap spatial dims, keep features in dim 1
    ta_end = blk.tri_att_end(z4; mask = nothing)
    z5 = z4 + ta_end
    z6 = permutedims(z5, (1, 3, 2))   # swap back
    pair_tr = blk.pair_transition(z6)
    z7 = z6 + pair_tr
    apb = blk.attention_pair_bias(s, z7, nothing)
    s1 = s + apb
    s_tr = blk.single_transition(s1)
    s2 = s1 + s_tr

    # Compare in features-first: permute Python (N,N,C) → (C,N,N), (N,C) → (C,N)
    _p2(x) = permutedims(x)
    _p3(x) = permutedims(x, (3, 1, 2))
    _report("tmu_out", _p3(_to_array_f32(raw["tmu_out"])), tmu_out)
    _report("z1", _p3(_to_array_f32(raw["z1"])), z1)
    _report("tmu_in", _p3(_to_array_f32(raw["tmu_in"])), tmu_in)
    _report("z2", _p3(_to_array_f32(raw["z2"])), z2)
    _report("ta_start", _p3(_to_array_f32(raw["ta_start"])), ta_start)
    _report("z3", _p3(_to_array_f32(raw["z3"])), z3)
    _report("ta_end", _p3(_to_array_f32(raw["ta_end"])), ta_end)
    _report("z6", _p3(_to_array_f32(raw["z6"])), z6)
    _report("pair_tr", _p3(_to_array_f32(raw["pair_tr"])), pair_tr)
    _report("z7", _p3(_to_array_f32(raw["z7"])), z7)
    _report("apb", _p2(_to_array_f32(raw["apb"])), apb)
    _report("s1", _p2(_to_array_f32(raw["s1"])), s1)
    _report("s_tr", _p2(_to_array_f32(raw["s_tr"])), s_tr)
    _report("s2", _p2(_to_array_f32(raw["s2"])), s2)
end

main()
