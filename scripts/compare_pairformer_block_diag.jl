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

    s = _to_array_f32(raw["s_in"])
    z = _to_array_f32(raw["z_in"])

    w = PXDesign.Model.load_safetensors_weights(joinpath(pwd(), "weights_safetensors_protenix_mini_default_v0.5.0"))
    m = PXDesign.ProtenixMini.build_protenix_mini_model(w)
    PXDesign.ProtenixMini.load_protenix_mini_model!(m, w; strict = true)
    blk = m.pairformer_stack.blocks[1]

    tmu_out = blk.tri_mul_out(z; mask = nothing)
    z1 = z + tmu_out
    tmu_in = blk.tri_mul_in(z1; mask = nothing)
    z2 = z1 + tmu_in
    ta_start = blk.tri_att_start(z2; mask = nothing)
    z3 = z2 + ta_start
    z4 = permutedims(z3, (2, 1, 3))
    ta_end = blk.tri_att_end(z4; mask = nothing)
    z5 = z4 + ta_end
    z6 = permutedims(z5, (2, 1, 3))
    pair_tr = blk.pair_transition(z6)
    z7 = z6 + pair_tr
    apb = blk.attention_pair_bias(s, z7, nothing)
    s1 = s + apb
    s_tr = blk.single_transition(s1)
    s2 = s1 + s_tr

    _report("tmu_out", _to_array_f32(raw["tmu_out"]), tmu_out)
    _report("z1", _to_array_f32(raw["z1"]), z1)
    _report("tmu_in", _to_array_f32(raw["tmu_in"]), tmu_in)
    _report("z2", _to_array_f32(raw["z2"]), z2)
    _report("ta_start", _to_array_f32(raw["ta_start"]), ta_start)
    _report("z3", _to_array_f32(raw["z3"]), z3)
    _report("ta_end", _to_array_f32(raw["ta_end"]), ta_end)
    _report("z6", _to_array_f32(raw["z6"]), z6)
    _report("pair_tr", _to_array_f32(raw["pair_tr"]), pair_tr)
    _report("z7", _to_array_f32(raw["z7"]), z7)
    _report("apb", _to_array_f32(raw["apb"]), apb)
    _report("s1", _to_array_f32(raw["s1"]), s1)
    _report("s_tr", _to_array_f32(raw["s_tr"]), s_tr)
    _report("s2", _to_array_f32(raw["s2"]), s2)
end

main()
