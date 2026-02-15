#!/usr/bin/env julia

using LinearAlgebra
using Statistics
using Random
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
    x isa AbstractVector || return Int(x)
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
    path = get(ENV, "MSA_DIAG", "/tmp/py_msa_diag.json")
    weights_dir = get(
        ENV,
        "MSA_WEIGHTS_DIR",
        _default_weights_dir("weights_safetensors_protenix_mini_default_v0.5.0"),
    )
    raw = PXDesign.JSONLite.parse_json(read(path, String))

    z_in = _to_array_f32(raw["z_in"])
    s_inputs = _to_array_f32(raw["s_inputs"])
    msa = _to_array_i(raw["msa"])
    has_del = _to_array_f32(raw["has_deletion"])
    del_val = _to_array_f32(raw["deletion_value"])
    m0_py = _to_array_f32(raw["m0"])
    opm_py = _to_array_f32(raw["opm"])
    z1_py = _to_array_f32(raw["z1"])
    z2_py = _to_array_f32(raw["z2"])
    z_out_py = _to_array_f32(raw["z_out"])
    z_out_module_py = haskey(raw, "z_out_module") ? _to_array_f32(raw["z_out_module"]) : z_out_py
    z_blocks_py = haskey(raw, "z_blocks") ? raw["z_blocks"] : Any[]
    m_blocks_py = haskey(raw, "m_blocks") ? raw["m_blocks"] : Any[]
    block_debug_py = haskey(raw, "block_debug") ? raw["block_debug"] : Any[]

    w = PXDesign.Model.load_safetensors_weights(weights_dir)
    m = PXDesign.ProtenixMini.build_protenix_mini_model(w)
    PXDesign.ProtenixMini.load_protenix_mini_model!(m, w; strict = true)

    feat = Dict{String,Any}(
        "msa" => msa,
        "has_deletion" => has_del,
        "deletion_value" => del_val,
    )

    msa_oh = PXDesign.ProtenixMini.one_hot_int(msa, 32)
    msa_feat = cat(
        msa_oh,
        reshape(has_del, size(has_del, 1), size(has_del, 2), 1),
        reshape(del_val, size(del_val, 1), size(del_val, 2), 1);
        dims = 3,
    )
    m0_jl = m.msa_module.linear_no_bias_m(msa_feat)
    m0_jl .+= reshape(m.msa_module.linear_no_bias_s(s_inputs), 1, size(s_inputs, 1), m.msa_module.c_m)
    opm_jl = m.msa_module.blocks[1].outer_product_mean_msa(m0_jl)
    z1_jl = z_in + opm_jl
    _, z2_jl = m.msa_module.blocks[1].pair_stack(nothing, z1_jl; pair_mask = nothing)
    msa_cur = m0_jl
    z_cur = z_in
    z_blocks_jl = Any[]
    m_blocks_jl = Any[]
    for (i_blk, blk) in enumerate(m.msa_module.blocks)
        opm_blk_jl = blk.outer_product_mean_msa(msa_cur)
        z_pre_pair_jl = z_cur + opm_blk_jl
        m_pair_jl = nothing
        m_after_pair_jl = nothing
        m_trans_jl = nothing
        m_after_trans_jl = nothing
        if blk.msa_stack !== nothing
            m_pair_jl = blk.msa_stack.msa_pair_weighted_averaging(msa_cur, z_pre_pair_jl)
            m_after_pair_jl = msa_cur + m_pair_jl
            m_trans_jl = blk.msa_stack.transition_m(m_after_pair_jl)
            m_after_trans_jl = m_after_pair_jl + m_trans_jl
        end
        msa_cur, z_cur = blk(msa_cur, z_cur; pair_mask = nothing)
        push!(m_blocks_jl, msa_cur)
        push!(z_blocks_jl, z_cur)

        if i_blk <= length(block_debug_py)
            dbg = block_debug_py[i_blk]
            if haskey(dbg, "opm")
                _report("msa.block_$(i_blk).opm", _to_array_f32(dbg["opm"]), opm_blk_jl)
            end
            if haskey(dbg, "z_pre_pair")
                _report("msa.block_$(i_blk).z_pre_pair", _to_array_f32(dbg["z_pre_pair"]), z_pre_pair_jl)
            end
            if m_pair_jl !== nothing && haskey(dbg, "m_pair") && dbg["m_pair"] !== nothing
                _report("msa.block_$(i_blk).m_pair", _to_array_f32(dbg["m_pair"]), m_pair_jl)
            end
            if m_after_pair_jl !== nothing && haskey(dbg, "m_after_pair") && dbg["m_after_pair"] !== nothing
                _report("msa.block_$(i_blk).m_after_pair", _to_array_f32(dbg["m_after_pair"]), m_after_pair_jl)
            end
            if m_trans_jl !== nothing && haskey(dbg, "m_trans") && dbg["m_trans"] !== nothing
                _report("msa.block_$(i_blk).m_trans", _to_array_f32(dbg["m_trans"]), m_trans_jl)
            end
            if m_after_trans_jl !== nothing && haskey(dbg, "m_after_trans") && dbg["m_after_trans"] !== nothing
                _report("msa.block_$(i_blk).m_after_trans", _to_array_f32(dbg["m_after_trans"]), m_after_trans_jl)
            end
            if haskey(dbg, "z_post_pair")
                _report("msa.block_$(i_blk).z_post_pair", _to_array_f32(dbg["z_post_pair"]), z_cur)
            end
        end
    end
    z_out_jl = m.msa_module(feat, z_in, s_inputs; pair_mask = nothing, rng = MersenneTwister(0))

    _report("msa.m0", m0_py, m0_jl)
    _report("msa.opm", opm_py, opm_jl)
    _report("msa.z1", z1_py, z1_jl)
    _report("msa.z2", z2_py, z2_jl)
    for i in 1:min(length(z_blocks_py), length(z_blocks_jl))
        _report("msa.z_block_$(i)", _to_array_f32(z_blocks_py[i]), z_blocks_jl[i])
    end
    for i in 1:min(length(m_blocks_py), length(m_blocks_jl))
        if m_blocks_py[i] !== nothing && m_blocks_jl[i] !== nothing
            _report("msa.m_block_$(i)", _to_array_f32(m_blocks_py[i]), m_blocks_jl[i])
        end
    end
    _report("msa.z_out_loop", z_out_py, z_cur)
    _report("msa.z_out_module_py", z_out_module_py, z_out_jl)
    _report("msa.z_out", z_out_py, z_out_jl)
end

main()
