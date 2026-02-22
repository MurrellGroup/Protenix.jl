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

    # Python reference data is features-last; load and permute to features-first
    z_in_py_fl = _to_array_f32(raw["z_in"])             # (N, N, c_z) features-last
    s_inputs_py_fl = _to_array_f32(raw["s_inputs"])     # (N, c_s_inputs) features-last
    msa_py = _to_array_i(raw["msa"])                    # (N_msa, N_tok)
    has_del_py = _to_array_f32(raw["has_deletion"])     # (N_msa, N_tok)
    del_val_py = _to_array_f32(raw["deletion_value"])   # (N_msa, N_tok)
    m0_py = _to_array_f32(raw["m0"])                    # features-last
    opm_py = _to_array_f32(raw["opm"])
    z1_py = _to_array_f32(raw["z1"])
    z2_py = _to_array_f32(raw["z2"])
    z_out_py = _to_array_f32(raw["z_out"])
    z_out_module_py = haskey(raw, "z_out_module") ? _to_array_f32(raw["z_out_module"]) : z_out_py
    z_blocks_py = haskey(raw, "z_blocks") ? raw["z_blocks"] : Any[]
    m_blocks_py = haskey(raw, "m_blocks") ? raw["m_blocks"] : Any[]
    block_debug_py = haskey(raw, "block_debug") ? raw["block_debug"] : Any[]

    # Convert to features-first for Julia
    z_in = permutedims(z_in_py_fl, (3, 1, 2))          # (c_z, N, N)
    s_inputs = permutedims(s_inputs_py_fl)              # (c_s_inputs, N)
    msa = permutedims(msa_py)                           # (N_tok, N_msa)
    has_del = permutedims(has_del_py)                   # (N_tok, N_msa)
    del_val = permutedims(del_val_py)                   # (N_tok, N_msa)

    w = PXDesign.Model.load_safetensors_weights(weights_dir)
    m = PXDesign.ProtenixMini.build_protenix_mini_model(w)
    PXDesign.ProtenixMini.load_protenix_mini_model!(m, w; strict = true)

    # Features-first MSA feature construction
    # MSA module uses features-first: msa (N_tok, N_msa), one_hot (32, N_tok, N_msa)
    msa_oh = PXDesign.ProtenixMini.one_hot_int(msa, 32)  # (32, N_tok, N_msa)
    n_tok = size(msa, 1)
    n_msa = size(msa, 2)
    msa_feat = cat(
        msa_oh,
        reshape(has_del, 1, n_tok, n_msa),
        reshape(del_val, 1, n_tok, n_msa);
        dims = 1,
    )  # (34, N_tok, N_msa) features-first
    m0_jl = m.msa_module.linear_no_bias_m(msa_feat)  # (c_m, N_tok, N_msa)
    c_m = m.msa_module.c_m
    m0_jl .+= reshape(m.msa_module.linear_no_bias_s(s_inputs), c_m, n_tok, 1)  # broadcast over N_msa
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
            _perm3(x) = permutedims(_to_array_f32(x), (3, 1, 2))
            if haskey(dbg, "opm")
                _report("msa.block_$(i_blk).opm", _perm3(dbg["opm"]), opm_blk_jl)
            end
            if haskey(dbg, "z_pre_pair")
                _report("msa.block_$(i_blk).z_pre_pair", _perm3(dbg["z_pre_pair"]), z_pre_pair_jl)
            end
            if m_pair_jl !== nothing && haskey(dbg, "m_pair") && dbg["m_pair"] !== nothing
                _report("msa.block_$(i_blk).m_pair", _perm3(dbg["m_pair"]), m_pair_jl)
            end
            if m_after_pair_jl !== nothing && haskey(dbg, "m_after_pair") && dbg["m_after_pair"] !== nothing
                _report("msa.block_$(i_blk).m_after_pair", _perm3(dbg["m_after_pair"]), m_after_pair_jl)
            end
            if m_trans_jl !== nothing && haskey(dbg, "m_trans") && dbg["m_trans"] !== nothing
                _report("msa.block_$(i_blk).m_trans", _perm3(dbg["m_trans"]), m_trans_jl)
            end
            if m_after_trans_jl !== nothing && haskey(dbg, "m_after_trans") && dbg["m_after_trans"] !== nothing
                _report("msa.block_$(i_blk).m_after_trans", _perm3(dbg["m_after_trans"]), m_after_trans_jl)
            end
            if haskey(dbg, "z_post_pair")
                _report("msa.block_$(i_blk).z_post_pair", _perm3(dbg["z_post_pair"]), z_cur)
            end
        end
    end
    # MSA module expects raw dict with (N_msa, N_tok) convention (pre-as_protenix_features)
    feat_raw = Dict{String,Any}(
        "msa" => msa_py,               # (N_msa, N_tok) Python convention
        "has_deletion" => has_del_py,   # (N_msa, N_tok)
        "deletion_value" => del_val_py, # (N_msa, N_tok)
    )
    z_out_jl = m.msa_module(feat_raw, z_in, s_inputs; pair_mask = nothing, rng = MersenneTwister(0))

    # Compare in features-first space (permute Python references)
    _perm2(x) = permutedims(x)
    _perm3(x) = permutedims(x, (3, 1, 2))
    _report("msa.m0", _perm3(m0_py), m0_jl)
    _report("msa.opm", _perm3(opm_py), opm_jl)
    _report("msa.z1", _perm3(z1_py), z1_jl)
    _report("msa.z2", _perm3(z2_py), z2_jl)
    for i in 1:min(length(z_blocks_py), length(z_blocks_jl))
        _report("msa.z_block_$(i)", _perm3(_to_array_f32(z_blocks_py[i])), z_blocks_jl[i])
    end
    for i in 1:min(length(m_blocks_py), length(m_blocks_jl))
        if m_blocks_py[i] !== nothing && m_blocks_jl[i] !== nothing
            _report("msa.m_block_$(i)", _perm3(_to_array_f32(m_blocks_py[i])), m_blocks_jl[i])
        end
    end
    _report("msa.z_out_loop", _perm3(z_out_py), z_cur)
    _report("msa.z_out_module_py", _perm3(z_out_module_py), z_out_jl)
    _report("msa.z_out", _perm3(z_out_py), z_out_jl)
end

main()
