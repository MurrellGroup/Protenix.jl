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
    diag_path = get(ENV, "PBASE_TRUNK_DENOISE_DIAG", "/tmp/py_protenix_base_trunk_denoise_diag.json")
    raw = PXDesign.JSONLite.parse_json(read(diag_path, String))
    feat_raw = raw["feat"]

    feat = Dict{String,Any}()
    for k in ("token_index", "residue_index", "asym_id", "entity_id", "sym_id", "msa", "atom_to_token_idx", "atom_to_tokatom_idx", "ref_space_uid")
        feat[k] = _to_array_i(feat_raw[k])
    end
    for k in ("token_bonds", "restype", "profile", "deletion_mean", "has_deletion", "deletion_value", "ref_pos", "ref_charge", "ref_mask", "ref_element", "ref_atom_name_chars")
        feat[k] = _to_array_f32(feat_raw[k])
    end
    feat["distogram_rep_atom_mask"] = _to_bool_vec(feat_raw["distogram_rep_atom_mask"])

    s_inputs_py = _to_array_f32(raw["s_inputs"])
    s_init_py = _to_array_f32(raw["s_init"])
    z_init_lin_py = _to_array_f32(raw["z_init_lin"])
    z_relpos_py = _to_array_f32(raw["z_relpos"])
    z_token_bond_py = _to_array_f32(raw["z_token_bond"])
    z_init_py = _to_array_f32(raw["z_init"])
    s_trunk_py = _to_array_f32(raw["s_trunk"])
    z_trunk_py = _to_array_f32(raw["z_trunk"])
    x_noisy = _to_array_f32(raw["x_noisy"])
    t_hat = vec(_to_array_f32(raw["t_hat"]))
    x_denoised_py = _to_array_f32(raw["x_denoised"])
    distogram_py = _to_array_f32(raw["distogram_logits"])
    plddt_py = _to_array_f32(raw["plddt"])
    pae_py = _to_array_f32(raw["pae"])
    pde_py = _to_array_f32(raw["pde"])
    resolved_py = _to_array_f32(raw["resolved"])
    n_cycle = Int(round(Float64(raw["n_cycle"])))

    weights_dir = get(
        ENV,
        "PBASE_WEIGHTS_DIR",
        _default_weights_dir("weights_safetensors_protenix_base_default_v0.5.0"),
    )
    w = PXDesign.Model.load_safetensors_weights(weights_dir)
    m = PXDesign.ProtenixBase.build_protenix_base_model(w)
    PXDesign.ProtenixBase.load_protenix_base_model!(m, w; strict = true)

    trunk = PXDesign.ProtenixMini.get_pairformer_output(m, feat; n_cycle = n_cycle)
    relpos = PXDesign.Model.as_relpos_input(feat)
    atom_input = PXDesign.Model.as_atom_attention_input(feat)
    atom_to_token_idx = vec(Int.(feat["atom_to_token_idx"]))

    s_init_jl = m.linear_no_bias_sinit(trunk.s_inputs)
    n_tok = size(s_init_jl, 1)
    z1_jl = m.linear_no_bias_zinit1(s_init_jl)
    z2_jl = m.linear_no_bias_zinit2(s_init_jl)
    z_init_lin_jl = reshape(z1_jl, n_tok, 1, size(z1_jl, 2)) .+ reshape(z2_jl, 1, n_tok, size(z2_jl, 2))
    z_relpos_jl = m.relative_position_encoding(feat)
    token_bonds = Float32.(feat["token_bonds"])
    z_token_bond_jl = m.linear_no_bias_token_bond(reshape(token_bonds, n_tok, n_tok, 1))
    z_init_jl = z_init_lin_jl .+ z_relpos_jl .+ z_token_bond_jl
    x_denoised_jl = m.diffusion_module(
        x_noisy,
        t_hat;
        relpos_input = relpos,
        s_inputs = trunk.s_inputs,
        s_trunk = trunk.s,
        z_trunk = trunk.z,
        atom_to_token_idx = atom_to_token_idx,
        input_feature_dict = atom_input,
    )
    distogram_jl = m.distogram_head(trunk.z)
    plddt_jl, pae_jl, pde_jl, resolved_jl = m.confidence_head(
        input_feature_dict = feat,
        s_inputs = trunk.s_inputs,
        s_trunk = trunk.s,
        z_trunk = trunk.z,
        pair_mask = nothing,
        x_pred_coords = x_denoised_jl,
        use_embedding = true,
    )

    _report("trunk.s_inputs", s_inputs_py, trunk.s_inputs)
    _report("trunk.s_init", s_init_py, s_init_jl)
    _report("trunk.z_init_lin", z_init_lin_py, z_init_lin_jl)
    _report("trunk.z_relpos", z_relpos_py, z_relpos_jl)
    _report("trunk.z_token_bond", z_token_bond_py, z_token_bond_jl)
    _report("trunk.z_init", z_init_py, z_init_jl)
    _report("trunk.s", s_trunk_py, trunk.s)
    _report("trunk.z", z_trunk_py, trunk.z)
    _report("diffusion.x_denoised", x_denoised_py, x_denoised_jl)
    _report("distogram.logits", distogram_py, distogram_jl)
    _report("confidence.plddt", plddt_py, plddt_jl)
    _report("confidence.pae", pae_py, pae_jl)
    _report("confidence.pde", pde_py, pde_jl)
    _report("confidence.resolved", resolved_py, resolved_jl)
end

main()
