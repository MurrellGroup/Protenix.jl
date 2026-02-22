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
    diag_path = get(ENV, "PMINI_TRUNK_DENOISE_DIAG", "/tmp/py_protenix_mini_trunk_denoise_diag.json")
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

    weights_ref = get(ENV, "PMINI_WEIGHTS_DIR", nothing)
    if weights_ref === nothing
        weights_ref = PXDesign.default_weights_path("protenix_mini_default_v0.5.0")
    end
    w = PXDesign.Model.load_safetensors_weights(weights_ref)
    m = PXDesign.ProtenixMini.build_protenix_mini_model(w)
    PXDesign.ProtenixMini.load_protenix_mini_model!(m, w; strict = true)

    trunk = PXDesign.ProtenixMini.get_pairformer_output(m, feat; n_cycle = n_cycle)
    relpos = PXDesign.Model.as_relpos_input(feat)
    atom_input = PXDesign.Model.as_atom_attention_input(feat)
    atom_to_token_idx = vec(Int.(feat["atom_to_token_idx"]))
    # Everything is features-first now; convert Python reference (features-last) for comparison
    x_noisy_ff = permutedims(x_noisy, (3, 2, 1))          # Python (N_sample, N_atom, 3) → (3, N_atom, N_sample)
    x_denoised_ff = m.diffusion_module(
        x_noisy_ff,
        t_hat;
        relpos_input = relpos,
        s_inputs = trunk.s_inputs,       # (c_s_inputs, N_tok) features-first
        s_trunk = trunk.s,               # (c_s, N_tok) features-first
        z_trunk = trunk.z,               # (c_z, N_tok, N_tok) features-first
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
        x_pred_coords = x_denoised_ff,   # (3, N_atom, N_sample) features-first
        use_embedding = true,
    )

    # Compare with Python reference (permute Julia features-first → Python features-last for comparison)
    _report("trunk.s_inputs", s_inputs_py, permutedims(trunk.s_inputs))
    _report("trunk.s", s_trunk_py, permutedims(trunk.s))
    _report("trunk.z", z_trunk_py, permutedims(trunk.z, (2, 3, 1)))
    _report("diffusion.x_denoised", x_denoised_py, permutedims(x_denoised_ff, (3, 2, 1)))
    _report("distogram.logits", distogram_py, permutedims(distogram_jl, (2, 3, 1)))
    _report("confidence.plddt", plddt_py, permutedims(plddt_jl, (2, 3, 1)))
    _report("confidence.pae", pae_py, permutedims(pae_jl, (2, 3, 4, 1)))
    _report("confidence.pde", pde_py, permutedims(pde_jl, (2, 3, 4, 1)))
    _report("confidence.resolved", resolved_py, permutedims(resolved_jl, (2, 3, 1)))
end

main()
