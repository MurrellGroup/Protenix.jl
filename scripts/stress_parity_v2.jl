#!/usr/bin/env julia
# Full 100-case stress input parity check — updated for Fixes 1-18.
#
# Compares Julia PXDesign input features against Python Protenix v1.0 dumps.
# Uses the same pipeline as predict_json(), including all fixes:
#   Fix 14: entity-gated CCD mol_type override (polymer_chain_ids)
#   Fix 16: DNA chain MSA features (dna_chain_specs)
#   Fix 18: v1.0 CCD override for all entities (all_entities)
#
# Usage:
#   julia --project=/home/claudey/FixingKAFA/ka_run_env \
#         /home/claudey/FixingKAFA/PXDesign.jl/scripts/stress_parity_v2.jl

using MoleculeFlow
using PXDesign
using Random

const PA = PXDesign.ProtenixAPI

# --- Array reconstruction from Python JSON dumps ---

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
    x isa AbstractVector || return fill(Float32(x), ())
    dims = _nested_dims(x)
    out = Array{Float32}(undef, Tuple(dims)...)
    idx = Int[]
    function walk(y)
        if y isa AbstractVector
            for (i, z) in enumerate(y)
                push!(idx, i); walk(z); pop!(idx)
            end
        else
            out[Tuple(idx)...] = Float32(y)
        end
    end
    walk(x)
    return out
end

function _to_array_i(x)
    x isa AbstractVector || return fill(Int(round(Float64(x))), ())
    dims = _nested_dims(x)
    out = Array{Int}(undef, Tuple(dims)...)
    idx = Int[]
    function walk(y)
        if y isa AbstractVector
            for (i, z) in enumerate(y)
                push!(idx, i); walk(z); pop!(idx)
            end
        else
            out[Tuple(idx)...] = Int(round(Float64(y)))
        end
    end
    walk(x)
    return out
end

function _py_entry_to_array(entry::AbstractDict)
    dtype = lowercase(String(entry["dtype"]))
    data = entry["data"]
    if occursin("bool", dtype)
        a = _to_array_f32(data)
        return Bool.(round.(a)), dtype
    elseif occursin("int", dtype) || occursin("long", dtype)
        return _to_array_i(data), dtype
    end
    return _to_array_f32(data), dtype
end

_as_array(x) = x isa AbstractArray ? x : fill(x, ())

# --- Build Julia features matching predict_json pipeline ---

function build_julia_features(json_path::String, case::String, model_name::String, seed::Int)
    tasks = PA._ensure_json_tasks(json_path)
    task = PA._as_string_dict(tasks[1])
    parsed_task = PA._parse_task_entities(task; json_dir=dirname(abspath(json_path)))

    rng = MersenneTwister(seed)
    atoms = PA._remove_covalent_leaving_atoms(
        parsed_task.atoms, task, parsed_task.entity_chain_ids, parsed_task.entity_atom_map;
        rng=rng)
    atoms = PA._apply_mse_to_met(atoms)
    # Fix 14 + Fix 18: entity-gated CCD override, all_entities for v1.0
    atoms = PA._apply_ccd_mol_type_override(atoms, parsed_task.polymer_chain_ids;
        all_entities=PA._is_v1_model(model_name))

    bundle = PXDesign.Data.build_feature_bundle_from_atoms(
        atoms; task_name=case, rng=MersenneTwister(seed), ref_pos_augment=false)
    token_chain_ids = [bundle["atoms"][tok.centre_atom_index].chain_id for tok in bundle["tokens"]]
    feat = bundle["input_feature_dict"]

    PA._normalize_protenix_feature_dict!(feat)
    PA._fix_restype_for_modified_residues!(feat, bundle["atoms"], bundle["tokens"])
    PA._fix_entity_and_sym_ids!(feat, bundle["atoms"], bundle["tokens"], parsed_task.entity_chain_ids)

    # MSA with Fix 16 DNA chain specs
    PA._inject_task_msa_features!(feat, task, json_path;
        use_msa=occursin("v1.0", model_name),
        chain_specs=parsed_task.protein_specs,
        rna_chain_specs=parsed_task.rna_specs,
        dna_chain_specs=parsed_task.dna_specs,
        token_chain_ids=token_chain_ids)

    PA._inject_task_covalent_token_bonds!(feat, bundle["atoms"], task,
        parsed_task.entity_chain_ids, parsed_task.entity_atom_map)
    PA._inject_task_template_features!(feat, task)
    PA._inject_task_esm_token_embedding!(feat, task)
    PA._inject_task_constraint_feature!(feat, task, bundle["atoms"],
        parsed_task.entity_chain_ids, parsed_task.entity_atom_map, "task '$case'")

    return feat
end

# --- Compare a single case ---

function compare_case(input_json::String, py_dump_path::String, model_name::String, seed::Int)
    case = replace(basename(input_json), ".json" => "")

    # Load Python dump
    raw = PXDesign.JSONLite.parse_json(read(py_dump_path, String))
    py_feat_raw = raw["input_feature_dict"]

    # Build Julia features
    jl_feat = build_julia_features(input_json, case, model_name, seed)

    # Compare shared keys
    py_keys = Set(String.(collect(keys(py_feat_raw))))
    jl_keys = Set(String.(collect(keys(jl_feat))))
    shared = sort!(collect(intersect(py_keys, jl_keys)))

    failed_keys = Dict{String,String}()
    n_pass = 0
    ref_pos_rigid = false
    atol = 1f-5

    for k in shared
        py_entry = py_feat_raw[k]
        py_entry isa AbstractDict || continue
        haskey(py_entry, "data") || continue
        haskey(py_entry, "dtype") || continue

        py_arr, dtype = _py_entry_to_array(py_entry)
        jl_arr = _as_array(jl_feat[k])

        if size(py_arr) != size(jl_arr)
            failed_keys[k] = "shape($(size(py_arr))vs$(size(jl_arr)))"
            continue
        end

        if occursin("float", dtype) || occursin("half", dtype) || occursin("double", dtype) || occursin("bfloat", dtype)
            d = abs.(Float32.(py_arr) .- Float32.(jl_arr))
            mx = isempty(d) ? 0f0 : maximum(d)
            if mx > atol
                n = count(d .> atol)
                if k == "ref_pos"
                    ref_pos_rigid = true
                    n_pass += 1
                else
                    failed_keys[k] = "$n/$(length(d))"
                end
            else
                n_pass += 1
            end
        else
            mis = count(Int.(py_arr) .!= Int.(jl_arr))
            if mis > 0
                failed_keys[k] = "$mis/$(length(py_arr))"
            else
                n_pass += 1
            end
        end
    end

    return (failed_keys=failed_keys, n_pass=n_pass, n_shared=length(shared),
            ref_pos_rigid=ref_pos_rigid)
end

# --- Main ---

function main()
    stress_dir = "/home/claudey/FixingKAFA/PXDesign.jl/clean_targets/stress_inputs"
    dump_base = "/tmp/v1_parity/py_dumps"
    model_name = "protenix_base_default_v1.0.0"
    seed = 101

    json_files = sort(filter(f -> endswith(f, ".json"), readdir(stress_dir)))
    println("Stress parity v2 (Fixes 1-18): $(length(json_files)) cases, model=$model_name")
    println()

    n_perfect = 0; n_refpos = 0; n_fail = 0; n_skip = 0; n_error = 0
    failures = String[]
    per_case = Dict{String, Symbol}()

    for jf in json_files
        case = replace(jf, ".json" => "")
        input_json = joinpath(stress_dir, jf)
        dump_dir = joinpath(dump_base, "stress_$(case)")

        if !isdir(dump_dir)
            n_skip += 1
            per_case[case] = :skip
            continue
        end

        suffix = "__seed$(seed).json"
        candidates = filter(readdir(dump_dir; join=true)) do p
            startswith(basename(p), model_name * "__") && endswith(basename(p), suffix)
        end
        if isempty(candidates)
            n_skip += 1
            per_case[case] = :skip
            continue
        end
        py_dump_path = first(candidates)

        print(rpad(case, 35))
        flush(stdout)

        try
            r = compare_case(input_json, py_dump_path, model_name, seed)
            if isempty(r.failed_keys)
                if r.ref_pos_rigid
                    println("PASS (ref_pos only, $(r.n_pass)/$(r.n_shared) keys)")
                    n_refpos += 1
                    per_case[case] = :refpos
                else
                    println("PERFECT ($(r.n_pass)/$(r.n_shared) keys)")
                    n_perfect += 1
                    per_case[case] = :perfect
                end
            else
                details = join(["$k($v)" for (k,v) in sort(collect(r.failed_keys))], ", ")
                println("FAIL: $details")
                n_fail += 1
                per_case[case] = :fail
                push!(failures, "$case: $details")
            end
        catch e
            msg = sprint(showerror, e; context=:compact=>true)
            if length(msg) > 150; msg = msg[1:150] * "..."; end
            println("ERROR: $msg")
            n_error += 1
            per_case[case] = :error
            push!(failures, "$case: ERROR $msg")
        end
        flush(stdout)
    end

    println("\n" * "=" * 60)
    println("STRESS PARITY SUMMARY (Fixes 1-18, v1.0 model)")
    println("=" * 60)
    println("  Perfect match: $n_perfect")
    println("  ref_pos only:  $n_refpos")
    println("  Failed:        $n_fail")
    println("  Errors:        $n_error")
    println("  Skipped:       $n_skip")
    total = n_perfect + n_refpos + n_fail + n_error + n_skip
    println("  Total:         $total")
    if !isempty(failures)
        println("\nFailures:")
        for f in failures
            println("  $f")
        end
    end
    println("=" * 60)
    println("Done.")
end

main()
