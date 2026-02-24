#!/usr/bin/env julia
"""
PXDesign input feature parity test: Julia vs Python.

Compares input_feature_dict from Julia's build_basic_feature_bundle()
against Python dumps from dump_pxdesign_features.py.

Usage:
    julia --project=/home/claudey/FixingKAFA/ka_run_env \\
        /home/claudey/FixingKAFA/PXDesign.jl/scripts/pxdesign_parity.jl
"""

using PXDesign
using PXDesign.JSONLite: parse_json
using PXDesign.Schema: parse_tasks, InputTask
using PXDesign.Inputs: load_input_tasks
using PXDesign.Data: build_basic_feature_bundle
using Random

const INPUTS_DIR = "/home/claudey/FixingKAFA/PXDesign.jl/clean_targets/inputs"
const RUN_DIR = "/home/claudey/FixingKAFA/PXDesign.jl/clean_targets/run_20260224/clean_targets"
const PY_DUMP_DIR = "/tmp/pxdesign_parity/py_dumps"

# Features where values are random (augmented ref_pos) — compare shape only
const SHAPE_ONLY_FEATURES = Set(["ref_pos"])

# Features that may differ in representation between Julia/Python
# (e.g. plddt might not exist in Julia unconditional, or vice versa)
const OPTIONAL_FEATURES = Set(["plddt", "struct_cb_coords", "struct_cb_mask"])

# Features produced by Python pipeline but NOT consumed by PXDesign model.
# These are infrastructure/training features and don't affect inference.
const NON_MODEL_FEATURES = Set([
    "bond_mask", "mol_id", "mol_atom_index", "entity_mol_id",
    "modified_res_mask", "resolution", "pae_rep_atom_mask",
    "plddt_m_rep_atom_mask", "prot_pair_num_alignments",
    "prot_unpair_num_alignments", "rna_pair_num_alignments",
    "rna_unpair_num_alignments", "label_dict",
])

function load_py_dump(case_name::String)
    path = joinpath(PY_DUMP_DIR, "$(case_name).json")
    isfile(path) || error("Python dump not found: $path")
    raw = parse_json(read(path, String))
    return raw
end

function py_tensor_to_array(entry)
    shape_raw = entry["shape"]
    shape = Int.(shape_raw)
    dtype = string(entry["dtype"])
    data = entry["data"]

    # Scalar
    if isempty(shape)
        return _convert_scalar(data, dtype)
    end

    # 1D
    if length(shape) == 1
        arr = _collect_flat(data, dtype)
        return arr
    end

    # Multi-dimensional: data is nested lists, reshape
    flat = _flatten(data)
    arr = _convert_flat(flat, dtype)
    # Python is row-major, Julia is column-major
    # Python shape [a, b, c] → Julia needs permutedims of reshape(..., reverse(shape)...)
    jl_shape = Tuple(reverse(shape))
    reshaped = reshape(arr, jl_shape)
    # Permute back to match Python's logical layout
    ndims_val = length(shape)
    perm = Tuple(ndims_val:-1:1)
    return permutedims(reshaped, perm)
end

function _convert_scalar(data, dtype)
    if occursin("int", dtype)
        return Int64(data)
    elseif occursin("float", dtype)
        return Float32(data)
    elseif occursin("bool", dtype)
        return Bool(data)
    else
        return data
    end
end

function _collect_flat(data, dtype)
    if occursin("int", dtype)
        return Int64.(data)
    elseif occursin("float", dtype)
        return Float32.(data)
    elseif occursin("bool", dtype)
        return Bool.(data)
    else
        return collect(data)
    end
end

function _flatten(data)
    out = Any[]
    _flatten!(out, data)
    return out
end

function _flatten!(out, data)
    if data isa AbstractVector
        for x in data
            _flatten!(out, x)
        end
    else
        push!(out, data)
    end
end

function _convert_flat(flat, dtype)
    if occursin("int", dtype)
        return Int64.(flat)
    elseif occursin("float", dtype)
        return Float32.(flat)
    elseif occursin("bool", dtype)
        return Bool.(flat)
    else
        return collect(flat)
    end
end

function compare_feature(name::String, jl_val, py_entry; atol=1e-5)
    py_shape = haskey(py_entry, "shape") ? Tuple(Int.(py_entry["shape"])) : ()
    py_dtype = haskey(py_entry, "dtype") ? string(py_entry["dtype"]) : "unknown"

    # Get Julia shape
    jl_shape = if jl_val isa AbstractArray
        size(jl_val)
    elseif jl_val isa Number
        ()
    else
        (length(jl_val),)
    end

    # Shape comparison
    if jl_shape != py_shape
        return (match=false, reason="shape mismatch: Julia=$jl_shape vs Python=$py_shape")
    end

    # For shape-only features, we're done
    if name in SHAPE_ONLY_FEATURES
        return (match=true, reason="shape-only ($(jl_shape))")
    end

    # Convert Python data to array for comparison
    py_arr = try
        py_tensor_to_array(py_entry)
    catch e
        return (match=false, reason="failed to convert Python data: $e")
    end

    # Value comparison
    if jl_val isa AbstractArray && py_arr isa AbstractArray
        if eltype(jl_val) <: AbstractFloat || eltype(py_arr) <: AbstractFloat
            jl_f = Float64.(jl_val)
            py_f = Float64.(py_arr)
            max_diff = maximum(abs.(jl_f .- py_f))
            if max_diff <= atol
                return (match=true, reason="float match (max_diff=$max_diff)")
            else
                # Find first mismatch location
                idx = findfirst(i -> abs(jl_f[i] - py_f[i]) > atol, eachindex(jl_f))
                return (match=false, reason="value mismatch: max_diff=$max_diff at idx=$idx (jl=$(jl_f[idx]) py=$(py_f[idx]))")
            end
        else
            n_diff = count(jl_val .!= py_arr)
            if n_diff == 0
                return (match=true, reason="exact match")
            else
                total = length(jl_val)
                # Find first mismatch
                idx = findfirst(i -> jl_val[i] != py_arr[i], eachindex(jl_val))
                return (match=false, reason="$n_diff/$total mismatches, first at idx=$idx (jl=$(jl_val[idx]) py=$(py_arr[idx]))")
            end
        end
    elseif jl_val isa Number && py_arr isa Number
        if abs(Float64(jl_val) - Float64(py_arr)) <= atol
            return (match=true, reason="scalar match")
        else
            return (match=false, reason="scalar mismatch: jl=$jl_val py=$py_arr")
        end
    else
        return (match=false, reason="type mismatch: jl=$(typeof(jl_val)) py=$(typeof(py_arr))")
    end
end

function get_test_cases()
    cases = Tuple{String, String}[]  # (json_path, case_name)

    # Existing design targets (22-32): use input_source_snapshot.json
    for i in 22:32
        yamls = filter(f -> startswith(basename(f), "$(i)_") && endswith(f, ".yaml"),
                       readdir(INPUTS_DIR; join=true))
        if !isempty(yamls)
            name = splitext(basename(yamls[1]))[1]
            snap = joinpath(RUN_DIR, "$(name)__pxdesign_v0.1.0", "input_source_snapshot.json")
            if isfile(snap)
                push!(cases, (snap, name))
            else
                println("  Skipping $name: no snapshot JSON")
            end
        end
    end

    # New design targets (38-45): direct JSON files
    for i in 38:45
        jsons = filter(f -> startswith(basename(f), "$(i)_") && endswith(f, ".json"),
                       readdir(INPUTS_DIR; join=true))
        if !isempty(jsons)
            name = splitext(basename(jsons[1]))[1]
            push!(cases, (jsons[1], name))
        end
    end

    return cases
end

function run_parity_test(json_path::String, case_name::String)
    println("\n=== $case_name ===")
    println("  JSON: $json_path")

    # Check Python dump exists
    py_dump_path = joinpath(PY_DUMP_DIR, "$(case_name).json")
    if !isfile(py_dump_path)
        println("  SKIP: no Python dump at $py_dump_path")
        return :skip
    end

    # Load Python dump
    py_dump = load_py_dump(case_name)
    py_feat = py_dump["input_feature_dict"]
    py_n_token = Int(py_dump["N_token"])
    py_n_atom = Int(py_dump["N_atom"])
    println("  Python: N_token=$py_n_token, N_atom=$py_n_atom, $(length(keys(py_feat))) features")

    # Build Julia features
    raw_tasks = load_input_tasks(json_path)
    # Resolve relative structure_file paths relative to the JSON directory
    json_dir = dirname(abspath(json_path))
    for rt in raw_tasks
        if haskey(rt, "condition") && haskey(rt["condition"], "structure_file")
            sf = rt["condition"]["structure_file"]
            if !isempty(sf) && !isabspath(sf)
                rt["condition"]["structure_file"] = normpath(joinpath(json_dir, sf))
            end
        end
    end
    tasks = parse_tasks(raw_tasks)
    task = tasks[1]

    # Use fixed RNG for reproducibility, ref_pos_augment=false to get clean CCD positions
    rng = Random.MersenneTwister(42)
    bundle = build_basic_feature_bundle(task; rng=rng, ref_pos_augment=false)
    jl_feat = bundle["input_feature_dict"]
    jl_dims = bundle["dims"]
    jl_n_token = jl_dims["N_token"]
    jl_n_atom = jl_dims["N_atom"]
    println("  Julia:  N_token=$jl_n_token, N_atom=$jl_n_atom, $(length(keys(jl_feat))) features")

    # Dimension check
    if jl_n_token != py_n_token
        println("  FAIL: N_token mismatch! Julia=$jl_n_token Python=$py_n_token")
        return :fail
    end
    if jl_n_atom != py_n_atom
        println("  FAIL: N_atom mismatch! Julia=$jl_n_atom Python=$py_n_atom")
        return :fail
    end

    # Compare features
    py_keys = Set(String.(keys(py_feat)))
    jl_keys = Set(String.(keys(jl_feat)))

    only_py = setdiff(py_keys, jl_keys)
    only_jl = setdiff(jl_keys, py_keys)
    common = intersect(py_keys, jl_keys)

    if !isempty(only_py)
        # Filter out optional features
        required_only_py = setdiff(only_py, OPTIONAL_FEATURES)
        if !isempty(required_only_py)
            println("  Keys only in Python (required): $required_only_py")
        end
        optional_only_py = intersect(only_py, OPTIONAL_FEATURES)
        if !isempty(optional_only_py)
            println("  Keys only in Python (optional): $optional_only_py")
        end
    end
    if !isempty(only_jl)
        println("  Keys only in Julia: $only_jl")
    end

    n_match = 0
    n_fail = 0
    failures = String[]

    for key in sort(collect(common))
        jl_val = jl_feat[key]
        py_entry = py_feat[key]
        result = compare_feature(key, jl_val, py_entry)
        if result.match
            n_match += 1
        else
            n_fail += 1
            push!(failures, key)
            println("    MISMATCH $key: $(result.reason)")
        end
    end

    # Classify missing keys
    missing_non_model = intersect(only_py, NON_MODEL_FEATURES)
    missing_optional = intersect(only_py, OPTIONAL_FEATURES)
    missing_critical = setdiff(only_py, union(NON_MODEL_FEATURES, OPTIONAL_FEATURES))

    n_missing_critical = length(missing_critical)
    n_fail += n_missing_critical

    total_common = length(common)
    println("  Common features: $n_match/$total_common match, $(total_common - n_match) mismatch")
    if !isempty(missing_non_model)
        println("  Non-model Python features (ok to skip): $(length(missing_non_model))")
    end
    if !isempty(missing_critical)
        println("  CRITICAL missing features: $missing_critical")
    end
    if !isempty(failures)
        println("  Failed features: $(join(failures, ", "))")
    end

    return n_fail == 0 ? :pass : :fail
end

function main()
    cases = get_test_cases()
    println("Found $(length(cases)) PXDesign test cases:")
    for (_, name) in cases
        println("  $name")
    end

    results = Dict{String, Symbol}()
    n_pass = 0
    n_fail = 0
    n_skip = 0

    for (path, name) in cases
        try
            status = run_parity_test(path, name)
            results[name] = status
            if status == :pass
                n_pass += 1
            elseif status == :fail
                n_fail += 1
            else
                n_skip += 1
            end
        catch e
            println("  EXCEPTION: $e")
            for (exc, bt) in Base.catch_stack()
                showerror(stdout, exc, bt)
                println()
            end
            results[name] = :fail
            n_fail += 1
        end
    end

    println("\n" * "="^60)
    println("PXDesign Input Feature Parity Summary")
    println("="^60)
    println("  Pass: $n_pass")
    println("  Fail: $n_fail")
    println("  Skip: $n_skip")
    println("  Total: $(n_pass + n_fail + n_skip)")

    if n_fail > 0
        println("\nFailures:")
        for (name, status) in sort(collect(results); by=first)
            status == :fail && println("  $name")
        end
    end
    println("="^60)
end

main()
