#!/usr/bin/env julia

using PXDesign
using Random
using Statistics
using LinearAlgebra

const PA = PXDesign.ProtenixAPI

function _usage()
    println(
        "Usage: julia --project=. scripts/compare_python_input_tensors.jl " *
        "--input-json <path> --python-dump-dir <dir> --model-name <name> [--seed <int>] " *
        "[--use-default-params <true|false>] [--use-msa <auto|true|false>] " *
        "[--atol <float>] [--rtol <float>] [--report <path>]",
    )
end

function _parse_bool(s::AbstractString)
    t = lowercase(strip(String(s)))
    if t in ("1", "true", "yes", "y")
        return true
    elseif t in ("0", "false", "no", "n")
        return false
    end
    error("Invalid bool value: '$s'")
end

function _parse_args(argv::Vector{String})
    opts = Dict{String, String}(
        "seed" => "101",
        "use-default-params" => "true",
        "use-msa" => "auto",
        "force-needs-esm" => "auto",
        "inject-python-esm" => "false",
        "ref-pos-augment" => "true",
        "allow-ref-pos-rigid-equiv" => "false",
        "strict-keyset" => "true",
        "atol" => "1e-5",
        "rtol" => "1e-5",
    )
    i = 1
    while i <= length(argv)
        arg = argv[i]
        startswith(arg, "--") || error("Unexpected argument '$arg'")
        key = arg[3:end]
        i == length(argv) && error("Missing value for --$key")
        val = argv[i + 1]
        opts[key] = val
        i += 2
    end

    for req in ("input-json", "python-dump-dir", "model-name")
        haskey(opts, req) || error("Missing required argument --$req")
    end
    return (
        input_json = String(opts["input-json"]),
        python_dump_dir = String(opts["python-dump-dir"]),
        model_name = String(opts["model-name"]),
        seed = parse(Int, opts["seed"]),
        use_default_params = _parse_bool(opts["use-default-params"]),
        use_msa = String(opts["use-msa"]),
        force_needs_esm = String(opts["force-needs-esm"]),
        inject_python_esm = _parse_bool(opts["inject-python-esm"]),
        ref_pos_augment = _parse_bool(opts["ref-pos-augment"]),
        allow_ref_pos_rigid_equiv = _parse_bool(opts["allow-ref-pos-rigid-equiv"]),
        strict_keyset = _parse_bool(opts["strict-keyset"]),
        atol = Float32(parse(Float64, opts["atol"])),
        rtol = Float32(parse(Float64, opts["rtol"])),
        report = get(opts, "report", ""),
    )
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

function _to_array_bool(x)
    x isa AbstractVector || return Bool(x)
    dims = _nested_dims(x)
    out = Array{Bool}(undef, Tuple(dims)...)
    idx = Int[]
    function walk(y)
        if y isa AbstractVector
            for (i, z) in enumerate(y)
                push!(idx, i)
                walk(z)
                pop!(idx)
            end
        else
            out[Tuple(idx)...] = Bool(y)
        end
    end
    walk(x)
    return out
end

function _py_entry_to_array(entry::AbstractDict{<:AbstractString, <:Any})
    dtype = lowercase(String(entry["dtype"]))
    data = entry["data"]
    if occursin("bool", dtype)
        return _to_array_bool(data), dtype
    elseif occursin("int", dtype) || occursin("long", dtype)
        return _to_array_i(data), dtype
    end
    return _to_array_f32(data), dtype
end

function _find_python_dump(dump_dir::AbstractString, model_name::AbstractString, seed::Int)
    files = readdir(dump_dir; join = true)
    suffix = "__seed$(seed).json"
    candidates = filter(files) do p
        b = basename(p)
        startswith(b, model_name * "__") && endswith(b, suffix)
    end
    isempty(candidates) && error("No python dump found for model=$model_name seed=$seed in $dump_dir")
    length(candidates) == 1 || error("Expected one python dump for model=$model_name seed=$seed, found $(length(candidates))")
    return only(candidates)
end

function _predict_task_name(task::Dict{String, Any}, input_json::AbstractString, task_idx::Int)
    if haskey(task, "name")
        return String(task["name"])
    end
    return "$(splitext(basename(input_json))[1])_$(task_idx - 1)"
end

function _resolve_use_msa_arg(use_msa_arg::AbstractString)
    t = lowercase(strip(String(use_msa_arg)))
    if t == "auto"
        return nothing
    elseif t in ("true", "1", "yes", "y")
        return true
    elseif t in ("false", "0", "no", "n")
        return false
    end
    error("Invalid --use-msa value '$use_msa_arg'; expected auto|true|false")
end

function _resolve_force_needs_esm_arg(s::AbstractString)
    t = lowercase(strip(String(s)))
    if t == "auto"
        return nothing
    elseif t in ("true", "1", "yes", "y")
        return true
    elseif t in ("false", "0", "no", "n")
        return false
    end
    error("Invalid --force-needs-esm value '$s'; expected auto|true|false")
end

function _build_julia_feature_dict(
    input_json::AbstractString,
    model_name::AbstractString,
    seed::Int;
    use_default_params::Bool,
    use_msa_override::Union{Nothing, Bool},
    force_needs_esm::Union{Nothing, Bool},
    python_esm_embedding::Union{Nothing, AbstractArray{<:Real}},
    ref_pos_augment::Bool,
)
    params = PA.recommended_params(
        model_name;
        use_default_params = use_default_params,
        use_msa = use_msa_override,
    )
    if force_needs_esm !== nothing
        params = (; params..., needs_esm_embedding = force_needs_esm)
    end
    tasks = PA._ensure_json_tasks(input_json)
    isempty(tasks) && error("No tasks in input JSON: $input_json")

    task_any = tasks[1]
    task_any isa AbstractDict || error("Task 0 is not an object in $input_json")
    task = PA._as_string_dict(task_any)
    task_name = _predict_task_name(task, input_json, 1)
    parsed_task = PA._parse_task_entities(task; json_dir = dirname(abspath(input_json)))
    chain_sequences = PA._protein_chain_sequence_map(parsed_task.protein_specs)

    rng = MersenneTwister(seed)
    atoms = PA._remove_covalent_leaving_atoms(
        parsed_task.atoms, task, parsed_task.entity_chain_ids, parsed_task.entity_atom_map;
        rng = rng,
    )
    atoms = PA._apply_mse_to_met(atoms)
    atoms = PA._apply_ccd_mol_type_override(atoms, parsed_task.polymer_chain_ids)
    bundle = PXDesign.Data.build_feature_bundle_from_atoms(
        atoms;
        task_name = task_name,
        rng = rng,
        ref_pos_augment = ref_pos_augment,
    )
    token_chain_ids = [bundle["atoms"][tok.centre_atom_index].chain_id for tok in bundle["tokens"]]
    feat = bundle["input_feature_dict"]
    PA._normalize_protenix_feature_dict!(feat)
    PA._inject_task_msa_features!(
        feat,
        task,
        input_json;
        use_msa = params.use_msa,
        msa_pair_as_unpair = params.msa_pair_as_unpair,
        chain_specs = parsed_task.protein_specs,
        token_chain_ids = token_chain_ids,
    )
    PA._inject_task_covalent_token_bonds!(
        feat,
        bundle["atoms"],
        task,
        parsed_task.entity_chain_ids,
        parsed_task.entity_atom_map,
    )
    PA._inject_task_template_features!(feat, task)
    PA._inject_task_esm_token_embedding!(feat, task)
    if python_esm_embedding !== nothing
        size(python_esm_embedding, 1) == size(feat["restype"], 1) || error(
            "python esm_token_embedding length mismatch for $model_name: expected $(size(feat["restype"], 1)), got $(size(python_esm_embedding, 1))",
        )
        feat["esm_token_embedding"] = Float32.(python_esm_embedding)
    end
    PA._inject_auto_esm_token_embedding!(
        feat,
        bundle["atoms"],
        bundle["tokens"],
        chain_sequences,
        params,
        "task '$task_name' in $(basename(input_json))",
    )
    PA._inject_task_constraint_feature!(
        feat,
        task,
        bundle["atoms"],
        parsed_task.entity_chain_ids,
        parsed_task.entity_atom_map,
        "task '$task_name' in $(basename(input_json))",
    )
    PA._validate_required_model_inputs!(
        params,
        feat,
        "task '$task_name' in $(basename(input_json))",
    )
    return feat
end

function _as_array(x)
    x isa AbstractArray && return x
    return fill(x, ())
end

function _key_report_float(key::AbstractString, py_arr, jl_arr, atol::Float32, rtol::Float32)
    a = Float32.(py_arr)
    b = Float32.(jl_arr)
    d = abs.(a .- b)
    max_abs = isempty(d) ? 0f0 : maximum(d)
    mean_abs = isempty(d) ? 0f0 : Float32(mean(d))
    denom = max.(abs.(a), 1f-6)
    rel = d ./ denom
    max_rel = isempty(rel) ? 0f0 : maximum(rel)
    pass = all(d .<= (atol .+ rtol .* abs.(a)))
    return (
        key = String(key),
        kind = "float",
        pass = pass,
        max_abs = max_abs,
        mean_abs = mean_abs,
        max_rel = max_rel,
    )
end

function _key_report_int(key::AbstractString, py_arr, jl_arr)
    a = Int.(py_arr)
    b = Int.(jl_arr)
    mismatch = count(a .!= b)
    total = length(a)
    return (
        key = String(key),
        kind = "int_or_bool",
        pass = mismatch == 0,
        mismatch = mismatch,
        total = total,
    )
end

function _ref_pos_pairwise_report(py_ref_pos, jl_ref_pos, ref_space_uid, atol::Float32, rtol::Float32)
    p = Float32.(py_ref_pos)
    q = Float32.(jl_ref_pos)
    uid = Int.(ref_space_uid)
    size(p) == size(q) || return (pass = false, pairwise_max_abs = Inf32, pairwise_max_rel = Inf32)
    length(uid) == size(p, 1) || return (pass = false, pairwise_max_abs = Inf32, pairwise_max_rel = Inf32)

    max_abs = 0f0
    max_rel = 0f0
    for u in sort!(unique(uid))
        idx = findall(==(u), uid)
        n = length(idx)
        n <= 1 && continue
        for i in 1:(n - 1)
            ii = idx[i]
            pi = @view p[ii, :]
            qi = @view q[ii, :]
            for j in (i + 1):n
                jj = idx[j]
                dp = norm(pi .- @view(p[jj, :]))
                dq = norm(qi .- @view(q[jj, :]))
                d = abs(dp - dq)
                denom = max(abs(dp), 1f-6)
                r = d / denom
                max_abs = max(max_abs, d)
                max_rel = max(max_rel, r)
            end
        end
    end
    pass = max_abs <= (atol + rtol)
    return (pass = pass, pairwise_max_abs = max_abs, pairwise_max_rel = max_rel)
end

function main(argv::Vector{String})
    isempty(argv) && (_usage(); return 1)
    opts = _parse_args(argv)
    py_dump_path = _find_python_dump(opts.python_dump_dir, opts.model_name, opts.seed)
    raw = PXDesign.JSONLite.parse_json(read(py_dump_path, String))
    py_feat_raw = raw["input_feature_dict"]
    py_feat_raw isa AbstractDict || error("Python dump missing input_feature_dict map: $py_dump_path")

    use_msa_override = _resolve_use_msa_arg(opts.use_msa)
    force_needs_esm = _resolve_force_needs_esm_arg(opts.force_needs_esm)
    py_esm_emb = nothing
    if opts.inject_python_esm
        if haskey(py_feat_raw, "esm_token_embedding")
            py_emb_entry = py_feat_raw["esm_token_embedding"]
            py_emb_entry isa AbstractDict || error("python esm_token_embedding entry is not a tensor object.")
            py_emb_arr, _ = _py_entry_to_array(py_emb_entry)
            py_esm_emb = Float32.(py_emb_arr)
        else
            error("--inject-python-esm=true requested but python dump has no esm_token_embedding.")
        end
    end
    jl_feat = _build_julia_feature_dict(
        opts.input_json,
        opts.model_name,
        opts.seed;
        use_default_params = opts.use_default_params,
        use_msa_override = use_msa_override,
        force_needs_esm = force_needs_esm,
        python_esm_embedding = py_esm_emb,
        ref_pos_augment = opts.ref_pos_augment,
    )

    py_keys = Set(String.(collect(keys(py_feat_raw))))
    jl_keys = Set(String.(collect(keys(jl_feat))))
    shared = sort!(collect(intersect(py_keys, jl_keys)))
    only_py = sort!(collect(setdiff(py_keys, jl_keys)))
    only_jl = sort!(collect(setdiff(jl_keys, py_keys)))

    reports = NamedTuple[]
    py_ref_uid = nothing
    if haskey(py_feat_raw, "ref_space_uid")
        py_ref_uid_entry = py_feat_raw["ref_space_uid"]
        if py_ref_uid_entry isa AbstractDict
            py_ref_uid, _ = _py_entry_to_array(py_ref_uid_entry)
        end
    end
    for k in shared
        py_entry_any = py_feat_raw[k]
        py_entry_any isa AbstractDict || continue
        py_arr, py_dtype = _py_entry_to_array(py_entry_any)
        jl_arr = _as_array(jl_feat[k])
        size(py_arr) == size(jl_arr) || begin
            push!(
                reports,
                (
                    key = k,
                    kind = "shape_mismatch",
                    pass = false,
                    py_shape = collect(size(py_arr)),
                    jl_shape = collect(size(jl_arr)),
                ),
            )
            continue
        end

        if occursin("float", py_dtype) || occursin("half", py_dtype) || occursin("double", py_dtype) || occursin("bfloat", py_dtype)
            rep = _key_report_float(k, py_arr, jl_arr, opts.atol, opts.rtol)
            if opts.allow_ref_pos_rigid_equiv && k == "ref_pos" && !rep.pass && py_ref_uid !== nothing && haskey(jl_feat, "ref_space_uid")
                rigid = _ref_pos_pairwise_report(py_arr, jl_arr, py_ref_uid, opts.atol, opts.rtol)
                if rigid.pass
                    rep = (
                        key = rep.key,
                        kind = "float_rigid_equivalent",
                        pass = true,
                        max_abs = rep.max_abs,
                        mean_abs = rep.mean_abs,
                        max_rel = rep.max_rel,
                        pairwise_max_abs = rigid.pairwise_max_abs,
                        pairwise_max_rel = rigid.pairwise_max_rel,
                    )
                else
                    rep = (
                        key = rep.key,
                        kind = rep.kind,
                        pass = rep.pass,
                        max_abs = rep.max_abs,
                        mean_abs = rep.mean_abs,
                        max_rel = rep.max_rel,
                        pairwise_max_abs = rigid.pairwise_max_abs,
                        pairwise_max_rel = rigid.pairwise_max_rel,
                    )
                end
            end
            push!(reports, rep)
        else
            push!(reports, _key_report_int(k, py_arr, jl_arr))
        end
    end

    failed = filter(r -> !Bool(getfield(r, :pass)), reports)
    rigid_equiv = filter(r -> (:kind in fieldnames(typeof(r))) && (String(getfield(r, :kind)) == "float_rigid_equivalent"), reports)

    println("model_name=$(opts.model_name)")
    println("python_dump=$(py_dump_path)")
    println("keys_shared=$(length(shared))")
    println("keys_only_python=$(length(only_py))")
    println("keys_only_julia=$(length(only_jl))")
    println("keys_failed=$(length(failed))")
    println("keys_rigid_equivalent=$(length(rigid_equiv))")
    println("allow_ref_pos_rigid_equiv=$(opts.allow_ref_pos_rigid_equiv)")
    if !isempty(only_py)
        n = min(length(only_py), 20)
        println("only_python_keys=$(join(first(only_py, n), ','))")
    end
    if !isempty(only_jl)
        n = min(length(only_jl), 20)
        println("only_julia_keys=$(join(first(only_jl, n), ','))")
    end

    if !isempty(failed)
        println("top_failures:")
        for r in first(failed, min(length(failed), 20))
            if :max_abs in fieldnames(typeof(r))
                println(
                    "  $(r.key): kind=$(r.kind) max_abs=$(r.max_abs) max_rel=$(r.max_rel) mean_abs=$(r.mean_abs)",
                )
                if :pairwise_max_abs in fieldnames(typeof(r))
                    println("    pairwise_max_abs=$(r.pairwise_max_abs) pairwise_max_rel=$(r.pairwise_max_rel)")
                end
            elseif :mismatch in fieldnames(typeof(r))
                println("  $(r.key): kind=$(r.kind) mismatch=$(r.mismatch)/$(r.total)")
            else
                println("  $(r.key): kind=$(r.kind) py_shape=$(r.py_shape) jl_shape=$(r.jl_shape)")
            end
        end
    end
    if !isempty(rigid_equiv)
        n = min(length(rigid_equiv), 20)
        println("rigid_equivalent_keys=$(join([r.key for r in first(rigid_equiv, n)], ','))")
    end

    if !isempty(opts.report)
        rep = Dict{String, Any}()
        rep["model_name"] = opts.model_name
        rep["python_dump_path"] = py_dump_path
        rep["shared_keys"] = length(shared)
        rep["only_python_keys"] = only_py
        rep["only_julia_keys"] = only_jl
        rep["failed_keys"] = length(failed)
        rep["results"] = [Dict{String, Any}(string(k) => getfield(r, k) for k in fieldnames(typeof(r))) for r in reports]
        mkpath(dirname(abspath(opts.report)))
        PXDesign.JSONLite.write_json(opts.report, rep)
    end

    keyset_ok = !opts.strict_keyset || (isempty(only_py) && isempty(only_jl))
    return isempty(failed) && keyset_ok ? 0 : 2
end

exit(main(ARGS))
