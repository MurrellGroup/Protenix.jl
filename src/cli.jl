module CLI

using Printf

import ..Config: default_config, set_by_key!
import ..Infer: run_infer
import ..Inputs: parse_yaml_to_json
import ..JSONLite: parse_json
import ..Model: compare_raw_weight_dirs
import ..ProtenixAPI:
    add_precomputed_msa_to_json,
    convert_structure_to_infer_json,
    list_supported_models,
    predict_json,
    predict_sequence

export main

function _usage(io::IO = stdout)
    println(io, "PXDesign.jl")
    println(io, "")
    println(io, "Usage:")
    println(io, "  pxdesign infer -i <input.json|yaml|yml> -o <dump_dir> [options]")
    println(io, "  pxdesign check-input --yaml <input.yaml|input.yml>")
    println(io, "  pxdesign parity-check <reference_raw_dir> <actual_raw_dir> [options]")
    println(io, "  pxdesign predict --input <json|json_dir> [options]")
    println(io, "  pxdesign predict --sequence <AA_SEQUENCE> [options]")
    println(io, "  pxdesign tojson --input <pdb|cif|dir> [options]")
    println(io, "  pxdesign msa --input <json> --precomputed_msa_dir <dir> [options]")
    println(io, "")
    println(io, "Infer options:")
    println(io, "  --model_name <name>")
    println(io, "  --load_checkpoint_dir <dir>")
    println(io, "  --dtype <bf16|fp32>")
    println(io, "  --N_sample <int>")
    println(io, "  --N_step <int>")
    println(io, "  --eta_type <name>")
    println(io, "  --eta_min <float>")
    println(io, "  --eta_max <float>")
    println(io, "  --seed <int>                    (repeatable)")
    println(io, "  --use_fast_ln <true|false>")
    println(io, "  --use_deepspeed_evo_attention <true|false>")
    println(io, "  --include_protenix_checkpoints")
    println(io, "  --dry_run")
    println(io, "  --set key=value                 (repeatable)")
    println(io, "")
    println(io, "Parity-check options:")
    println(io, "  --atol <float>                  (default: 1e-5)")
    println(io, "  --rtol <float>                  (default: 1e-4)")
    println(io, "  --top <int>                     (default: 20)")
    println(io, "")
    println(io, "Predict options:")
    println(io, "  --input <json|json_dir>")
    println(io, "  --sequence <AA_SEQUENCE>")
    println(io, "  --out_dir <dir>                 (default: ./output)")
    println(io, "  --model_name <name>             (default: protenix_base_default_v0.5.0)")
    println(io, "  --weights_path <dir>")
    println(io, "  --seeds <i,j,k>                 (default: 101)")
    println(io, "  --cycle <int>")
    println(io, "  --step <int>")
    println(io, "  --sample <int>")
    println(io, "  --use_msa <true|false>")
    println(io, "  --use_default_params <true|false>  (default: true)")
    println(io, "  --strict <true|false>           (default: true)")
    println(io, "  --list-models                   (print supported model variants)")
    println(io, "  --task_name <name>              (sequence mode only)")
    println(io, "  --chain_id <id>                 (sequence mode only)")
    println(io, "  --esm_token_embedding_json <file> (sequence mode only; JSON matrix [N_token,D])")
    println(io, "")
    println(io, "Tojson options:")
    println(io, "  --input <pdb|cif|dir>")
    println(io, "  --out_dir <dir>                 (default: ./output)")
    println(io, "  --altloc <first>                (default: first)")
    println(io, "  --assembly_id <id>              (currently unsupported)")
    println(io, "")
    println(io, "MSA options:")
    println(io, "  --input <json>")
    println(io, "  --precomputed_msa_dir <dir>")
    println(io, "  --pairing_db <name>             (default: uniref100)")
    println(io, "  --out_dir <dir>                 (default: ./output)")
end

function _parse_bool(s::AbstractString)
    t = lowercase(strip(s))
    if t in ("true", "1", "yes", "y", "on")
        return true
    elseif t in ("false", "0", "no", "n", "off")
        return false
    end
    error("Invalid boolean value: $s")
end

function _require_value(args::Vector{String}, i::Int, opt::String)
    i >= length(args) && error("Missing value for $opt")
    return args[i + 1], i + 1
end

function _parse_infer_args(args::Vector{String})
    parsed = Dict{String, Any}(
        "model_name" => "pxdesign_v0.1.0",
        "dtype" => "bf16",
        "N_sample" => 5,
        "N_step" => 400,
        "eta_type" => "const",
        "eta_min" => 2.5,
        "eta_max" => 2.5,
        "seed" => Int[],
        "use_fast_ln" => true,
        "use_deepspeed_evo_attention" => false,
        "include_protenix_checkpoints" => false,
        "dry_run" => false,
        "set" => String[],
    )

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "-i" || arg == "--input"
            value, i = _require_value(args, i, arg)
            parsed["input"] = value
        elseif arg == "-o" || arg == "--dump_dir"
            value, i = _require_value(args, i, arg)
            parsed["dump_dir"] = value
        elseif arg == "--model_name"
            value, i = _require_value(args, i, arg)
            parsed["model_name"] = value
        elseif arg == "--load_checkpoint_dir"
            value, i = _require_value(args, i, arg)
            parsed["load_checkpoint_dir"] = value
        elseif arg == "--dtype"
            value, i = _require_value(args, i, arg)
            parsed["dtype"] = value
        elseif arg == "--N_sample"
            value, i = _require_value(args, i, arg)
            parsed["N_sample"] = parse(Int, value)
        elseif arg == "--N_step"
            value, i = _require_value(args, i, arg)
            parsed["N_step"] = parse(Int, value)
        elseif arg == "--eta_type"
            value, i = _require_value(args, i, arg)
            parsed["eta_type"] = value
        elseif arg == "--eta_min"
            value, i = _require_value(args, i, arg)
            parsed["eta_min"] = parse(Float64, value)
        elseif arg == "--eta_max"
            value, i = _require_value(args, i, arg)
            parsed["eta_max"] = parse(Float64, value)
        elseif arg == "--seed"
            value, i = _require_value(args, i, arg)
            push!(parsed["seed"], parse(Int, value))
        elseif arg == "--use_fast_ln"
            value, i = _require_value(args, i, arg)
            parsed["use_fast_ln"] = _parse_bool(value)
        elseif arg == "--use_deepspeed_evo_attention"
            value, i = _require_value(args, i, arg)
            parsed["use_deepspeed_evo_attention"] = _parse_bool(value)
        elseif arg == "--include_protenix_checkpoints"
            parsed["include_protenix_checkpoints"] = true
        elseif arg == "--dry_run"
            parsed["dry_run"] = true
        elseif arg == "--set"
            value, i = _require_value(args, i, arg)
            push!(parsed["set"], value)
        elseif arg == "--help" || arg == "-h"
            _usage()
            return Dict{String, Any}("__help__" => true)
        else
            error("Unknown option: $arg")
        end
        i += 1
    end

    !haskey(parsed, "input") && error("Missing required option: --input/-i")
    !haskey(parsed, "dump_dir") && error("Missing required option: --dump_dir/-o")
    return parsed
end

function _apply_set_overrides!(cfg::Dict{String, Any}, pairs::Vector{String})
    for kv in pairs
        if !occursin('=', kv)
            error("--set must be key=value, got: $kv")
        end
        k, v = split(kv, '='; limit = 2)
        set_by_key!(cfg, strip(k), strip(v))
    end
end

function _run_infer(parsed::Dict{String, Any})
    project_root = abspath(joinpath(@__DIR__, ".."))
    cfg = default_config(; project_root = project_root)

    set_by_key!(cfg, "input_json_path", parsed["input"])
    set_by_key!(cfg, "dump_dir", parsed["dump_dir"])
    set_by_key!(cfg, "model_name", parsed["model_name"])
    set_by_key!(cfg, "dtype", parsed["dtype"])
    set_by_key!(cfg, "N_sample", parsed["N_sample"])
    set_by_key!(cfg, "N_step", parsed["N_step"])
    set_by_key!(cfg, "eta_type", parsed["eta_type"])
    set_by_key!(cfg, "eta_min", parsed["eta_min"])
    set_by_key!(cfg, "eta_max", parsed["eta_max"])
    set_by_key!(cfg, "use_fast_ln", parsed["use_fast_ln"])
    set_by_key!(cfg, "use_deepspeed_evo_attention", parsed["use_deepspeed_evo_attention"])
    set_by_key!(cfg, "include_protenix_checkpoints", parsed["include_protenix_checkpoints"])

    if haskey(parsed, "load_checkpoint_dir")
        set_by_key!(cfg, "load_checkpoint_dir", parsed["load_checkpoint_dir"])
    end

    if !isempty(parsed["seed"])
        cfg["seeds"] = Int.(parsed["seed"])
    end

    _apply_set_overrides!(cfg, parsed["set"])
    return run_infer(cfg; dry_run = parsed["dry_run"])
end

function _parse_parity_args(args::Vector{String})
    length(args) >= 2 || error("Usage: pxdesign parity-check <reference_raw_dir> <actual_raw_dir> [--atol <float>] [--rtol <float>] [--top <int>]")
    parsed = Dict{String, Any}(
        "reference_raw_dir" => args[1],
        "actual_raw_dir" => args[2],
        "atol" => 1f-5,
        "rtol" => 1f-4,
        "top" => 20,
    )
    i = 3
    while i <= length(args)
        arg = args[i]
        if arg == "--atol"
            value, i = _require_value(args, i, arg)
            parsed["atol"] = Float32(parse(Float64, value))
        elseif arg == "--rtol"
            value, i = _require_value(args, i, arg)
            parsed["rtol"] = Float32(parse(Float64, value))
        elseif arg == "--top"
            value, i = _require_value(args, i, arg)
            parsed["top"] = parse(Int, value)
        elseif arg == "--help" || arg == "-h"
            _usage()
            return Dict{String, Any}("__help__" => true)
        else
            error("Unknown option: $arg")
        end
        i += 1
    end
    parsed["top"] > 0 || error("--top must be positive.")
    return parsed
end

function _parse_seeds_csv(raw::AbstractString)
    parts = split(strip(raw), ',')
    out = Int[]
    for p in parts
        s = strip(p)
        isempty(s) && continue
        push!(out, parse(Int, s))
    end
    isempty(out) && error("Expected at least one seed in --seeds")
    return out
end

function _parse_predict_args(args::Vector{String})
    parsed = Dict{String, Any}(
        "out_dir" => "./output",
        "model_name" => "protenix_base_default_v0.5.0",
        "weights_path" => "",
        "seeds" => [101],
        "use_default_params" => true,
        "strict" => true,
        "task_name" => "protenix_sequence",
        "chain_id" => "A0",
        "list_models" => false,
    )

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--input"
            value, i = _require_value(args, i, arg)
            parsed["input"] = value
        elseif arg == "--sequence"
            value, i = _require_value(args, i, arg)
            parsed["sequence"] = value
        elseif arg == "--out_dir"
            value, i = _require_value(args, i, arg)
            parsed["out_dir"] = value
        elseif arg == "--model_name"
            value, i = _require_value(args, i, arg)
            parsed["model_name"] = value
        elseif arg == "--weights_path"
            value, i = _require_value(args, i, arg)
            parsed["weights_path"] = value
        elseif arg == "--seeds"
            value, i = _require_value(args, i, arg)
            parsed["seeds"] = _parse_seeds_csv(value)
        elseif arg == "--cycle"
            value, i = _require_value(args, i, arg)
            parsed["cycle"] = parse(Int, value)
        elseif arg == "--step"
            value, i = _require_value(args, i, arg)
            parsed["step"] = parse(Int, value)
        elseif arg == "--sample"
            value, i = _require_value(args, i, arg)
            parsed["sample"] = parse(Int, value)
        elseif arg == "--use_msa"
            value, i = _require_value(args, i, arg)
            parsed["use_msa"] = _parse_bool(value)
        elseif arg == "--use_default_params"
            value, i = _require_value(args, i, arg)
            parsed["use_default_params"] = _parse_bool(value)
        elseif arg == "--strict"
            value, i = _require_value(args, i, arg)
            parsed["strict"] = _parse_bool(value)
        elseif arg == "--list-models"
            parsed["list_models"] = true
        elseif arg == "--task_name"
            value, i = _require_value(args, i, arg)
            parsed["task_name"] = value
        elseif arg == "--chain_id"
            value, i = _require_value(args, i, arg)
            parsed["chain_id"] = value
        elseif arg == "--esm_token_embedding_json"
            value, i = _require_value(args, i, arg)
            parsed["esm_token_embedding_json"] = value
        elseif arg == "--help" || arg == "-h"
            _usage()
            return Dict{String, Any}("__help__" => true)
        else
            error("Unknown option: $arg")
        end
        i += 1
    end

    has_input = haskey(parsed, "input")
    has_seq = haskey(parsed, "sequence")
    if parsed["list_models"]
        !(has_input || has_seq) || error("--list-models cannot be combined with --input/--sequence")
    else
        (has_input || has_seq) || error("predict requires either --input or --sequence")
        !(has_input && has_seq) || error("predict accepts only one of --input or --sequence")
    end
    return parsed
end

function _load_embedding_json_matrix(path::AbstractString)
    abspath_path = abspath(path)
    isfile(abspath_path) || error("Embedding JSON file not found: $abspath_path")
    value = parse_json(read(abspath_path, String))
    value isa AbstractVector || error("Embedding JSON must be a rank-2 array [N_token,D].")
    n_row = length(value)
    n_row > 0 || error("Embedding JSON matrix must be non-empty.")
    first_row = value[1]
    first_row isa AbstractVector || error("Embedding JSON must be a rank-2 array [N_token,D].")
    n_col = length(first_row)
    n_col > 0 || error("Embedding JSON matrix must have at least one feature column.")
    out = Matrix{Float32}(undef, n_row, n_col)
    for i in 1:n_row
        row = value[i]
        row isa AbstractVector || error("Embedding JSON row $i is not an array.")
        length(row) == n_col || error("Embedding JSON rows must be rectangular (row $i length mismatch).")
        for j in 1:n_col
            out[i, j] = Float32(row[j])
        end
    end
    return out
end

function _run_predict(parsed::Dict{String, Any})
    if parsed["list_models"]
        println("supported_models:")
        for spec in list_supported_models()
            println(
                "  - $(spec.model_name) family=$(spec.family) cycle=$(spec.default_cycle) step=$(spec.default_step) sample=$(spec.default_sample) use_msa=$(spec.default_use_msa) needs_esm_embedding=$(spec.needs_esm_embedding)",
            )
        end
        return
    end

    common_kwargs = (
        out_dir = parsed["out_dir"],
        model_name = parsed["model_name"],
        weights_path = parsed["weights_path"],
        seeds = parsed["seeds"],
        use_default_params = parsed["use_default_params"],
        cycle = get(parsed, "cycle", nothing),
        step = get(parsed, "step", nothing),
        sample = get(parsed, "sample", nothing),
        use_msa = get(parsed, "use_msa", nothing),
        strict = parsed["strict"],
    )

    if haskey(parsed, "sequence")
        emb = haskey(parsed, "esm_token_embedding_json") ?
              _load_embedding_json_matrix(String(parsed["esm_token_embedding_json"])) :
              nothing
        records = predict_sequence(
            parsed["sequence"];
            out_dir = common_kwargs.out_dir,
            model_name = common_kwargs.model_name,
            weights_path = common_kwargs.weights_path,
            task_name = parsed["task_name"],
            chain_id = parsed["chain_id"],
            seeds = common_kwargs.seeds,
            use_default_params = common_kwargs.use_default_params,
            cycle = common_kwargs.cycle,
            step = common_kwargs.step,
            sample = common_kwargs.sample,
            use_msa = common_kwargs.use_msa,
            esm_token_embedding = emb,
            strict = common_kwargs.strict,
        )
        for r in records
            println("task_name=$(r.task_name) seed=$(r.seed)")
            println("prediction_dir=$(r.prediction_dir)")
            for p in r.cif_paths
                println("cif_path=$(p)")
            end
        end
        return
    end

    records = predict_json(
        parsed["input"];
        out_dir = common_kwargs.out_dir,
        model_name = common_kwargs.model_name,
        weights_path = common_kwargs.weights_path,
        seeds = common_kwargs.seeds,
        use_default_params = common_kwargs.use_default_params,
        cycle = common_kwargs.cycle,
        step = common_kwargs.step,
        sample = common_kwargs.sample,
        use_msa = common_kwargs.use_msa,
        strict = common_kwargs.strict,
    )
    for r in records
        println("input_json=$(r.input_json) task_name=$(r.task_name) seed=$(r.seed)")
        println("prediction_dir=$(r.prediction_dir)")
        for p in r.cif_paths
            println("cif_path=$(p)")
        end
    end
end

function _parse_tojson_args(args::Vector{String})
    parsed = Dict{String, Any}(
        "out_dir" => "./output",
        "altloc" => "first",
    )
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--input"
            value, i = _require_value(args, i, arg)
            parsed["input"] = value
        elseif arg == "--out_dir"
            value, i = _require_value(args, i, arg)
            parsed["out_dir"] = value
        elseif arg == "--altloc"
            value, i = _require_value(args, i, arg)
            parsed["altloc"] = value
        elseif arg == "--assembly_id"
            value, i = _require_value(args, i, arg)
            parsed["assembly_id"] = value
        elseif arg == "--help" || arg == "-h"
            _usage()
            return Dict{String, Any}("__help__" => true)
        else
            error("Unknown option: $arg")
        end
        i += 1
    end
    haskey(parsed, "input") || error("tojson requires --input")
    return parsed
end

function _run_tojson(parsed::Dict{String, Any})
    out_paths = convert_structure_to_infer_json(
        parsed["input"];
        out_dir = parsed["out_dir"],
        altloc = parsed["altloc"],
        assembly_id = get(parsed, "assembly_id", nothing),
    )
    for p in out_paths
        println("json_path=$(p)")
    end
end

function _parse_msa_args(args::Vector{String})
    parsed = Dict{String, Any}(
        "out_dir" => "./output",
        "pairing_db" => "uniref100",
    )
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--input"
            value, i = _require_value(args, i, arg)
            parsed["input"] = value
        elseif arg == "--precomputed_msa_dir"
            value, i = _require_value(args, i, arg)
            parsed["precomputed_msa_dir"] = value
        elseif arg == "--pairing_db"
            value, i = _require_value(args, i, arg)
            parsed["pairing_db"] = value
        elseif arg == "--out_dir"
            value, i = _require_value(args, i, arg)
            parsed["out_dir"] = value
        elseif arg == "--help" || arg == "-h"
            _usage()
            return Dict{String, Any}("__help__" => true)
        else
            error("Unknown option: $arg")
        end
        i += 1
    end
    haskey(parsed, "input") || error("msa requires --input")
    haskey(parsed, "precomputed_msa_dir") || error("msa requires --precomputed_msa_dir")
    return parsed
end

function _run_msa(parsed::Dict{String, Any})
    input_path = String(parsed["input"])
    ext = lowercase(splitext(input_path)[2])
    if ext == ".json"
        out_path = add_precomputed_msa_to_json(
            input_path;
            out_dir = parsed["out_dir"],
            precomputed_msa_dir = parsed["precomputed_msa_dir"],
            pairing_db = parsed["pairing_db"],
        )
        println("updated_json=$(out_path)")
        return
    elseif ext in (".fasta", ".fa")
        error(
            "FASTA MSA search is not yet implemented in Julia. " *
            "Provide --input <json> with --precomputed_msa_dir to attach existing MSA paths.",
        )
    end
    error("msa currently supports JSON input only. Got: $input_path")
end

function _run_parity_check(parsed::Dict{String, Any}; io::IO = stdout)
    report = compare_raw_weight_dirs(
        parsed["reference_raw_dir"],
        parsed["actual_raw_dir"];
        atol = parsed["atol"],
        rtol = parsed["rtol"],
    )
    println(io, "Parity report:")
    println(io, "  compared: $(length(report.compared))")
    println(io, "  failed: $(length(report.failed))")
    println(io, "  missing_in_actual: $(length(report.missing_in_actual))")
    println(io, "  missing_in_reference: $(length(report.missing_in_reference))")
    println(io, @sprintf("  tolerances: atol=%.3e rtol=%.3e", report.atol, report.rtol))

    top = parsed["top"]
    if !isempty(report.failed)
        nshow = min(length(report.failed), top)
        println(io, "Top failed tensors by max_abs:")
        ranked = sort(report.failed; by = x -> x.max_abs, rev = true)
        for s in ranked[1:nshow]
            println(
                io,
                @sprintf(
                    "  - %s | max_abs=%.6e max_rel=%.6e mean_abs=%.6e numel=%d",
                    s.key,
                    s.max_abs,
                    s.max_rel,
                    s.mean_abs,
                    s.numel,
                ),
            )
        end
    end

    if !isempty(report.missing_in_actual)
        nshow = min(length(report.missing_in_actual), top)
        println(io, "Missing in actual:")
        for k in report.missing_in_actual[1:nshow]
            println(io, "  - $k")
        end
    end
    if !isempty(report.missing_in_reference)
        nshow = min(length(report.missing_in_reference), top)
        println(io, "Missing in reference:")
        for k in report.missing_in_reference[1:nshow]
            println(io, "  - $k")
        end
    end

    ok = isempty(report.failed) && isempty(report.missing_in_actual) && isempty(report.missing_in_reference)
    return ok ? 0 : 2
end

function main(argv::Vector{String} = copy(ARGS))
    if isempty(argv)
        _usage()
        return 0
    end

    cmd = argv[1]
    if cmd == "--help" || cmd == "-h"
        _usage()
        return 0
    elseif cmd == "infer"
        parsed = _parse_infer_args(argv[2:end])
        if haskey(parsed, "__help__")
            return 0
        end
        _run_infer(parsed)
        return 0
    elseif cmd == "check-input"
        args = argv[2:end]
        length(args) == 2 || error("Usage: pxdesign check-input --yaml <input.yaml>")
        (args[1] == "--yaml" || args[1] == "-y") || error("Usage: pxdesign check-input --yaml <input.yaml>")
        parse_yaml_to_json(args[2], nothing)
        println("YAML input is valid.")
        return 0
    elseif cmd == "parity-check"
        parsed = _parse_parity_args(argv[2:end])
        haskey(parsed, "__help__") && return 0
        return _run_parity_check(parsed)
    elseif cmd == "predict"
        parsed = _parse_predict_args(argv[2:end])
        haskey(parsed, "__help__") && return 0
        _run_predict(parsed)
        return 0
    elseif cmd == "tojson"
        parsed = _parse_tojson_args(argv[2:end])
        haskey(parsed, "__help__") && return 0
        _run_tojson(parsed)
        return 0
    elseif cmd == "msa"
        parsed = _parse_msa_args(argv[2:end])
        haskey(parsed, "__help__") && return 0
        _run_msa(parsed)
        return 0
    end

    error("Unknown command: $cmd")
end

end
