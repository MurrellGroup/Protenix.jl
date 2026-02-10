module CLI

using Printf

import ..Config: default_config, set_by_key!
import ..Infer: run_infer
import ..Inputs: parse_yaml_to_json
import ..Model: compare_raw_weight_dirs

export main

function _usage(io::IO = stdout)
    println(io, "PXDesign.jl")
    println(io, "")
    println(io, "Usage:")
    println(io, "  pxdesign infer -i <input.json|yaml|yml> -o <dump_dir> [options]")
    println(io, "  pxdesign check-input --yaml <input.yaml|input.yml>")
    println(io, "  pxdesign parity-check <reference_raw_dir> <actual_raw_dir> [options]")
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
    end

    error("Unknown command: $cmd")
end

end
