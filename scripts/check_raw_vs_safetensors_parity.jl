#!/usr/bin/env julia

using PXDesign
using TOML

function _usage()
    println(
        "Usage: check_raw_vs_safetensors_parity.jl <raw_dir> <safetensors_path_or_dir> " *
        "[--atol <float>] [--rtol <float>] [--report <path>]",
    )
end

function _parse_args(args::Vector{String})
    length(args) >= 2 || error("Missing required arguments.")
    raw_dir = args[1]
    safetensors_path = args[2]
    atol = 0.0
    rtol = 0.0
    report_path = nothing

    i = 3
    while i <= length(args)
        flag = args[i]
        if flag == "--atol"
            i < length(args) || error("--atol requires a value")
            i += 1
            atol = parse(Float64, args[i])
        elseif flag == "--rtol"
            i < length(args) || error("--rtol requires a value")
            i += 1
            rtol = parse(Float64, args[i])
        elseif flag == "--report"
            i < length(args) || error("--report requires a value")
            i += 1
            report_path = args[i]
        else
            error("Unknown flag: $flag")
        end
        i += 1
    end

    return (
        raw_dir = raw_dir,
        safetensors_path = safetensors_path,
        atol = atol,
        rtol = rtol,
        report_path = report_path,
    )
end

function main(args::Vector{String})
    opts = try
        _parse_args(args)
    catch err
        _usage()
        rethrow(err)
    end

    raw = PXDesign.Model.load_raw_weights(opts.raw_dir)
    safetensors = PXDesign.Model.load_safetensors_weights(opts.safetensors_path)
    report = PXDesign.Model.tensor_parity_report(raw, safetensors; atol = opts.atol, rtol = opts.rtol)

    summary = Dict{String, Any}(
        "raw_dir" => abspath(opts.raw_dir),
        "safetensors_path" => abspath(opts.safetensors_path),
        "atol" => Float64(report.atol),
        "rtol" => Float64(report.rtol),
        "num_compared" => length(report.compared),
        "num_failed" => length(report.failed),
        "missing_in_safetensors" => length(report.missing_in_actual),
        "missing_in_raw" => length(report.missing_in_reference),
    )

    println("compared=$(summary["num_compared"])")
    println("failed=$(summary["num_failed"])")
    println("missing_in_safetensors=$(summary["missing_in_safetensors"])")
    println("missing_in_raw=$(summary["missing_in_raw"])")

    if opts.report_path !== nothing
        report_dir = dirname(opts.report_path)
        isempty(report_dir) || mkpath(report_dir)
        open(opts.report_path, "w") do io
            TOML.print(io, summary)
        end
        println("report=$(abspath(opts.report_path))")
    end

    if !isempty(report.failed)
        first_fail = report.failed[1]
        println(
            "first_failed_key=$(first_fail.key) max_abs=$(first_fail.max_abs) max_rel=$(first_fail.max_rel)",
        )
        return 1
    end
    if !isempty(report.missing_in_actual) || !isempty(report.missing_in_reference)
        return 1
    end
    return 0
end

exit(main(ARGS))
