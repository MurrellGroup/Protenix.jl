#!/usr/bin/env julia

using Printf
using PXDesign

function _usage()
    println(
        "Usage: julia --project=. scripts/compare_parity_raw.jl <reference_dir> <actual_dir> [--atol=<x>] [--rtol=<x>] [--top=<n>]",
    )
end

function _parse_args(args::Vector{String})
    length(args) >= 2 || error("expected <reference_dir> and <actual_dir>")
    ref_dir = args[1]
    act_dir = args[2]
    atol = 1f-5
    rtol = 1f-4
    top = 20
    for a in args[3:end]
        if startswith(a, "--atol=")
            atol = Float32(parse(Float64, split(a, "=", limit = 2)[2]))
        elseif startswith(a, "--rtol=")
            rtol = Float32(parse(Float64, split(a, "=", limit = 2)[2]))
        elseif startswith(a, "--top=")
            top = parse(Int, split(a, "=", limit = 2)[2])
        else
            error("unknown argument: $a")
        end
    end
    top > 0 || error("--top must be positive")
    return (ref_dir = ref_dir, act_dir = act_dir, atol = atol, rtol = rtol, top = top)
end

function main(args::Vector{String})
    if any(x -> x in ("-h", "--help"), args)
        _usage()
        return 0
    end

    opts = _parse_args(args)
    report = PXDesign.Model.compare_raw_weight_dirs(
        opts.ref_dir,
        opts.act_dir;
        atol = opts.atol,
        rtol = opts.rtol,
    )

    println("Parity report:")
    println("  compared: $(length(report.compared))")
    println("  failed: $(length(report.failed))")
    println("  missing_in_actual: $(length(report.missing_in_actual))")
    println("  missing_in_reference: $(length(report.missing_in_reference))")
    println(@sprintf("  tolerances: atol=%.3e rtol=%.3e", report.atol, report.rtol))

    if !isempty(report.missing_in_actual)
        nshow = min(length(report.missing_in_actual), opts.top)
        println("Missing in actual (first $nshow):")
        for k in report.missing_in_actual[1:nshow]
            println("  - $k")
        end
    end
    if !isempty(report.missing_in_reference)
        nshow = min(length(report.missing_in_reference), opts.top)
        println("Missing in reference (first $nshow):")
        for k in report.missing_in_reference[1:nshow]
            println("  - $k")
        end
    end
    if !isempty(report.failed)
        nshow = min(length(report.failed), opts.top)
        println("Failed tensors (top $nshow by max_abs):")
        ranked = sort(report.failed; by = x -> x.max_abs, rev = true)
        for s in ranked[1:nshow]
            println(
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

    ok = isempty(report.failed) && isempty(report.missing_in_actual) && isempty(report.missing_in_reference)
    return ok ? 0 : 2
end

try
    exit(main(ARGS))
catch err
    _usage()
    println(stderr, "\nError: ", err)
    exit(1)
end
