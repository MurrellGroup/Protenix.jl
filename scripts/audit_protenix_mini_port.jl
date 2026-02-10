#!/usr/bin/env julia

using PXDesign
using Random
using TOML

function _usage()
    println(
        "Usage: audit_protenix_mini_port.jl <safetensors_path_or_dir> " *
        "[--raw-dir <path>] [--report <path>]",
    )
end

function _parse_args(args::Vector{String})
    length(args) >= 1 || error("Missing required safetensors path.")
    st_path = args[1]
    raw_dir = nothing
    report_path = nothing

    i = 2
    while i <= length(args)
        flag = args[i]
        if flag == "--raw-dir"
            i < length(args) || error("--raw-dir requires a value")
            i += 1
            raw_dir = args[i]
        elseif flag == "--report"
            i < length(args) || error("--report requires a value")
            i += 1
            report_path = args[i]
        else
            error("Unknown flag: $flag")
        end
        i += 1
    end

    return (st_path = st_path, raw_dir = raw_dir, report_path = report_path)
end

function _root_prefix_counts(weights::AbstractDict{<:AbstractString, <:Any})
    counts = Dict{String, Int}()
    for k_any in keys(weights)
        k = String(k_any)
        root = first(split(k, '.'))
        counts[root] = get(counts, root, 0) + 1
    end
    return counts
end

function _sort_counts_desc(counts::Dict{String, Int})
    rows = collect(counts)
    sort!(rows; by = kv -> (-kv[2], kv[1]))
    return rows
end

function main(args::Vector{String})
    opts = try
        _parse_args(args)
    catch err
        _usage()
        rethrow(err)
    end

    weights = PXDesign.Model.load_safetensors_weights(opts.st_path)
    dims = PXDesign.Model.infer_model_scaffold_dims(weights)
    dm = PXDesign.Model.DiffusionModule(
        dims.c_token,
        dims.c_s,
        dims.c_z,
        dims.c_s_inputs;
        c_atom = dims.c_atom,
        c_atompair = dims.c_atompair,
        atom_encoder_blocks = dims.atom_encoder_blocks,
        atom_encoder_heads = dims.atom_encoder_heads,
        n_blocks = dims.n_blocks,
        n_heads = dims.n_heads,
        atom_decoder_blocks = dims.atom_decoder_blocks,
        atom_decoder_heads = dims.atom_decoder_heads,
        rng = MersenneTwister(1),
    )
    coverage = PXDesign.Model.checkpoint_coverage_report(dm, nothing, weights)
    total = length(keys(weights))
    diffusion_present = coverage.n_present
    non_diffusion = total - diffusion_present

    println("safetensors=$(abspath(opts.st_path))")
    println("total_tensors=$total")
    println("diffusion_tensors=$diffusion_present")
    println("non_diffusion_tensors=$non_diffusion")
    println("diffusion_missing=$(length(coverage.missing))")
    println("diffusion_unused=$(length(coverage.unused))")

    root_counts = _root_prefix_counts(weights)
    println("top_roots:")
    for (k, v) in _sort_counts_desc(root_counts)[1:min(end, 8)]
        println("  $k => $v")
    end

    parity_summary = Dict{String, Any}()
    if opts.raw_dir !== nothing
        raw = PXDesign.Model.load_raw_weights(opts.raw_dir)
        parity = PXDesign.Model.tensor_parity_report(raw, weights; atol = 0f0, rtol = 0f0)
        parity_summary = Dict(
            "num_compared" => length(parity.compared),
            "num_failed" => length(parity.failed),
            "missing_in_safetensors" => length(parity.missing_in_actual),
            "missing_in_raw" => length(parity.missing_in_reference),
        )
        println("parity_num_compared=$(parity_summary["num_compared"])")
        println("parity_num_failed=$(parity_summary["num_failed"])")
        println("parity_missing_in_safetensors=$(parity_summary["missing_in_safetensors"])")
        println("parity_missing_in_raw=$(parity_summary["missing_in_raw"])")
    end

    summary = Dict{String, Any}(
        "safetensors_path" => abspath(opts.st_path),
        "raw_dir" => opts.raw_dir === nothing ? "" : abspath(opts.raw_dir),
        "total_tensors" => total,
        "diffusion_tensors" => diffusion_present,
        "non_diffusion_tensors" => non_diffusion,
        "diffusion_missing" => length(coverage.missing),
        "diffusion_unused" => length(coverage.unused),
        "diffusion_coverage_fraction" => total == 0 ? 0.0 : diffusion_present / total,
        "root_prefix_counts" => Dict(root_counts),
        "parity" => parity_summary,
    )

    if opts.report_path !== nothing
        report_dir = dirname(opts.report_path)
        isempty(report_dir) || mkpath(report_dir)
        open(opts.report_path, "w") do io
            TOML.print(io, summary)
        end
        println("report=$(abspath(opts.report_path))")
    end

    return 0
end

exit(main(ARGS))
