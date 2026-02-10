#!/usr/bin/env julia

using PXDesign
using Random
using TOML

function _usage()
    println(
        "Usage: audit_protenix_v05_port.jl <safetensors_path_or_dir> " *
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

function _root_prefix_counts(weights::AbstractDict{<:AbstractString,<:Any})
    counts = Dict{String,Int}()
    for k_any in keys(weights)
        k = String(k_any)
        root = first(split(k, '.'))
        counts[root] = get(counts, root, 0) + 1
    end
    return counts
end

function _sort_counts_desc(counts::Dict{String,Int})
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
    total = length(keys(weights))
    root_counts = _root_prefix_counts(weights)

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
    diffusion_cov = PXDesign.Model.checkpoint_coverage_report(dm, nothing, weights)

    strict_load_ok = false
    strict_load_error = ""
    mini_dims = nothing
    try
        mini_dims = PXDesign.ProtenixMini.infer_protenix_mini_dims(weights)
        m = PXDesign.ProtenixMini.build_protenix_mini_model(weights)
        PXDesign.ProtenixMini.load_protenix_mini_model!(m, weights; strict = true)
        strict_load_ok = true
    catch err
        strict_load_error = sprint(showerror, err)
    end

    expected_roots = Set([
        "pairformer_stack",
        "diffusion_module",
        "confidence_head",
        "msa_module",
        "input_embedder",
        "template_embedder",
        "distogram_head",
        "layernorm_s",
        "layernorm_z_cycle",
        "linear_no_bias_s",
        "linear_no_bias_sinit",
        "linear_no_bias_token_bond",
        "linear_no_bias_z_cycle",
        "linear_no_bias_zinit1",
        "linear_no_bias_zinit2",
        "relative_position_encoding",
    ])
    observed_roots = Set(keys(root_counts))
    unexpected_roots = sort!(collect(setdiff(observed_roots, expected_roots)))

    println("safetensors=$(abspath(opts.st_path))")
    println("total_tensors=$total")
    println("strict_protenix_load_ok=$strict_load_ok")
    isempty(strict_load_error) || println("strict_protenix_load_error=$(strict_load_error)")
    if mini_dims !== nothing
        println("inferred_protenix_dims=$(mini_dims)")
    end
    println("diffusion_tensors=$(diffusion_cov.n_present)")
    println("diffusion_missing=$(length(diffusion_cov.missing))")
    println("diffusion_unused=$(length(diffusion_cov.unused))")
    println("unexpected_roots=$(length(unexpected_roots))")
    for r in unexpected_roots
        println("  unexpected_root=$r")
    end

    println("top_roots:")
    for (k, v) in _sort_counts_desc(root_counts)[1:min(end, 12)]
        println("  $k => $v")
    end

    parity_summary = Dict{String,Any}()
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

    summary = Dict{String,Any}(
        "safetensors_path" => abspath(opts.st_path),
        "raw_dir" => opts.raw_dir === nothing ? "" : abspath(opts.raw_dir),
        "total_tensors" => total,
        "strict_protenix_load_ok" => strict_load_ok,
        "strict_protenix_load_error" => strict_load_error,
        "inferred_protenix_dims" => mini_dims === nothing ? Dict{String,Any}() : Dict{String,Any}(string(k) => v for (k, v) in pairs(mini_dims)),
        "diffusion_coverage" => Dict(
            "n_expected" => diffusion_cov.n_expected,
            "n_present" => diffusion_cov.n_present,
            "missing" => length(diffusion_cov.missing),
            "unused" => length(diffusion_cov.unused),
        ),
        "unexpected_roots" => unexpected_roots,
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

    return strict_load_ok ? 0 : 1
end

exit(main(ARGS))
