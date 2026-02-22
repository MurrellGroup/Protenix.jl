#!/usr/bin/env julia
#
# Stress test: run 100 diverse inputs through both protenix_mini and protenix_base.
#
# Usage:
#   cd /home/claudey/FixingKAFA/ka_run_env
#   /home/claudey/.julia/juliaup/julia-1.11.8+0.aarch64.linux.gnu/bin/julia --project=. ../PXDesign.jl/clean_targets/scripts/run_stress_test.jl

using CUDA, cuDNN
using Flux
using Random
using MoleculeFlow  # triggers PXDesignMoleculeFlowExt for SMILES ligand bonds + 3D coords
using PXDesign

const ROOT = joinpath(@__DIR__, "..", "..")
const INPUTS = joinpath(ROOT, "clean_targets", "stress_inputs")

# Clean output directories
const OUTBASE = joinpath(ROOT, "clean_targets", "stress_outputs")
rm(OUTBASE; force=true, recursive=true)
mkpath(OUTBASE)

const CIFDIR = joinpath(ROOT, "clean_targets", "stress_cif_results")
rm(CIFDIR; force=true, recursive=true)
mkpath(CIFDIR)

println("CUDA: ", CUDA.functional(), "  GPU: ", CUDA.name(CUDA.device()))
println("Free GPU memory: ", round(CUDA.available_memory() / 1024^3, digits=1), " GB")
println()

const MODELS = [
    ("protenix_mini_default_v0.5.0", 20, 4),
    ("protenix_base_default_v0.5.0", 200, 10),
]

println("Stress test: 100 diverse inputs × $(length(MODELS)) models")
for (m, s, c) in MODELS
    println("  $m  step=$s  cycle=$c")
end
println("Input:  $INPUTS")
println("Output: $OUTBASE")
println("CIFs:   $CIFDIR")
println()

# key = "inputname__modelname" => :pass or :fail
results = Dict{String, Symbol}()
errors  = Dict{String, String}()

json_files = sort(filter(f -> endswith(f, ".json"), readdir(INPUTS; join=true)))
total_runs = length(json_files) * length(MODELS)
println("Found $(length(json_files)) test cases → $total_runs total runs\n")

run_idx = 0
for (i, json_path) in enumerate(json_files)
    input_name = splitext(basename(json_path))[1]

    for (model_name, step, cycle) in MODELS
        global run_idx += 1
        key = "$(input_name)__$(model_name)"
        outdir = joinpath(OUTBASE, key)

        println("=" ^ 60)
        println("  [$run_idx/$total_runs] $input_name")
        println("  Model: $model_name  step=$step  cycle=$cycle")
        println("=" ^ 60)

        try
            records = PXDesign.predict_json(
                json_path;
                out_dir = outdir,
                model_name = model_name,
                seeds = [101],
                use_default_params = false,
                step = step,
                cycle = cycle,
                sample = 1,
                use_msa = false,
                gpu = true,
            )

            cif_found = false
            for r in records
                for cif in r.cif_paths
                    println("    CIF: $(basename(cif)) ($(filesize(cif)) bytes)")
                    dst = joinpath(CIFDIR, "$(key)__$(basename(cif))")
                    cp(cif, dst; force=true)
                    cif_found = true
                end
            end

            if cif_found
                results[key] = :pass
                println("    PASS")
            else
                results[key] = :fail
                errors[key] = "No CIF files produced"
                println("    FAIL: No CIF files produced")
            end
        catch e
            results[key] = :fail
            errors[key] = sprint(showerror, e)
            println("    FAIL: $e")
            for (exc, bt) in current_exceptions()
                showerror(stdout, exc, bt)
                println()
            end
        end

        GC.gc(); CUDA.reclaim()
        println()
    end
end

# ============================================================
# Summary
# ============================================================
println("\n" * "=" ^ 60)
println("  STRESS TEST SUMMARY")
println("=" ^ 60)

passed = count(v -> v == :pass, values(results))
failed = count(v -> v == :fail, values(results))
total  = length(results)

println("\n  $passed / $total PASSED")
println("  $failed / $total FAILED\n")

# Per-model breakdown
for (model_name, _, _) in MODELS
    model_results = filter(p -> endswith(first(p), model_name), results)
    mp = count(v -> v == :pass, values(model_results))
    mf = count(v -> v == :fail, values(model_results))
    println("  $model_name: $mp passed, $mf failed")
end
println()

if failed > 0
    println("  Failed cases:")
    for key in sort(collect(keys(filter(p -> p.second == :fail, results))))
        msg = get(errors, key, "unknown")
        if length(msg) > 200
            msg = msg[1:200] * "..."
        end
        println("    $key")
        println("      $msg")
    end
end

println("\n  CIFs: $CIFDIR")
println("=" ^ 60)
