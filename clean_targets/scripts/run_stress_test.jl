#!/usr/bin/env julia
#
# Stress test: run 100 diverse inputs through protenix_mini_default_v0.5.0
# with low step/cycle count (speed over quality — this is a crash/sanity test).
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
println("Stress test: 100 diverse inputs")
println("Model: protenix_mini_default_v0.5.0  step=20  cycle=4")
println("Input:  $INPUTS")
println("Output: $OUTBASE")
println("CIFs:   $CIFDIR")
println()

const MODEL = "protenix_mini_default_v0.5.0"
const STEP = 20
const CYCLE = 4

results = Dict{String, Symbol}()   # name => :pass or :fail
errors  = Dict{String, String}()   # name => error message

json_files = sort(filter(f -> endswith(f, ".json"), readdir(INPUTS; join=true)))
println("Found $(length(json_files)) test cases\n")

for (i, json_path) in enumerate(json_files)
    name = splitext(basename(json_path))[1]
    outdir = joinpath(OUTBASE, name)

    println("=" ^ 60)
    println("  [$i/$(length(json_files))] $name")
    println("=" ^ 60)

    try
        records = PXDesign.predict_json(
            json_path;
            out_dir = outdir,
            model_name = MODEL,
            seeds = [101],
            use_default_params = false,
            step = STEP,
            cycle = CYCLE,
            sample = 1,
            use_msa = false,
            gpu = true,
        )

        cif_found = false
        for r in records
            for cif in r.cif_paths
                println("    CIF: $(basename(cif)) ($(filesize(cif)) bytes)")
                dst = joinpath(CIFDIR, "$(name)__$(basename(cif))")
                cp(cif, dst; force=true)
                cif_found = true
            end
        end

        if cif_found
            results[name] = :pass
            println("    PASS")
        else
            results[name] = :fail
            errors[name] = "No CIF files produced"
            println("    FAIL: No CIF files produced")
        end
    catch e
        results[name] = :fail
        errors[name] = sprint(showerror, e)
        println("    FAIL: $e")
        for (exc, bt) in current_exceptions()
            showerror(stdout, exc, bt)
            println()
        end
    end

    GC.gc(); CUDA.reclaim()
    println()
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

if failed > 0
    println("  Failed cases:")
    for name in sort(collect(keys(filter(p -> p.second == :fail, results))))
        msg = get(errors, name, "unknown")
        # Truncate long error messages
        if length(msg) > 200
            msg = msg[1:200] * "..."
        end
        println("    $name")
        println("      $msg")
    end
end

println("\n  CIFs: $CIFDIR")
println("=" ^ 60)
