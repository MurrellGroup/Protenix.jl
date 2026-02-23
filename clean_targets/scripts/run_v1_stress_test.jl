#!/usr/bin/env julia
#
# Stress test: run 100 diverse inputs through protenix_base_default_v1.0.0.
#
# Usage: julia --project=<env> clean_targets/scripts/run_v1_stress_test.jl

using CUDA, cuDNN
using Flux
using Random
using MoleculeFlow  # triggers PXDesignMoleculeFlowExt for SMILES ligand bonds + 3D coords
using PXDesign

const ROOT = joinpath(@__DIR__, "..", "..")
const INPUTS = joinpath(ROOT, "clean_targets", "stress_inputs")
const OUTBASE = joinpath(ROOT, "clean_targets", "v1_stress_outputs")
const CIFDIR = joinpath(ROOT, "clean_targets", "v1_stress_cif_results")

# Local safetensors path (pre-HuggingFace upload)
const V1_WEIGHTS = joinpath(ROOT, "weights_safetensors_protenix_base_default_v1.0.0")
const MODEL_NAME = "protenix_base_default_v1.0.0"
const STEP = 200
const CYCLE = 10

rm(OUTBASE; force=true, recursive=true)
mkpath(OUTBASE)
rm(CIFDIR; force=true, recursive=true)
mkpath(CIFDIR)

println("CUDA: ", CUDA.functional(), "  GPU: ", CUDA.name(CUDA.device()))
println("Free GPU memory: ", round(CUDA.available_memory() / 1024^3, digits=1), " GB")
println()
println("V1 Stress test: 100 diverse inputs")
println("  Model: $MODEL_NAME  step=$STEP  cycle=$CYCLE")
println("  Weights: $V1_WEIGHTS")
println("  Input:  $INPUTS")
println("  Output: $OUTBASE")
println("  CIFs:   $CIFDIR")
println()

results = Dict{String, Symbol}()
errors  = Dict{String, String}()

json_files = sort(filter(f -> endswith(f, ".json"), readdir(INPUTS; join=true)))
println("Found $(length(json_files)) test cases\n")

for (i, json_path) in enumerate(json_files)
    input_name = splitext(basename(json_path))[1]
    key = "$(input_name)__$(MODEL_NAME)"
    outdir = joinpath(OUTBASE, key)

    println("=" ^ 60)
    println("  [$i/$(length(json_files))] $input_name")
    println("  Model: $MODEL_NAME  step=$STEP  cycle=$CYCLE")
    println("=" ^ 60)

    try
        records = PXDesign.predict_json(
            json_path;
            out_dir = outdir,
            model_name = MODEL_NAME,
            weights_path = V1_WEIGHTS,
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

println("\n" * "=" ^ 60)
println("  V1 STRESS TEST SUMMARY")
println("=" ^ 60)

passed = count(v -> v == :pass, values(results))
failed = count(v -> v == :fail, values(results))
total  = length(results)

println("\n  $passed / $total PASSED")
println("  $failed / $total FAILED\n")

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
