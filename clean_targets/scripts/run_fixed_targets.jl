#!/usr/bin/env julia
#
# Run previously-broken targets with all fixes applied.
# Output goes to a CLEAN directory with no old artifacts.
#
# Usage:
#   cd /home/claudey/FixingKAFA/ka_run_env
#   julia --project=. ../PXDesign.jl/clean_targets/scripts/run_fixed_targets.jl

using CUDA, cuDNN
using Flux
using Random
using MoleculeFlow  # triggers PXDesignMoleculeFlowExt for SMILES ligand bonds + 3D coords
using PXDesign

const ROOT = joinpath(@__DIR__, "..", "..")
const INPUTS = joinpath(ROOT, "clean_targets", "inputs")

# CLEAN output directory — delete any previous contents
const OUTBASE = joinpath(ROOT, "clean_targets", "julia_outputs_fixed")
rm(OUTBASE; force=true, recursive=true)
mkpath(OUTBASE)

# Clean CIF collection directory
const CIFDIR = joinpath(ROOT, "clean_targets", "cif_results_fixed")
rm(CIFDIR; force=true, recursive=true)
mkpath(CIFDIR)

println("CUDA: ", CUDA.functional(), "  GPU: ", CUDA.name(CUDA.device()))
println("Free GPU memory: ", round(CUDA.available_memory() / 1024^3, digits=1), " GB")
println("Output: $OUTBASE")
println("CIFs:   $CIFDIR")
println()

const FOLDING_MODELS = Dict(
    "protenix_base_default_v0.5.0" => (step=200, cycle=10),
    "protenix_base_constraint_v0.5.0" => (step=200, cycle=10),
    "protenix_mini_default_v0.5.0" => (step=20, cycle=4),
)

function run_target(json_path::String, model_name::String)
    name = splitext(basename(json_path))[1]
    params = FOLDING_MODELS[model_name]
    outdir = joinpath(OUTBASE, "$(name)__$(model_name)")

    println("=" ^ 60)
    println("  Target: $name")
    println("  Model:  $model_name")
    println("  Steps:  $(params.step)   Cycles: $(params.cycle)")
    println("=" ^ 60)

    try
        records = PXDesign.predict_json(
            json_path;
            out_dir = outdir,
            model_name = model_name,
            seeds = [101],
            use_default_params = false,
            step = params.step,
            cycle = params.cycle,
            sample = 1,
            use_msa = false,
            gpu = true,
        )
        for r in records
            println("  Output: $(r.prediction_dir)")
            for cif in r.cif_paths
                println("    CIF: $cif ($(filesize(cif)) bytes)")
                # Copy CIF to clean collection
                dst = joinpath(CIFDIR, "$(name)__$(model_name)__$(basename(cif))")
                cp(cif, dst; force=true)
                println("    -> $dst")
            end
        end
    catch e
        println("  FAILED: $name / $model_name")
        println("  Error: ", e)
        for (exc, bt) in current_exceptions()
            showerror(stdout, exc, bt)
            println()
        end
    end

    GC.gc(); CUDA.reclaim()
    println()
end

# Previously-broken targets: 04 (DNA), 05 (RNA), 06 (CCD ligand), 07 (SMILES ligand),
# 11 (modified residue), 12 (DNA modified), 16 (constraint+ligand), 21 (SDF ligand)
broken_targets = ["04", "05", "06", "07", "11", "12", "16", "21"]

for n in broken_targets
    for f in sort(filter(x -> startswith(basename(x), n) && endswith(x, ".json"), readdir(INPUTS; join=true)))
        if n == "16"
            # Constraint target uses constraint model
            run_target(f, "protenix_base_constraint_v0.5.0")
        else
            run_target(f, "protenix_base_default_v0.5.0")
            run_target(f, "protenix_mini_default_v0.5.0")
        end
    end
end

println("=" ^ 60)
println("  Fixed targets run complete")
println("  Output: $OUTBASE")
println("  CIFs:   $CIFDIR")
println("=" ^ 60)
