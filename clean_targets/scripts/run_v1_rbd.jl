#!/usr/bin/env julia
#
# Re-run RBD glycosylated with v1 + MSA enabled.
#
# Usage: julia --project=<env> clean_targets/scripts/run_v1_rbd.jl

using CUDA, cuDNN
using Flux
using Random
using MoleculeFlow
using PXDesign

const ROOT = joinpath(@__DIR__, "..", "..")
const JSON_PATH = joinpath(ROOT, "clean_targets", "inputs", "33_rbd_glycosylated.json")
const OUTDIR = joinpath(ROOT, "clean_targets", "v1_rbd_msa_output")
const V1_WEIGHTS = joinpath(ROOT, "weights_safetensors_protenix_base_default_v1.0.0")
const MODEL_NAME = "protenix_base_default_v1.0.0"

rm(OUTDIR; force=true, recursive=true)
mkpath(OUTDIR)

println("CUDA: ", CUDA.functional(), "  GPU: ", CUDA.name(CUDA.device()))
println("RBD glycosylated with v1 + MSA")
println("  Model: $MODEL_NAME  step=200  cycle=10")
println("  Weights: $V1_WEIGHTS")
println("  Input: $JSON_PATH")
println("  Output: $OUTDIR")
println()

records = PXDesign.predict_json(
    JSON_PATH;
    out_dir = OUTDIR,
    model_name = MODEL_NAME,
    weights_path = V1_WEIGHTS,
    seeds = [101],
    use_default_params = false,
    step = 200,
    cycle = 10,
    sample = 1,
    use_msa = true,
    gpu = true,
)

for r in records
    for cif in r.cif_paths
        println("CIF: $cif ($(filesize(cif)) bytes)")
    end
end
println("Done.")
