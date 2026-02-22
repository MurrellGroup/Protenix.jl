#!/usr/bin/env julia
#
# Run all clean_targets through Julia PXDesign.
#
# Usage:
#   cd /home/claudey/FixingKAFA/ka_run_env
#   julia --project=. ../PXDesign.jl/clean_targets/scripts/run_all_julia.jl [target_number]
#
# If a target number is given (e.g. "01"), only that target is run.
# Otherwise, all targets that Julia currently supports are attempted.

using CUDA, cuDNN
using Flux
using Random
using PXDesign

# No allowscalar — scalar GPU indexing is forbidden

const ROOT = joinpath(@__DIR__, "..", "..")
const INPUTS = joinpath(ROOT, "clean_targets", "inputs")
const OUTBASE = joinpath(ROOT, "clean_targets", "julia_outputs")
mkpath(OUTBASE)

const FILTER = length(ARGS) >= 1 ? ARGS[1] : ""

println("CUDA: ", CUDA.functional(), "  GPU: ", CUDA.name(CUDA.device()))
println("Free GPU memory: ", round(CUDA.available_memory() / 1024^3, digits=1), " GB")
println()

should_run(n::String) = isempty(FILTER) || FILTER == n

# ── Folding targets (JSON) ────────────────────────────────────────────────────

# Model → (step, cycle) mapping — matches Python per-model defaults from
# configs_model_type.py exactly.  Do NOT override these with arbitrary values;
# Python's config merge order (model overrides after CLI args) silently
# forces model defaults, so Julia must use the same values for parity.
const FOLDING_MODELS = Dict(
    "protenix_base_default_v0.5.0" => (step=200, cycle=10),
    "protenix_base_constraint_v0.5.0" => (step=200, cycle=10),
    "protenix_mini_default_v0.5.0" => (step=20, cycle=4),
    "protenix_mini_esm_v0.5.0" => (step=20, cycle=4),
    "protenix_mini_ism_v0.5.0" => (step=20, cycle=4),
    "protenix_mini_tmpl_v0.5.0" => (step=20, cycle=4),
    "protenix_tiny_default_v0.5.0" => (step=20, cycle=4),
)

function run_folding_target(json_path::String, model_name::String)
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
            end
        end
    catch e
        println("  FAILED: $name / $model_name")
        println("  Error: ", e)
    end

    GC.gc(); CUDA.reclaim()
    println()
end

function run_folding_target_msa(json_path::String, model_name::String)
    name = splitext(basename(json_path))[1]
    params = FOLDING_MODELS[model_name]
    outdir = joinpath(OUTBASE, "$(name)__$(model_name)")

    println("=" ^ 60)
    println("  Target: $name (MSA enabled)")
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
            use_msa = true,
            gpu = true,
        )
        for r in records
            println("  Output: $(r.prediction_dir)")
            for cif in r.cif_paths
                println("    CIF: $cif ($(filesize(cif)) bytes)")
            end
        end
    catch e
        println("  FAILED: $name / $model_name")
        println("  Error: ", e)
    end

    GC.gc(); CUDA.reclaim()
    println()
end

# ── Category 1: Protein-Only (01-03) ──────────────────────────────────────────
for n in ("01", "02", "03")
    should_run(n) || continue
    for f in sort(filter(x -> startswith(basename(x), n) && endswith(x, ".json"), readdir(INPUTS; join=true)))
        run_folding_target(f, "protenix_base_default_v0.5.0")
        run_folding_target(f, "protenix_mini_default_v0.5.0")
    end
end

# Target 01 also runs on tiny
if should_run("01")
    f = joinpath(INPUTS, "01_protein_monomer.json")
    isfile(f) && run_folding_target(f, "protenix_tiny_default_v0.5.0")
end

# ── Category 2: Nucleic Acids (04-05) — may not be supported yet ──────────────
for n in ("04", "05")
    should_run(n) || continue
    for f in sort(filter(x -> startswith(basename(x), n) && endswith(x, ".json"), readdir(INPUTS; join=true)))
        run_folding_target(f, "protenix_base_default_v0.5.0")
        run_folding_target(f, "protenix_mini_default_v0.5.0")
    end
end

# ── Category 3: Small Molecules (06-08) — may not be supported yet ────────────
for n in ("06", "07", "08")
    should_run(n) || continue
    for f in sort(filter(x -> startswith(basename(x), n) && endswith(x, ".json"), readdir(INPUTS; join=true)))
        run_folding_target(f, "protenix_base_default_v0.5.0")
        run_folding_target(f, "protenix_mini_default_v0.5.0")
    end
end

# ── Category 4: Multi-Entity (09-10) — may not be supported yet ───────────────
for n in ("09", "10")
    should_run(n) || continue
    for f in sort(filter(x -> startswith(basename(x), n) && endswith(x, ".json"), readdir(INPUTS; join=true)))
        run_folding_target(f, "protenix_base_default_v0.5.0")
        run_folding_target(f, "protenix_mini_default_v0.5.0")
    end
end

# ── Category 5: Modifications (11-13) — may not be supported yet ──────────────
for n in ("11", "12", "13")
    should_run(n) || continue
    for f in sort(filter(x -> startswith(basename(x), n) && endswith(x, ".json"), readdir(INPUTS; join=true)))
        run_folding_target(f, "protenix_base_default_v0.5.0")
        run_folding_target(f, "protenix_mini_default_v0.5.0")
    end
end

# ── Category 6: Constraints (14-16) ──────────────────────────────────────────
for n in ("14", "15", "16")
    should_run(n) || continue
    for f in sort(filter(x -> startswith(basename(x), n) && endswith(x, ".json"), readdir(INPUTS; join=true)))
        run_folding_target(f, "protenix_base_constraint_v0.5.0")
    end
end

# ── Category 7: Input Modalities (17-19) ──────────────────────────────────────
# Target 17: MSA
if should_run("17")
    f = joinpath(INPUTS, "17_protein_msa.json")
    if isfile(f)
        run_folding_target_msa(f, "protenix_base_default_v0.5.0")
        run_folding_target_msa(f, "protenix_mini_default_v0.5.0")
    end
end

# Target 18: ESM/ISM
if should_run("18")
    f = joinpath(INPUTS, "18_protein_esm.json")
    if isfile(f)
        run_folding_target(f, "protenix_mini_esm_v0.5.0")
        run_folding_target(f, "protenix_mini_ism_v0.5.0")
    end
end

# Target 19: Template
if should_run("19")
    f = joinpath(INPUTS, "19_protein_template.json")
    if isfile(f)
        run_folding_target(f, "protenix_mini_tmpl_v0.5.0")
    end
end

# ── Category 8: Complex Assembly (20) ─────────────────────────────────────────
# Commented out: 1190 tokens OOMs flash attention bias tensor (~27 GB)
#if should_run("20")
#    f = joinpath(INPUTS, "20_complex_multichain.json")
#    if isfile(f)
#        run_folding_target(f, "protenix_base_default_v0.5.0")
#        run_folding_target(f, "protenix_mini_default_v0.5.0")
#    end
#end

# ── Category 9: File-Based Ligand (21) ────────────────────────────────────────
if should_run("21")
    f = joinpath(INPUTS, "21_ligand_file_sdf.json")
    if isfile(f)
        run_folding_target(f, "protenix_base_default_v0.5.0")
        run_folding_target(f, "protenix_mini_default_v0.5.0")
    end
end

# ── Category 10: Design (22-32) ── Julia PXDesign inference ──────────────────
# Design targets use YAML format and the PXDesign design pipeline (run_infer),
# NOT the Protenix folding pipeline (predict_json).
const DESIGN_N_STEP = 200
const DESIGN_N_SAMPLE = 1

function run_design_target(yaml_path::String)
    name = splitext(basename(yaml_path))[1]
    outdir = joinpath(OUTBASE, "$(name)__pxdesign")

    println("=" ^ 60)
    println("  Design Target: $name")
    println("  Model:  pxdesign_v0.1.0")
    println("  Steps:  $DESIGN_N_STEP   Samples: $DESIGN_N_SAMPLE")
    println("=" ^ 60)

    try
        cfg = PXDesign.Config.default_config(; project_root = ROOT)
        cfg["input_json_path"] = yaml_path
        cfg["dump_dir"] = outdir
        cfg["model_name"] = "pxdesign_v0.1.0"
        cfg["seeds"] = [101]
        cfg["gpu"] = true
        PXDesign.Config.set_nested!(cfg, "sample_diffusion.N_step", DESIGN_N_STEP)
        PXDesign.Config.set_nested!(cfg, "sample_diffusion.N_sample", DESIGN_N_SAMPLE)
        # model_scaffold.enabled defaults to true in default_config

        result = PXDesign.run_infer(cfg)
        println("  Result: $(result["status"])  tasks=$(get(result, "num_tasks", "?"))")
        println("  Output: $outdir")
    catch e
        println("  FAILED: $name")
        println("  Error: ", e)
        for (exc, bt) in current_exceptions()
            showerror(stdout, exc, bt)
            println()
        end
    end

    GC.gc(); CUDA.reclaim()
    println()
end

for n in ("22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32")
    should_run(n) || continue
    for f in sort(filter(x -> startswith(basename(x), n) && endswith(x, ".yaml"), readdir(INPUTS; join=true)))
        run_design_target(f)
    end
end

# ── Summary ───────────────────────────────────────────────────────────────────
println("=" ^ 60)
println("  Julia clean_targets run complete")
println("  Output: $OUTBASE")
println("=" ^ 60)
