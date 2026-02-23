#!/usr/bin/env julia
#
# Full rerun: all test sets with all models.
#
# After Fixes 1-13 for v1.0 input feature parity, this establishes a fresh baseline
# with all outputs in one isolated directory.
#
# Usage:
#   julia --project=/home/claudey/FixingKAFA/ka_run_env \
#         /home/claudey/FixingKAFA/PXDesign.jl/clean_targets/scripts/run_full_rerun.jl
#
# Run sets:
#   1. Clean targets (27 JSON + 11 YAML design)
#   2. Stress test (100 JSON × 3 models = 300 runs)
#   3. RBD+glycan+MSA dedicated (3 runs)
#   4. Structure checks + comparison report

using Dates
using CUDA, cuDNN
using Flux
using Random
using MoleculeFlow  # triggers PXDesignMoleculeFlowExt for SMILES ligand bonds + 3D coords
using PXDesign
using ProtInterop
const SC = ProtInterop.StructureChecking

# ============================================================
# Paths
# ============================================================
const ROOT = joinpath(@__DIR__, "..", "..")
const INPUTS = joinpath(ROOT, "clean_targets", "inputs")
const STRESS_INPUTS = joinpath(ROOT, "clean_targets", "stress_inputs")

const RUN_DIR = joinpath(ROOT, "clean_targets", "run_20260223")

# Nested prediction output directories
const CLEAN_OUTDIR   = joinpath(RUN_DIR, "clean_targets")
const STRESS_OUTDIR  = joinpath(RUN_DIR, "stress_outputs")
const RBD_OUTDIR     = joinpath(RUN_DIR, "rbd_outputs")

# Flat CIF directories
const CIFS_CLEAN   = joinpath(RUN_DIR, "cifs_clean")
const CIFS_STRESS  = joinpath(RUN_DIR, "cifs_stress")
const CIFS_RBD     = joinpath(RUN_DIR, "cifs_rbd")

# Structure check directories
const SC_CLEAN   = joinpath(RUN_DIR, "structure_checks", "clean")
const SC_STRESS  = joinpath(RUN_DIR, "structure_checks", "stress")
const SC_RBD     = joinpath(RUN_DIR, "structure_checks", "rbd")

# Reference reports (read only — never overwrite)
const REF_CLEAN  = joinpath(ROOT, "clean_targets", "structure_check_reference", "clean_targets")
const REF_STRESS = joinpath(ROOT, "clean_targets", "structure_check_reference", "stress")

# V1.0 weights (local safetensors)
const V1_DEFAULT_WEIGHTS  = joinpath(ROOT, "weights_safetensors_protenix_base_default_v1.0.0")
const V1_20250630_WEIGHTS = joinpath(ROOT, "weights_safetensors_protenix_base_20250630_v1.0.0")

# Create all directories
for d in (CLEAN_OUTDIR, STRESS_OUTDIR, RBD_OUTDIR,
          CIFS_CLEAN, CIFS_STRESS, CIFS_RBD,
          SC_CLEAN, SC_STRESS, SC_RBD)
    mkpath(d)
end

# ============================================================
# Model parameters
# ============================================================
const MODEL_PARAMS = Dict(
    "protenix_base_default_v0.5.0"      => (step=200, cycle=10),
    "protenix_base_constraint_v0.5.0"    => (step=200, cycle=10),
    "protenix_mini_default_v0.5.0"       => (step=20,  cycle=4),
    "protenix_mini_esm_v0.5.0"           => (step=20,  cycle=4),
    "protenix_mini_ism_v0.5.0"           => (step=20,  cycle=4),
    "protenix_mini_tmpl_v0.5.0"          => (step=20,  cycle=4),
    "protenix_tiny_default_v0.5.0"       => (step=20,  cycle=4),
    "protenix_base_default_v1.0.0"       => (step=200, cycle=10),
    "protenix_base_20250630_v1.0.0"      => (step=200, cycle=10),
)

# Models that need explicit weights_path (local safetensors, not yet on HuggingFace)
const WEIGHTS_PATHS = Dict(
    "protenix_base_default_v1.0.0"  => V1_DEFAULT_WEIGHTS,
    "protenix_base_20250630_v1.0.0" => V1_20250630_WEIGHTS,
)

# ============================================================
# Tracking
# ============================================================
mutable struct RunTracker
    results::Dict{String, Symbol}   # key => :pass/:fail
    errors::Dict{String, String}    # key => error message
    cif_map::Dict{String, Vector{String}}  # key => [flat cif paths]
    run_count::Int
end
RunTracker() = RunTracker(Dict(), Dict(), Dict(), 0)

function record_pass!(t::RunTracker, key::String, cif_paths::Vector{String})
    t.results[key] = :pass
    t.cif_map[key] = cif_paths
end
function record_fail!(t::RunTracker, key::String, msg::String)
    t.results[key] = :fail
    t.errors[key] = length(msg) > 500 ? msg[1:500] * "..." : msg
end

const CLEAN_TRACKER   = RunTracker()
const STRESS_TRACKER  = RunTracker()
const RBD_TRACKER     = RunTracker()

# ============================================================
# Helpers
# ============================================================

function get_input_name(path::String)
    splitext(basename(path))[1]
end

"""
Copy CIF files from prediction output to a flat directory with model-identifying names.
Returns list of flat CIF paths.
"""
function copy_cifs_flat(cif_paths::Vector, input_name::String, model_name::String, flat_dir::String)
    flat_paths = String[]
    for cif in cif_paths
        bn = basename(cif)
        # Extract sample number from original filename (e.g., "protein_monomer_sample_0.cif")
        m = match(r"sample_(\d+)\.cif$", bn)
        sample_str = m !== nothing ? "sample_$(m.captures[1])" : replace(splitext(bn)[1], r"[^a-zA-Z0-9_]" => "_")
        flat_name = "$(input_name)__$(model_name)__seed101__$(sample_str).cif"
        flat_path = joinpath(flat_dir, flat_name)
        cp(cif, flat_path; force=true)
        push!(flat_paths, flat_path)
    end
    return flat_paths
end

"""
Find all CIF files in a directory tree.
"""
function find_cifs(dir::String)
    paths = String[]
    isdir(dir) || return paths
    for (d, _, files) in walkdir(dir)
        for f in files
            endswith(f, ".cif") && push!(paths, joinpath(d, f))
        end
    end
    sort!(paths)
end

"""
Run a folding target (JSON) and copy CIFs to flat dir.
"""
function run_folding(json_path::String, model_name::String;
                     use_msa::Bool=false, outbase::String, flat_dir::String,
                     tracker::RunTracker)
    input_name = get_input_name(json_path)
    params = MODEL_PARAMS[model_name]
    outdir = joinpath(outbase, "$(input_name)__$(model_name)")
    key = "$(input_name)__$(model_name)"
    tracker.run_count += 1
    n = tracker.run_count

    println("=" ^ 70)
    println("  [$n] $input_name")
    println("  Model:  $model_name   step=$(params.step)  cycle=$(params.cycle)  msa=$use_msa")
    println("=" ^ 70)
    flush(stdout)

    try
        kwargs = Dict{Symbol,Any}(
            :out_dir => outdir,
            :model_name => model_name,
            :seeds => [101],
            :use_default_params => false,
            :step => params.step,
            :cycle => params.cycle,
            :sample => 1,
            :use_msa => use_msa,
            :gpu => true,
        )
        if haskey(WEIGHTS_PATHS, model_name)
            kwargs[:weights_path] = WEIGHTS_PATHS[model_name]
        end

        records = PXDesign.predict_json(json_path; kwargs...)

        all_cifs = String[]
        for r in records
            for cif in r.cif_paths
                println("    CIF: $(basename(cif)) ($(filesize(cif)) bytes)")
                push!(all_cifs, cif)
            end
        end

        if !isempty(all_cifs)
            flat_paths = copy_cifs_flat(all_cifs, input_name, model_name, flat_dir)
            record_pass!(tracker, key, flat_paths)
            println("    PASS ($(length(all_cifs)) CIFs)")
        else
            record_fail!(tracker, key, "No CIF files produced")
            println("    FAIL: No CIF files produced")
        end
    catch e
        msg = sprint(showerror, e)
        record_fail!(tracker, key, msg)
        println("    FAIL: $e")
        for (exc, bt) in current_exceptions()
            showerror(stdout, exc, bt)
            println()
        end
    end

    GC.gc(); CUDA.reclaim()
    println()
    flush(stdout)
end

"""
Run a design target (YAML) and copy CIFs to flat dir.
"""
function run_design(yaml_path::String; outbase::String, flat_dir::String, tracker::RunTracker)
    input_name = get_input_name(yaml_path)
    model_name = "pxdesign_v0.1.0"
    outdir = joinpath(outbase, "$(input_name)__$(model_name)")
    key = "$(input_name)__$(model_name)"
    tracker.run_count += 1
    n = tracker.run_count

    println("=" ^ 70)
    println("  [$n] Design: $input_name")
    println("  Model:  $model_name   step=200  sample=1")
    println("=" ^ 70)
    flush(stdout)

    try
        cfg = PXDesign.Config.default_config(; project_root = ROOT)
        cfg["input_json_path"] = yaml_path
        cfg["dump_dir"] = outdir
        cfg["model_name"] = model_name
        cfg["seeds"] = [101]
        cfg["gpu"] = true
        PXDesign.Config.set_nested!(cfg, "sample_diffusion.N_step", 200)
        PXDesign.Config.set_nested!(cfg, "sample_diffusion.N_sample", 1)

        result = PXDesign.run_infer(cfg)
        println("    Result: $(result["status"])  tasks=$(get(result, "num_tasks", "?"))")

        # Collect CIFs from output tree
        all_cifs = find_cifs(outdir)
        if !isempty(all_cifs)
            flat_paths = copy_cifs_flat(all_cifs, input_name, model_name, flat_dir)
            record_pass!(tracker, key, flat_paths)
            println("    PASS ($(length(all_cifs)) CIFs)")
        else
            record_fail!(tracker, key, "No CIF files produced")
            println("    FAIL: No CIF files produced")
        end
    catch e
        msg = sprint(showerror, e)
        record_fail!(tracker, key, msg)
        println("    FAIL: $e")
        for (exc, bt) in current_exceptions()
            showerror(stdout, exc, bt)
            println()
        end
    end

    GC.gc(); CUDA.reclaim()
    println()
    flush(stdout)
end

# ============================================================
# Print startup info
# ============================================================
println("=" ^ 70)
println("  FULL RERUN: ALL TEST SETS WITH ALL MODELS")
println("  Date: ", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))
println("=" ^ 70)
println()
println("CUDA: ", CUDA.functional(), "  GPU: ", CUDA.name(CUDA.device()))
println("Free GPU memory: ", round(CUDA.available_memory() / 1024^3, digits=1), " GB")
println("Output: $RUN_DIR")
println()
flush(stdout)

const T_START = Dates.now()

# ############################################################
# TEST SET 1: CLEAN TARGETS
# ############################################################
println("\n", "#" ^ 70)
println("# TEST SET 1: CLEAN TARGETS")
println("#" ^ 70, "\n")
flush(stdout)

# Helper to get JSON input path
inp(name) = joinpath(INPUTS, name)

# ── Standard folding: inputs 01-13, 21 with base + mini (skip 20) ──
for n in ("01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "21")
    for f in sort(filter(x -> startswith(basename(x), n * "_") && endswith(x, ".json"), readdir(INPUTS; join=true)))
        run_folding(f, "protenix_base_default_v0.5.0";
                    outbase=CLEAN_OUTDIR, flat_dir=CIFS_CLEAN, tracker=CLEAN_TRACKER)
        run_folding(f, "protenix_mini_default_v0.5.0";
                    outbase=CLEAN_OUTDIR, flat_dir=CIFS_CLEAN, tracker=CLEAN_TRACKER)
    end
end

# ── Input 01 also on tiny ──
let f = inp("01_protein_monomer.json")
    if isfile(f)
        run_folding(f, "protenix_tiny_default_v0.5.0";
                    outbase=CLEAN_OUTDIR, flat_dir=CIFS_CLEAN, tracker=CLEAN_TRACKER)
    end
end

# ── Constraints: inputs 14-16 with constraint model ──
for n in ("14", "15", "16")
    for f in sort(filter(x -> startswith(basename(x), n * "_") && endswith(x, ".json"), readdir(INPUTS; join=true)))
        run_folding(f, "protenix_base_constraint_v0.5.0";
                    outbase=CLEAN_OUTDIR, flat_dir=CIFS_CLEAN, tracker=CLEAN_TRACKER)
    end
end

# ── Input 17: MSA test (base + mini with MSA) ──
let f = inp("17_protein_msa.json")
    if isfile(f)
        run_folding(f, "protenix_base_default_v0.5.0"; use_msa=true,
                    outbase=CLEAN_OUTDIR, flat_dir=CIFS_CLEAN, tracker=CLEAN_TRACKER)
        run_folding(f, "protenix_mini_default_v0.5.0"; use_msa=true,
                    outbase=CLEAN_OUTDIR, flat_dir=CIFS_CLEAN, tracker=CLEAN_TRACKER)
    end
end

# ── Input 18: ESM / ISM ──
let f = inp("18_protein_esm.json")
    if isfile(f)
        run_folding(f, "protenix_mini_esm_v0.5.0";
                    outbase=CLEAN_OUTDIR, flat_dir=CIFS_CLEAN, tracker=CLEAN_TRACKER)
        run_folding(f, "protenix_mini_ism_v0.5.0";
                    outbase=CLEAN_OUTDIR, flat_dir=CIFS_CLEAN, tracker=CLEAN_TRACKER)
    end
end

# ── Input 19: Template ──
let f = inp("19_protein_template.json")
    if isfile(f)
        run_folding(f, "protenix_mini_tmpl_v0.5.0";
                    outbase=CLEAN_OUTDIR, flat_dir=CIFS_CLEAN, tracker=CLEAN_TRACKER)
    end
end

# ── Input 33: RBD glycosylated with base v0.5 + MSA ──
let f = inp("33_rbd_glycosylated.json")
    if isfile(f)
        run_folding(f, "protenix_base_default_v0.5.0"; use_msa=true,
                    outbase=CLEAN_OUTDIR, flat_dir=CIFS_CLEAN, tracker=CLEAN_TRACKER)
    end
end

# ── Inputs 33b, 34-37: RNA MSA tests with base + mini (use_msa=true since they have MSA data) ──
for n in ("33b", "34", "35", "36", "37")
    for f in sort(filter(x -> startswith(basename(x), n * "_") && endswith(x, ".json"), readdir(INPUTS; join=true)))
        run_folding(f, "protenix_base_default_v0.5.0"; use_msa=true,
                    outbase=CLEAN_OUTDIR, flat_dir=CIFS_CLEAN, tracker=CLEAN_TRACKER)
        run_folding(f, "protenix_mini_default_v0.5.0"; use_msa=true,
                    outbase=CLEAN_OUTDIR, flat_dir=CIFS_CLEAN, tracker=CLEAN_TRACKER)
    end
end

# ── ALL JSON inputs with v1.0 (use_msa=true, skip 20) ──
println("\n── Clean targets: v1.0 model on all JSON inputs ──\n")
flush(stdout)
for f in sort(filter(x -> endswith(x, ".json"), readdir(INPUTS; join=true)))
    bn = basename(f)
    # Skip input 20 (complex_multichain — OOMs)
    startswith(bn, "20_") && continue
    run_folding(f, "protenix_base_default_v1.0.0"; use_msa=true,
                outbase=CLEAN_OUTDIR, flat_dir=CIFS_CLEAN, tracker=CLEAN_TRACKER)
end

# ── Design targets (YAML 22-32) ──
println("\n── Clean targets: design targets ──\n")
flush(stdout)
for n in ("22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32")
    for f in sort(filter(x -> startswith(basename(x), n * "_") && endswith(x, ".yaml"), readdir(INPUTS; join=true)))
        run_design(f; outbase=CLEAN_OUTDIR, flat_dir=CIFS_CLEAN, tracker=CLEAN_TRACKER)
    end
end

println("\n── Clean targets complete ──")
let passed = count(v -> v == :pass, values(CLEAN_TRACKER.results)),
    failed = count(v -> v == :fail, values(CLEAN_TRACKER.results))
    println("  $(CLEAN_TRACKER.run_count) runs: $passed passed, $failed failed")
end
println()
flush(stdout)

# ############################################################
# TEST SET 2: STRESS TEST (100 inputs × 3 models)
# ############################################################
println("\n", "#" ^ 70)
println("# TEST SET 2: STRESS TEST")
println("#" ^ 70, "\n")
flush(stdout)

stress_files = sort(filter(f -> endswith(f, ".json"), readdir(STRESS_INPUTS; join=true)))
println("Found $(length(stress_files)) stress inputs")
println("Models: mini v0.5 (step=20/cycle=4), base v0.5 (step=200/cycle=10), base v1.0 (step=200/cycle=10)")
println("Expected: $(length(stress_files) * 3) total runs\n")
flush(stdout)

for json_path in stress_files
    # Mini v0.5
    run_folding(json_path, "protenix_mini_default_v0.5.0";
                outbase=STRESS_OUTDIR, flat_dir=CIFS_STRESS, tracker=STRESS_TRACKER)
    # Base v0.5
    run_folding(json_path, "protenix_base_default_v0.5.0";
                outbase=STRESS_OUTDIR, flat_dir=CIFS_STRESS, tracker=STRESS_TRACKER)
    # Base v1.0 (use_msa=true to match model defaults)
    run_folding(json_path, "protenix_base_default_v1.0.0"; use_msa=true,
                outbase=STRESS_OUTDIR, flat_dir=CIFS_STRESS, tracker=STRESS_TRACKER)
end

println("\n── Stress test complete ──")
let passed = count(v -> v == :pass, values(STRESS_TRACKER.results)),
    failed = count(v -> v == :fail, values(STRESS_TRACKER.results))
    println("  $(STRESS_TRACKER.run_count) runs: $passed passed, $failed failed")
end
println()
flush(stdout)

# ############################################################
# TEST SET 3: RBD + GLYCAN + MSA
# ############################################################
println("\n", "#" ^ 70)
println("# TEST SET 3: RBD + GLYCAN + MSA")
println("#" ^ 70, "\n")
flush(stdout)

let f = inp("33_rbd_glycosylated.json")
    if isfile(f)
        run_folding(f, "protenix_base_default_v0.5.0"; use_msa=true,
                    outbase=RBD_OUTDIR, flat_dir=CIFS_RBD, tracker=RBD_TRACKER)
        run_folding(f, "protenix_base_default_v1.0.0"; use_msa=true,
                    outbase=RBD_OUTDIR, flat_dir=CIFS_RBD, tracker=RBD_TRACKER)
        run_folding(f, "protenix_base_20250630_v1.0.0"; use_msa=true,
                    outbase=RBD_OUTDIR, flat_dir=CIFS_RBD, tracker=RBD_TRACKER)
    else
        println("WARNING: 33_rbd_glycosylated.json not found!")
    end
end

println("\n── RBD test complete ──")
let passed = count(v -> v == :pass, values(RBD_TRACKER.results)),
    failed = count(v -> v == :fail, values(RBD_TRACKER.results))
    println("  $(RBD_TRACKER.run_count) runs: $passed passed, $failed failed")
end
println()
flush(stdout)

# ############################################################
# STRUCTURE CHECKS
# ############################################################
println("\n", "#" ^ 70)
println("# STRUCTURE CHECKS")
println("#" ^ 70, "\n")
flush(stdout)

"""
Run structure checks on all CIFs in a flat directory.
Returns dict of report_name => StructureIssueReport.
"""
function run_structure_checks(flat_dir::String, sc_dir::String, label::String)
    cifs = sort(filter(f -> endswith(f, ".cif"), readdir(flat_dir; join=true)))
    if isempty(cifs)
        println("  No CIFs in $flat_dir — skipping $label structure checks")
        return Dict{String, Any}()
    end

    mkpath(sc_dir)
    reports = Dict{String, Any}()
    n_ok = 0
    n_fail = 0

    println("  Checking $(length(cifs)) CIFs in $label...")
    flush(stdout)

    for cif in cifs
        name = splitext(basename(cif))[1]
        report_path = joinpath(sc_dir, name * ".txt")

        try
            report = SC.check_structure(cif)
            open(report_path, "w") do io
                SC.print_issue_report(report; io=io)
                println(io)
                println(io, "=" ^ 60)
                println(io, "RAW SCORES (for regression checking)")
                println(io, "=" ^ 60)
                s = report.scores
                println(io, "n_atoms: ", report.n_atoms)
                println(io, "n_residues: ", report.n_residues)
                println(io, "expected_bonds: ", s.expected_bonds)
                println(io, "checked_bonds: ", s.checked_bonds)
                println(io, "bond_length_violations: ", s.bond_length_violations)
                println(io, "missing_bonds: ", s.missing_bonds)
                println(io, "missing_atoms_for_bonds: ", s.missing_atoms_for_bonds)
                println(io, "nonbonded_pairs_evaluated: ", s.nonbonded_pairs_evaluated)
                println(io, "clashes: ", s.clashes)
                println(io, "mild_clashes: ", s.mild_clashes)
                println(io, "moderate_clashes: ", s.moderate_clashes)
                println(io, "severe_clashes: ", s.severe_clashes)
                println(io, "inter_residue_clashes: ", s.inter_residue_clashes)
                println(io, "percent_bond_length_violations: ", s.percent_bond_length_violations)
                println(io, "percent_missing_bonds: ", s.percent_missing_bonds)
                println(io, "percent_clashes: ", s.percent_clashes)
                println(io, "percent_atoms_in_clashes: ", s.percent_atoms_in_clashes)
                println(io, "clashscore_per_1000_atoms: ", s.clashscore_per_1000_atoms)
                println(io, "inter_residue_clashscore_per_1000_atoms: ", s.inter_residue_clashscore_per_1000_atoms)
                println(io, "severe_clashscore_per_1000_atoms: ", s.severe_clashscore_per_1000_atoms)
                println(io, "inter_residue_severe_clashscore_per_1000_atoms: ", s.inter_residue_severe_clashscore_per_1000_atoms)
                println(io, "overall_issue_score: ", s.overall_issue_score)
            end
            reports[name] = report
            n_ok += 1
        catch e
            n_fail += 1
            println("    ERROR checking $name: $e")
            open(report_path, "w") do io
                println(io, "ERROR processing $cif")
                println(io, e)
            end
        end
    end

    println("  $label: $n_ok OK, $n_fail errors")
    flush(stdout)
    return reports
end

clean_reports  = run_structure_checks(CIFS_CLEAN, SC_CLEAN, "clean")
stress_reports = run_structure_checks(CIFS_STRESS, SC_STRESS, "stress")
rbd_reports    = run_structure_checks(CIFS_RBD, SC_RBD, "rbd")

# ############################################################
# COMPARISON REPORT vs PREVIOUS REFERENCE
# ############################################################
println("\n", "#" ^ 70)
println("# COMPARISON vs REFERENCE")
println("#" ^ 70, "\n")
flush(stdout)

"""
Parse RAW SCORES from a reference report text file.
Returns Dict{String, Float64} of score_name => value.
"""
function parse_raw_scores(report_path::String)
    scores = Dict{String, Float64}()
    in_raw = false
    for line in eachline(report_path)
        if contains(line, "RAW SCORES")
            in_raw = true
            continue
        end
        if in_raw && contains(line, ":")
            parts = split(line, ":", limit=2)
            if length(parts) == 2
                key = strip(parts[1])
                val = tryparse(Float64, strip(parts[2]))
                if val !== nothing
                    scores[key] = val
                end
            end
        end
    end
    return scores
end

"""
Try to find a matching reference report for a new report name.
The new name format: {input_name}__{model_name}__seed101__sample_{N}
The reference format: {input_name}__{model_name}__{task_name}__seed_{seed}__predictions__{task_sample}.
We match on input_name + model_name prefix.
"""
function find_reference_match(new_name::String, ref_dir::String)
    isdir(ref_dir) || return nothing
    # Extract input_name and model_name from the new name
    parts = split(new_name, "__")
    length(parts) >= 2 || return nothing
    input_name = parts[1]
    model_name = parts[2]
    prefix = "$(input_name)__$(model_name)__"

    # Look for reference files with matching prefix
    for f in readdir(ref_dir)
        endswith(f, ".txt") || continue
        if startswith(f, prefix)
            return joinpath(ref_dir, f)
        end
    end
    return nothing
end

comparison_path = joinpath(RUN_DIR, "comparison_report.txt")
open(comparison_path, "w") do io
    println(io, "=" ^ 80)
    println(io, "STRUCTURE CHECK COMPARISON REPORT")
    println(io, "Generated: ", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))
    println(io, "New run: $RUN_DIR")
    println(io, "Reference: structure_check_reference/")
    println(io, "=" ^ 80)
    println(io)

    compare_metrics = [
        "overall_issue_score",
        "bond_length_violations",
        "severe_clashes",
        "clashscore_per_1000_atoms",
        "missing_bonds",
    ]

    for (label, sc_dir, ref_dir) in [
        ("CLEAN TARGETS", SC_CLEAN, REF_CLEAN),
        ("STRESS TEST", SC_STRESS, REF_STRESS),
    ]
        println(io, "\n", "─" ^ 80)
        println(io, "  $label")
        println(io, "─" ^ 80)

        new_reports = sort(filter(f -> endswith(f, ".txt"), readdir(sc_dir)))
        matched = 0
        new_only = 0
        regressions = 0
        improvements = 0

        for report_file in new_reports
            new_name = splitext(report_file)[1]
            new_path = joinpath(sc_dir, report_file)
            ref_path = find_reference_match(new_name, ref_dir)

            if ref_path === nothing
                new_only += 1
                continue
            end

            matched += 1
            new_scores = parse_raw_scores(new_path)
            ref_scores = parse_raw_scores(ref_path)

            # Check for significant changes
            has_regression = false
            has_improvement = false
            changes = String[]

            for metric in compare_metrics
                new_val = get(new_scores, metric, NaN)
                ref_val = get(ref_scores, metric, NaN)
                isnan(new_val) && continue
                isnan(ref_val) && continue

                delta = new_val - ref_val
                # Threshold: 10% relative change or absolute change > 0.5
                if abs(delta) > max(0.5, abs(ref_val) * 0.1)
                    direction = delta > 0 ? "WORSE" : "BETTER"
                    push!(changes, "    $metric: $(round(ref_val, digits=2)) → $(round(new_val, digits=2)) ($direction, Δ=$(round(delta, sigdigits=3)))")
                    if delta > 0
                        has_regression = true
                    else
                        has_improvement = true
                    end
                end
            end

            if has_regression
                regressions += 1
                println(io, "\n  REGRESSION: $new_name")
                println(io, "    Reference: $(basename(ref_path))")
                for c in changes
                    println(io, c)
                end
            elseif has_improvement
                improvements += 1
                println(io, "\n  IMPROVED: $new_name")
                println(io, "    Reference: $(basename(ref_path))")
                for c in changes
                    println(io, c)
                end
            end
        end

        println(io, "\n  Summary for $label:")
        println(io, "    New reports: $(length(new_reports))")
        println(io, "    Matched to reference: $matched")
        println(io, "    New (no reference): $new_only")
        println(io, "    Regressions: $regressions")
        println(io, "    Improvements: $improvements")
        println(io, "    Unchanged: $(matched - regressions - improvements)")
    end

    # RBD section — no previous reference for v1.0 / 20250630
    println(io, "\n", "─" ^ 80)
    println(io, "  RBD + GLYCAN + MSA (new — no reference comparison)")
    println(io, "─" ^ 80)
    rbd_reports_files = sort(filter(f -> endswith(f, ".txt"), readdir(SC_RBD)))
    for report_file in rbd_reports_files
        rbd_path = joinpath(SC_RBD, report_file)
        scores = parse_raw_scores(rbd_path)
        issue = get(scores, "overall_issue_score", NaN)
        bonds = get(scores, "bond_length_violations", NaN)
        clashes = get(scores, "severe_clashes", NaN)
        println(io, "  $(splitext(report_file)[1])")
        println(io, "    overall_issue_score=$(round(issue, digits=3))  bond_violations=$(bonds)  severe_clashes=$(clashes)")
    end
end

println("Comparison report written: $comparison_path")
flush(stdout)

# ############################################################
# SUMMARY
# ############################################################
println("\n", "#" ^ 70)
println("# FINAL SUMMARY")
println("#" ^ 70, "\n")

const T_END = Dates.now()
elapsed = Dates.canonicalize(Dates.CompoundPeriod(T_END - T_START))

println("Total elapsed: $elapsed\n")

# Per-tracker summary
function print_tracker_summary(name::String, tracker::RunTracker)
    passed = count(v -> v == :pass, values(tracker.results))
    failed = count(v -> v == :fail, values(tracker.results))
    total  = length(tracker.results)
    println("  $name: $passed/$total passed, $failed failed")

    if failed > 0
        println("    Failed:")
        for key in sort(collect(keys(filter(p -> p.second == :fail, tracker.results))))
            msg = get(tracker.errors, key, "unknown")
            if length(msg) > 120
                msg = msg[1:120] * "..."
            end
            println("      $key: $msg")
        end
    end

    # Per-model breakdown
    model_counts = Dict{String, @NamedTuple{pass::Int, fail::Int}}()
    for (key, status) in tracker.results
        parts = split(key, "__")
        model = length(parts) >= 2 ? parts[2] : "unknown"
        curr = get(model_counts, model, (pass=0, fail=0))
        if status == :pass
            model_counts[model] = (pass=curr.pass + 1, fail=curr.fail)
        else
            model_counts[model] = (pass=curr.pass, fail=curr.fail + 1)
        end
    end
    for model in sort(collect(keys(model_counts)))
        c = model_counts[model]
        println("    $model: $(c.pass) pass, $(c.fail) fail")
    end
    println()
end

print_tracker_summary("Clean Targets", CLEAN_TRACKER)
print_tracker_summary("Stress Test", STRESS_TRACKER)
print_tracker_summary("RBD", RBD_TRACKER)

# CIF counts
println("  CIF counts:")
for (label, dir) in [("clean", CIFS_CLEAN), ("stress", CIFS_STRESS), ("rbd", CIFS_RBD)]
    n = length(filter(f -> endswith(f, ".cif"), readdir(dir)))
    println("    $label: $n CIFs")
end

# Structure check score summary
function score_summary(reports::Dict, label::String)
    if isempty(reports)
        println("  $label: no reports")
        return
    end
    scores = Float64[]
    for (_, r) in reports
        push!(scores, r.scores.overall_issue_score)
    end
    sort!(scores)
    println("  $label structure check scores (overall_issue_score):")
    println("    n=$(length(scores))  min=$(round(minimum(scores), digits=3))  median=$(round(scores[div(length(scores)+1,2)], digits=3))  max=$(round(maximum(scores), digits=3))  mean=$(round(sum(scores)/length(scores), digits=3))")
end

println()
score_summary(clean_reports, "Clean")
score_summary(stress_reports, "Stress")
score_summary(rbd_reports, "RBD")

println("\n  Comparison report: $comparison_path")
println("\n", "=" ^ 70)
println("  FULL RERUN COMPLETE")
println("=" ^ 70)
flush(stdout)
