#!/usr/bin/env julia
#
# Generate structure check reference reports for all existing CIF outputs.
#
# Usage: julia --project=<env> clean_targets/scripts/generate_structure_check_reference.jl
#
# Runs ProtInterop.StructureChecking.check_structure on every CIF file in:
#   - clean_targets/julia_outputs/
#   - clean_targets/stress_outputs/
#   - clean_targets/stress_cif_results/
#
# Outputs reference reports to:
#   - clean_targets/structure_check_reference/clean_targets/
#   - clean_targets/structure_check_reference/stress/

using ProtInterop
const SC = ProtInterop.StructureChecking

const ROOT = joinpath(@__DIR__, "..", "..")
const CLEAN = joinpath(ROOT, "clean_targets")

const REF_DIR = joinpath(CLEAN, "structure_check_reference")

# Output subdirectories
const REF_CLEAN = joinpath(REF_DIR, "clean_targets")
const REF_STRESS = joinpath(REF_DIR, "stress")

function collect_cifs(dir::String)
    paths = String[]
    isdir(dir) || return paths
    for (d, _, files) in walkdir(dir)
        for f in files
            endswith(f, ".cif") && push!(paths, joinpath(d, f))
        end
    end
    sort!(paths)
end

function relative_key(cif_path::String, base_dir::String)
    # Create a unique key from the relative path, replacing / with __
    rel = relpath(cif_path, base_dir)
    # Replace path separators and .cif extension
    key = replace(rel, "/" => "__", "\\" => "__")
    key = replace(key, ".cif" => "")
    return key
end

function generate_reports(source_dir::String, output_dir::String, label::String)
    cifs = collect_cifs(source_dir)
    if isempty(cifs)
        println("  No CIF files found in $source_dir")
        return 0, 0
    end

    mkpath(output_dir)
    n_ok = 0
    n_fail = 0

    for cif in cifs
        key = relative_key(cif, source_dir)
        report_path = joinpath(output_dir, key * ".txt")

        try
            report = SC.check_structure(cif)
            open(report_path, "w") do io
                SC.print_issue_report(report; io=io)
                # Also print the raw scores for machine-readable comparison
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
            n_ok += 1
            println("  ✓ $key  (score: $(round(report.scores.overall_issue_score, digits=2)))")
        catch e
            n_fail += 1
            println("  ✗ $key  ERROR: $e")
            # Still write the error to file for record
            open(report_path, "w") do io
                println(io, "ERROR processing $cif")
                println(io, e)
            end
        end
    end

    return n_ok, n_fail
end

function main()
    println("=" ^ 80)
    println("  STRUCTURE CHECK REFERENCE GENERATION")
    println("=" ^ 80)
    println()

    # Wipe any existing reference (fresh start)
    if isdir(REF_DIR)
        println("Removing existing reference directory: $REF_DIR")
        rm(REF_DIR; force=true, recursive=true)
    end

    total_ok = 0
    total_fail = 0

    # 1. Clean targets (julia_outputs)
    println("\n── Clean Targets (julia_outputs) ──────────────────────────────────────")
    ok, fail = generate_reports(joinpath(CLEAN, "julia_outputs"), REF_CLEAN, "clean")
    total_ok += ok
    total_fail += fail

    # 2. Stress test outputs (stress_cif_results is a flat copy of these, skip it)
    println("\n── Stress Outputs (stress_outputs) ────────────────────────────────────")
    ok, fail = generate_reports(joinpath(CLEAN, "stress_outputs"), REF_STRESS, "stress")
    total_ok += ok
    total_fail += fail

    println("\n" * "=" ^ 80)
    println("TOTAL: $total_ok reports generated, $total_fail errors")
    println("Reference stored in: $REF_DIR")
    println("=" ^ 80)
end

main()
