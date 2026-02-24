#!/usr/bin/env julia
"""
Round 2: Test Julia template feature extraction against Python reference for 20 PDB structures.
10 random/common + 10 intentionally tricky.
"""

using NPZ
using JSON3
using PXDesign

const _extract = PXDesign.ProtenixAPI._extract_template_from_cif
const _derive = PXDesign.ProtenixAPI._derive_template_features!

const ROUND2_DIR = "/tmp/template_parity/round2"
const CIF_DIR = joinpath(@__DIR__, "..", "clean_targets", "structures", "round2_pdbs")

const PDBS = [
    # Random (10)
    ("1l2y", "A", "Trp-cage miniprotein (NMR, 20 res)"),
    ("1mbn", "A", "Myoglobin (classic, 153 res)"),
    ("1ema", "A", "GFP (beta-barrel, 236 res)"),
    ("4pti", "A", "BPTI (small, 58 res, disulfides)"),
    ("2cba", "A", "Carbonic anhydrase II (260 res)"),
    ("3lyz", "A", "Lysozyme (129 res)"),
    ("2rns", "A", "Ribonuclease S (124 res)"),
    ("7acn", "A", "Aconitase (754 res, large)"),
    ("1hho", "A", "Hemoglobin oxy (141 res)"),
    ("1ten", "A", "Tenascin FN3 domain (90 res)"),
    # Tricky (10)
    ("1aoi", "A", "Nucleosome protein+DNA complex"),
    ("4rcn", "A", "Long chain (1093 res)"),
    ("1htq", "A", "Extensive missing density (477 res)"),
    ("1mag", "A", "D-amino acids gramicidin (16 res)"),
    ("3k0n", "A", "Multiple conformations throughout (165 res)"),
    ("1m17", "A", "auth_seq_id large offset EGFR (333 res)"),
    ("132l", "A", "Non-standard MLY residues (129 res)"),
    ("1nsa", "A", "Insertion codes throughout (395 res)"),
    ("3evp", "A", "Circular permutant GCaMP2 (243 res)"),
    ("4rxn", "A", "Very old PDB entry rubredoxin (54 res)"),
]

function compare_arrays(name::String, julia_arr, python_arr; atol=1e-5, rtol=1e-4)
    if size(julia_arr) != size(python_arr)
        println("    FAIL $name: shape Julia=$(size(julia_arr)) vs Python=$(size(python_arr))")
        return false
    end

    if eltype(julia_arr) <: Integer && eltype(python_arr) <: Integer
        n_diff = sum(julia_arr .!= python_arr)
        if n_diff > 0
            println("    FAIL $name: $n_diff / $(length(julia_arr)) elements differ")
            diffs = findall(julia_arr .!= python_arr)
            for idx in diffs[1:min(3, length(diffs))]
                println("      [$idx]: Julia=$(julia_arr[idx]) Python=$(python_arr[idx])")
            end
            return false
        end
        println("    PASS $name ($(length(julia_arr)) elements, exact)")
        return true
    end

    max_abs = maximum(abs.(Float64.(julia_arr) .- Float64.(python_arr)))
    nonzero = python_arr .!= 0
    max_rel = any(nonzero) ?
        maximum(abs.(Float64.(julia_arr[nonzero]) .- Float64.(python_arr[nonzero])) ./ abs.(Float64.(python_arr[nonzero]))) : 0.0
    n_nonzero_julia = sum(julia_arr .!= 0)
    n_nonzero_python = sum(python_arr .!= 0)

    ok = max_abs < atol || max_rel < rtol
    status = ok ? "PASS" : "FAIL"
    println("    $status $name: max_abs=$(round(max_abs; sigdigits=3)) " *
            "nnz=$(n_nonzero_julia)/$(n_nonzero_python)")
    return ok
end

function run_test(pdb_id, chain_id, description)
    println("\n═══ $pdb_id chain $chain_id: $description ═══")

    meta_path = joinpath(ROUND2_DIR, "$(pdb_id)_meta.json")
    npz_path = joinpath(ROUND2_DIR, "$(pdb_id)_template_features.npz")
    cif_path = joinpath(CIF_DIR, "$(pdb_id).cif")

    if !isfile(npz_path)
        println("  SKIP: Python reference not found")
        return true
    end

    meta = JSON3.read(read(meta_path, String))
    query_seq = meta[:query_sequence]
    println("  Query: $(length(query_seq)) residues")

    # Run Julia extraction
    local julia_raw
    try
        julia_raw = _extract(query_seq, cif_path, chain_id)
    catch e
        println("  FAIL: Julia extraction error: $e")
        return false
    end

    info = julia_raw["_alignment_info"]
    println("  Julia: $(info.n_aligned) aligned, $(info.n_identical) identical")

    # Load Python reference
    py_feats = npzread(npz_path)

    # Build derived features
    julia_feat = Dict{String, Any}(
        "template_restype" => julia_raw["template_restype"],
        "template_all_atom_mask" => julia_raw["template_all_atom_mask"],
        "template_all_atom_positions" => julia_raw["template_all_atom_positions"],
    )
    _derive(julia_feat)

    all_pass = true

    # Compare raw features
    if haskey(py_feats, "template_aatype")
        all_pass &= compare_arrays("aatype", julia_feat["template_restype"], py_feats["template_aatype"])
    end
    if haskey(py_feats, "template_atom_mask")
        all_pass &= compare_arrays("atom_mask", julia_feat["template_all_atom_mask"], py_feats["template_atom_mask"])
    end
    if haskey(py_feats, "template_atom_positions")
        all_pass &= compare_arrays("atom_positions", julia_feat["template_all_atom_positions"], py_feats["template_atom_positions"])
    end

    # Compare derived features
    for key in ("template_distogram", "template_pseudo_beta_mask", "template_unit_vector", "template_backbone_frame_mask")
        if haskey(py_feats, key) && haskey(julia_feat, key)
            all_pass &= compare_arrays(key, julia_feat[key], py_feats[key])
        end
    end

    return all_pass
end

function main()
    println("═══ Round 2: Template Parity Test (20 structures) ═══")
    println("  10 random/common + 10 intentionally tricky\n")

    n_pass = 0
    n_fail = 0
    failed = String[]

    for (pdb_id, chain_id, desc) in PDBS
        ok = run_test(pdb_id, chain_id, desc)
        if ok
            n_pass += 1
        else
            n_fail += 1
            push!(failed, pdb_id)
        end
    end

    println("\n\n═══ RESULTS: $n_pass/$(n_pass + n_fail) passed ═══")
    if n_fail > 0
        println("Failed: $(join(failed, ", "))")
    end
    return n_fail == 0
end

success = main()
exit(success ? 0 : 1)
