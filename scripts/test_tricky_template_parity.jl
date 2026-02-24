#!/usr/bin/env julia
"""
Test Julia template feature extraction against Python reference for 10 tricky PDB structures.
"""

using NPZ
using JSON3
using PXDesign

const _extract = PXDesign.ProtenixAPI._extract_template_from_cif
const _derive = PXDesign.ProtenixAPI._derive_template_features!

const TRICKY_DIR = "/tmp/template_parity/tricky"
const CIF_DIR = joinpath(@__DIR__, "..", "clean_targets", "structures", "tricky_pdbs")

const PDBS = [
    ("1igy", "B", "Insertion codes (antibody Kabat numbering)"),
    ("2ace", "A", "Missing internal residues (disordered loop)"),
    ("3ckr", "A", "Negative residue numbers (purification tag)"),
    ("1a8o", "A", "Selenomethionine (MSE → MET)"),
    ("2beg", "A", "NMR multi-model, N-terminal truncation"),
    ("5awl", "A", "Very short chain (10 residues)"),
    ("3nir", "A", "Alternate conformations (ultra-high-res crambin)"),
    ("1en2", "A", "Point-mutation disorder (SER/CYS at same position)"),
    ("1bkx", "A", "Phosphorylated residues (SEP, TPO)"),
    ("4hhb", "A", "auth_seq_num vs label_seq_id (hemoglobin)"),
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

    meta_path = joinpath(TRICKY_DIR, "$(pdb_id)_meta.json")
    npz_path = joinpath(TRICKY_DIR, "$(pdb_id)_template_features.npz")
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
    println("═══ Tricky PDB Template Parity Test (10 structures) ═══")

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
