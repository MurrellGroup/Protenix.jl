#!/usr/bin/env julia
"""
Test Julia template feature extraction against Python reference .npz files.

Compares the output of _extract_template_from_cif (Julia) against
dump_template_features.py (Python) for 3 test cases.
"""

using NPZ
using PXDesign

# Access internal functions
const _extract = PXDesign.ProtenixAPI._extract_template_from_cif
const _derive = PXDesign.ProtenixAPI._derive_template_features!

struct TestCase
    name::String
    query_sequence::String
    cif_path::String
    chain_id::String
    python_npz::String
    python_raw_npz::String
end

const STRUCTURES_DIR = joinpath(@__DIR__, "..", "clean_targets", "structures")
const PARITY_DIR = "/tmp/template_parity"

const TEST_CASES = [
    TestCase(
        "ubq_self_template",
        "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
        joinpath(STRUCTURES_DIR, "1ubq.cif"),
        "A",
        joinpath(PARITY_DIR, "ubq_self_template_template_features.npz"),
        joinpath(PARITY_DIR, "ubq_self_template_template_raw37.npz"),
    ),
    TestCase(
        "hemo_ubq_cross_template",
        "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
        joinpath(STRUCTURES_DIR, "1ubq.cif"),
        "A",
        joinpath(PARITY_DIR, "hemo_ubq_cross_template_template_features.npz"),
        joinpath(PARITY_DIR, "hemo_ubq_cross_template_template_raw37.npz"),
    ),
    TestCase(
        "barnase_partial_template",
        "AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGY",
        joinpath(STRUCTURES_DIR, "1brs.cif"),
        "A",
        joinpath(PARITY_DIR, "barnase_partial_template_template_features.npz"),
        joinpath(PARITY_DIR, "barnase_partial_template_template_raw37.npz"),
    ),
]

function compare_arrays(name::String, julia_arr, python_arr; atol=1e-5, rtol=1e-4)
    if size(julia_arr) != size(python_arr)
        println("  FAIL $name: shape mismatch Julia=$(size(julia_arr)) vs Python=$(size(python_arr))")
        return false
    end

    # For integer arrays
    if eltype(julia_arr) <: Integer && eltype(python_arr) <: Integer
        n_diff = sum(julia_arr .!= python_arr)
        if n_diff > 0
            println("  FAIL $name: $n_diff / $(length(julia_arr)) elements differ")
            # Show first few differences
            diffs = findall(julia_arr .!= python_arr)
            for idx in diffs[1:min(5, length(diffs))]
                println("    [$idx]: Julia=$(julia_arr[idx]) Python=$(python_arr[idx])")
            end
            return false
        end
        println("  PASS $name: exact match ($(length(julia_arr)) elements)")
        return true
    end

    # For float arrays
    max_abs = maximum(abs.(Float64.(julia_arr) .- Float64.(python_arr)))
    nonzero = python_arr .!= 0
    max_rel = if any(nonzero)
        maximum(abs.(Float64.(julia_arr[nonzero]) .- Float64.(python_arr[nonzero])) ./ abs.(Float64.(python_arr[nonzero])))
    else
        0.0
    end
    n_total = length(julia_arr)
    n_nonzero_julia = sum(julia_arr .!= 0)
    n_nonzero_python = sum(python_arr .!= 0)

    ok = max_abs < atol || max_rel < rtol
    status = ok ? "PASS" : "FAIL"
    println("  $status $name: max_abs=$(round(max_abs; sigdigits=4)) max_rel=$(round(max_rel; sigdigits=4)) " *
            "nnz_jl=$n_nonzero_julia nnz_py=$n_nonzero_python / $n_total")
    return ok
end

function run_test(tc::TestCase)
    println("\n═══ $(tc.name) ═══")
    println("  Query: $(length(tc.query_sequence)) residues")
    println("  Template: $(tc.cif_path) chain $(tc.chain_id)")

    # Run Julia extraction
    julia_raw = _extract(tc.query_sequence, tc.cif_path, tc.chain_id)
    info = julia_raw["_alignment_info"]
    println("  Julia alignment: $(info.n_aligned) aligned, $(info.n_identical) identical")

    # Load Python reference (derived features)
    py_feats = npzread(tc.python_npz)

    # Build derived features from Julia raw
    julia_feat = Dict{String, Any}(
        "template_restype" => julia_raw["template_restype"],
        "template_all_atom_mask" => julia_raw["template_all_atom_mask"],
        "template_all_atom_positions" => julia_raw["template_all_atom_positions"],
    )
    _derive(julia_feat)

    # Compare all features
    all_pass = true
    println("\n  --- Raw features ---")
    # Python keys use "template_aatype" while Julia uses "template_restype"
    if haskey(py_feats, "template_aatype")
        all_pass &= compare_arrays("template_aatype/restype", julia_feat["template_restype"], py_feats["template_aatype"])
    end
    if haskey(py_feats, "template_atom_mask")
        all_pass &= compare_arrays("template_atom_mask", julia_feat["template_all_atom_mask"], py_feats["template_atom_mask"])
    end
    if haskey(py_feats, "template_atom_positions")
        all_pass &= compare_arrays("template_atom_positions", julia_feat["template_all_atom_positions"], py_feats["template_atom_positions"])
    end

    println("\n  --- Derived features ---")
    if haskey(py_feats, "template_distogram")
        all_pass &= compare_arrays("template_distogram", julia_feat["template_distogram"], py_feats["template_distogram"])
    end
    if haskey(py_feats, "template_pseudo_beta_mask")
        all_pass &= compare_arrays("template_pseudo_beta_mask", julia_feat["template_pseudo_beta_mask"], py_feats["template_pseudo_beta_mask"])
    end
    if haskey(py_feats, "template_unit_vector")
        all_pass &= compare_arrays("template_unit_vector", julia_feat["template_unit_vector"], py_feats["template_unit_vector"])
    end
    if haskey(py_feats, "template_backbone_frame_mask")
        all_pass &= compare_arrays("template_backbone_frame_mask", julia_feat["template_backbone_frame_mask"], py_feats["template_backbone_frame_mask"])
    end

    # Also compare raw 37-atom data if available
    if isfile(tc.python_raw_npz)
        println("\n  --- Raw atom37 comparison ---")
        py_raw = npzread(tc.python_raw_npz)
        if haskey(py_raw, "template_aatype_raw")
            # Compare the 1D aatype before batch dimension was added
            jl_aatype_1d = vec(julia_feat["template_restype"])
            py_aatype_1d = vec(py_raw["template_aatype_raw"])
            all_pass &= compare_arrays("raw_aatype", jl_aatype_1d, py_aatype_1d)
        end
    end

    return all_pass
end

function main()
    println("═══ Template Feature Parity Test ═══")
    println("Comparing Julia _extract_template_from_cif + _derive_template_features!")
    println("against Python dump_template_features.py outputs")

    all_pass = true
    for tc in TEST_CASES
        if !isfile(tc.python_npz)
            println("\n  SKIP $(tc.name): Python reference not found at $(tc.python_npz)")
            continue
        end
        if !isfile(tc.cif_path)
            println("\n  SKIP $(tc.name): CIF file not found at $(tc.cif_path)")
            continue
        end
        all_pass &= run_test(tc)
    end

    println("\n\n═══ $(all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED") ═══")
    return all_pass
end

success = main()
exit(success ? 0 : 1)
