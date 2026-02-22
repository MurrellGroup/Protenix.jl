#!/usr/bin/env julia
#
# Dump features for key targets into a clean directory.
#
# Usage: julia --project=<env> clean_targets/scripts/dump_all_features.jl

using MoleculeFlow
using PXDesign
using Random

const ROOT = joinpath(@__DIR__, "..", "..")
const INPUTS = joinpath(ROOT, "clean_targets", "inputs")

# Include the dump functions (but not the ARGS-based main)
include(joinpath(@__DIR__, "dump_julia_features.jl"))

const DUMP_BASE = joinpath(ROOT, "clean_targets", "feature_dumps", "julia_final")
rm(DUMP_BASE; force=true, recursive=true)

# Key targets to dump
targets = [
    ("04", "protein_dna"),
    ("06", "protein_ligand_ccd"),
    ("07", "protein_ligand_smiles"),
    ("21", "ligand_file_sdf"),
]

for (num, name) in targets
    json_path = joinpath(INPUTS, "$(num)_$(name).json")
    dump_dir = joinpath(DUMP_BASE, name)
    println("\n>>> Dumping features for $name...")
    dump_features(json_path, dump_dir)
end

println("\n>>> All feature dumps complete: $DUMP_BASE")
