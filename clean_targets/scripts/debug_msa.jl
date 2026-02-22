#!/usr/bin/env julia
# Debug MSA feature building for target 17.
# Usage: julia --project=<env> clean_targets/scripts/debug_msa.jl

using PXDesign
using Statistics

const PA = PXDesign.ProtenixAPI  # private API access

const ROOT = joinpath(@__DIR__, "..", "..")
const JSON_PATH = joinpath(ROOT, "clean_targets", "inputs", "17_protein_msa.json")

println("Loading JSON: $JSON_PATH")
isfile(JSON_PATH) || error("File not found: $JSON_PATH")

# Parse the JSON tasks
tasks_json = PA._ensure_json_tasks(JSON_PATH)
task = PA._as_string_dict(tasks_json[1])

# Get chain specs
chain_specs = PA._extract_protein_chain_specs(task)
println("Number of protein chains: ", length(chain_specs))
for (i, spec) in enumerate(chain_specs)
    println("  Chain $i: id=$(spec.chain_id), seq length=$(length(spec.sequence))")
    msa_dir = spec.msa_cfg.precomputed_msa_dir
    println("    MSA dir: $msa_dir")
end

# Build chain MSA features directly
spec = chain_specs[1]
msa_dir_raw = String(spec.msa_cfg.precomputed_msa_dir)
json_dir = dirname(abspath(JSON_PATH))
msa_dir = isabspath(msa_dir_raw) ? msa_dir_raw : normpath(joinpath(json_dir, msa_dir_raw))
println("  Resolved MSA dir: $msa_dir")
println("  MSA dir exists: ", isdir(msa_dir))

non_pair_path = joinpath(msa_dir, "non_pairing.a3m")
println("  non_pairing.a3m exists: ", isfile(non_pair_path))

if isfile(non_pair_path)
    # Parse A3M
    seqs, descs = PA._parse_a3m(non_pair_path; seq_limit=-1)
    println("  Parsed $(length(seqs)) sequences from A3M")
    println("  Query length: $(length(seqs[1]))")

    # Check query sequence
    query = seqs[1]
    println("  Query first 40 chars: ", query[1:min(40, length(query))])

    # Check for gap chars in query
    gap_in_query = count(==('-'), query)
    println("  Gaps in query: $gap_in_query")

    # Get aligned + deletion
    aln, del = PA._aligned_and_deletions_from_a3m(seqs)
    println("  Aligned rows: $(length(aln)), length: $(length(aln[1]))")

    # Build chain MSA features
    block = PA._build_chain_msa_features(spec.sequence, aln, del)
    println("\n--- ChainMSABlock ---")
    println("  msa shape: ", size(block.msa), " (N_msa, N_tok)")
    println("  msa unique values: ", sort(unique(block.msa)))
    println("  msa value counts:")
    for v in sort(unique(block.msa))
        c = count(==(v), block.msa)
        pct = round(100.0 * c / length(block.msa); digits=1)
        println("    value $v: $c ($pct%)")
    end

    # Gap statistics
    n_msa_rows = size(block.msa, 1)
    n_tok = size(block.msa, 2)
    println("\n  Per-row gap fractions:")
    for i in 1:min(n_msa_rows, 5)
        row_gaps = count(==(21), @view block.msa[i, :])
        println("    Row $i: $(round(100.0 * row_gaps / n_tok; digits=1))% gaps ($row_gaps / $n_tok)")
    end
    if n_msa_rows > 10
        println("    ...")
        for i in (n_msa_rows - 2):n_msa_rows
            row_gaps = count(==(21), @view block.msa[i, :])
            println("    Row $i: $(round(100.0 * row_gaps / n_tok; digits=1))% gaps ($row_gaps / $n_tok)")
        end
    end

    # Profile
    println("\n  Profile shape: ", size(block.profile), " (N_tok, 32)")
    println("  Profile sum per token range: ", extrema(sum(block.profile; dims=2)))
    nonzero_cols = findall(vec(sum(abs, block.profile; dims=1)) .> 0)
    println("  Non-zero profile columns (1-based): ", nonzero_cols)

    # Average profile across tokens
    avg_prof = vec(mean(block.profile; dims=1))
    println("  Average profile (non-zero cols):")
    for col in nonzero_cols
        println("    Col $col: $(round(avg_prof[col]; digits=4))")
    end

    # Deletion features
    println("\n  has_deletion shape: ", size(block.has_deletion))
    println("  has_deletion mean: ", round(mean(block.has_deletion); digits=4))
    println("  deletion_value range: ", extrema(block.deletion_value))
    println("  deletion_mean range: ", extrema(block.deletion_mean))
end

# Also check what Python's MAP_HHBLITS_AATYPE_TO_OUR_AATYPE produces
println("\n--- _HHBLITS_TO_PROTENIX mapping ---")
mapping = PA._HHBLITS_TO_PROTENIX
id_to_aa = PA._HHBLITS_ID_TO_AA
for i in eachindex(mapping)
    println("  HHblits[$i] = '$(id_to_aa[i])' → Protenix $(mapping[i])")
end

# Compare with Python reference
println("\n--- Python MAP_HHBLITS_AATYPE_TO_OUR_AATYPE (expected) ---")
# Python restypes: A=0,R=1,N=2,D=3,C=4,Q=5,E=6,G=7,H=8,I=9,L=10,K=11,M=12,F=13,P=14,S=15,T=16,W=17,Y=18,V=19,X=20,gap=21
python_expected = Dict(
    'A' => 0, 'C' => 4, 'D' => 3, 'E' => 6, 'F' => 13,
    'G' => 7, 'H' => 8, 'I' => 9, 'K' => 11, 'L' => 10,
    'M' => 12, 'N' => 2, 'P' => 14, 'Q' => 5, 'R' => 1,
    'S' => 15, 'T' => 16, 'V' => 19, 'W' => 17, 'Y' => 18,
    'X' => 20, '-' => 21,
)
for i in eachindex(id_to_aa)
    aa = id_to_aa[i]
    expected = python_expected[aa]
    actual = mapping[i]
    status = expected == actual ? "OK" : "MISMATCH"
    if status == "MISMATCH"
        println("  *** $status: HHblits[$i]='$aa' expected=$expected got=$actual ***")
    end
end
println("  All mappings match!" * (all(python_expected[id_to_aa[i]] == mapping[i] for i in eachindex(id_to_aa)) ? " ✓" : " ERRORS FOUND"))

println("\nDone.")
