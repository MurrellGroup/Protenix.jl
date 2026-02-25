# ── RBD + MSA + Glycan via REPL API ──────────────────────────────────────────
#
# Folds the SARS-CoV-2 receptor binding domain (RBD) with:
#   - precomputed MSA (pairing + non-pairing alignments)
#   - two N-linked glycan trees (Asn13 and Asn25)
#   - covalent bonds wiring the glycans to the protein
#
# This is the programmatic equivalent of clean_targets/inputs/33_rbd_glycosylated.json,
# but built entirely from Julia without touching a JSON file.
#
# Usage:
#   julia --project=Protenix.jl examples/rbd_glycosylated_repl.jl

using Protenix

# ── 1. RBD sequence (SARS-CoV-2, residues 333–526) ──────────────────────────
const RBD_SEQ = "RVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFK" *
                "CYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNS" *
                "NNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQ" *
                "PTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF"

# ── 2. MSA directory (absolute path so it resolves from any pwd) ─────────────
const MSA_DIR = abspath(joinpath(@__DIR__, "msa", "rbd_glyco"))

# ── 3. Build entities ────────────────────────────────────────────────────────
#
# Entity 1: RBD protein with precomputed MSA
rbd = protein_chain(RBD_SEQ;
    msa = Dict(
        "precomputed_msa_dir" => MSA_DIR,
        "pairing_db"          => "uniref100",
    ),
)

# Entities 2–4: first N-glycan tree (attached to Asn13 = N331 in full spike)
glycan_tree_1_core   = ligand("CCD_NAG_NAG_BMA_MAN_NAG_GAL")  # entity 2
glycan_tree_1_branch = ligand("CCD_MAN_NAG_GAL")               # entity 3
glycan_tree_1_fuc    = ligand("CCD_FUC")                        # entity 4

# Entities 5–7: second N-glycan tree (attached to Asn25 = N343 in full spike)
glycan_tree_2_core   = ligand("CCD_NAG_NAG_BMA_MAN_NAG_GAL")  # entity 5
glycan_tree_2_branch = ligand("CCD_MAN_NAG_GAL")               # entity 6
glycan_tree_2_fuc    = ligand("CCD_FUC")                        # entity 7

# ── 4. Covalent bonds ───────────────────────────────────────────────────────
#
# Two N-linked glycosylation sites, each with a branched sugar tree.
# entity/position/atom triplets follow the Protenix JSON convention:
#   entity = 1-based index into the sequences array
#   position = 1-based residue/sugar index within that entity
#   atom = PDB atom name
#
covalent_bonds = [
    # ── Glycan tree 1 (Asn13 → entity 2 core) ──
    Dict("entity1"=>"1", "position1"=>"13",  "atom1"=>"ND2", "entity2"=>"2", "position2"=>"1", "atom2"=>"C1"),
    # intra-core linkages (entity 2 internal)
    Dict("entity1"=>"2", "position1"=>"2",   "atom1"=>"C1",  "entity2"=>"2", "position2"=>"1", "atom2"=>"O4"),
    Dict("entity1"=>"2", "position1"=>"3",   "atom1"=>"C1",  "entity2"=>"2", "position2"=>"2", "atom2"=>"O4"),
    Dict("entity1"=>"2", "position1"=>"4",   "atom1"=>"C1",  "entity2"=>"2", "position2"=>"3", "atom2"=>"O3"),
    Dict("entity1"=>"2", "position1"=>"5",   "atom1"=>"C1",  "entity2"=>"2", "position2"=>"4", "atom2"=>"O2"),
    Dict("entity1"=>"2", "position1"=>"6",   "atom1"=>"C1",  "entity2"=>"2", "position2"=>"5", "atom2"=>"O4"),
    # branch (entity 3) attaches to core (entity 2)
    Dict("entity1"=>"3", "position1"=>"1",   "atom1"=>"C1",  "entity2"=>"2", "position2"=>"3", "atom2"=>"O6"),
    Dict("entity1"=>"3", "position1"=>"2",   "atom1"=>"C1",  "entity2"=>"3", "position2"=>"1", "atom2"=>"O2"),
    Dict("entity1"=>"3", "position1"=>"3",   "atom1"=>"C1",  "entity2"=>"3", "position2"=>"2", "atom2"=>"O4"),
    # fucose (entity 4) attaches to core (entity 2)
    Dict("entity1"=>"4", "position1"=>"1",   "atom1"=>"C1",  "entity2"=>"2", "position2"=>"1", "atom2"=>"O6"),

    # ── Glycan tree 2 (Asn25 → entity 5 core) ──
    Dict("entity1"=>"1", "position1"=>"25",  "atom1"=>"ND2", "entity2"=>"5", "position2"=>"1", "atom2"=>"C1"),
    # intra-core linkages (entity 5 internal)
    Dict("entity1"=>"5", "position1"=>"2",   "atom1"=>"C1",  "entity2"=>"5", "position2"=>"1", "atom2"=>"O4"),
    Dict("entity1"=>"5", "position1"=>"3",   "atom1"=>"C1",  "entity2"=>"5", "position2"=>"2", "atom2"=>"O4"),
    Dict("entity1"=>"5", "position1"=>"4",   "atom1"=>"C1",  "entity2"=>"5", "position2"=>"3", "atom2"=>"O3"),
    Dict("entity1"=>"5", "position1"=>"5",   "atom1"=>"C1",  "entity2"=>"5", "position2"=>"4", "atom2"=>"O2"),
    Dict("entity1"=>"5", "position1"=>"6",   "atom1"=>"C1",  "entity2"=>"5", "position2"=>"5", "atom2"=>"O4"),
    # branch (entity 6) attaches to core (entity 5)
    Dict("entity1"=>"6", "position1"=>"1",   "atom1"=>"C1",  "entity2"=>"5", "position2"=>"3", "atom2"=>"O6"),
    Dict("entity1"=>"6", "position1"=>"2",   "atom1"=>"C1",  "entity2"=>"6", "position2"=>"1", "atom2"=>"O2"),
    Dict("entity1"=>"6", "position1"=>"3",   "atom1"=>"C1",  "entity2"=>"6", "position2"=>"2", "atom2"=>"O4"),
    # fucose (entity 7) attaches to core (entity 5)
    Dict("entity1"=>"7", "position1"=>"1",   "atom1"=>"C1",  "entity2"=>"5", "position2"=>"1", "atom2"=>"O6"),
]

# ── 5. Assemble the task ────────────────────────────────────────────────────
task = protenix_task(
    rbd,
    glycan_tree_1_core, glycan_tree_1_branch, glycan_tree_1_fuc,
    glycan_tree_2_core, glycan_tree_2_branch, glycan_tree_2_fuc;
    name = "sars_cov2_rbd_glycosylated",
    covalent_bonds = covalent_bonds,
)

# ── 6. Load model & fold ────────────────────────────────────────────────────
println("Loading protenix_base_default_v0.5.0 …")
h = load_protenix("protenix_base_default_v0.5.0"; gpu = true)

out_dir = joinpath(@__DIR__, "..", "e2e_output", "rbd_glyco_repl")
println("Folding RBD + glycans (seed 101) …")
result = fold(h, task; seed = 101, out_dir = out_dir)

# ── 7. Report ────────────────────────────────────────────────────────────────
println()
println("═══ Results ═══")
println("  mean pLDDT : ", round(result.mean_plddt; digits = 1))
println("  mean PAE   : ", round(result.mean_pae; digits = 1))
println("  seed       : ", result.seed)
println("  task       : ", result.task_name)
println("  CIF files  : ")
for p in result.cif_paths
    println("    ", p)
end
println("  output dir : ", result.prediction_dir)
