#!/usr/bin/env python3
"""Generate 100 stress test JSON files for PXDesign.jl."""

import json
import os

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stress_inputs")
os.makedirs(OUTDIR, exist_ok=True)


def write_json(name, task):
    path = os.path.join(OUTDIR, f"{name}.json")
    with open(path, 'w') as f:
        json.dump([task], f, indent=2)
    print(f"  {name}.json")


def protein(seq="MAGSTYLK", count=1, modifications=None):
    d = {"sequence": seq, "count": count}
    if modifications:
        d["modifications"] = modifications
    return {"proteinChain": d}


def ccd_ligand(code, count=1):
    return {"ligand": {"ligand": f"CCD_{code}", "count": count}}


def smiles_ligand(smi, count=1):
    return {"ligand": {"ligand": smi, "count": count}}


def dna(seq, count=1, modifications=None):
    d = {"sequence": seq, "count": count}
    if modifications:
        d["modifications"] = modifications
    return {"dnaSequence": d}


def rna(seq, count=1, modifications=None):
    d = {"sequence": seq, "count": count}
    if modifications:
        d["modifications"] = modifications
    return {"rnaSequence": d}


def ion(symbol, count=1):
    return {"ion": {"ion": symbol, "count": count}}


def cov_bond(e1, p1, a1, e2, p2, a2):
    return {
        "entity1": str(e1), "position1": str(p1), "atom1": a1,
        "entity2": str(e2), "position2": str(p2), "atom2": a2,
    }


print("Generating stress test inputs...")

# ================================================================
# A. CCD Ligands (25 cases) — protein(8AA) + CCD ligand
# ================================================================
ccd_cases = [
    ("s001_ccd_atp", "ATP"),
    ("s002_ccd_gtp", "GTP"),
    ("s003_ccd_adp", "ADP"),
    ("s004_ccd_nad", "NAD"),
    ("s005_ccd_fad", "FAD"),
    ("s006_ccd_hem", "HEM"),
    ("s007_ccd_fmn", "FMN"),
    ("s008_ccd_plp", "PLP"),
    ("s009_ccd_coa", "COA"),
    ("s010_ccd_sam", "SAM"),
    ("s011_ccd_btn", "BTN"),
    ("s012_ccd_tpp", "TPP"),
    ("s013_ccd_sti", "STI"),
    ("s014_ccd_4ht", "4HT"),
    ("s015_ccd_ret", "RET"),
    ("s016_ccd_nag", "NAG"),
    ("s017_ccd_bma", "BMA"),
    ("s018_ccd_gal", "GAL"),
    ("s019_ccd_fuc", "FUC"),
    ("s020_ccd_sf4", "SF4"),
    ("s021_ccd_fes", "FES"),
    ("s022_ccd_gol", "GOL"),
    ("s023_ccd_so4", "SO4"),
    ("s024_ccd_cit", "CIT"),
    ("s025_ccd_asa", "ASA"),
]
for name, code in ccd_cases:
    write_json(name, {"name": name, "sequences": [protein(), ccd_ligand(code)]})

# ================================================================
# B. SMILES Ligands (15 cases) — protein(8AA) + SMILES string
# ================================================================
smiles_cases = [
    ("s026_smi_benzene", "c1ccccc1"),
    ("s027_smi_caffeine", "Cn1cnc2c1c(=O)[nH]c(=O)n2C"),
    ("s028_smi_aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("s029_smi_ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
    ("s030_smi_glucose", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"),
    ("s031_smi_cholesterol", "C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C"),
    ("s032_smi_ethanol", "CCO"),
    ("s033_smi_acetone", "CC(=O)C"),
    ("s034_smi_pyridine", "c1ccncc1"),
    ("s035_smi_indole", "c1ccc2c(c1)[nH]cc2"),
    ("s036_smi_naphthalene", "c1ccc2ccccc2c1"),
    ("s037_smi_morphine", "CN1CC[C@]23c4c5ccc(c4O[C@H]2C(=C[C@@H]31)C=C5)O"),
    ("s038_smi_penicillin_g", "CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@@H]1C(=O)O"),
    ("s039_smi_metformin", "CN(C)C(=N)NC(=N)N"),
    ("s040_smi_dopamine", "NCCc1ccc(O)c(O)c1"),
]
for name, smi in smiles_cases:
    write_json(name, {"name": name, "sequences": [protein(), smiles_ligand(smi)]})

# ================================================================
# C. PTMs / Modified Residues (15 cases)
# ================================================================
# (name, sequence, ptm_ccd_code, 1-indexed position of target residue)
ptm_cases = [
    ("s041_ptm_sep", "MAGSTYLK", "SEP", 4),   # S at pos 4
    ("s042_ptm_tpo", "MAGTTYLK", "TPO", 4),   # T at pos 4
    ("s043_ptm_ptr", "MAGATYLK", "PTR", 6),   # Y at pos 6
    ("s044_ptm_mse", "MAMGTYLK", "MSE", 3),   # M at pos 3
    ("s045_ptm_cso", "MACGTYLK", "CSO", 3),   # C at pos 3
    ("s046_ptm_hyp", "MAGPTYLK", "HYP", 4),   # P at pos 4
    ("s047_ptm_mly", "MAGSTYLK", "MLY", 8),   # K at pos 8
    ("s048_ptm_aly", "MAGSTYLK", "ALY", 8),   # K at pos 8
    ("s049_ptm_cme", "MACGTYLK", "CME", 3),   # C at pos 3
    ("s050_ptm_dal", "AAGSTYLK", "DAL", 1),   # A at pos 1
    ("s051_ptm_nle", "MALGSTYK", "NLE", 3),   # L at pos 3
    ("s052_ptm_aib", "MAAGSTYK", "AIB", 3),   # A at pos 3
    ("s053_ptm_dha", "MAAGSTYK", "DHA", 3),   # A at pos 3
    ("s054_ptm_kcx", "MAGSTYLK", "KCX", 8),   # K at pos 8
    ("s055_ptm_tys", "MAGATYLK", "TYS", 6),   # Y at pos 6
]
for name, seq, ptm_code, pos in ptm_cases:
    write_json(name, {
        "name": name,
        "sequences": [protein(seq, modifications=[{"ptmType": f"CCD_{ptm_code}", "ptmPosition": pos}])],
    })

# ================================================================
# D. Covalent Bonds (15 cases) — protein + CCD ligand + covalent bond
# ================================================================
# (name, protein_seq, ligand_ccd, list_of_bonds)
# Bond tuple: (entity1, position1, atom1, entity2, position2, atom2)
cov_cases = [
    ("s056_cov_cys_atp", "MACGTYLK", "ATP", [(1, 3, "SG", 2, 1, "PA")]),
    ("s057_cov_cys_gtp", "MACGTYLK", "GTP", [(1, 3, "SG", 2, 1, "PG")]),
    ("s058_cov_cys_hem", "MACHGYLK", "HEM", [(1, 3, "SG", 2, 1, "FE")]),
    ("s059_cov_his_hem", "MACHGYLK", "HEM", [(1, 4, "NE2", 2, 1, "FE")]),
    ("s060_cov_lys_plp", "MAGKTYLK", "PLP", [(1, 4, "NZ", 2, 1, "C4A")]),
    ("s061_cov_cys_fad", "MACGTYLK", "FAD", [(1, 3, "SG", 2, 1, "C8M")]),
    ("s062_cov_cys_coa", "MACGTYLK", "COA", [(1, 3, "SG", 2, 1, "S1P")]),
    ("s063_cov_cys_sf4", "MCCGTYLK", "SF4", [
        (1, 2, "SG", 2, 1, "FE1"),
        (1, 3, "SG", 2, 1, "FE2"),
    ]),
    ("s064_cov_cys_fes", "MCCGTYLK", "FES", [
        (1, 2, "SG", 2, 1, "FE1"),
        (1, 3, "SG", 2, 1, "FE2"),
    ]),
    ("s065_cov_cys_nag", "MACGTYLK", "NAG", [(1, 3, "SG", 2, 1, "C1")]),
    ("s066_cov_cys_sti", "MACGTYLK", "STI", [(1, 3, "SG", 2, 1, "C9")]),
    ("s067_cov_ser_so4", "MASGKYLK", "SO4", [(1, 3, "OG", 2, 1, "S")]),
    ("s068_cov_cys_btn", "MACGTYLK", "BTN", [(1, 3, "SG", 2, 1, "C2")]),
    ("s069_cov_lys_sam", "MAGKTYLK", "SAM", [(1, 4, "NZ", 2, 1, "C")]),
    ("s070_cov_cys_adp", "MACGTYLK", "ADP", [(1, 3, "SG", 2, 1, "PA")]),
]
for name, seq, lig_code, bonds in cov_cases:
    write_json(name, {
        "name": name,
        "sequences": [protein(seq), ccd_ligand(lig_code)],
        "covalent_bonds": [cov_bond(*b) for b in bonds],
    })

# ================================================================
# E. DNA/RNA Modifications (7 cases)
# ================================================================
# (name, na_type, sequence, mod_ccd_code, base_position)
dna_rna_mod_cases = [
    ("s071_dna_mod_6og", "dna", "ATGC", "6OG", 3),
    ("s072_dna_mod_5mc", "dna", "ATGC", "5MC", 4),
    ("s073_rna_mod_psu", "rna", "AUGC", "PSU", 3),
    ("s074_rna_mod_5mu", "rna", "AUGC", "5MU", 3),
    ("s075_rna_mod_1ma", "rna", "AUGC", "1MA", 1),
    ("s076_rna_mod_7mg", "rna", "AUGC", "7MG", 3),
    ("s077_dna_mod_5bu", "dna", "ATGC", "5BU", 3),
]
for name, na_type, seq, mod_code, pos in dna_rna_mod_cases:
    mod = [{"modificationType": f"CCD_{mod_code}", "basePosition": pos}]
    if na_type == "dna":
        na_entity = dna(seq, modifications=mod)
    else:
        na_entity = rna(seq, modifications=mod)
    write_json(name, {"name": name, "sequences": [protein(), na_entity]})

# ================================================================
# F. Multi-Entity Combinations (8 cases)
# ================================================================

# s078: Protein + DNA + ATP
write_json("s078_multi_prot_dna_atp", {
    "name": "s078_multi_prot_dna_atp",
    "sequences": [protein(), dna("ATGC"), ccd_ligand("ATP")],
})

# s079: Protein + RNA + MG ion
write_json("s079_multi_prot_rna_mg", {
    "name": "s079_multi_prot_rna_mg",
    "sequences": [protein(), rna("AUGC"), ion("MG")],
})

# s080: Protein + DNA + RNA
write_json("s080_multi_prot_dna_rna", {
    "name": "s080_multi_prot_dna_rna",
    "sequences": [protein(), dna("ATGC"), rna("AUGC")],
})

# s081: Protein + ATP + NAG (two ligands)
write_json("s081_multi_prot_2lig", {
    "name": "s081_multi_prot_2lig",
    "sequences": [protein(), ccd_ligand("ATP"), ccd_ligand("NAG")],
})

# s082: Protein + HEM + ZN + FE (metal-rich)
write_json("s082_multi_prot_lig_ion", {
    "name": "s082_multi_prot_lig_ion",
    "sequences": [protein(), ccd_ligand("HEM"), ion("ZN"), ion("FE")],
})

# s083: Protein(count=2) + FAD (homodimer + cofactor)
write_json("s083_multi_homodimer_lig", {
    "name": "s083_multi_homodimer_lig",
    "sequences": [protein(count=2), ccd_ligand("FAD")],
})

# s084: Protein + NAG + BMA + GAL (multi-glycan)
write_json("s084_multi_prot_glycan", {
    "name": "s084_multi_prot_glycan",
    "sequences": [protein(), ccd_ligand("NAG"), ccd_ligand("BMA"), ccd_ligand("GAL")],
})

# s085: Protein(Cys) + DNA + ATP + covalent bond (entity 3 = ATP)
write_json("s085_multi_prot_dna_cov", {
    "name": "s085_multi_prot_dna_cov",
    "sequences": [protein("MACGTYLK"), dna("ATGC"), ccd_ligand("ATP")],
    "covalent_bonds": [cov_bond(1, 3, "SG", 3, 1, "PA")],
})

# ================================================================
# G. Edge Cases / Stress Tests (15 cases)
# ================================================================

# s086: Single residue "M" + GOL (minimal protein)
write_json("s086_edge_single_aa", {
    "name": "s086_edge_single_aa",
    "sequences": [protein("M"), ccd_ligand("GOL")],
})

# s087: Protein + very long SMILES (cholesterol)
write_json("s087_edge_long_smi", {
    "name": "s087_edge_long_smi",
    "sequences": [protein(), smiles_ligand(
        "C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C"
    )],
})

# s088: Protein + tiny SMILES "O" (water-like)
write_json("s088_edge_tiny_smi", {
    "name": "s088_edge_tiny_smi",
    "sequences": [protein(), smiles_ligand("O")],
})

# s089: Protein with 3 PTMs: SEP@2, TPO@4, MSE@5 on MSATMYLK
write_json("s089_edge_multi_ptm", {
    "name": "s089_edge_multi_ptm",
    "sequences": [protein("MSATMYLK", modifications=[
        {"ptmType": "CCD_SEP", "ptmPosition": 2},
        {"ptmType": "CCD_TPO", "ptmPosition": 4},
        {"ptmType": "CCD_MSE", "ptmPosition": 5},
    ])],
})

# s090: Protein + 4 ions (CA, MG, ZN, FE)
write_json("s090_edge_all_ions", {
    "name": "s090_edge_all_ions",
    "sequences": [protein(), ion("CA"), ion("MG"), ion("ZN"), ion("FE")],
})

# s091: Protein + large CCD ligand (RIF = rifampicin, ~67 heavy atoms)
write_json("s091_edge_large_lig", {
    "name": "s091_edge_large_lig",
    "sequences": [protein(), ccd_ligand("RIF")],
})

# s092: Protein + PO4 (inorganic phosphate)
write_json("s092_edge_po4", {
    "name": "s092_edge_po4",
    "sequences": [protein(), ccd_ligand("PO4")],
})

# s093: Protein + ACT (acetate, tiny)
write_json("s093_edge_act", {
    "name": "s093_edge_act",
    "sequences": [protein(), ccd_ligand("ACT")],
})

# s094: Protein(MCCCGYLK) + SF4 + 3 covalent bonds
write_json("s094_edge_multi_cov", {
    "name": "s094_edge_multi_cov",
    "sequences": [protein("MCCCGYLK"), ccd_ligand("SF4")],
    "covalent_bonds": [
        cov_bond(1, 2, "SG", 2, 1, "FE1"),
        cov_bond(1, 3, "SG", 2, 1, "FE2"),
        cov_bond(1, 4, "SG", 2, 1, "FE3"),
    ],
})

# s095: Protein + cyclohexane SMILES (saturated ring)
write_json("s095_edge_smi_ring", {
    "name": "s095_edge_smi_ring",
    "sequences": [protein(), smiles_ligand("C1CCCCC1")],
})

# s096: Protein + sulfonamide drug fragment SMILES
write_json("s096_edge_smi_sulfonamide", {
    "name": "s096_edge_smi_sulfonamide",
    "sequences": [protein(), smiles_ligand("NS(=O)(=O)c1ccc(Cl)cc1")],
})

# s097: Protein + glycine as SMILES (amino acid as ligand)
write_json("s097_edge_smi_amino_acid", {
    "name": "s097_edge_smi_amino_acid",
    "sequences": [protein(), smiles_ligand("NCC(=O)O")],
})

# s098: DNA only, no protein
write_json("s098_edge_dna_only", {
    "name": "s098_edge_dna_only",
    "sequences": [dna("ATGCATGC")],
})

# s099: RNA only, no protein
write_json("s099_edge_rna_only", {
    "name": "s099_edge_rna_only",
    "sequences": [rna("AUGCAUGC")],
})

# s100: Heterodimer(count=2) + monomer + ATP + ZN ion
write_json("s100_edge_prot_2chain", {
    "name": "s100_edge_prot_2chain",
    "sequences": [
        protein("MACGTYLK", count=2),
        protein("GKLYTSAM"),
        ccd_ligand("ATP"),
        ion("ZN"),
    ],
})

count = len(os.listdir(OUTDIR))
print(f"\nDone! Generated {count} files in {OUTDIR}")
