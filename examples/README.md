# Clean Targets — Comprehensive Protenix & PXDesign Test Suite

This directory contains the definitive validation suite for achieving full feature parity between Python Protenix/PXDesign and Julia Protenix.jl. It replaces all previous scattered test outputs (`e2e_output/`, `cif_comparison/`, `/tmp/` scripts).

## Quick Start

```bash
# Run all targets through Python Protenix (generates reference outputs)
cd path/to/Protenix.jl
bash clean_targets/scripts/run_all_python.sh

# Run a single target
bash clean_targets/scripts/run_all_python.sh 01

# Run all targets through Julia Protenix
cd path/to/your-julia-env
julia --project=. path/to/Protenix.jl/clean_targets/scripts/run_all_julia.jl

# Validate all outputs (bond geometry + comparison)
julia --project=. ../Protenix.jl/clean_targets/scripts/validate_all.jl
```

## Directory Layout

```
clean_targets/
├── README.md                   # This file
├── inputs/                     # All input files (JSON for Protenix, YAML for PXDesign)
│   ├── 01_protein_monomer.json
│   ├── ...
│   └── 32_design_many_hotspots.yaml
├── msa/                        # Precomputed MSA files
│   ├── 7r6r_chain1/            # For target 17 (protein with MSA)
│   └── PDL1_chain_A/           # For target 29 (design with MSA)
├── structures/                 # Target CIF structures for design targets
│   └── 5o45.cif                # PD-L1 structure (used by targets 23-32)
├── ligands/                    # Ligand structure files
│   ├── compounds-3d-R.sdf      # For target 21 (SDF ligand)
│   └── compounds-3d-RS.sdf
├── python_outputs/             # Reference outputs from Python Protenix
│   └── <target>__<model>/seed_101/...
├── julia_outputs/              # Julia Protenix outputs for comparison
│   └── <target>__<model>/seed_101/...
└── scripts/
    ├── run_all_python.sh       # Run all targets through Python
    ├── run_all_julia.jl        # Run all targets through Julia
    └── validate_all.jl         # Bond check + comparison
```

## Target Inventory

### Category 1: Protein-Only Folding

| # | Target | Description | Entities | Tests |
|---|--------|-------------|----------|-------|
| 01 | `protein_monomer` | Single 50-residue hemoglobin alpha fragment | 1 protein | Basic folding pipeline |
| 02 | `protein_heterodimer` | Two different protein chains | 2 proteins | Multi-chain, interface prediction |
| 03 | `protein_homodimer` | Single protein with count=2 | 1 protein (x2) | Symmetric copy handling |

### Category 2: Nucleic Acid Complexes

| # | Target | Description | Entities | Tests |
|---|--------|-------------|----------|-------|
| 04 | `protein_dna` | Protein + dsDNA (two complementary strands) | protein + 2 DNA | DNA entity handling |
| 05 | `protein_rna` | Protein + ssRNA | protein + RNA | RNA entity handling |

### Category 3: Small Molecules

| # | Target | Description | Entities | Tests |
|---|--------|-------------|----------|-------|
| 06 | `protein_ligand_ccd` | Protein + ATP (CCD code) | protein + ligand | CCD lookup, tokenization |
| 07 | `protein_ligand_smiles` | Protein + ATP (SMILES) | protein + ligand | SMILES parsing, conformer generation |
| 08 | `protein_ion` | Protein + Mg²⁺ + Zn²⁺ | protein + ions | Ion entity handling |

### Category 4: Multi-Entity Complexes

| # | Target | Description | Entities | Tests |
|---|--------|-------------|----------|-------|
| 09 | `protein_dna_ligand` | Protein + dsDNA + PCG ligand (7pzb) | protein + DNA + ligand | 3+ entity types |
| 10 | `protein_ligand_covalent` | Protein + ATP with covalent bond | protein + ligand + bond | Covalent bond parsing |

### Category 5: Modifications

| # | Target | Description | Entities | Tests |
|---|--------|-------------|----------|-------|
| 11 | `protein_ptm` | Protein with HY3 and P1L PTMs | modified protein | PTM CCD resolution |
| 12 | `dna_modification` | DNA with 6OG base modification | protein + modified DNA | DNA modification handling |
| 13 | `multi_ccd_glycan` | NAG-BMA-BGC glycan ligand | protein + multi-CCD ligand | Glycan component handling |

### Category 6: Constraints

| # | Target | Description | Model | Tests |
|---|--------|-------------|-------|-------|
| 14 | `constraint_pocket` | Pocket constraint (7st3) | base_constraint | Pocket featurization |
| 15 | `constraint_token_contact` | Residue-residue distance constraint | base_constraint | Token contact constraint |
| 16 | `constraint_atom_contact` | Atom-level distance constraint (5sak) | base_constraint | Atom contact constraint |

### Category 7: Input Modalities

| # | Target | Description | Model | Tests |
|---|--------|-------------|-------|-------|
| 17 | `protein_msa` | Protein with precomputed MSA (7r6r) | base/mini default | MSA loading + featurization |
| 18 | `protein_esm` | Protein for ESM/ISM embedding | mini_esm / mini_ism | ESM embedding generation |
| 19 | `protein_template` | Protein for template-guided prediction | mini_tmpl | Template embedding |

### Category 8: Complex Assembly

| # | Target | Description | Entities | Tests |
|---|--------|-------------|----------|-------|
| 20 | `complex_multichain` | 7wux: 2 protein types (x2) + 3 ligand types + ions | protein + protein + ligands + ions | Full assembly |

### Category 9: File-Based Ligand

| # | Target | Description | Entities | Tests |
|---|--------|-------------|----------|-------|
| 21 | `ligand_file_sdf` | Protein + ligand from SDF file | protein + file ligand | SDF file loading |

### Category 10: Design (PXDesign)

All design targets use YAML format and the PXDesign diffusion generator.

| # | Target | Description | Tests |
|---|--------|-------------|-------|
| 22 | `design_unconditional` | No target, pure de novo generation (60 AA) | Unconditional diffusion |
| 23 | `design_pdl1_hotspots` | PDL1 + hotspots [40,99,107], crop 1-116 (80 AA) | Condition + hotspot baseline |
| 24 | `design_no_hotspots` | PDL1 without hotspots, crop 1-116 (80 AA) | Conditioning without guidance |
| 25 | `design_full_chain` | PDL1 full chain, hotspots (80 AA) | Full-length target |
| 26 | `design_discontinuous_crop` | PDL1, crop [1-50, 80-116], hotspots (80 AA) | Gap handling |
| 27 | `design_multichain` | Multi-chain target, hotspots per chain (100 AA) | Multi-chain conditioning |
| 28 | `design_multichain_mixed_crop` | Multi-chain: one cropped, one full (100 AA) | Per-chain independence |
| 29 | `design_with_msa` | PDL1 + precomputed MSA (80 AA) | MSA in design context |
| 30 | `design_short_binder` | PDL1, 40 AA binder | Small scaffold |
| 31 | `design_long_binder` | PDL1, 150 AA binder | Scaling + memory |
| 32 | `design_many_hotspots` | PDL1, 20 hotspot residues (80 AA) | Dense guidance |

## Model-Target Matrix

Each target runs through specific model variants:

| Target | base_default | base_constraint | mini_default | mini_esm | mini_ism | mini_tmpl | tiny_default | PXDesign |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 01 protein_monomer | X | | X | | | | X | |
| 02 protein_heterodimer | X | | X | | | | | |
| 03 protein_homodimer | X | | X | | | | | |
| 04 protein_dna | X | | X | | | | | |
| 05 protein_rna | X | | X | | | | | |
| 06 protein_ligand_ccd | X | | X | | | | | |
| 07 protein_ligand_smiles | X | | X | | | | | |
| 08 protein_ion | X | | X | | | | | |
| 09 protein_dna_ligand | X | | X | | | | | |
| 10 protein_ligand_covalent | X | | X | | | | | |
| 11 protein_ptm | X | | X | | | | | |
| 12 dna_modification | X | | X | | | | | |
| 13 multi_ccd_glycan | X | | X | | | | | |
| 14 constraint_pocket | | X | | | | | | |
| 15 constraint_token_contact | | X | | | | | | |
| 16 constraint_atom_contact | | X | | | | | | |
| 17 protein_msa | X | | X | | | | | |
| 18 protein_esm | | | | X | X | | | |
| 19 protein_template | | | | | | X | | |
| 20 complex_multichain | X | | X | | | | | |
| 21 ligand_file_sdf | X | | X | | | | | |
| 22-32 design_* | | | | | | | | X |

## Inference Parameters

All targets use consistent parameters for reproducibility:

| Parameter | base models | mini/tiny models | Design |
|-----------|:-----------:|:----------------:|:------:|
| seed | 101 | 101 | 101 |
| N_step | 200 | 20 | 200 |
| N_sample | 1 | 1 | 1 |
| N_cycle | 10 | 4 | N/A |
| use_msa | false* | false* | N/A |

\* Except target 17 which specifically tests MSA.

## Validation Criteria

For each target, validation includes:

1. **Bond check** (protein residues only):
   - **PERFECT**: 0 violations out of all checked bonds
   - **GREEN**: <1% of total bonds violated
   - **ORANGE**: 1-5% of total bonds violated — investigate
   - **RED**: >5% of total bonds violated — something is wrong

2. **Output completeness**: CIF file produced with all expected chains/entities

3. **Confidence metrics**: JSON confidence file produced with valid scores

4. **No RMSD comparison**: These are diffusion models with different random noise — Julia and Python outputs will NOT match numerically and that's expected. Bond geometry is the quality metric.

## Julia Feature Parity Status

Features that are **working** in Julia:
- Protein-only folding (targets 01-03): all 7 model variants
- Constraints (targets 14-16): pocket, token contact, atom contact
- MSA (target 17): precomputed MSA loading + featurization
- ESM/ISM embeddings (target 18): via ESMFold.jl
- Template (target 19): template embedding + noisy structure conditioning
- Design/scaffold inference (targets 22-32): DesignConditionEmbedder + DiffusionModule

Features that **need implementing** for full parity:
1. DNA/RNA entity handling (targets 04, 05, 09, 12)
2. Ligand/Ion handling (targets 06, 07, 08, 10, 13, 21)
3. Covalent bond support (target 10)
4. PTM/modification support (targets 11, 12)

## Archived Outputs

Previous test outputs are preserved in `archived_targets/`:
- `archived_targets/e2e_output/` — former `e2e_output/`
- `archived_targets/cif_comparison/` — former `cif_comparison/`
