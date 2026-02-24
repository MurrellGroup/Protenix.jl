# PXDesign.jl

Pure Julia implementation of [Protenix](https://github.com/bytedance/Protenix) structure prediction and [PXDesign](https://github.com/bytedance/PXDesign) protein binder design. Supports all Protenix v0.5 and v1.0 folding models plus the PXDesign diffusion design model.

## Installation

```julia
using Pkg
Pkg.develop(path="path/to/PXDesign.jl")
```

If shared dependencies aren't resolved automatically:

```julia
Pkg.develop(path="path/to/Onion.jl")
Pkg.develop(path="path/to/ProtInterop.jl")
Pkg.develop(path="path/to/PXDesign.jl")
Pkg.instantiate()
```

CUDA GPU support requires `CUDA.jl` and `cuDNN.jl` in your environment. SMILES ligand support requires `MoleculeFlow.jl`.

### cuTile Kernels

To use cuTile/OnionTile accelerated kernels, add `OnionTile.jl` to your environment. You will also need the [BFloat16s-compatible SafeTensors fork](https://github.com/AntonOresten/SafeTensors.jl/tree/bfloat16s-v0.6) to resolve a dependency conflict with BFloat16s:

```julia
Pkg.add(url="https://github.com/AntonOresten/SafeTensors.jl", rev="bfloat16s-v0.6")
```

## Quickstart: Folding

```julia
using PXDesign

# Load the latest v1.0 model
h = load_protenix("protenix_v1"; gpu=true)

# Fold a sequence
result = fold(h, "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH")
result.mean_plddt   # predicted local distance difference test (0-100)
result.cif_paths    # paths to output CIF files
```

### Multi-Entity Complex

```julia
task = protenix_task(
    protein_chain("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"),
    ligand("CCD_ATP"),
    ion("MG"),
)
result = fold(h, task; seed=101, out_dir="./output")
```

### With MSA

```julia
task = protenix_task(
    protein_chain("MGSSHHHHHHSSGLVPRGSH...";
        msa = Dict(
            "precomputed_msa_dir" => "path/to/msa_dir",
            "pairing_db" => "uniref100",
        ),
    ),
)
result = fold(h, task; seed=101)
```

### With Covalent Bonds (Glycoprotein)

```julia
rbd = protein_chain(rbd_sequence; msa = Dict("precomputed_msa_dir" => msa_dir))
glycan = ligand("CCD_NAG_NAG_BMA")

task = protenix_task(rbd, glycan;
    covalent_bonds = [
        Dict("entity1"=>"1", "position1"=>"13", "atom1"=>"ND2",
             "entity2"=>"2", "position2"=>"1", "atom2"=>"C1"),
    ],
)
result = fold(h, task)
```

### Batch JSON Prediction

```julia
records = predict_json("inputs/complex.json";
    model_name = "protenix_v1",
    out_dir = "./output",
    seeds = [101, 102],
    use_msa = true,
    gpu = true,
)
```

## Quickstart: Design

```julia
using PXDesign

dh = load_pxdesign("pxdesign_v0.1.0"; gpu=true)

# Unconditional (de novo)
result = design(dh; binder_length=60, seed=42, n_sample=5)

# Conditional with target + hotspots
target = design_target("structures/1ubq.cif"; chains=["A"])
result = design(dh;
    binder_length=80,
    target=target,
    hotspots=Dict("A" => [8, 44, 48]),
    seed=42,
)
result.cif_paths  # designed structures
```

### Design with Target Cropping and MSA

```julia
target = design_target("structures/5o45.cif";
    chains=["A"],
    crop=Dict("A" => "1-116"),
    msa=Dict("A" => Dict("precomputed_msa_dir" => "msa/PDL1_chain_A")),
)
result = design(dh; binder_length=80, target=target, seed=42)
```

## Models

| Model | Alias | Family | Description |
|-------|-------|--------|-------------|
| `protenix_base_20250630_v1.0.0` | `protenix_v1` | Folding | Latest v1.0 model. Best quality. |
| `protenix_base_default_v1.0.0` | | Folding | Original v1.0 release. |
| `protenix_base_default_v0.5.0` | | Folding | v0.5 base model. |
| `protenix_base_constraint_v0.5.0` | | Folding | Supports pocket/distance constraints. |
| `protenix_mini_default_v0.5.0` | | Folding | Fast mini model (~40x faster). |
| `protenix_mini_esm_v0.5.0` | | Folding | Mini + ESM2 embeddings. |
| `protenix_mini_ism_v0.5.0` | | Folding | Mini + ISM embeddings. |
| `protenix_mini_tmpl_v0.5.0` | | Folding | Mini + template embedder. |
| `protenix_tiny_default_v0.5.0` | | Folding | Smallest model (smoke tests). |
| `pxdesign_v0.1.0` | | Design | Diffusion binder design. |

See [docs/models.md](docs/models.md) for detailed model descriptions, capabilities, and recommended usage.

## Supported Input Types

| Entity | REPL Builder | JSON Key | Example |
|--------|-------------|----------|---------|
| Protein | `protein_chain(seq)` | `proteinChain` | Single chain or homo-oligomer |
| DNA | `dna_chain(seq)` | `dnaSequence` | Single-stranded DNA |
| RNA | `rna_chain(seq)` | `rnaSequence` | Single-stranded RNA |
| Ligand (CCD) | `ligand("CCD_ATP")` | `ligand` | By Chemical Component Dictionary code |
| Ligand (SMILES) | `ligand("Nc1ncnc2...")` | `ligand` | By SMILES string |
| Ligand (file) | `ligand("FILE_x.sdf")` | `ligand` | From SDF file |
| Ion | `ion("MG")` | `ion` | Metal ion by CCD code |

Additional features: covalent bonds, constraints (pocket/distance), MSA, template structures, PTMs, DNA/RNA modifications.

See [docs/inputs.md](docs/inputs.md) for complete input format documentation.

## Documentation

- **[Models](docs/models.md)** — Detailed model descriptions, capabilities, parameters, and selection guide
- **[Input Formats](docs/inputs.md)** — Complete reference for REPL API, JSON, and YAML inputs, including MSA, templates, covalent bonds, constraints, modifications
- **[API Reference](docs/api.md)** — Full function signatures, arguments, return types, environment variables, CLI

## Model Weights

Weights are downloaded automatically from HuggingFace on first use and cached locally.

- Repository: [`MurrellLab/PXDesign.jl`](https://huggingface.co/MurrellLab/PXDesign.jl)
- CCD components file: auto-downloaded alongside weights

For offline use:
```bash
export PXDESIGN_WEIGHTS_LOCAL_FILES_ONLY=true
```

## Examples

The `examples/` directory contains runnable inputs for all supported entity types:

- `examples/inputs/` — 45 folding and design JSON/YAML inputs
- `examples/stress_inputs/` — 100 stress test inputs (CCD compounds, SMILES, PTMs, covalent bonds, edge cases)
- `examples/structures/` — CIF structures for design conditioning
- `examples/msa/` — Precomputed MSA directories
- `examples/ligands/` — SDF ligand files
- `examples/rbd_glycosylated_repl.jl` — Full REPL example: RBD + MSA + glycans + covalent bonds

### Running an Example

```julia
using PXDesign

# From JSON
records = predict_json("examples/inputs/01_protein_monomer.json";
    model_name="protenix_v1", gpu=true)

# Design from YAML
cfg = PXDesign.Config.default_config()
cfg["input_json_path"] = "examples/inputs/23_design_pdl1_hotspots.yaml"
cfg["dump_dir"] = "./output"
cfg["model_name"] = "pxdesign_v0.1.0"
cfg["gpu"] = true
cfg["seeds"] = [101]
PXDesign.run_infer(cfg)
```

## Known Limitations

1. **Online MSA search**: Not implemented. Only precomputed A3M files are supported.
2. **Amber relax**: Not implemented.
3. **Target 20 (complex multichain)**: 1190-token complexes exceed GPU memory for flash attention bias tensors on smaller GPUs.
4. **SMILES conformers**: SMILES-to-3D generation produces different conformers than Python's RDKit (different internal RNG). This does not affect model behavior (trained with random conformer augmentation). CCD ligands are unaffected.

## Testing

```julia
using Pkg
Pkg.test("PXDesign")
```
