# Models

Protenix.jl supports two model families: **Protenix** (structure prediction / folding) and **PXDesign** (protein binder design). All weights are hosted on HuggingFace and downloaded automatically on first use.

## Folding Models (Protenix)

Folding models predict 3D structures from sequences and optional context (MSA, templates, constraints, ligands, ions, covalent bonds).

### v1.0 Models (Recommended)

| Model | Alias | Parameters | Notes |
|-------|-------|:----------:|-------|
| `protenix_base_20250630_v1.0.0` | `protenix_v1` | ~600M | Latest v1.0 checkpoint. Best overall quality. |
| `protenix_base_default_v1.0.0` | *(none)* | ~600M | Original v1.0 release. |

v1.0 models support: proteins, DNA, RNA, ligands (CCD + SMILES + SDF), ions, covalent bonds, MSA, post-translational modifications, modified nucleotides, and template structures.

**Key v1.0 differences from v0.5:**
- Uses CCD mol_type lookup for **all** entities (including ligands), improving tokenization of modified residues and non-standard components.
- Active template embedder (`n_blocks=2`) that conditions on experimental structures.
- Paired MSA is treated as unpaired internally (`msa_pair_as_unpair=true`), matching the training regime.

### v0.5 Models

| Model | Parameters | Notes |
|-------|:----------:|-------|
| `protenix_base_default_v0.5.0` | ~600M | General-purpose base model. |
| `protenix_base_constraint_v0.5.0` | ~600M | Supports pocket and distance constraints. |
| `protenix_mini_default_v0.5.0` | ~15M | Fast mini model for testing and iteration. |
| `protenix_mini_tmpl_v0.5.0` | ~15M | Mini with template embedder. |
| `protenix_mini_esm_v0.5.0` | ~15M | Mini requiring ESM2 token embeddings. |
| `protenix_mini_ism_v0.5.0` | ~15M | Mini requiring ISM-tuned ESM2 embeddings. |
| `protenix_tiny_default_v0.5.0` | ~5M | Smallest model, for smoke tests only. |

### Default Inference Parameters

| Model | Cycles | Steps | Samples |
|-------|:------:|:-----:|:-------:|
| Base models (v0.5 + v1.0) | 10 | 200 | 5 |
| Mini/tiny models | 4 | 5 | 5 |

- **Cycles**: Number of recycling iterations in the trunk.
- **Steps**: Number of diffusion denoising steps.
- **Samples**: Number of independent structure samples per seed.

Parameters can be overridden:
```julia
result = fold(h, seq; step=100, sample=3, cycle=5)
```

### Model Aliases

Short names that resolve to canonical model names:

| Alias | Resolves to |
|-------|-------------|
| `protenix_v1` | `protenix_base_20250630_v1.0.0` |

```julia
h = load_protenix("protenix_v1"; gpu=true)  # loads protenix_base_20250630_v1.0.0
```

### Feature Support Matrix

| Feature | v1.0 Base | v0.5 Base | v0.5 Constraint | v0.5 Mini | Mini ESM | Mini ISM | Mini Tmpl |
|---------|:---------:|:---------:|:---------------:|:---------:|:--------:|:--------:|:---------:|
| Protein sequences | Y | Y | Y | Y | Y | Y | Y |
| DNA sequences | Y | Y | Y | Y | Y | Y | Y |
| RNA sequences | Y | Y | Y | Y | Y | Y | Y |
| Ligands (CCD) | Y | Y | Y | Y | Y | Y | Y |
| Ligands (SMILES) | Y | Y | Y | Y | Y | Y | Y |
| Ligands (SDF file) | Y | Y | Y | Y | Y | Y | Y |
| Ions | Y | Y | Y | Y | Y | Y | Y |
| Covalent bonds | Y | Y | Y | Y | Y | Y | Y |
| MSA (precomputed) | Y | Y | Y | Y | -- | -- | Y |
| PTMs / modifications | Y | Y | Y | Y | Y | Y | Y |
| DNA/RNA modifications | Y | Y | Y | Y | Y | Y | Y |
| Template structures | Y | -- | -- | -- | -- | -- | Y |
| ESM2 embeddings | -- | -- | -- | -- | Y | -- | -- |
| ISM embeddings | -- | -- | -- | -- | -- | Y | -- |
| Pocket constraints | -- | -- | Y | -- | -- | -- | -- |
| Distance constraints | -- | -- | Y | -- | -- | -- | -- |

## Design Models (PXDesign)

Design models generate novel protein binders conditioned on a target structure.

| Model | Parameters | Notes |
|-------|:----------:|-------|
| `pxdesign_v0.1.0` | ~150M | Diffusion-based binder design with optional target conditioning. |

### Design Capabilities

- **Unconditional design**: Generate de novo proteins of specified length.
- **Conditional design**: Design binders against a target structure (CIF file).
- **Hotspot specification**: Focus the binder interface on specific target residues.
- **Cropping**: Use a subset of target chains/residues as the conditioning context.
- **MSA conditioning**: Provide precomputed MSA for target chains to improve design quality.
- **Multi-chain targets**: Condition on multiple chains simultaneously.

### Default Design Parameters

| Parameter | Default | Description |
|-----------|:-------:|-------------|
| `N_step` | 200 | Diffusion denoising steps |
| `N_sample` | 5 | Independent design samples per seed |
| `gamma0` | 1.0 | Initial noise scale |
| `gamma_min` | 0.01 | Minimum noise scale |
| `noise_scale_lambda` | 1.003 | Noise scale decay rate |

## Choosing a Model

**For production folding**: Use `protenix_v1` (alias for the latest v1.0 model). It has the best structure quality and supports all input types.

**For constrained folding**: Use `protenix_base_constraint_v0.5.0`. This is the only model trained with pocket and distance constraint support.

**For fast iteration / testing**: Use `protenix_mini_default_v0.5.0`. It runs ~40x faster than base models.

**For binder design**: Use `pxdesign_v0.1.0` via `load_pxdesign()` and `design()`.

**For ESM-conditioned folding**: Use `protenix_mini_esm_v0.5.0` with pre-computed or auto-generated ESM2 embeddings.

## Listing Models at Runtime

```julia
using Protenix
for m in list_supported_models()
    println(m.model_name, "  family=", m.family, "  step=", m.default_step)
end
```

## Weight Storage

All weights are stored as SafeTensors on HuggingFace (`MurrellLab/PXDesign.jl`). Base models use sharded SafeTensors (multiple files + index JSON); mini/design models use single SafeTensors files.

Weights are cached after first download. For offline use:
```bash
export PROTENIX_WEIGHTS_LOCAL_FILES_ONLY=true
```
