# PXDesign.jl

Pure Julia implementation of Protenix structure prediction and PXDesign binder design.

## Installation

From a local checkout:

```julia
using Pkg
Pkg.develop(path="path/to/PXDesign.jl")
Pkg.instantiate()
```

If your resolver cannot find shared dependencies, add them as path dependencies first:

```julia
using Pkg
Pkg.develop(path="path/to/Onion.jl")
Pkg.develop(path="path/to/ProtInterop.jl")
Pkg.develop(path="path/to/PXDesign.jl")
Pkg.instantiate()
```

## Quickstart

### Load and Fold

```julia
using PXDesign

h = load_protenix(gpu=true)
result = fold(h, "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH")
result.mean_plddt   # average predicted local distance difference test (0-100)
result.cif          # mmCIF text as String
result.cif_paths    # paths to written CIF files
```

### Confidence Metrics

```julia
m = confidence_metrics(result)
m.mean_plddt  # average pLDDT (0-100)
m.mean_pae    # average predicted aligned error (Angstroms)
m.pde         # predicted distance error
m.resolved    # predicted experimentally-resolved mask
```

### Batch JSON Prediction (Multi-Entity)

```julia
records = predict_json("inputs/complex.json";
    model_name = "protenix_base_default_v0.5.0",
    out_dir = "./output",
    seeds = [101, 102],
    gpu = true,
)
for r in records
    println("$(r.task_name) seed=$(r.seed): $(r.prediction_dir)")
end
```

### Sequence Prediction

```julia
records = predict_sequence("ACDEFGHIKLMNPQRSTVWY";
    model_name = "protenix_mini_default_v0.5.0",
    out_dir = "./output",
    gpu = true,
)
```

## Supported Models

| Model | Family | Cycle | Step | Sample | MSA | ESM |
|-------|--------|:-----:|:----:|:------:|:---:|:---:|
| `protenix_base_default_v0.5.0` | base | 10 | 200 | 5 | yes | no |
| `protenix_base_constraint_v0.5.0` | base | 10 | 200 | 5 | yes | no |
| `protenix_mini_default_v0.5.0` | mini | 4 | 5 | 5 | yes | no |
| `protenix_mini_tmpl_v0.5.0` | mini | 4 | 5 | 5 | yes | no |
| `protenix_mini_esm_v0.5.0` | mini | 4 | 5 | 5 | no | yes |
| `protenix_mini_ism_v0.5.0` | mini | 4 | 5 | 5 | no | yes |
| `protenix_tiny_default_v0.5.0` | mini | 4 | 5 | 5 | yes | no |
| `protenix_base_default_v1.0.0` | base | 10 | 200 | 5 | yes | no |
| `protenix_base_20250630_v1.0.0` | base | 10 | 200 | 5 | yes | no |
| `pxdesign_v0.1.0` | design | — | 200 | — | — | no |

### Discovering Models at Runtime

```julia
for m in list_supported_models()
    println("$(m.model_name)  family=$(m.family)  cycle=$(m.default_cycle) step=$(m.default_step)")
end
```

## API Reference

### Core REPL Functions

**`load_protenix(model_name="protenix_base_default_v0.5.0"; gpu=false, strict=true) → ProtenixHandle`**

Load a Protenix model and return a reusable handle. Weights are downloaded from
HuggingFace on first use and cached locally.

- `model_name`: one of the supported model names (see table above)
- `gpu`: move model to GPU after loading
- `strict`: enforce strict weight key coverage (recommended)
- Returns: `ProtenixHandle` — pass to `fold()` for repeated predictions

**`fold(handle, sequence; seed=101, step=nothing, sample=nothing, cycle=nothing, out_dir=nothing, task_name="protenix_sequence", chain_id="A", esm_token_embedding=nothing) → NamedTuple`**

Fold a protein sequence using a loaded model handle.

- `seed`: RNG seed for diffusion sampling
- `step`, `sample`, `cycle`: override model defaults (or `nothing` to use defaults)
- `out_dir`: directory for CIF output (temp dir if `nothing`)
- `esm_token_embedding`: explicit ESM embedding matrix `[N_token, D]` (overrides auto-generation)

Returns a NamedTuple with fields:
- `coordinate` — predicted 3D coordinates
- `cif` — mmCIF text as String
- `cif_paths` — paths to written CIF files
- `prediction_dir` — output directory path
- `plddt` — per-residue pLDDT scores (0-100)
- `mean_plddt` — average pLDDT
- `pae` — predicted aligned error matrix (Angstroms)
- `mean_pae` — average PAE
- `pde` — predicted distance error
- `resolved` — predicted experimentally-resolved mask
- `distogram_logits`, `plddt_logits`, `pae_logits` — raw logits
- `seed`, `task_name` — echo of inputs

**`confidence_metrics(result) → NamedTuple`**

Extract confidence metrics from a fold result. Returns `(plddt, mean_plddt, pae, mean_pae, pde, resolved)`.

### Batch Prediction

**`predict_json(input; out_dir, model_name, seeds, gpu, cycle, step, sample, use_msa, strict) → Vector{PredictJSONRecord}`**

Run prediction on one or more JSON input files. `input` can be a file path or directory.
Each record contains `(input_json, task_name, seed, prediction_dir, cif_paths)`.

**`predict_sequence(sequence; out_dir, model_name, seeds, gpu, task_name, chain_id, esm_token_embedding, cycle, step, sample, use_msa, strict) → Vector{PredictSequenceRecord}`**

Run prediction on a single protein sequence. Each record contains
`(task_name, seed, prediction_dir, cif_paths)`.

### Utilities

**`list_supported_models() → Vector{NamedTuple}`**

Return sorted metadata for all registered models. Each entry has fields:
`model_name`, `family`, `default_cycle`, `default_step`, `default_sample`,
`default_use_msa`, `needs_esm_embedding`.

**`recommended_params(model_name; use_default_params=true, cycle, step, sample, use_msa) → NamedTuple`**

Return recommended inference parameters for a model. When `use_default_params=true`,
returns the model's registered defaults. Override individual parameters as needed.

**`convert_structure_to_infer_json(input; out_dir="./output", altloc="first", assembly_id=nothing) → Vector{String}`**

Convert PDB/mmCIF structure files to Protenix inference JSON format. Returns paths
to the written JSON files. Supports mmCIF bioassembly expansion via `assembly_id`.

**`add_precomputed_msa_to_json(input_json; out_dir="./output", precomputed_msa_dir, pairing_db="uniref100") → Vector{String}`**

Attach a precomputed MSA directory to an existing inference JSON. Adds
`msa.precomputed_msa_dir` and `msa.pairing_db` to each `proteinChain` entity.

### Types

- `ProtenixHandle` — loaded model state (model, family, model_name, on_gpu, params)
- `ProtenixModelSpec` — model metadata (name, family, defaults)
- `ProtenixPredictOptions` — shared options for `predict_json` / `predict_sequence`
- `ProtenixSequenceOptions` — sequence-specific options (wraps `ProtenixPredictOptions`)
- `PredictJSONRecord` — result record from `predict_json`
- `PredictSequenceRecord` — result record from `predict_sequence`

## Supported Entities

| Entity | JSON Key | Description |
|--------|----------|-------------|
| Protein | `proteinChain` | Amino acid sequence with optional `count` for homo-oligomers |
| DNA | `dnaSequence` | Single-stranded DNA sequence |
| RNA | `rnaSequence` | Single-stranded RNA sequence |
| Ligand (CCD) | `ligand` with `"CCD_XXX"` | Ligand by Chemical Component Dictionary code |
| Ligand (SMILES) | `ligand` with `"SMILES_..."` or SMILES string | Ligand by SMILES notation |
| Ligand (file) | `ligand` with `"FILE_path.sdf"` | Ligand from local structure file |
| Ion | `ion` | Metal ion by CCD code (e.g. `"MG"`, `"ZN"`) |

### Constraints

Supported via `protenix_base_constraint_v0.5.0`:

- `constraint.contact` — inter-chain residue/atom distance constraints
- `constraint.pocket` — pocket-definition constraints
- `constraint.structure` — accepted and treated as no-op (matches Python v0.5)

## JSON Input Format

Minimal single-task example:

```json
[{
  "name": "my_prediction",
  "sequences": [
    {"proteinChain": {"sequence": "MKQLLED...", "count": 1}},
    {"ligand": {"ligand": "CCD_ATP", "count": 1}}
  ]
}]
```

Accepted input shapes:
- A single task object `{...}`
- An array of task objects `[{...}, {...}]`
- A wrapper object `{"tasks": [{...}, {...}]}` (Python-compatible)

## Design Workflow

Design targets use YAML format:

```yaml
target:
  structure_path: structures/target.cif
  chains:
    - chain_id: A
      crop: "1-116"
      hotspot_residues: [40, 99, 107]
binder:
  n_residues: 80
```

Invoke via the CLI:

```julia
PXDesign.main(["infer", "-i", "design_input.yaml", "-o", "./output"])
```

## CLI Reference

All commands are invoked via `PXDesign.main(args)`:

| Command | Description |
|---------|-------------|
| `predict --input <json> --out_dir <dir>` | Run JSON or sequence prediction |
| `predict --sequence <seq> --out_dir <dir>` | Predict from a raw sequence |
| `predict --list-models` | List supported model variants |
| `tojson --input <pdb/cif> --out_dir <dir>` | Convert structure to inference JSON |
| `msa --input <json> --precomputed_msa_dir <dir>` | Attach precomputed MSA to JSON |
| `infer -i <json/yaml> -o <dir>` | Low-level inference (JSON or YAML design) |
| `check-input --yaml <yaml>` | Validate a YAML design input |
| `parity-check <ref_dir> <actual_dir>` | Numeric parity comparison |

## Model Weights

Weights are downloaded automatically from HuggingFace on first use:

- Repository: `MurrellLab/PXDesign.jl`
- Revision: `main`

For offline mode, prefetch weights once, then set:

```bash
export PXDESIGN_WEIGHTS_LOCAL_FILES_ONLY=true
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PXDESIGN_WEIGHTS_REPO_ID` | `MurrellLab/PXDesign.jl` | HuggingFace repository for model weights |
| `PXDESIGN_WEIGHTS_REVISION` | `main` | Git revision/branch for weights |
| `PXDESIGN_WEIGHTS_LOCAL_FILES_ONLY` | `false` | Skip network; use cached weights only |
| `PXDESIGN_ESM_LOCAL_FILES_ONLY` | `false` | Skip network for ESM weights |
| `PXDESIGN_ESM_REPO_ID` | `facebook/esmfold_v1` | ESM2 weight source |
| `PXDESIGN_ESM_FILENAME` | `model.safetensors` | ESM2 weight file |
| `PXDESIGN_ESM_REVISION` | `ba837a3` | ESM2 weight revision |
| `PXDESIGN_ESM_ISM_REPO_ID` | (from weights repo) | ISM-tuned ESM2 source |
| `PXDESIGN_ESM_ISM_FILENAME` | (model-specific) | ISM weight file |
| `PXDESIGN_ESM_ISM_REVISION` | (from weights revision) | ISM weight revision |
| `PXDESIGN_ESM_ISM_LOADER` | `fair_esm2` | ISM loader backend |
| `PXDESIGN_PYTHON_VENV` | — | Python venv path (for reference scripts only) |

## ESM / ISM Embeddings

ESM embeddings are required by `protenix_mini_esm_v0.5.0` and `protenix_mini_ism_v0.5.0`.

**Automatic mode** (default): When using `fold()` or `predict_*`, ESM embeddings are
generated automatically via `ESMFold.jl`. No user action needed.

**Explicit mode**: Supply a pre-computed embedding matrix:
- REPL: `fold(h, seq; esm_token_embedding=my_matrix)`
- JSON: add `task.esm_token_embedding` field with shape `[N_token, D]`

ISM variant uses an ISM-tuned ESM2 checkpoint. Configure source via
`PXDESIGN_ESM_ISM_*` environment variables.

## MSA Support

Precomputed MSA can be attached to protein chains:

```json
{"proteinChain": {
  "sequence": "MKQLLED...",
  "msa": {
    "precomputed_msa_dir": "path/to/msa_dir",
    "pairing_db": "uniref100"
  }
}}
```

The MSA directory should contain `non_pairing.a3m` (and `pairing.a3m` for multi-chain tasks).
Enable MSA consumption with `use_msa=true` in predict options.

Online/local MSA search is not implemented — only precomputed A3M files are supported.

## Known Gaps

1. **Template features for v1.0 models**: v1.0 models have an active TemplateEmbedder
   (`n_blocks=2`), but Julia does not yet compute template features (distogram, unit
   vector, pseudo-beta mask, backbone frame mask). The embedder currently receives zeros.
   v0.5 models are unaffected (templates disabled).
2. **REPL API for complex inputs**: `fold()` only accepts a single protein sequence string.
   Multi-chain complexes, ligands, ions, covalent bonds, and constraints require JSON
   input via `predict_json()`. A richer REPL API is planned.
3. **Online MSA search**: Not implemented. Only precomputed A3M files are supported.
4. **Amber relax**: Not implemented.
5. **Heteromer MSA pairing**: Simplified pairing by inferred taxonomy keys with row-index
   fallback. Full OpenFold species/taxonomic pairing is out of scope.
6. **SMILES ligand conformers**: SMILES-to-3D coordinate generation produces different
   3D coordinates than Python's RDKit pipeline (RDKit's internal C++ RNG is not seeded
   by Python's `random.seed()`). This causes `ref_pos` and `frame_atom_index` to differ
   for SMILES ligands. All other features (atom names, bonds, ref_mask, restype) match.
   CCD ligands are unaffected. This is an inherent difference that does not affect model
   behavior (the model is trained with random conformer augmentation).
7. **Modified residue tokenization**: Non-standard amino acids (PTMs like SEP, MSE) and
   modified nucleic acid bases (6OG, PSU, etc.) are tokenized differently from Python.
   Python queries CCD `_chem_comp.type` to classify components as "PEPTIDE LINKING" (1
   protein token) vs ligand (per-atom tokens). Julia does not yet implement this CCD
   mol_type lookup, causing wrong token counts, MSA shape mismatches, and restype errors
   for inputs with modified residues.
8. **Multi-chain MSA parity**: Parity has not yet been verified for inputs with separate
   precomputed MSAs per chain (e.g., different A3M files for chain A and chain B in a
   heterodimer). Single-chain MSA and homomer MSA are verified.

### Covalent Bond Support

Covalent bonds are supported for **Protenix prediction models** (protenix_base, protenix_mini)
via the `covalent_bonds` field in JSON input. PXDesign design models (`pxdesign_v0.1.0`)
do not use covalent bonds — they are a binder-design model family that takes YAML input
specifying a target structure and hotspot residues.

## Testing

```julia
using Pkg
Pkg.test("PXDesign")
```

Or run the test suite directly:

```bash
julia --project=<env> test/runtests.jl
```

This covers: config validation, model listing, mixed-entity parsing, covalent bonds,
MSA ingestion, template/ESM features, end-to-end smoke forwards for all model families,
and CLI smoke behavior.

## Developer Documentation

Internal porting notes, audit logs, and architecture docs are in `porting/`.
