# API Reference

## Model Loading

### `load_protenix`

```julia
load_protenix(model_name="protenix_base_default_v0.5.0"; gpu=false, strict=true) → ProtenixHandle
```

Load a Protenix folding model. Weights are downloaded from HuggingFace on first use and cached locally.

**Arguments:**
- `model_name::AbstractString`: Model identifier. Supports aliases (e.g., `"protenix_v1"`). See [Models](models.md) for the full list.
- `gpu::Bool`: Move model to GPU after loading.
- `strict::Bool`: Require all weight keys to be present and consumed. Set `false` only for debugging.

**Returns:** `ProtenixHandle` — reusable handle for `fold()`.

```julia
h = load_protenix("protenix_v1"; gpu=true)
```

### `load_pxdesign`

```julia
load_pxdesign(model_name="pxdesign_v0.1.0"; gpu=false, strict=true) → PXDesignHandle
```

Load a PXDesign diffusion model for binder design.

**Returns:** `PXDesignHandle` — reusable handle for `design()`.

```julia
dh = load_pxdesign("pxdesign_v0.1.0"; gpu=true)
```

---

## Folding

### `fold` (single sequence)

```julia
fold(handle::ProtenixHandle, sequence::AbstractString;
     seed=101, step=nothing, sample=nothing, cycle=nothing,
     out_dir=nothing, task_name="protenix_sequence", chain_id="A",
     esm_token_embedding=nothing) → NamedTuple
```

Fold a single protein sequence.

**Arguments:**
- `handle`: From `load_protenix()`
- `sequence`: Amino acid sequence (one-letter codes)
- `seed::Integer`: RNG seed for diffusion sampling
- `step`, `sample`, `cycle`: Override model defaults (`nothing` = use defaults)
- `out_dir`: Directory for CIF output (temp dir if `nothing`)
- `task_name`: Identifier for this prediction
- `chain_id`: Chain label in output CIF
- `esm_token_embedding`: Pre-computed ESM embedding matrix `(N_token, D)`. Only for ESM/ISM models. If `nothing`, auto-generated when needed.

**Returns:** NamedTuple — see [Output Format](inputs.md#folding-output).

```julia
h = load_protenix("protenix_v1"; gpu=true)
result = fold(h, "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH")
println("pLDDT: ", result.mean_plddt)
```

### `fold` (multi-entity task)

```julia
fold(handle::ProtenixHandle, task::AbstractDict;
     seed=101, step=nothing, sample=nothing, cycle=nothing,
     out_dir=nothing) → NamedTuple
```

Fold a multi-entity complex from a task Dict built with `protenix_task()`.

```julia
task = protenix_task(
    protein_chain("MVLSPAD..."),
    ligand("CCD_ATP"),
    ion("MG"),
)
result = fold(h, task; seed=101)
```

---

## Design

### `design` (from InputTask)

```julia
design(handle::PXDesignHandle, task::InputTask;
       seed=42, n_step=nothing, n_sample=nothing, out_dir=nothing,
       gamma0=1.0, gamma_min=0.01, noise_scale_lambda=1.003,
       diffusion_chunk_size=0, eta_type="const", eta_min=1.5, eta_max=1.5)
       → NamedTuple
```

Run binder design on a pre-built `InputTask`.

**Arguments:**
- `handle`: From `load_pxdesign()`
- `task`: From `design_task()`
- `seed`: RNG seed
- `n_step`: Diffusion steps (default: handle's `default_n_step`, typically 200)
- `n_sample`: Number of samples (default: handle's `default_n_sample`, typically 5)
- `out_dir`: Output directory (temp dir if `nothing`)
- `gamma0`, `gamma_min`, `noise_scale_lambda`: Noise schedule parameters
- `eta_type`: Noise schedule type (`"const"` or `"piecewise_65"`)
- `eta_min`, `eta_max`: Eta range

**Returns:** NamedTuple — see [Output Format](inputs.md#design-output).

### `design` (convenience kwargs)

```julia
design(handle::PXDesignHandle;
       binder_length::Integer,
       target=nothing, hotspots=nothing, name=nothing,
       seed=42, n_step=nothing, n_sample=nothing, out_dir=nothing, ...)
       → NamedTuple
```

Convenience method that builds a `design_task()` internally.

```julia
dh = load_pxdesign(gpu=true)

# Unconditional
result = design(dh; binder_length=60, seed=42, n_sample=5)

# Conditional
target = design_target("1ubq.cif"; chains=["A"])
result = design(dh;
    binder_length=80,
    target=target,
    hotspots=Dict("A" => [8, 44, 48]),
    seed=42,
)
```

---

## Task Builders

### `protein_chain`

```julia
protein_chain(sequence; count=1, msa=nothing, modifications=nothing) → Dict
```

See [Entity Builders](inputs.md#entity-builders).

### `dna_chain`

```julia
dna_chain(sequence; count=1, modifications=nothing) → Dict
```

### `rna_chain`

```julia
rna_chain(sequence; count=1, unpaired_msa=nothing, unpaired_msa_path=nothing, modifications=nothing) → Dict
```

### `ligand`

```julia
ligand(name; count=1) → Dict
```

### `ion`

```julia
ion(name; count=1) → Dict
```

### `protenix_task`

```julia
protenix_task(entities...; name=nothing, constraint=nothing, covalent_bonds=nothing, template_features=nothing) → Dict
```

### `template_structure`

```julia
template_structure(cif_path, chain_id) → Dict
```

Build template features from an experimental structure for v1.0 model conditioning.

### `design_target`

```julia
design_target(structure_file; chains=String[], crop=nothing, msa=nothing) → NamedTuple
```

### `design_task`

```julia
design_task(; binder_length, target=nothing, hotspots=nothing, name=nothing) → InputTask
```

---

## Batch Prediction

### `predict_json`

```julia
predict_json(input;
    out_dir="./output",
    model_name="protenix_base_default_v0.5.0",
    weights_path="",
    seeds=[101],
    use_default_params=true,
    cycle=nothing, step=nothing, sample=nothing,
    use_msa=nothing,
    strict=true,
    gpu=false)
    → Vector{PredictJSONRecord}
```

Run prediction on JSON input files. Loads the model, runs all tasks/seeds, writes CIFs.

**Arguments:**
- `input`: Path to a JSON file or directory of JSON files
- `model_name`: Supports aliases (e.g., `"protenix_v1"`)
- `weights_path`: Explicit path to weights (empty = auto-download from HuggingFace)
- `seeds`: Vector of RNG seeds to run
- `use_default_params`: Use model's registered defaults for step/cycle/sample
- `use_msa`: Enable/disable MSA (`nothing` = model default)

**Returns:** Vector of `(input_json, task_name, seed, prediction_dir, cif_paths)`.

### `predict_sequence`

```julia
predict_sequence(sequence;
    out_dir="./output",
    model_name="protenix_base_default_v0.5.0",
    weights_path="",
    task_name="protenix_sequence",
    chain_id="A",
    seeds=[101],
    use_default_params=true,
    cycle=nothing, step=nothing, sample=nothing,
    use_msa=nothing,
    esm_token_embedding=nothing,
    strict=true,
    gpu=false)
    → Vector{PredictSequenceRecord}
```

**Returns:** Vector of `(task_name, seed, prediction_dir, cif_paths)`.

---

## Design Pipeline (Config-Based)

### `run_infer`

```julia
Protenix.run_infer(cfg::Dict{String, Any}; dry_run=false) → Dict
```

Low-level design inference from a config Dict. Used internally by the CLI and for YAML-based design.

```julia
cfg = Protenix.Config.default_config(; project_root=".")
cfg["input_json_path"] = "design_input.yaml"
cfg["dump_dir"] = "./output"
cfg["model_name"] = "pxdesign_v0.1.0"
cfg["seeds"] = [101]
cfg["gpu"] = true
Protenix.Config.set_nested!(cfg, "sample_diffusion.N_step", 200)
Protenix.Config.set_nested!(cfg, "sample_diffusion.N_sample", 5)

result = Protenix.run_infer(cfg)
# result["status"] == "ok_scaffold_model"
```

---

## Utilities

### `list_supported_models`

```julia
list_supported_models() → Vector{NamedTuple}
```

Return metadata for all registered models. Each entry has:
`model_name`, `family`, `default_cycle`, `default_step`, `default_sample`, `default_use_msa`, `needs_esm_embedding`, `msa_pair_as_unpair`.

### `recommended_params`

```julia
recommended_params(model_name; use_default_params=true, cycle=nothing, step=nothing, sample=nothing, use_msa=nothing) → NamedTuple
```

### `confidence_metrics`

```julia
confidence_metrics(result) → NamedTuple
```

Extract `(plddt, mean_plddt, pae, mean_pae, pde, resolved)` from a fold result.

### `convert_structure_to_infer_json`

```julia
convert_structure_to_infer_json(input; out_dir="./output", altloc="first", assembly_id=nothing) → Vector{String}
```

Convert PDB/mmCIF files to Protenix inference JSON. Supports biological assembly expansion.

### `add_precomputed_msa_to_json`

```julia
add_precomputed_msa_to_json(input_json; out_dir="./output", precomputed_msa_dir, pairing_db="uniref100") → Vector{String}
```

Attach MSA to an existing JSON file. Writes a new JSON with MSA fields added.

---

## Types

| Type | Description |
|------|-------------|
| `ProtenixHandle` | Loaded folding model (model, family, model_name, on_gpu, params) |
| `PXDesignHandle` | Loaded design model (model, design_condition_embedder, dims, defaults) |
| `ProtenixModelSpec` | Model metadata (name, family, defaults, feature flags) |
| `PredictJSONRecord` | Result from `predict_json` (input_json, task_name, seed, prediction_dir, cif_paths) |
| `PredictSequenceRecord` | Result from `predict_sequence` (task_name, seed, prediction_dir, cif_paths) |
| `Schema.InputTask` | Design task specification (name, structure_file, chain_ids, crop, hotspots, msa, generation) |
| `Schema.GenerationSpec` | What to generate (type, length, count) |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PROTENIX_WEIGHTS_REPO_ID` | `MurrellLab/PXDesign.jl` | HuggingFace repo for weights |
| `PROTENIX_WEIGHTS_REVISION` | `main` | Git revision for weights |
| `PROTENIX_WEIGHTS_LOCAL_FILES_ONLY` | `false` | Offline mode (cached only) |
| `PROTENIX_ESM_LOCAL_FILES_ONLY` | `false` | Offline mode for ESM weights |
| `PROTENIX_ESM_REPO_ID` | `facebook/esmfold_v1` | ESM2 weight source |
| `PROTENIX_ESM_ISM_REPO_ID` | *(from weights repo)* | ISM weight source |
| `PROTENIX_DATA_ROOT_DIR` | *(auto)* | Override data cache directory |

---

## CLI

All commands via `Protenix.main(args)`:

| Command | Description |
|---------|-------------|
| `predict --input <json> --out_dir <dir>` | JSON prediction |
| `predict --sequence <seq> --out_dir <dir>` | Sequence prediction |
| `predict --list-models` | List models |
| `tojson --input <pdb/cif> --out_dir <dir>` | Convert structure to JSON |
| `msa --input <json> --precomputed_msa_dir <dir>` | Attach MSA to JSON |
| `infer -i <json/yaml> -o <dir>` | Design inference |
| `check-input --yaml <yaml>` | Validate YAML |
