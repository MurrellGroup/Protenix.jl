# PXDesign.jl

Infer-only Julia port of PXDesign.

## Status

- Implemented:
  - package scaffold and `bin/pxdesign` CLI (`infer`, `check-input`, `parity-check`, `predict`, `tojson`, `msa`)
  - YAML/JSON input normalization (native Julia via `YAML.jl`)
  - native mmCIF/PDB atom parsing + chain/crop filtering
  - condition+binder feature bundle assembly
  - diffusion scheduler/sampler parity primitives
  - typed model scaffolding modules:
    - embedders (`RelativePositionEncoding`, `ConditionTemplateEmbedder`)
    - primitives (`LinearNoBias`, `LayerNormNoOffset`, `AdaptiveLayerNorm`, `Transition`)
    - transformer blocks (`AttentionPairBias`, `ConditionedTransitionBlock`, `DiffusionTransformer`)
    - atom attention stack (`AtomAttentionEncoder`, `AtomAttentionDecoder`) with shared cross-attention blocks
    - design condition embedder (`InputFeatureEmbedderDesign`, `DesignConditionEmbedder`)
    - diffusion conditioning + diffusion module scaffold
    - raw weight loading for design + diffusion module trees
  - infer scaffold that writes real CIF predictions and PXDesign-compatible output tree
  - numeric parity harness for raw snapshot bundles:
    - `PXDesign.Model.tensor_parity_report`
    - `PXDesign.Model.compare_raw_weight_dirs`
    - `scripts/compare_parity_raw.jl`
- In progress:
  - exact Python model architecture port (`pxdesign/model/*.py`)
  - ESM2 integration for Protenix-mini ESM/ISM variants:
    - automatic `esm_token_embedding` production from Julia `ESMFold.jl` is implemented
    - JSON/sequence API still accepts explicit user-provided `esm_token_embedding` (`[N_token, D]`) and uses it as override

Parity policy:

- Python-vs-Julia parity is validated locally via scripts/tests.
- Reference parity artifacts are not committed and are out of scope for release packaging.

### Protenix v0.5 ESM Notes

- ESM is variant-dependent in Protenix v0.5:
  - ESM enabled: `protenix_mini_esm_v0.5.0`, `protenix_mini_ism_v0.5.0`
  - ESM disabled by default: `protenix_base_default_v0.5.0`, `protenix_base_constraint_v0.5.0`, `protenix_mini_default_v0.5.0`, `protenix_mini_tmpl_v0.5.0`, `protenix_tiny_default_v0.5.0`
- `esm2-3b-ism` refers to an ISM-tuned ESM2 checkpoint (ISM = Implicit Structure Model in the upstream cited work).
- For ESM/ISM model variants:
  - default behavior is automatic ESM embedding generation in Julia via `ESMFold.jl`
  - explicit embeddings remain supported and take precedence over auto-generation:
    - JSON mode: add top-level `task.esm_token_embedding` (`[N_token, D]`)
    - sequence mode: pass `--esm_token_embedding_json /path/to/embedding.json`
- ISM variant source configuration:
  - default `esm2_3b` path uses `ESMFold.load_ESM` defaults (`facebook/esmfold_v1` safetensors)
  - for `protenix_mini_ism_v0.5.0`, set:
    - `PXDESIGN_ESM_ISM_REPO_ID`
    - `PXDESIGN_ESM_ISM_FILENAME`
    - `PXDESIGN_ESM_ISM_REVISION`
  - optional offline mode:
    - `PXDESIGN_ESM_LOCAL_FILES_ONLY=true`

### Protenix v0.5 User API (Julia)

Python-like Protenix commands are available via `bin/pxdesign`:

```bash
# JSON predict (single file or directory)
bin/pxdesign predict --input /path/to/input.json --out_dir ./output \
  --model_name protenix_base_default_v0.5.0 --seeds 101,102

# Sequence-only predict
bin/pxdesign predict --sequence ACDEFGHIKLMNPQRSTVWY --out_dir ./output \
  --model_name protenix_mini_default_v0.5.0 --step 5 --sample 1

# List supported model variants + defaults
bin/pxdesign predict --list-models

# PDB/CIF -> infer JSON conversion
bin/pxdesign tojson --input /path/to/structure.cif --out_dir ./output

# mmCIF bioassembly expansion (single id or all)
bin/pxdesign tojson --input /path/to/structure.cif --assembly_id 1 --out_dir ./output

# Attach precomputed MSA path to an infer JSON
bin/pxdesign msa --input /path/to/input.json \
  --precomputed_msa_dir /path/to/msa_dir --out_dir ./output
```

`predict --use_msa true` now consumes precomputed A3Ms from JSON `proteinChain.msa.precomputed_msa_dir` (reads `non_pairing.a3m`, and `pairing.a3m` for multi-chain tasks). Julia still does not run online/local MSA search.
For heteromer assemblies, Julia merges `pairing.a3m` rows across chains by inferred keys (`TaxID`/`ncbi_taxid`/`OX`/`Tax`/`OS`) when available, with fallback to row index; full OpenFold species/taxonomic pairing parity is currently out of scope.

Detailed API coverage vs Python is documented in:

- `docs/PROTENIX_API_SURFACE_AUDIT.md`

Current Julia `predict` infer-JSON path supports mixed entities:

- `proteinChain`
- `dnaSequence`
- `rnaSequence`
- `ligand`:
  - `CCD_*` ligands
  - `SMILES` ligands (Julia-native parser path)
  - `SMILES_*` prefixed ligand strings (Python-compatible input alias)
  - `FILE_*` ligands (local structure-file ligand path)
  - `condition_ligand` alias is accepted for compatibility
- `ion`
- `covalent_bonds`:
  - atom-name fields (`left_atom`/`right_atom` or `atom1`/`atom2`)
  - numeric atom indices for ligand entities via `atom_map_to_atom_name`

Input-shape compatibility note:

- `predict` / `msa` accept task JSON as:
  - a single task object
  - an array of task objects
  - a wrapper object with `tasks: [...]` (Python-compatible layout)
- `msa` preserves the input task-container shape in its output JSON (object/array/tasks-wrapper).

Constraint path status in this runtime:

- `constraint.contact` / `constraint.pocket` ingestion + typed constraint embedder plumbing: implemented
- constraint JSON validation now matches Python v0.5 semantics:
  - same-chain `constraint.contact` pairs are rejected
  - same-chain binder/contact residue in `constraint.pocket` is rejected
  - `max_distance >= min_distance` is enforced
- `constraint.structure`: accepted for JSON inference and treated as a no-op (matches current Python v0.5 behavior)
- real checkpoint conversion/load coverage validated for `protenix_base_constraint_v0.5.0`
- forward numerical parity checks are now wired:
  - `scripts/dump_python_protenix_base_constraint_trunk_denoise_parity.py`
  - `scripts/compare_protenix_base_constraint_trunk_denoise_parity.jl`
- practical quality note: very shallow constraint sampling (for example `cycle=2, step=6`) can produce poor geometry in both Python and Julia; use recommended settings (`cycle=10, step=200`) for realistic folds.

### Typed Protenix Features (Julia-first runtime path)

For model runtime, Protenix-mini/base now supports a typed feature container (`ProtenixFeatures`) so hot inference paths avoid `Dict{String,Any}` dispatch.
`Dict` remains at I/O boundaries (JSON/YAML ingestion), then converts once:

```julia
bundle = PXDesign.ProtenixMini.build_sequence_feature_bundle("ACDEFG")
feat = PXDesign.ProtenixMini.as_protenix_features(bundle["input_feature_dict"])
pred = PXDesign.run_inference(PXDesign.ProtenixMiniModel(...), feat; n_cycle=1, n_step=5, n_sample=1)
```

## Quick Start

```bash
cd /Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl
JULIA_DEPOT_PATH=$PWD/.julia_depot JULIAUP_DEPOT_PATH=$PWD/.julia_depot \
  ~/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia --project=. bin/pxdesign infer \
  -i /path/to/input.json \
  -o ./output \
  --set sample_diffusion.N_sample=2 \
  --set sample_diffusion.N_step=10
```

Validate YAML inputs:

```bash
JULIA_DEPOT_PATH=$PWD/.julia_depot JULIAUP_DEPOT_PATH=$PWD/.julia_depot \
  ~/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia --project=. bin/pxdesign check-input --yaml /path/to/input.yaml
```

Optional parser parity check against `python3 + PyYAML` (for supported PXDesign YAML subset):

```bash
~/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia --project=. \
  scripts/check_yaml_parity.jl /path/to/input.yaml
```

Default test/inference runtime is pure Julia. Python parity checks are opt-in.

Default config sets `download_cache=false` so infer runs without network. Enable PXDesign cache download explicitly with:

```bash
--set download_cache=true
```

Enable typed model-scaffold denoiser path:

```bash
--set model_scaffold.enabled=true
```

Model weights are now loaded from HuggingFace only (no local fallback directories).
Default source:

- `repo_id`: `MurrellLab/PXDesign.jl`
- `revision`: `main`

Configure weight source / offline mode with:

```bash
export PXDESIGN_WEIGHTS_REPO_ID=MurrellLab/PXDesign.jl
export PXDESIGN_WEIGHTS_REVISION=main
export PXDESIGN_WEIGHTS_LOCAL_FILES_ONLY=false
```

For fully offline execution after prefetching, set:

```bash
export PXDESIGN_WEIGHTS_LOCAL_FILES_ONLY=true
```

Legacy local weight overrides are intentionally disabled:

- `predict --weights_path ...`
- `infer --set raw_weights_dir=...`
- `infer --set safetensors_weights_path=...`

Checkpoint conversion bridge (for parity/audit workflows only):

```bash
python3 scripts/export_checkpoint_raw.py \
  --checkpoint /path/to/pxdesign_v0.1.0.pt \
  --outdir ./weights_raw \
  --cast-float32
```

Then load in Julia via `PXDesign.Model.load_raw_weights("./weights_raw")` for conversion/parity scripts.

Convert raw weights to safetensors (single file or shards):

```bash
python3 scripts/convert_raw_to_safetensors.py \
  --raw-dir ./weights_raw \
  --out-dir ./weights_safetensors
```

To avoid manual scaffold shape mismatches during conversion checks, infer model dimensions directly from raw weights (diffusion + design embedder):

```bash
--set model_scaffold.enabled=true \
--set model_scaffold.auto_dims_from_weights=true \
--set raw_weights_dir=./weights_raw
```

Enforce strict key coverage (all expected keys present, no unexpected keys in the loaded model subtrees):

```bash
--set strict_weight_load=true
```

With `strict_weight_load=true` and both model scaffold + design condition embedder enabled, key coverage is validated against the full infer-only checkpoint tree.

Use chunked diffusion sampling (matches Python runner structure for large `N_sample`) via:

```bash
--set infer_setting.sample_diffusion_chunk_size=10
```

Compare raw snapshot bundles for numeric parity:

```bash
~/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia --project=. \
  bin/pxdesign parity-check ./reference_raw ./actual_raw --atol 1e-5 --rtol 1e-4
```

Compare a raw bundle directly against a safetensors bundle:

```bash
~/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia --project=. \
  scripts/check_raw_vs_safetensors_parity.jl ./weights_raw ./weights_safetensors
```

Prepare Protenix-Mini safetensors and run parity + coverage audits in one command:

```bash
scripts/prepare_protenix_mini_safetensors.sh
```

This writes audit reports under:

- `output/protenix_mini_audit/`

See detailed status and coverage numbers in:

- `docs/PROTENIX_MINI_PORT_STATUS.md`

Prepare Protenix-Base v0.5.0 safetensors and run conversion + coverage audits:

```bash
scripts/prepare_protenix_base_safetensors.sh
```

Prepare Protenix-Base constraint v0.5.0 safetensors and run conversion + coverage audits:

```bash
scripts/prepare_protenix_base_constraint_safetensors.sh
```

See status/details in:

- `docs/PROTENIX_BASE_PORT_STATUS.md`
- `docs/PROTENIX_API_SURFACE_AUDIT.md`

Run Julia-only Protenix-Mini sequence folding:

```bash
JULIA_DEPOT_PATH=$PWD/.julia_depot JULIAUP_DEPOT_PATH=$PWD/.julia_depot \
~/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia --project=. \
scripts/fold_sequence_protenix_mini.jl "ACDEFGHIKLMNPQRSTVWY"
```

Run Julia-only Protenix-Base v0.5.0 sequence folding:

```bash
JULIA_DEPOT_PATH=$PWD/.julia_depot JULIAUP_DEPOT_PATH=$PWD/.julia_depot \
~/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia --project=. \
scripts/fold_sequence_protenix_base.jl "ACDEFGHIKLMNPQRSTVWY"
```

Both fold scripts resolve safetensors from HuggingFace (`MurrellLab/PXDesign.jl`) and accept source overrides via:

- `PXDESIGN_WEIGHTS_REPO_ID`
- `PXDESIGN_WEIGHTS_REVISION`
- `PXDESIGN_WEIGHTS_LOCAL_FILES_ONLY`

Run a tiny end-to-end CPU smoke with strict safetensors loading (`N_sample=1`, `N_step=2`):

```bash
~/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia --project=. \
  scripts/run_e2e_cpu_smoke.jl
```

Run the same tiny CPU smoke through safetensors (writes to a stable repo path under `output/safetensors_smoke`):

```bash
~/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia --project=. \
  scripts/run_e2e_safetensors_smoke.jl
```

### Sandbox-safe Julia runs

When running inside sandboxed environments, prefer a writable local depot while still reading preinstalled packages:

```bash
JULIA_DEPOT_PATH=$PWD/.julia_depot:$HOME/.julia \
JULIAUP_DEPOT_PATH=$PWD/.julia_depot \
~/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia --project=. test/runtests.jl
```

`test/runtests.jl` includes end-to-end smoke paths for all three model families in this repo:

- PXDesign infer scaffold (`Infer.run_infer` -> CIF output tree)
- Protenix-mini sequence fold (`ProtenixMini.fold_sequence` -> CIF)
- Protenix-base v0.5 sequence fold (`ProtenixBase.fold_sequence` -> CIF)

For Python-vs-Julia numeric checks across Protenix modules (MSA, pairformer, mini trunk+denoise, base trunk+denoise):

```bash
~/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia --project=. \
scripts/run_protenix_parity_suite.jl
```

Parity compare scripts still accept explicit local override env vars for parity workflows:

- `MSA_WEIGHTS_DIR`
- `PAIRFORMER_WEIGHTS_DIR`
- `PMINI_WEIGHTS_DIR`
- `PBASE_WEIGHTS_DIR`
- `PBASE_CONSTRAINT_WEIGHTS_DIR`

To run Python-backed parity checks inside `test/runtests.jl`, opt in explicitly:

```bash
PXDESIGN_ENABLE_PYTHON_PARITY_TESTS=1 \
JULIA_DEPOT_PATH=$PWD/.julia_depot:$HOME/.julia \
JULIAUP_DEPOT_PATH=$PWD/.julia_depot \
~/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia --project=. test/runtests.jl
```

### Frozen layer regression fixtures

The test suite includes locked layer-level regression fixtures across PXDesign + Protenix-mini/v0.5 paths:

```bash
JULIA_DEPOT_PATH=$PWD/.julia_depot:$HOME/.julia \
JULIAUP_DEPOT_PATH=$PWD/.julia_depot \
~/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia --project=. test/layer_regression.jl
```

Fixture regeneration is intentionally guarded and requires explicit opt-in:

```bash
PXDESIGN_LAYER_FIXTURE_REGEN=do-not-set-this \
~/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia --project=. \
scripts/generate_layer_regression_fixtures.jl
```

State-dict assignment helpers are in:

- `PXDesign.Model.load_condition_template_embedder!`
- `PXDesign.Model.load_design_condition_embedder!`
- `PXDesign.Model.load_relative_position_encoding!`
- `PXDesign.Model.load_diffusion_conditioning!`
- `PXDesign.Model.load_diffusion_transformer!`
- `PXDesign.Model.load_diffusion_module!`

## Port Plan

See:

- `docs/PORTING_PLAN.md`
- `docs/INFER_ONLY_AUDIT.md`
- `docs/MODEL_PORT_MAP.md`
- `docs/CODEBASE_ISSUES_TRACKER.md`
