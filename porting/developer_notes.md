# Developer Handoff Notes

Last updated: 2026-02-15

This document is the practical handoff for the mixed Python + Julia PXDesign workspace at:

- `/Users/benmurrell/JuliaM3/PXDesign`

It covers:

1. Repo structure and model scope
2. Exact weight/checkpoint files and what they are used for
3. How to run Julia tests/checks/parity
4. How to run end-to-end inference/folding examples
5. Geometry sanity checks for CIF outputs
6. How to set up the official Python reference on a separate system
7. What is still not implemented

## 1) Repository Layout

Top-level repository contains both:

- Official Python PXDesign code (reference implementation)
  - `pxdesign/`
  - `examples/`
  - `install.sh`
  - `download_tool_weights.sh`
- Julia port and tooling
  - `PXDesign.jl/`

Important Julia paths:

- Package source: `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/src`
- CLI entrypoint: `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/bin/pxdesign`
- Tests: `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/test`
- Scripts/parity tooling: `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/scripts`
- Status docs/audits: `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/docs`

## 2) Model Families in This Workspace

Three inference families are actively wired:

1. PXDesign infer scaffold (Julia)
2. Protenix-Mini v0.5 (Julia)
3. Protenix-Base v0.5 (Julia), including constraint variant

Current reality:

- Protenix mini/base paths are the strongest parity paths in this repo.
- PXDesign infer runs end-to-end and writes CIFs, but exact Python architecture parity is still in progress.

## 3) Checkpoints and Weights (Exact Filenames)

### 3.1 Upstream `.pt` checkpoints (source of truth)

Located in:

- `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/release_data/checkpoint`

Files:

- `pxdesign_v0.1.0.pt`
- `protenix_mini_default_v0.5.0.pt`
- `protenix_mini_tmpl_v0.5.0.pt`
- `protenix_base_default_v0.5.0.pt`
- `protenix_base_constraint_v0.5.0.pt`

### 3.2 PXDesign safetensors (converted)

Directory:

- `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/weights_safetensors`

File:

- `model.safetensors`

### 3.3 Protenix safetensors (converted)

Mini default:

- `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/weights_safetensors_protenix_mini_default_v0.5.0/model.safetensors`

Mini tmpl:

- `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/weights_safetensors_protenix_mini_tmpl_v0.5.0/model.safetensors`

Base default (sharded):

- `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/weights_safetensors_protenix_base_default_v0.5.0/model-00001-of-00002.safetensors`
- `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/weights_safetensors_protenix_base_default_v0.5.0/model-00002-of-00002.safetensors`
- `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/weights_safetensors_protenix_base_default_v0.5.0/model.safetensors.index.json`

Base constraint (sharded):

- `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/weights_safetensors_protenix_base_constraint_v0.5.0/model-00001-of-00002.safetensors`
- `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/weights_safetensors_protenix_base_constraint_v0.5.0/model-00002-of-00002.safetensors`
- `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/weights_safetensors_protenix_base_constraint_v0.5.0/model.safetensors.index.json`

### 3.4 Raw tensor bundles (manifest + bin files)

Each raw bundle contains:

- `manifest.json`
- many `tensor_XXXXXX.bin`

Key directories:

- `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/weights_raw`
- `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/weights_raw_protenix_mini_default_v0.5.0`
- `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/weights_raw_protenix_mini_tmpl_v0.5.0`
- `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/weights_raw_protenix_base_default_v0.5.0`
- `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/weights_raw_protenix_base_constraint_v0.5.0`

### 3.5 Non-model data cache files

Located in:

- `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/release_data/ccd_cache`

Files:

- `components.v20240608.cif`
- `components.v20240608.cif.rdkit_mol.pkl`
- `clusters-by-entity-40.txt`

## 4) Julia Environment and Runtime

Pinned Julia used in this workspace:

- `~/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia`

Recommended sandbox-safe depot setup:

```bash
cd /Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl
JULIA_DEPOT_PATH=$PWD/.julia_depot:$HOME/.julia \
JULIAUP_DEPOT_PATH=$PWD/.julia_depot \
~/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia --project=. -e 'import Pkg; Pkg.instantiate()'
```

CPU-only note for this machine:

- Do not use GPU/MPS for this repo on this Mac.

## 5) Core CLI Commands (Julia)

From `PXDesign.jl` directory:

### 5.1 PXDesign infer

```bash
JULIA_DEPOT_PATH=$PWD/.julia_depot:$HOME/.julia \
JULIAUP_DEPOT_PATH=$PWD/.julia_depot \
~/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia --project=. \
bin/pxdesign infer -i /path/to/input.json -o ./output
```

### 5.2 Input YAML syntax check

```bash
~/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia --project=. \
bin/pxdesign check-input --yaml /path/to/input.yaml
```

### 5.3 Protenix JSON/sequence prediction

```bash
# JSON tasks
bin/pxdesign predict --input /path/to/input.json --out_dir ./output --model_name protenix_base_default_v0.5.0

# Sequence-only
bin/pxdesign predict --sequence ACDEFGHIKLMNPQRSTVWY --out_dir ./output --model_name protenix_mini_default_v0.5.0
```

### 5.4 Model listing

```bash
bin/pxdesign predict --list-models
```

## 6) End-to-End Examples

### 6.1 Protenix-mini fold (Julia-only)

```bash
cd /Users/benmurrell/JuliaM3/PXDesign
JULIA_DEPOT_PATH=$PWD/PXDesign.jl/.julia_depot:$HOME/.julia \
JULIAUP_DEPOT_PATH=$PWD/PXDesign.jl/.julia_depot \
~/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia --project=PXDesign.jl \
PXDesign.jl/scripts/fold_sequence_protenix_mini.jl "ACDEFGHIKLMNPQRSTVWY"
```

Output location pattern:

- `/Users/benmurrell/JuliaM3/PXDesign/output/protenix_mini_sequence/seed_<seed>/predictions/*.cif`

### 6.2 Protenix-base fold (Julia-only)

```bash
cd /Users/benmurrell/JuliaM3/PXDesign
JULIA_DEPOT_PATH=$PWD/PXDesign.jl/.julia_depot:$HOME/.julia \
JULIAUP_DEPOT_PATH=$PWD/PXDesign.jl/.julia_depot \
~/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia --project=PXDesign.jl \
PXDesign.jl/scripts/fold_sequence_protenix_base.jl "ACDEFGHIKLMNPQRSTVWY"
```

### 6.3 PXDesign conditional design run (target + binder)

Example command used successfully with 400 default steps (`N_step=400`) and 1 sample:

```bash
cd /Users/benmurrell/JuliaM3/PXDesign
cat > /tmp/pxd_conditional_short.yaml <<'YAML'
task_name: pxd_cond_short
binder_length: 32
target:
  file: /Users/benmurrell/JuliaM3/PXDesign/examples/5o45.cif
  chains:
    A:
      hotspots: [40, 99, 107]
YAML

JULIA_DEPOT_PATH=$PWD/PXDesign.jl/.julia_depot:$HOME/.julia \
JULIAUP_DEPOT_PATH=$PWD/PXDesign.jl/.julia_depot \
~/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia --project=PXDesign.jl \
PXDesign.jl/bin/pxdesign infer \
  -i /tmp/pxd_conditional_short.yaml \
  -o /Users/benmurrell/JuliaM3/PXDesign/output/pxdesign_conditional_defaultsteps \
  --N_sample 1 \
  --set model_scaffold.enabled=true \
  --set model_scaffold.auto_dims_from_weights=true \
  --set model_scaffold.use_design_condition_embedder=true \
  --set safetensors_weights_path=/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/weights_safetensors \
  --set strict_weight_load=true
```

Output example:

- `/Users/benmurrell/JuliaM3/PXDesign/output/pxdesign_conditional_defaultsteps/global_run_0/pxd_cond_short/seed_906451/predictions/pxd_cond_short_sample_0.cif`

### 6.4 Tiny E2E smoke scripts

- Raw-weights path: `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/scripts/run_e2e_cpu_smoke.jl`
- Safetensors path: `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/scripts/run_e2e_safetensors_smoke.jl`

These are small `N_step=2` smoke tests (not quality tests).

## 7) Geometry Sanity Checking

Geometry checker script:

- `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/scripts/geometry_check.py`

Usage:

```bash
python /Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/scripts/geometry_check.py /path/to/prediction.cif
```

It reports summary stats for:

- `N-CA`
- `CA-C`
- `C-O`
- `C-N(next)`

And flags a coarse count:

- `out_of_range_0.9_1.8`

Interpretation:

- This is a coarse guardrail, not a full structural validation pipeline.
- Use it to catch obviously broken outputs (exploded bonds / disconnected geometry artifacts).

## 8) Test and Validation Matrix

### 8.1 Main Julia test suite

```bash
cd /Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl
JULIA_DEPOT_PATH=$PWD/.julia_depot:$HOME/.julia \
JULIAUP_DEPOT_PATH=$PWD/.julia_depot \
~/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia --project=. test/runtests.jl
```

Covers:

- YAML/JSON input handling
- feature construction
- CIF output structure
- Protenix/PXDesign smoke paths
- many typed API regressions

### 8.2 Layer regression fixture test (frozen goldens)

```bash
cd /Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl
JULIA_DEPOT_PATH=$PWD/.julia_depot:$HOME/.julia \
JULIAUP_DEPOT_PATH=$PWD/.julia_depot \
~/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia --project=. test/layer_regression.jl
```

Fixtures:

- `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/test/regression_fixtures/layer_regression_v1.bin`

Intentional regen is guarded and requires explicit magic env var:

```bash
cd /Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl
PXDESIGN_LAYER_FIXTURE_REGEN=do-not-set-this \
~/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia --project=. \
scripts/generate_layer_regression_fixtures.jl
```

### 8.3 Optional Python YAML parity check in tests

Disabled by default. To enable:

```bash
PXDESIGN_ENABLE_PYTHON_PARITY_TESTS=1 \
JULIA_DEPOT_PATH=$PWD/.julia_depot:$HOME/.julia \
JULIAUP_DEPOT_PATH=$PWD/.julia_depot \
~/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia --project=. test/runtests.jl
```

## 9) Parity Checks in This Repo

### 9.1 Raw-vs-raw tensor parity (general)

- Script: `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/scripts/compare_parity_raw.jl`
- CLI wrapper: `bin/pxdesign parity-check`

Compares two raw bundles and reports failed/missing keys.

### 9.2 Raw-vs-safetensors parity

- Script: `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/scripts/check_raw_vs_safetensors_parity.jl`

Use this after conversion to verify zero mismatches.

### 9.3 Protenix parity suite (Python dump + Julia compare)

- Orchestrator: `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/scripts/run_protenix_parity_suite.jl`
- Requires Python env at: `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/.venv_pyref/bin/python`

Runs parity for:

- MSA module
- Pairformer module
- Protenix-mini trunk+denoise(+heads)
- Protenix-base trunk+denoise(+heads)
- Protenix-base-constraint trunk+denoise(+heads)

### 9.4 PXDesign step/block debug compares (manual)

- `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/scripts/compare_python_blocks.jl`
- `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/scripts/compare_python_step.jl`

These consume diagnostics at:

- `/tmp/py_step_blocks.json`
- `/tmp/py_step_diag.json`

Use only when those artifacts are intentionally generated.

### 9.5 Input tensor parity caveats (`compare_python_input_tensors.jl`)

The script `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/scripts/compare_python_input_tensors.jl` now has two important controls:

- `--allow-ref-pos-rigid-equiv true|false` (default `true`)
- `--strict-keyset true|false` (default `true`)

Why this exists:

- In Protenix Python inference, `ref_pos` can be SE(3)-transformed per residue in the JSON featurizer path, so direct coordinate values can differ even when residue-internal geometry is equivalent.
- With `--allow-ref-pos-rigid-equiv true`, `ref_pos` is accepted when per-`ref_space_uid` pairwise distances match within tolerance, and reported as `float_rigid_equivalent`.
- This avoids false negatives from augmentation-style coordinate transforms while still catching geometry drift.

Recommended local parity invocation (value parity focus):

```bash
cd /Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl
JULIA_DEPOT_PATH=$PWD/.julia_depot:$HOME/.julia \
JULIAUP_DEPOT_PATH=$PWD/.julia_depot \
~/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia --project=. \
scripts/compare_python_input_tensors.jl \
  --input-json output/input_tensor_parity_full/input_single_chain.json \
  --python-dump-dir output/input_tensor_parity_full/python_dumps \
  --model-name protenix_base_default_v0.5.0 \
  --seed 101 \
  --use-default-params true \
  --use-msa false \
  --ref-pos-augment false \
  --allow-ref-pos-rigid-equiv true \
  --strict-keyset false
```

For ESM models in network-restricted environments, inject Python-dumped ESM embeddings to avoid HF fetches:

- add `--inject-python-esm true`

### 9.6 Official Python parity runner (guarded)

Use:

- `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/scripts/run_input_tensor_parity_official.sh`

What it enforces:

- Uses official `.external/Protenix` runner pipeline.
- Fails fast if `.external/Protenix` has any code changes other than:
  - `M runner/inference.py` (tensor-dump instrumentation only)
  - `?? esm_embeddings/` (cache artifacts)

What it produces:

- Python dumps: `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/output/input_tensor_parity_official/python_dumps`
- Raw reports: `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/output/input_tensor_parity_official/reports_raw`
- Rigid-equivalent reports: `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/output/input_tensor_parity_official/reports_rigid`
- Logs: `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/output/input_tensor_parity_official/logs`

## 10) Weight Conversion Utilities

### 10.1 Export `.pt` to raw bundle

- Script: `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/scripts/export_checkpoint_raw.py`

Example:

```bash
python /Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/scripts/export_checkpoint_raw.py \
  --checkpoint /Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/release_data/checkpoint/pxdesign_v0.1.0.pt \
  --outdir /Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/weights_raw \
  --cast-float32
```

### 10.2 Convert raw bundle to safetensors

- Script: `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/scripts/convert_raw_to_safetensors.py`

Example:

```bash
python /Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/scripts/convert_raw_to_safetensors.py \
  --raw-dir /Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/weights_raw \
  --out-dir /Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/weights_safetensors
```

### 10.3 One-command prepare/audit scripts

- `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/scripts/prepare_protenix_mini_safetensors.sh`
- `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/scripts/prepare_protenix_base_safetensors.sh`
- `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/scripts/prepare_protenix_base_constraint_safetensors.sh`

These run conversion + parity checks + audit report generation.

## 11) Setting Up Official Python PXDesign on a Separate System

Use upstream path from top-level repository (outside Julia package):

### Option A: Conda installer script

From `/Users/benmurrell/JuliaM3/PXDesign`:

```bash
bash -x install.sh --env pxdesign --pkg_manager conda --cuda-version 12.1
```

Then install first-time tool weights:

```bash
bash download_tool_weights.sh
```

### Option B: Docker (official README path)

```bash
docker build -t pxdesign -f Dockerfile .
docker run -it --gpus all pxdesign bash
# inside container:
git clone https://github.com/bytedance/PXDesign.git
cd PXDesign
pip install --upgrade pip
pip install -e .
```

Then run:

```bash
bash download_tool_weights.sh
```

### Quick Python runtime smoke (this workspace helper)

The Julia workspace includes helper scripts for a local Python reference env:

- setup: `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/scripts/setup_python_reference_env.sh`
- activate env: `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/scripts/python_reference_env.sh`
- smoke run: `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/scripts/run_python_reference_smoke.sh`

Example:

```bash
cd /Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl
scripts/setup_python_reference_env.sh
scripts/run_python_reference_smoke.sh
```

## 12) Known Not Implemented / Out of Scope (Current)

### 12.1 PXDesign

Still not complete:

- Exact Python PXDesign architecture parity (`pxdesign/model/*.py`) remains in progress.
- Full Python feature pipeline parity remains incomplete (full featurizer/CCD/ref/permutation/mask suite).

Important behavior caveat:

- `pxdesign infer` default config has `model_scaffold.enabled=false`.
- If scaffold is disabled, denoiser fallback path can still produce CIF output (use scaffold+weights for meaningful model behavior).

### 12.2 Protenix API/runtime limits

Out-of-scope or not implemented right now:

- MSA search (online/local) is not implemented (precomputed A3M ingestion is supported).
- Template search is not implemented.
- Full OpenFold species/taxonomic pairing semantics are out of scope (key-based pairing + fallback exists).
- RDKit-equivalent ligand chemistry/conformer parity is out of scope (Julia-native SMILES path exists).

### 12.3 Internal cleanup still open

From tracker (`docs/CODEBASE_ISSUES_TRACKER.md`):

- Remaining `Dict{String,Any}` reduction in stable config internals.
- Continued Julia memory-locality cleanup in hot kernels.

## 13) High-Value Files to Read First

If you are onboarding and need orientation quickly:

1. `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/README.md`
2. `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/docs/PORTING_PLAN.md`
3. `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/docs/INFER_ONLY_AUDIT.md`
4. `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/docs/PROTENIX_API_SURFACE_AUDIT.md`
5. `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/docs/CODEBASE_ISSUES_TRACKER.md`

## 14) Suggested Daily Validation Workflow

1. Run core Julia tests:

```bash
cd /Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl
JULIA_DEPOT_PATH=$PWD/.julia_depot:$HOME/.julia \
JULIAUP_DEPOT_PATH=$PWD/.julia_depot \
~/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia --project=. test/runtests.jl
```

2. Run layer regression fixtures:

```bash
~/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia --project=. test/layer_regression.jl
```

3. Run Protenix parity suite when touching model math:

```bash
~/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia --project=. scripts/run_protenix_parity_suite.jl
```

4. Run one small end-to-end fold/design and geometry-check the CIF.
