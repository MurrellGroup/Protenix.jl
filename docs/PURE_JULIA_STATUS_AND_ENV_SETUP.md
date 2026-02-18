# Pure-Julia Status and Full Environment Setup

This document answers two practical questions:

1. Where is this repo not fully "pure Julia" today?
2. What exact environment setup is needed to run the full functionality present in this repo?

It is written for maintainers/operators who want reproducible local runs on a fresh machine.

## 1) Purity boundary: what is pure Julia vs not

### 1.1 Pure-Julia runtime paths

The runtime package paths used by `bin/pxdesign` inference and Protenix prediction are Julia-native:

- `src/`
- `bin/pxdesign`

Core runtime operations that are Julia-native:

- YAML/JSON ingest and normalization (`YAML.jl`, local JSON layer)
- mmCIF/PDB parsing and output writing
- PXDesign infer sampling path
- Protenix v0.5 model loading from safetensors
- Protenix predict APIs (`predict`, `tojson`, `msa` precomputed ingest)

### 1.2 Not pure Julia (repo-level tooling sidecars)

The following are intentional Python/tooling sidecars and are not part of pure Julia runtime inference:

- Python parity and diagnostics scripts in `scripts/`:
  - `scripts/dump_python_*.py`
  - `scripts/export_checkpoint_raw.py`
  - `scripts/convert_raw_to_safetensors.py`
  - `scripts/geometry_check.py`
  - `scripts/geometry_check_sidechain.py`
- Python reference env bootstrap:
  - `scripts/setup_python_reference_env.sh`
  - `scripts/python_reference_env.sh`
  - `.venv_pyref/`
  - `.pydeps/`
- Python reference source mirrors:
  - `.external/Protenix`
  - `.external/PXDesignBench`
- Optional Python-backed test/parity hooks:
  - `test/runtests.jl` testset `"Inputs YAML vs PyYAML parity (supported subset)"` (opt-in)
  - `scripts/run_protenix_parity_suite.jl`
  - `scripts/run_input_tensor_parity_official.sh`

### 1.3 Mixed dependency notes (still Julia runtime)

These are Julia dependencies, but still external model/data dependencies:

- `ESMFold.jl` + `Onion.jl` for auto ESM embeddings.
- HuggingFace-hosted safetensors weights.
- Optional CCD/cache files (`release_data/ccd_cache`).

This means runtime is Julia code, but not "self-contained without external assets."

## 2) Functional tiers and required setup

### 2.1 Tier A: Julia-only inference (no Python parity)

Use this for product/runtime usage.

Requirements:

- Julia `1.11.x` (repo currently validated on `1.11.2`)
- `Pkg.instantiate` for this project
- HuggingFace access (unless weights are already cached and local-only mode is enabled)

Setup:

```bash
cd /Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl
JULIA_DEPOT_PATH=$PWD/.julia_depot:$HOME/.julia \
JULIAUP_DEPOT_PATH=$PWD/.julia_depot \
~/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia --project=. -e 'import Pkg; Pkg.instantiate()'
```

Recommended weight source env:

```bash
export PXDESIGN_WEIGHTS_REPO_ID=MurrellLab/PXDesign.jl
export PXDESIGN_WEIGHTS_REVISION=main
export PXDESIGN_WEIGHTS_LOCAL_FILES_ONLY=false
```

Offline mode after prefetch:

```bash
export PXDESIGN_WEIGHTS_LOCAL_FILES_ONLY=true
```

### 2.2 Tier B: Julia-only inference including auto ESM paths

Needed for:

- `protenix_mini_esm_v0.5.0`
- `protenix_mini_ism_v0.5.0` with automatic embedding injection

Additional requirements:

- `ESMFold.jl` + `Onion.jl` resolved by `Project.toml`/`Manifest.toml`
- ESM weights available through configured source(s)

Optional ISM override env:

```bash
export PXDESIGN_ESM_ISM_REPO_ID=<repo>
export PXDESIGN_ESM_ISM_FILENAME=<filename>
export PXDESIGN_ESM_ISM_REVISION=<revision>
export PXDESIGN_ESM_LOCAL_FILES_ONLY=true
```

If auto ESM is unavailable, explicit `esm_token_embedding` injection remains supported.

### 2.3 Tier C: Full local parity/audit environment (Python + Julia)

Needed for "all functionality available in this repo," including official Python comparisons, checkpoint conversion, and Python/Julia parity scripts.

Additional requirements:

- Python 3.11 environment
- Torch + reference Python deps
- Optional RDKit for matching reference environment tooling
- Cloned/available `.external/Protenix` reference source

Bootstrap command:

```bash
cd /Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl
scripts/setup_python_reference_env.sh
source scripts/python_reference_env.sh
```

Reference smoke command:

```bash
bash scripts/run_python_reference_smoke.sh
```

## 3) CCD and cache data sources

### 3.1 Lookup order

CCD/component lookups use this precedence:

1. `PROTENIX_DATA_ROOT_DIR` (if set)
2. `release_data/ccd_cache` in repo

Files involved:

- `components.v20240608.cif` (or `components.cif`)
- `ref_coords_std.json` (used by feature construction path)

### 3.2 Download path

`bin/pxdesign infer` only downloads cache assets if:

- `download_cache=true`

Otherwise cache download is skipped.

## 4) What is required for customer-facing runtime package use

For deployment/runtime usage without parity workflows:

- Julia + project dependencies
- HuggingFace safetensors access (or pre-cached local files with local-only mode)
- Optional ESM setup if using ESM/ISM model variants

Not required:

- Python virtualenv
- `.external/Protenix`
- Python parity scripts
- RDKit-side tooling

## 5) Comprehensive validation workflows and prerequisites

### 5.1 Julia-only comprehensive test run

```bash
cd /Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl
JULIA_DEPOT_PATH=$PWD/.julia_depot:$HOME/.julia \
JULIAUP_DEPOT_PATH=$PWD/.julia_depot \
~/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia --project=. test/runtests.jl
```

Includes:

- config/schema/JSON/YAML parser checks
- mixed-entity Protenix parsing (protein/dna/rna/ligand/ion)
- covalent bond handling for ligand atom names/indices
- ESM embedding injection checks
- mini/base/constraint e2e smoke forwards
- infer scaffold non-dry-run smoke

### 5.2 Frozen layer regression fixtures

```bash
JULIA_DEPOT_PATH=$PWD/.julia_depot:$HOME/.julia \
JULIAUP_DEPOT_PATH=$PWD/.julia_depot \
~/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia --project=. test/layer_regression.jl
```

Fixture regeneration (guarded):

```bash
PXDESIGN_LAYER_FIXTURE_REGEN=do-not-set-this \
~/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia --project=. \
scripts/generate_layer_regression_fixtures.jl
```

### 5.3 Python-backed parity suites (optional)

Requires Tier C setup.

- Protenix parity suite:

```bash
~/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia --project=. \
scripts/run_protenix_parity_suite.jl
```

- Official Python input-tensor parity sweep:

```bash
bash scripts/run_input_tensor_parity_official.sh
```

- Optional PyYAML parity inside `test/runtests.jl`:

```bash
PXDESIGN_ENABLE_PYTHON_PARITY_TESTS=1 \
JULIA_DEPOT_PATH=$PWD/.julia_depot:$HOME/.julia \
JULIAUP_DEPOT_PATH=$PWD/.julia_depot \
~/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia --project=. test/runtests.jl
```

## 6) Current practical conclusion

- Runtime inference paths (`src/` + `bin/pxdesign`) are Julia-native.
- The repo is not "pure Julia only" at the tooling/ecosystem level because Python parity, conversion, and diagnostics tooling are intentionally included and maintained.
- To replicate all capabilities currently used in this development environment, set up both:
  - Julia runtime environment (Tier A/B), and
  - Python reference/parity environment (Tier C).
