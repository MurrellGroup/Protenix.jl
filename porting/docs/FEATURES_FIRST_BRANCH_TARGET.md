# features_first Branch — Development Target

This document defines the acceptance criteria for the `features_first` branch.
The branch converts PXDesign's shared model layers (`src/model/`) from features-last
`(N, features)` to features-first `(features, N, batch)` convention, replaces custom
primitives with Onion.jl layers, and enables GPU execution. Everything listed below
was working on `main` before the refactor started and must work identically (or better)
on this branch, on GPU.

## 1. Model Inventory

All models below must load, run inference, and produce correct protein geometry on GPU.

| Model name | Family | Status on `main` | Target on `features_first` |
|---|---|---|---|
| `pxdesign_v0.1.0` | design | Partially done; `bin/pxdesign infer` writes CIFs | Same functionality, GPU, features-first |
| `protenix_base_default_v0.5.0` | base | Working; strong Python parity | Working on GPU, features-first, parity preserved |
| `protenix_base_constraint_v0.5.0` | base | Working; contact/pocket constraint parity | Working on GPU, features-first, parity preserved |
| `protenix_mini_default_v0.5.0` | mini | Working; tight module/forward parity | Working on GPU, features-first, parity preserved |
| `protenix_mini_tmpl_v0.5.0` | mini | Working; template embedder no-op parity | Working on GPU, features-first |
| `protenix_tiny_default_v0.5.0` | mini | Working; lighter validation | Working on GPU, features-first |
| `protenix_mini_esm_v0.5.0` | mini | Working with ESM setup | Working on GPU, features-first, ESM path |
| `protenix_mini_ism_v0.5.0` | mini | Working with ISM ESM setup | Working on GPU, features-first, ISM path |

### Parity quality targets (from `main` baseline)

These tolerances were achieved on `main` against Python reference and must be preserved:

- Pairformer/MSA/Confidence: max abs `1e-5` to `1e-3`
- Trunk + denoise + heads: tight deterministic parity
- Constraint-conditioned: trunk `max_abs < 1.5e-5`, diffusion `max_abs < 1e-5`, confidence `max_abs < 4e-5`

## 2. Test Map

Every test/script below must pass on `features_first`, running on GPU where applicable.

### 2.1 Unit test suite

| Test | Script | What it validates |
|---|---|---|
| Full test suite | `test/runtests.jl` | Layer regression, config, API surface, parsing, MSA, templates, ESM, e2e smoke, constraints, state load, scheduler, sampler, embedders, primitives, diffusion, atom attention, parity harness |
| Layer regression fixtures | `test/layer_regression_reference.jl` | Frozen per-layer output fixtures (regenerated for features-first) |

### 2.2 End-to-end fold scripts

| Model(s) | Script | What it validates |
|---|---|---|
| PXDesign (`pxdesign_v0.1.0`) | `scripts/run_e2e_cpu_smoke.jl` | Design model infer smoke (writes CIF) |
| PXDesign (`pxdesign_v0.1.0`) | `scripts/run_e2e_safetensors_smoke.jl` | Design model infer via safetensors/HF path |
| PXDesign (`pxdesign_v0.1.0`) | `scripts/run_julia_reference_smoke.jl` | Design model reference smoke (tiny input, writes CIF) |
| Protenix-mini default | `scripts/fold_sequence_protenix_mini.jl` | Julia-only fold from sequence (writes CIF) |
| Protenix-base default | `scripts/fold_sequence_protenix_base.jl` | Julia-only fold from sequence (writes CIF) |

**GPU target**: Fold scripts must support `gpu=true` and produce correct protein geometry on GPU. Specifically:
- CA-CA backbone distances: mean ~3.8A, >95% within 3.0-4.5A
- N-CA bond lengths: mean ~1.47A
- Correct side chain placement (no exploded/random side chains)
- Realistic secondary structure (helices/sheets, not random coil)

### 2.3 Python-vs-Julia parity suite

| Model(s) | Script | What it validates |
|---|---|---|
| Protenix-mini/base/base-constraint | `scripts/run_protenix_parity_suite.jl` | End-to-end MSA/Pairformer/trunk+denoise+heads parity |
| All v0.5 variants | `scripts/run_input_tensor_parity_official.sh` | Official Python input-tensor parity sweep |

### 2.4 Per-model parity compare scripts

| Script | What it validates |
|---|---|
| `scripts/compare_protenix_mini_trunk_denoise_parity.jl` | Mini trunk + denoise + heads vs Python |
| `scripts/compare_protenix_base_trunk_denoise_parity.jl` | Base trunk + denoise + heads vs Python |
| `scripts/compare_protenix_base_constraint_trunk_denoise_parity.jl` | Constraint base trunk + denoise + heads vs Python |
| `scripts/compare_python_input_tensors.jl` | Input tensor construction parity |
| `scripts/compare_pairformer_parity.jl` | Pairformer module-level parity |
| `scripts/compare_msa_parity.jl` | MSA module-level parity |
| `scripts/compare_confidence_parity.jl` | Confidence head module-level parity |

## 3. What Changed in the Refactor

### 3.1 Tensor layout transformation

| Tensor | `main` (features-last) | `features_first` |
|---|---|---|
| single (s) | `(N_token, c_s)` | `(c_s, N_token, batch)` |
| pair (z) | `(N_token, N_token, c_z)` | `(c_z, N_token, N_token, batch)` |
| attention bias | `(N, N, n_heads)` | `(n_heads, N_q, N_k, batch)` |
| coordinates | `(N_sample, N_atom, 3)` | `(3, N_atom, batch)` |

### 3.2 Layer replacements

| `main` layer | `features_first` replacement | Source |
|---|---|---|
| `LayerNormNoOffset` | `Onion.LayerNormFirst` | `Onion.jl/src/protein/layernorm.jl` |
| `LinearNoBias` | `Onion.BGLinear(...; bias=false)` | `Onion.jl/src/protein/boltzgen/` |
| `Transition` (custom) | `Onion.Transition` | `Onion.jl/src/protein/boltzgen/transition.jl` |
| Naive O(N^2) attention | `Onion.AttentionPairBias` | `Onion.jl/src/protein/boltzgen/attention_pair_bias.jl` |

### 3.3 Bridge layer (ProtenixMini)

`src/protenix_mini/` still uses features-last convention internally. The bridge between
ProtenixMini (features-last) and the shared DiffusionModule (features-first) is in
`src/protenix_mini/model.jl:run_inference` via `permutedims` calls:

```julia
s_inputs_ff = permutedims(trunk.s_inputs)           # (N, c) -> (c, N)
s_trunk_ff = permutedims(trunk.s)                    # (N, c) -> (c, N)
z_trunk_ff = permutedims(trunk.z, (3, 1, 2))        # (N, N, c) -> (c, N, N)
```

### 3.4 GPU execution

- `predict_sequence(...; gpu=true)` and `predict_json(...; gpu=true)` move model to GPU
- `features_to_device` keeps int/bool arrays on CPU, moves float arrays to GPU
- `_model_dev_ref` detects GPU from model weights for automatic feature transfer

## 4. Current Status

### Done

- [x] Shared model primitives replaced with Onion layers (Step 1)
- [x] Embedders adapted for features-first (Step 2)
- [x] DiffusionConditioning adapted (Step 3)
- [x] TransformerBlocks adapted with Onion.AttentionPairBias (Step 4)
- [x] AtomAttention adapted (Step 5)
- [x] DiffusionModule adapted (Step 6)
- [x] DesignConditionEmbedder adapted (Step 7)
- [x] State loading updated for Onion field names (Step 8)
- [x] Sampler adapted (Step 9)
- [x] Sharded HuggingFace weight download fixed
- [x] GPU feature transfer fixed (keep int arrays on CPU)
- [x] ESM auto-detection fixed (use MODEL_SPECS)
- [x] Unit tests passing (`Pkg.test("PXDesign")`)
- [x] GPU e2e: mini, tiny, base all fold with real weights on GPU
- [x] Backbone geometry verified (CA-CA, N-CA, Rg all correct)
- [x] `dump_prediction_bundle` accepts GPU CuArray coordinates (`AbstractArray` signature)
- [x] **Bond length parity: main vs FF confirmed identical** (mini 200 steps: both 6/407, 1.5%)
- [x] **`protenix_mini_default_v0.5.0`**: GPU fold verified, bond parity with main confirmed
- [x] **`protenix_base_default_v0.5.0`**: GPU fold verified (200 steps)
- [x] **`protenix_tiny_default_v0.5.0`**: GPU fold verified via predict_sequence API
- [x] **`protenix_base_constraint_v0.5.0`**: GPU fold verified (200 steps, 178s)
- [x] **`protenix_mini_tmpl_v0.5.0`**: GPU fold verified (200 steps, 48.4s)
- [x] **`protenix_mini_esm_v0.5.0`**: GPU fold verified with auto ESM embedding (115.2s)
- [x] **`protenix_mini_ism_v0.5.0`**: GPU fold verified with auto ISM embedding (1347.6s)
- [x] GPU confirmed via CUDA.jl memory tracking (mini: 9.9 GB, base: 18.6 GB)
- [x] nvidia-smi "Not Supported" explained: GB10 Grace Blackwell unified memory limitation
- [x] **REPL-friendly API**: `load_protenix()` / `fold()` / `confidence_metrics()` implemented
  - `ProtenixHandle` struct for model reuse (load once, predict many)
  - Proper pLDDT conversion (logits → 0-100 score via softmax over bin centers)
  - `result.cif` for in-memory CIF text access
  - `result.plddt`, `result.mean_plddt`, `result.mean_pae` for confidence metrics
- [x] **Python Protenix reference env**: installed at `.venv_pyref/` (Python 3.14, CPU-only)
- [x] **Python reference outputs**: mini (200 steps), tiny (200 steps), base (200 steps) — CIF + confidence JSON
- [x] **`pxdesign_v0.1.0`**: GPU design model e2e validated (5-step smoke: 23.1s, 200-step full: 28.5s)
  - Added `gpu=true` config option to `Infer.run_infer()`
  - Model + features auto-transferred to GPU, `device_ref` passed to sampler
- [x] **Layer regression fixtures regenerated** for features-first (42 entries, all pass)

### Bond Geometry Results (200 steps, hemoglobin 51 aa)

| Model | Violations | Total | Rate% | BB | SC |
|-------|-----------|-------|-------|-----|-----|
| jl_mini_200_gpu | 54 | 410 | 13.2 | 47 | 7 |
| jl_constraint_200_gpu | 56 | 410 | 13.7 | 51 | 5 |
| py_base_200 | 56 | 410 | 13.7 | 51 | 5 |
| jl_base_200_gpu | 65 | 410 | 15.9 | 51 | 14 |
| py_tiny_200 | 88 | 410 | 21.5 | 52 | 36 |
| jl_ism_200_gpu | 94 | 410 | 22.9 | 53 | 41 |
| py_mini_200 | 102 | 410 | 24.9 | 54 | 48 |
| jl_esm_200_gpu | 128 | 410 | 31.2 | 63 | 65 |
| jl_tmpl_200_gpu | 139 | 410 | 33.9 | 73 | 66 |

Julia and Python outputs show comparable bond geometry (different seeds/RNG states).
Note: Initial bond check had a CIF column parsing bug — Python Protenix CIFs place `occupancy`
as the last column (col 20) while Julia CIFs put it at col 15, shifting coordinate columns.
Fixed parser reads `_atom_site.*` header dynamically.

### Python Reference Confidence Scores (seed=101, 200 steps)

| Model | pLDDT | ptm | ranking_score |
|-------|-------|-----|---------------|
| py_mini_200 | 52.69 | 0.296 | 0.059 |
| py_tiny_200 | 53.03 | 0.307 | 0.061 |
| py_base_200 | 56.59 | 0.300 | 0.060 |

### Not Yet Done

- [ ] ProtenixMini modules (Step 10) — still features-last, not yet converted to Onion layers
- [ ] Per-model parity compare scripts updated for features-first
- [ ] Timing benchmark: GPU `features_first` vs CPU `main`
- [ ] Small molecule support (BoltzGen pattern)

## 5. How to Run Parity Check

### 5.1 Julia-vs-main (same-codebase parity)

The `main` branch is accessible via git worktree at `/home/claudey/FixingKAFA/PXDesign_main`.

```bash
# Run original (main) code
julia --project=<main_env> scripts/fold_sequence_protenix_mini.jl "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"

# Run features_first code (GPU)
julia --project=ka_run_env scripts/fold_sequence_protenix_mini.jl "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"

# Compare output CIF coordinates
```

### 5.2 Python-vs-Julia parity

Requires Python reference env (Tier C setup). See `docs/PURE_JULIA_STATUS_AND_ENV_SETUP.md`.

## 6. Environment

- Julia: `/home/claudey/.julia/juliaup/julia-1.11.8+0.aarch64.linux.gnu/bin/julia`
- Run environment: `/home/claudey/FixingKAFA/ka_run_env`
- GPU: NVIDIA GB10 Grace Blackwell (128 GB unified memory; nvidia-smi cannot track memory/utilization)
- Branch: `features_first` (latest commit `c5fb46e`)
- Original: `main` (commit `555a905`)
- Worktree for `main`: `/home/claudey/FixingKAFA/PXDesign_main`
