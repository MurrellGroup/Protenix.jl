# Protenix-Mini Port Status

This document tracks Julia coverage for the Protenix-mini v0.5.0 checkpoints.

## Scope

- Full Protenix-mini infer-only architecture in Julia:
  - `input_embedder`
  - `relative_position_encoding`
  - `template_embedder` (no-op behavior parity for `n_blocks=0`)
  - `msa_module`
  - `pairformer_stack`
  - `diffusion_module`
  - `distogram_head`
  - `confidence_head`
  - recycling trunk linears/layer norms
- Checkpoint conversion:
  - PyTorch `.pt` -> raw tensor bundle -> SafeTensors
- Parity validation:
  - module-level parity (Pairformer, MSA, Confidence)
  - trunk + denoise + heads parity

## Current State

- `build_protenix_mini_model` + `load_protenix_mini_model!(...; strict=true)` loads
  both:
  - `protenix_mini_default_v0.5.0`
  - `protenix_mini_tmpl_v0.5.0`
- Pairformer/MSA/Confidence parity is tight (typical max abs in `1e-5` to `1e-3` range).
- Deterministic trunk+denoise+head parity against Python is tight.
- Julia-only sequence folding path is available:
  - `PXDesign.build_sequence_feature_bundle`
  - `PXDesign.fold_sequence`
  - CLI script: `scripts/fold_sequence_protenix_mini.jl`

## Repro Commands

From `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl`:

1. Prepare safetensors + audits:

```bash
scripts/prepare_protenix_mini_safetensors.sh
```

2. Pairformer parity:

```bash
source scripts/python_reference_env.sh
python scripts/dump_python_pairformer_parity.py \
  --checkpoint release_data/checkpoint/protenix_mini_default_v0.5.0.pt \
  --out /tmp/py_pairformer_diag.json

JULIA_DEPOT_PATH=$PWD/.julia_depot JULIAUP_DEPOT_PATH=$PWD/.julia_depot \
~/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia --project=. \
scripts/compare_pairformer_parity.jl
```

3. MSA parity:

```bash
source scripts/python_reference_env.sh
python scripts/dump_python_msa_parity.py \
  --checkpoint release_data/checkpoint/protenix_mini_default_v0.5.0.pt \
  --out /tmp/py_msa_diag.json

JULIA_DEPOT_PATH=$PWD/.julia_depot JULIAUP_DEPOT_PATH=$PWD/.julia_depot \
~/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia --project=. \
scripts/compare_msa_parity.jl
```

4. Confidence parity:

```bash
source scripts/python_reference_env.sh
python scripts/dump_python_confidence_parity.py \
  --checkpoint release_data/checkpoint/protenix_mini_default_v0.5.0.pt \
  --out /tmp/py_conf_diag.json

JULIA_DEPOT_PATH=$PWD/.julia_depot JULIAUP_DEPOT_PATH=$PWD/.julia_depot \
~/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia --project=. \
scripts/compare_confidence_parity.jl
```

5. Trunk + denoise + heads parity:

```bash
source scripts/python_reference_env.sh
python scripts/dump_python_protenix_mini_trunk_denoise_parity.py \
  --checkpoint release_data/checkpoint/protenix_mini_default_v0.5.0.pt \
  --out /tmp/py_protenix_mini_trunk_denoise_diag.json

JULIA_DEPOT_PATH=$PWD/.julia_depot JULIAUP_DEPOT_PATH=$PWD/.julia_depot \
~/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia --project=. \
scripts/compare_protenix_mini_trunk_denoise_parity.jl
```

6. Julia-only sequence folding:

```bash
JULIA_DEPOT_PATH=$PWD/.julia_depot JULIAUP_DEPOT_PATH=$PWD/.julia_depot \
~/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia --project=. \
scripts/fold_sequence_protenix_mini.jl "ACDEFGHIKLMNPQRSTVWY"
```

Outputs go to:

- `output/protenix_mini_sequence/seed_0/predictions/*.cif`
