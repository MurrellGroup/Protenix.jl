# Protenix Base v0.5.0 Port Audit

This document records safetensors conversion + coverage checks for:

- `protenix_base_default_v0.5.0`
- `protenix_base_constraint_v0.5.0`

## Artifacts

- Checkpoint:
  - `release_data/checkpoint/protenix_base_default_v0.5.0.pt`
- Raw export:
  - `weights_raw_protenix_base_default_v0.5.0`
- SafeTensors:
  - `weights_safetensors_protenix_base_default_v0.5.0`

## Conversion + Parity

- Raw tensor count: `4092`
- SafeTensors shard count: `2`
- Raw vs SafeTensors parity: exact (`4092/4092` compared, `0` failed, no missing keys)

Reports:

- `output/protenix_base_audit/protenix_base_default_v0.5.0_parity.toml`
- `output/protenix_base_audit/protenix_base_constraint_v0.5.0_parity.toml`

## Constraint Variant Forward Parity

Constraint-conditioned trunk/denoise/head parity is covered by Python-vs-Julia dump/compare scripts:

- Python dump:
  - `scripts/dump_python_protenix_base_constraint_trunk_denoise_parity.py`
- Julia compare:
  - `scripts/compare_protenix_base_constraint_trunk_denoise_parity.jl`

Latest saved report:

- `output/protenix_base_audit/protenix_base_constraint_forward_parity.txt`

Observed parity quality from the latest run:

- trunk substructure constraint path: max_abs `1.3828278e-5`
- trunk z after constraints: max_abs `1.4305115e-5`
- diffusion `x_denoised`: max_abs `8.285046e-6`
- confidence heads: max_abs `<= 3.9577484e-5`

## Component Coverage / Missing Check

Using current Julia Protenix stack (`PXDesign.ProtenixMini.*` loader):

- Strict state load: `PASS`
- Inferred model dims:
  - `c_token_diffusion=768`
  - `c_token_input=384`
  - `c_s=384`
  - `c_z=128`
  - `c_s_inputs=449`
  - `pairformer_blocks=48`
  - `msa_blocks=4`
  - `diffusion_blocks=24`
  - `diffusion_atom_encoder_blocks=3`
  - `diffusion_atom_decoder_blocks=3`

Root-key check:

- Unexpected root modules: `0`
- Observed roots are all in the currently ported infer-only Protenix stack:
  - `pairformer_stack`
  - `diffusion_module`
  - `confidence_head`
  - `msa_module`
  - `input_embedder`
  - `template_embedder`
  - `distogram_head`
  - plus trunk recycling/top-level linears + layernorms

Diffusion subtree check (legacy diffusion-only auditor):

- Diffusion tensors present: `766`
- Diffusion missing expected keys: `0`
- Diffusion unused keys: `0`

Reports:

- `output/protenix_base_audit/protenix_base_default_v0.5.0_diffusion_audit.toml`
- `output/protenix_base_audit/protenix_base_default_v0.5.0_port_audit.toml`

## Repro

One-command pipeline:

```bash
scripts/prepare_protenix_base_safetensors.sh
```

Julia-only end-to-end sequence folding:

```bash
JULIA_DEPOT_PATH=$PWD/.julia_depot JULIAUP_DEPOT_PATH=$PWD/.julia_depot \
~/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia --project=. \
scripts/fold_sequence_protenix_base.jl "ACDEFGHIKLMNPQRSTVWY"
```

Trunk + denoise + heads parity harness:

```bash
source scripts/python_reference_env.sh
python scripts/dump_python_protenix_base_trunk_denoise_parity.py \
  --checkpoint release_data/checkpoint/protenix_base_default_v0.5.0.pt \
  --out /tmp/py_protenix_base_trunk_denoise_diag.json

JULIA_DEPOT_PATH=$PWD/.julia_depot JULIAUP_DEPOT_PATH=$PWD/.julia_depot \
~/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia --project=. \
scripts/compare_protenix_base_trunk_denoise_parity.jl
```
