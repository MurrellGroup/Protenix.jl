# Protenix v0.5 API Surface Audit (Python vs Julia)

This document compares upstream `protenix` user-facing commands with the Julia port in this repository.

## Scope

- Upstream reference: `.external/Protenix/runner/batch_inference.py`
- Julia CLI entrypoint: `bin/pxdesign` (`src/cli.jl`)
- Julia API entrypoint: `src/protenix_api.jl`

## Coverage Matrix

| Python API | Python behavior | Julia status | Julia entrypoint |
|---|---|---|---|
| `predict --input <json|dir>` | Runs inference for JSON tasks | Supported (protein-only task entities) | `pxdesign predict --input ...` / `PXDesign.predict_json(...)` |
| `predict --model_name` | Model variant selection | Supported for v0.5 mini/base model names | same |
| `predict --seeds a,b,c` | Multi-seed inference | Supported | same |
| `predict --cycle --step --sample` | Inference controls | Supported | same |
| `predict --use_default_params` | Applies per-model recommended defaults | Supported | same |
| `predict --use_msa` | Enables MSA use/search path | Uses precomputed `non_pairing.a3m` (and `pairing.a3m` for multi-chain inputs); MSA search not implemented in Julia | same |
| `tojson --input <pdb|cif|dir>` | Convert structures to infer JSON | Supported for protein-chain extraction | `pxdesign tojson ...` / `PXDesign.convert_structure_to_infer_json(...)` |
| `tojson --altloc` | Altloc handling | `first` only | same |
| `tojson --assembly_id` | Bioassembly expansion | Not supported yet | same (errors explicitly) |
| `msa --input <json>` | Search/update MSA in JSON | Search not implemented; precomputed MSA attachment supported | `pxdesign msa ...` / `PXDesign.add_precomputed_msa_to_json(...)` |
| `msa --input <fasta>` | Run MSA search by FASTA | Not implemented yet | explicit error |

## Model Variant Handling

Supported model-name defaults in Julia:

- `protenix_base_default_v0.5.0`: cycle=10, step=200, sample=5, use_msa=true
- `protenix_base_constraint_v0.5.0`: cycle=10, step=200, sample=5, use_msa=true
- `protenix_mini_default_v0.5.0`: cycle=4, step=5, sample=5, use_msa=true
- `protenix_mini_tmpl_v0.5.0`: cycle=4, step=5, sample=5, use_msa=true
- `protenix_tiny_default_v0.5.0`: cycle=4, step=5, sample=5, use_msa=true

Recognized but currently blocked in Julia runtime until ESM2/ISM embedding wiring is completed:

- `protenix_mini_esm_v0.5.0`
- `protenix_mini_ism_v0.5.0`

## Current Functional Limits (Explicit)

`predict_json(...)` currently supports `proteinChain` entities only. The following input entities are not yet supported in the Julia infer JSON path and will raise clear errors:

- `dnaSequence`
- `rnaSequence`
- `ligand`
- `ion`
- `covalent_bonds`
- `constraint`

Template note (v0.5 parity): upstream Protenix v0.5 keeps `TemplateEmbedder` disabled (`forward` returns zero). Julia mirrors this behavior; template features are not an active signal path for these checkpoints.

## Output Notes

Julia prediction output preserves mmCIF generation and seed/sample directory structure and now writes per-sample confidence summary JSON files from Julia model outputs.

When `predict --use_msa true`, Julia consumes precomputed A3M files from JSON `proteinChain.msa.precomputed_msa_dir`:

- always reads `non_pairing.a3m`
- additionally requires/reads `pairing.a3m` for multi-chain assemblies
- applies A3M lowercase-deletion transforms (`has_deletion`, `deletion_value`, `deletion_mean`) and profile remapping to Protenix residue indices
- for multi-chain inputs, current merge is per-chain/block-concatenated (full OpenFold-style species pairing/merge parity is not fully ported yet)
