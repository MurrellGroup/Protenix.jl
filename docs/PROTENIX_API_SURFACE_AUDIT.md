# Protenix v0.5 API Surface Audit (Python vs Julia)

This document compares upstream `protenix` user-facing commands with the Julia port in this repository.

## Scope

- Upstream reference: `.external/Protenix/runner/batch_inference.py`
- Julia CLI entrypoint: `bin/pxdesign` (`src/cli.jl`)
- Julia API entrypoint: `src/protenix_api.jl`

## Coverage Matrix

| Python API | Python behavior | Julia status | Julia entrypoint |
|---|---|---|---|
| `predict --input <json|dir>` | Runs inference for JSON tasks | Supported for protein/dna/rna/ligand/ion mixed task entities | `pxdesign predict --input ...` / `PXDesign.predict_json(...)` |
| `predict --model_name` | Model variant selection | Supported for v0.5 mini/base model names | same |
| `predict --seeds a,b,c` | Multi-seed inference | Supported | same |
| `predict --cycle --step --sample` | Inference controls | Supported | same |
| `predict --use_default_params` | Applies per-model recommended defaults | Supported | same |
| `predict --list-models` | Discover model variants/defaults | Supported | `pxdesign predict --list-models` / `PXDesign.list_supported_models()` |
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

Recognized and runnable in Julia when `esm_token_embedding` is provided (automatic ESM2 feature generation is still pending):

- `protenix_mini_esm_v0.5.0`
- `protenix_mini_ism_v0.5.0`

## Current Functional Limits (Explicit)

`predict_json(...)` supports these task entities in Julia:

- `proteinChain`
- `dnaSequence`
- `rnaSequence`
- `ligand`:
  - `CCD_*`
  - `SMILES` (Julia-native parsing path)
  - `FILE_*` (structure file path)
  - `condition_ligand` alias support
- `ion`
- `covalent_bonds`:
  - name-based atom references
  - numeric ligand atom indices via `atom_map_to_atom_name`

Current remaining deltas in this infer path:

- `constraint` conditioning is partially wired:
  - `constraint.contact` and `constraint.pocket` are ingested into `constraint_feature`
  - pair `z` receives additive constraint embeddings when constraint embedder modules are enabled
  - `constraint.structure` is accepted and currently treated as a no-op for JSON inference (matching current Python v0.5 `generate_from_json` behavior)
  - substructure embedder supports `linear`/`mlp`/`transformer` architectures with state-load mapping
- pending: validation against the real constraint checkpoint bundle for full end-to-end numeric parity.
- `SMILES` ligand handling is Julia-native and currently not RDKit-equivalent in conformer generation/chemistry normalization.

Template note (v0.5 parity): upstream Protenix v0.5 keeps `TemplateEmbedder` disabled (`forward` returns zero). Julia mirrors this behavior; template features are not an active signal path for these checkpoints.

## Output Notes

Julia prediction output preserves mmCIF generation and seed/sample directory structure and now writes per-sample confidence summary JSON files from Julia model outputs.

When `predict --use_msa true`, Julia consumes precomputed A3M files from JSON `proteinChain.msa.precomputed_msa_dir`:

- always reads `non_pairing.a3m`
- additionally requires/reads `pairing.a3m` for multi-chain assemblies
- applies A3M lowercase-deletion transforms (`has_deletion`, `deletion_value`, `deletion_mean`) and profile remapping to Protenix residue indices
- for multi-chain inputs, current merge is per-chain/block-concatenated (full OpenFold-style species pairing/merge parity is not fully ported yet)
