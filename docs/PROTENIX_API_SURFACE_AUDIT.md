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
| `tojson --assembly_id` | Bioassembly expansion | Supported for mmCIF input (`id` or `all`) | same |
| `msa --input <json>` | Search/update MSA in JSON | Search not implemented; precomputed MSA attachment supported | `pxdesign msa ...` / `PXDesign.add_precomputed_msa_to_json(...)` |
| `msa --input <fasta>` | Run MSA search by FASTA | Not implemented yet | explicit error |

## Model Variant Handling

Supported model-name defaults in Julia:

- `protenix_base_default_v0.5.0`: cycle=10, step=200, sample=5, use_msa=true
- `protenix_base_constraint_v0.5.0`: cycle=10, step=200, sample=5, use_msa=true
- `protenix_mini_default_v0.5.0`: cycle=4, step=5, sample=5, use_msa=true
- `protenix_mini_tmpl_v0.5.0`: cycle=4, step=5, sample=5, use_msa=true
- `protenix_tiny_default_v0.5.0`: cycle=4, step=5, sample=5, use_msa=true

Recognized and runnable in Julia:

- `protenix_mini_esm_v0.5.0`
- `protenix_mini_ism_v0.5.0`

ESM note: automatic ESM2 token-embedding generation is implemented in Julia via `ESMFold.jl` (with explicit repo/file overrides for ISM variants).

## Current Functional Limits (Explicit)

Task JSON ingestion supports:

- single task object
- task array
- wrapper object with `tasks: [...]`
- `add_precomputed_msa_to_json(...)` preserves input container shape on write (object/array/tasks-wrapper)

`predict_json(...)` supports these task entities in Julia:

- `proteinChain`
- `dnaSequence`
- `rnaSequence`
- `ligand`:
  - `CCD_*`
  - `SMILES` (Julia-native parsing path)
  - `SMILES_*` (prefix alias normalized to SMILES payload for Python compatibility)
  - `FILE_*` (structure file path)
  - `condition_ligand` alias support
- `ion`
- `covalent_bonds`:
  - name-based atom references
  - numeric ligand atom indices via `atom_map_to_atom_name`

Constraint-path status in this infer path:

- `constraint` conditioning is implemented with parity checks:
  - `constraint.contact` and `constraint.pocket` are ingested into `constraint_feature`
  - same-chain `constraint.contact` pairs are rejected (Python parity)
  - same-chain binder/contact residue in `constraint.pocket` is rejected (Python parity)
  - `max_distance >= min_distance` validation is enforced (Python parity)
  - pair `z` receives additive constraint embeddings when constraint embedder modules are enabled
  - `constraint.structure` is accepted and currently treated as a no-op for JSON inference (matching current Python v0.5 `generate_from_json` behavior)
  - substructure embedder supports `linear`/`mlp`/`transformer` architectures with state-load mapping
- real constraint checkpoint conversion/load coverage is validated (`4109/4109` tensors parity raw vs safetensors).
- full end-to-end numeric parity for constraint-conditioned forwards is now covered via:
  - `scripts/dump_python_protenix_base_constraint_trunk_denoise_parity.py`
  - `scripts/compare_protenix_base_constraint_trunk_denoise_parity.jl`
- `SMILES` ligand handling is Julia-native and currently not RDKit-equivalent in conformer generation/chemistry normalization.

Template note (v0.5 parity): upstream Protenix v0.5 keeps `TemplateEmbedder` disabled (`forward` returns zero). Julia mirrors this behavior; template features are not an active signal path for these checkpoints.

Quality note (behavioral parity): very shallow constraint-model sampling (for example `cycle=2, step=6`) can yield poor geometry in both Python and Julia; recommended settings (`cycle=10, step=200`) recover realistic bond geometry.

## Output Notes

Julia prediction output preserves mmCIF generation and seed/sample directory structure and now writes per-sample confidence summary JSON files from Julia model outputs.

When `predict --use_msa true`, Julia consumes precomputed A3M files from JSON `proteinChain.msa.precomputed_msa_dir`:

- always reads `non_pairing.a3m`
- additionally requires/reads `pairing.a3m` for multi-chain assemblies
- applies A3M lowercase-deletion transforms (`has_deletion`, `deletion_value`, `deletion_mean`) and profile remapping to Protenix residue indices
- for heteromer inputs with `pairing.a3m`, paired rows are merged across chains by inferred keys (`TaxID`/`ncbi_taxid`/`OX`/`Tax`/`OS`) when available, with safe row-index fallback
- full OpenFold species/taxonomic pairing parity is currently out of scope
