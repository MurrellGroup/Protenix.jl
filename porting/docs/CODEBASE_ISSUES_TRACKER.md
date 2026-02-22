# Codebase Issues Tracker

This tracker focuses on two constraints:

1. user-facing Julia UX should be easy and explicit
2. internals should stay modular, typed, and reusable

Python is allowed only for explicit parity tooling, not runtime inference.

## Closed in Current Pass

1. Predict options were untyped kwargs spread across callsites.
   - Fixed by adding typed option structs in `src/protenix_api.jl`:
     - `ProtenixPredictOptions`
     - `ProtenixSequenceOptions`
   - `predict_json` / `predict_sequence` now have typed overloads.

2. Model discoverability from CLI was weak.
   - Added `PXDesign.list_supported_models()`.
   - Added CLI: `pxdesign predict --list-models`.

3. Default test path could opportunistically run Python checks.
   - PyYAML parity test now requires explicit opt-in:
     - `PXDESIGN_ENABLE_PYTHON_PARITY_TESTS=1`

4. Runtime Python boundary is now policy-defined.
   - Added explicit agent policy: runtime code in `src/` + `bin/` stays pure Julia.
   - Python remains limited to explicit parity/audit tooling in `scripts/`.

5. Infer JSON entity support expanded in `predict_json`.
   - Added mixed-entity parsing/runtime support:
     - `proteinChain`
     - `dnaSequence`
     - `rnaSequence`
     - `ligand` (`CCD_*`, `SMILES`, `FILE_*`)
     - `condition_ligand` (compat alias)
     - `ion`
   - Added covalent-bond injection support for:
     - atom-name bond fields
     - numeric ligand atom indices via `atom_map_to_atom_name`

6. Protenix task JSON ingestion/container handling was brittle.
   - Fixed by supporting all three input container layouts:
     - single task object
     - task array
     - wrapper object with `tasks: [...]`
   - `add_precomputed_msa_to_json(...)` now preserves input container shape on write.
   - Added regression tests for wrapper acceptance and invalid `tasks` type rejection.

7. Protenix predict return-path used abstract `Vector{NamedTuple}` accumulators.
   - Added concrete result record aliases:
     - `PredictJSONRecord`
     - `PredictSequenceRecord`
   - `predict_json` / `predict_sequence` now accumulate concrete record vectors.
   - Added end-to-end smoke coverage for wrapper-input `predict_json` and `predict_sequence` typed output records.

8. MSA runtime internals used untyped per-chain feature collections.
   - Added concrete MSA runtime record types in `src/protenix_api.jl`:
     - `ChainMSABlock`
     - `ChainMSAFeatures`
   - `_build_chain_msa_features` / `_chain_msa_features` now return concrete typed records.
   - `_inject_task_msa_features!` now uses `Vector{ChainMSAFeatures}` instead of generic `NamedTuple[]`.

9. Remaining generic `NamedTuple[]` helper collections in output/protein-chain utilities.
   - Added concrete record aliases:
     - `Output.ResidueRun`
     - `ProtenixAPI.ChainSequenceRecord`
   - Updated helper paths to use concrete vectors.

10. Schema MSA settings carried `Dict{String,Dict{String,Any}}`.
    - Added typed schema record `MSAChainOptions` and migrated `InputTask.msa` to `Dict{String,MSAChainOptions}`.
    - Preserves known keys (`precomputed_msa_dir`, `pairing_db`) and stores extra keys as `Dict{String,String}`.

11. JSON writer only supported vectors/dicts, forcing stable payloads through `Dict{String,Any}`.
    - Added `NamedTuple` JSON encoding support in `src/jsonlite.jl`.
    - Confidence-summary emission now uses typed `NamedTuple` payloads in `src/protenix_api.jl`.
    - Added regression test for NamedTuple JSON roundtrip.

12. CLI `tojson`/`msa` argument handling used generic `Dict{String,Any}` payloads.
   - Added typed option structs in `src/cli.jl`:
      - `ToJSONCLIOptions`
      - `MSACLIOptions`
   - Migrated parse/run paths for those subcommands to typed structs while preserving CLI behavior.

13. CLI `infer`/`parity-check` argument handling still used generic `Dict{String,Any}` payloads.
   - Added typed option structs in `src/cli.jl`:
     - `InferCLIOptions`
     - `ParityCLIOptions`
   - Removed legacy `Dict("__help__"=>true)` sentinels in favor of `nothing` help returns.
   - Migrated parse/run paths for both subcommands to typed structs.

14. Parity/fold helper scripts were CWD-sensitive for safetensors paths.
   - Parity scripts now resolve safetensor directories from repo-root defaults (`release_data/...`) and support env overrides.
   - Sequence fold helper scripts now use the same robust path resolution.
   - `scripts/run_protenix_parity_suite.jl` now runs cleanly from workspace root without manual path setup.

15. Infer runtime repeatedly read nested config sections as untyped dicts.
   - Added typed config adapters in `src/infer.jl`:
     - `RuntimeEnvOptions`
     - `NoiseSchedulerOptions`
     - `SampleDiffusionOptions` + `EtaScheduleOptions`
     - `InferSettingOptions`
     - `ModelScaffoldOptions`
     - `WeightLoadOptions`
   - Diffusion/scaffold execution now uses typed option records internally while keeping `Dict` at external config boundaries.

16. Cache download plumbing read runtime config as untyped nested dicts throughout.
   - Added typed cache adapter `InferenceCacheOptions` in `src/cache.jl`.
   - `ensure_inference_cache!` now parses at boundary, then runs downloads via typed fields.

17. Heteromer MSA pairing merge depended on row index alignment only.
   - Added species/taxonomic pairing-key extraction (`TaxID`, `ncbi_taxid`, `OX`, `Tax`, `OS`) from pairing A3M descriptions.
   - Heteromer merge now prefers key-based row matching and safely falls back to index-based merge when keys are unavailable.
   - Added regressions with out-of-order `TaxID` and `OS=` rows across chains to lock expected merge behavior.

18. Protein-chain sequence extraction kept residues in abstract NamedTuple vectors.
   - Added concrete alias `ChainResidueRecord` and switched residue accumulation to concrete vectors.

19. Constraint-transformer self-attention used avoidable `permutedims` copies in hot path.
   - Refactored `SubstructureSelfAttention` to compute per-head attention directly from head slices of `q/k/v`.
   - Removed intermediate `qh/kh/vh` and post-attention merge transpose allocation.
   - Preserved numerical parity (validated by full regression + parity suite).

20. Template-feature flattening used `Vector{Any}` internals.
   - Made nested flattening typed (`Vector{T}`) in `_to_dense_array` conversion path.
   - Removed dynamic `Any` accumulation while preserving template ingestion behavior.

21. Config override nesting could drop existing keys in non-`Dict{String,Any}` subtrees.
   - Fixed `set_nested!` to preserve and canonicalize existing nested dict content instead of replacing it.
   - Added regression coverage to prevent silent subtree clobbering during overrides.

## Open / Next

1. Continue Dict-boundary reduction:
   - recent pass:
     - `TaskEntityParseResult.entity_chain_ids` migrated from `Dict{Int,Vector{String}}` to `Vector{Vector{String}}`
     - `TaskEntityParseResult.entity_atom_map` migrated from `Dict{Int,Dict{Int,String}}` to `Vector{Dict{Int,String}}`
     - covalent/constraint entity lookup helpers now operate on typed vectors
     - task JSON ingestion now preserves input container shape and normalizes to typed task vectors internally
     - schema `msa` payloads migrated to typed `MSAChainOptions`
     - MSA runtime per-chain feature bundles migrated to typed records
     - CLI parsing paths (`predict`/`tojson`/`msa`/`infer`/`parity-check`) migrated to typed option structs
     - infer runtime nested config sections now parsed into typed option records
   - remaining:
     - replace remaining mutable `Dict{String,Any}` internals in `config.jl` default/override scaffolding where schema is stable.

2. Keep layer/layout cleanup moving toward Julia-first memory locality:
   - feature-first + batch-last in hot kernels with minimal `permutedims`.

## Out Of Scope (For Now)

1. Full OpenFold species/taxonomic pairing parity semantics.
   - current behavior:
     - heteromer `pairing.a3m` rows merge across chains by inferred keys (`TaxID`/`ncbi_taxid`/`OX`/`Tax`/`OS`) when available
     - safe fallback to row-index pairing when metadata is missing
