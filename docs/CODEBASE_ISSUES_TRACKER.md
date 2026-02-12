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

## Open / Next

1. Implement `constraint` conditioning parity in `predict_json` runtime path.
   - completed in part:
     - `constraint.contact` and `constraint.pocket` parsing + feature injection
     - typed constraint feature plumbing into Protenix trunk `z` path
     - `constraint.structure` acceptance as JSON-inference no-op (Python v0.5 parity)
     - substructure embedder architecture/weight-loading support for `linear`/`mlp`/`transformer`
     - real checkpoint conversion/load coverage validated for `protenix_base_constraint_v0.5.0`
      - full forward numeric parity against Python reference dumps for constraint-conditioned paths
        (`dump_python_protenix_base_constraint_trunk_denoise_parity.py` vs
        `compare_protenix_base_constraint_trunk_denoise_parity.jl`)

2. Improve multi-chain MSA merge parity:
   - implemented: heteromer `pairing.a3m` rows merge across chains by row index, then non-pair rows append chain-wise.
   - pending: full OpenFold species/taxonomic pairing parity.

3. Continue Dict-boundary reduction:
   - recent pass:
     - `TaskEntityParseResult.entity_chain_ids` migrated from `Dict{Int,Vector{String}}` to `Vector{Vector{String}}`
     - `TaskEntityParseResult.entity_atom_map` migrated from `Dict{Int,Dict{Int,String}}` to `Vector{Dict{Int,String}}`
     - covalent/constraint entity lookup helpers now operate on typed vectors
   - remaining:
     - replace remaining mutable `Dict{String,Any}` internals with typed records where schema is stable.

4. Keep layer/layout cleanup moving toward Julia-first memory locality:
   - feature-first + batch-last in hot kernels with minimal `permutedims`.
