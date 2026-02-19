# PXDesign Test Baseline

Date: 2026-02-19
Platform: aarch64 Linux (julia-1.11.8)
Note: Layer regression fixtures were regenerated for this platform.

## Summary

| Category | Pass | Fail | Error | Broken | Total | Time |
|---|---|---|---|---|---|---|
| Layer regression fixtures | 131 | 0 | 0 | 0 | 131 | 21.2s |
| Main test suite (below) | 434 | 0 | 2 | 2 | 438 | ~1m20s |

## Testset Details

| Testset | Pass | Error | Broken | Total | Time |
|---|---|---|---|---|---|
| Config defaults and overrides | 25 | | | 25 | 0.4s |
| Protenix API surface | 58 | 1 | | 59 | 28.3s |
| Protenix mixed-entity parsing and covalent bonds | 35 | | | 35 | 7.0s |
| Protenix precomputed MSA ingestion | 38 | | | 38 | 2.0s |
| Protenix template feature ingestion | 3 | | | 3 | 0.2s |
| Protenix ESM token embedding injection | 8 | 1 | | 9 | 0.1s |
| ProtenixBase sequence wrappers | 6 | | | 6 | 0.0s |
| Protenix mini/base end-to-end smoke | 12 | | | 12 | 0.3s |
| Protenix base-constraint end-to-end smoke | 5 | | | 5 | 0.7s |
| Protenix typed feature path parity | 7 | | | 7 | 0.6s |
| Protenix constraint embedder plumbing | 10 | | | 10 | 1.1s |
| Cache zero-byte checkpoint refresh | 4 | | | 4 | 0.1s |
| JSONLite | 10 | | | 10 | 0.1s |
| Range utils | 5 | | | 5 | 0.0s |
| Inputs JSON | 4 | | | 4 | 0.3s |
| Data tokenizer/features | 16 | | | 16 | 0.2s |
| Data design encoders | 3 | | | 3 | 0.0s |
| ProtenixMini sequence features | 12 | | | 12 | 0.0s |
| Inputs YAML native parser | 22 | | | 22 | 1.8s |
| Inputs YAML vs PyYAML parity (supported subset) | 1 | | | 1 | 0.0s |
| Scheduler | 3 | | | 3 | 0.0s |
| Sampler | 4 | | | 4 | 0.3s |
| Checkpoint map utilities | 5 | | 1 | 6 | 0.1s |
| Model embedders | 17 | | | 17 | 0.7s |
| Model primitives | 9 | | | 9 | 0.3s |
| Diffusion conditioning | 6 | | | 6 | 0.2s |
| Transformer blocks | 14 | | | 14 | 0.8s |
| Atom attention modules | 17 | | | 17 | 0.6s |
| Diffusion module | 2 | | | 2 | 0.3s |
| Raw weights loader | 9 | | | 9 | 0.3s |
| Parity harness | 10 | | | 10 | 0.2s |
| CLI parity-check | 2 | | | 2 | 0.8s |
| State load mapping | 38 | | | 38 | 0.0s |
| Real checkpoint coverage (optional) | | | 1 | 1 | 0.0s |
| Infer scaffold non-dry-run | 14 | | | 14 | 29.6s |

## Known Issues

### Error 1: "Unknown atom element: A" (affects 2 testsets)
- **Testsets**: Protenix API surface, Protenix ESM token embedding injection
- **Root cause**: Missing CCD components file (`components.v20240608.cif`)
- **Details**: When `_ccd_component_atoms` can't find the CCD file, it returns empty for CCD ligand codes like "ATP". The fallback path in `_build_ligand_atoms_from_codes` infers element from the code name "ATP" → "A", which is not a valid element.
- **Fix**: Either download the CCD file or improve the fallback to properly error when CCD data is unavailable for ligand codes.

### Broken 1: Checkpoint map utilities
- Likely a `@test_broken` marking for a known issue

### Broken 2: Real checkpoint coverage (optional)
- Guarded optional test that requires checkpoint files not present on this machine
