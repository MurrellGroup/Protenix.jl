# PXDesign Infer-Only Audit (Python Reference)

## Scope

This audit covers only `pxdesign infer` (no PXDesignBench pipeline, no AF2 ranking loop).

Primary Python call chain:

1. `pxdesign/runner/cli.py` (`infer`)
2. `pxdesign/runner/inference.py`
3. `pxdesign/model/pxdesign.py`
4. `pxdesign/data/infer_data_pipeline.py`

## Infer-Only Modules To Port

### Must-have PXDesign modules

- `pxdesign/utils/inputs.py`
- `pxdesign/utils/infer.py` (config aliases, cache URLs, structure conversion helpers)
- `pxdesign/data/infer_data_pipeline.py`
- `pxdesign/data/json_to_feature.py`
- `pxdesign/data/featurizer.py`
- `pxdesign/data/tokenizer.py`
- `pxdesign/data/constants.py`
- `pxdesign/model/generator.py`
- `pxdesign/model/embedders.py`
- `pxdesign/model/pxdesign.py`
- `pxdesign/runner/dumper.py`

### External Python dependencies used by infer-only path

- `torch` (required)
- `protenix` (required, major)
  - `protenix.model.modules.diffusion.DiffusionModule`
  - `protenix.model.modules.transformer.AtomAttentionEncoder`
  - `protenix.model.modules.primitives.LinearNoBias`
  - `protenix.data.parser.*` (MMCIF parsing / bioassembly)
  - `protenix.config.parse_configs`
  - `protenix.utils.*` runtime/device/seed helpers
- `biotite` (AtomArray + structure utils)
- `numpy`
- `ml_collections`
- `scikit-learn` (KDTree in frame construction)

## Dependency Risk Assessment

- Infer-only is **not self-contained** in Python.
- The largest dependency is `protenix`; PXDesign delegates core neural blocks and parts of data parsing to it.
- If we insist on self-contained Julia, we must replace both:
  1. Protenix model internals
  2. Protenix data/parser behaviors

## Minimum Viable Julia Port (to run infer)

Required runtime pieces:

1. Input normalization (YAML/JSON parity)
2. Structure parser + crop/hotspot handling
3. Feature builder producing model-required tensors
4. PXDesign neural graph (`embedders + diffusion module + sampler`)
5. Checkpoint loader and key mapping
6. CIF output writer and run layout

## Current PXDesign.jl Status vs Audit

Implemented now:

- 1) Input normalization
- 2) Native structure parser (mmCIF/PDB) + filtering
- partial 3) Reduced but working feature builder
- infer-only core of 4) neural modules (embedders + diffusion module + sampler)
- 5) checkpoint loading/key mapping + strict coverage and raw-snapshot parity harness
- 6) CIF output writer + infer output tree

Not yet implemented:

- full 3) Python-complete feature set (CCD/ref/permutation/mask suite)
- confidence/ranking heads beyond infer-only diffusion coordinates
- committed Python numeric snapshot artifacts for CI parity checks

## Hard Blockers In Current Environment

1. Network access for fetching fresh upstream `protenix` source/checkpoints is unavailable.
2. Direct Python-side snapshot generation depends on local Torch/runtime stability; Julia-side parity tooling is ready once snapshots are produced.
