# PXDesign.jl Infer-Only Port Plan

## Goal

Port `pxdesign infer` functionality from Python PXDesign to Julia, excluding PXDesignBench/AF2/Protenix ranking pipeline.

Current scope target:
- Input ingestion
- Feature construction
- Model forward (diffusion sampling)
- Output serialization

Explicitly out of scope in this port stage:
- `pipeline` command
- AF2/Protenix/MPNN evaluation and reranking
- PXDesignBench integration

## Upstream Commit Audited

- PXDesign: `f788441`

## Architecture Strategy

1. Keep inference orchestration self-contained in Julia.
2. Minimize transitive dependency on external projects.
3. Reuse prior Julia AF2/ESMFold work only where it maps directly to shared primitives.
4. Preserve output directory conventions to keep downstream compatibility.

## Phases

### Phase 0 (Completed): Skeleton + Infer Scaffolding

Implemented:
- `src/config.jl`: defaults, alias mapping, nested overrides.
- `src/cache.jl`: checkpoint/data cache downloader.
- `src/inputs.jl`: JSON/YAML task loading.
- `src/infer.jl`: infer orchestration scaffold with deterministic seeds and output layout.
- `src/cli.jl`: `infer` command.
- `src/data/*`: first data layer slice (constants, tokenization, canonical residue encoding, basic feature bundle).
- `bin/pxdesign`: executable entrypoint with local depot defaults.

Deliverable:
- Dry-run capable CLI validating config/cache/input flow.

### Phase 1: Data Model and Input Normalization

Python sources:
- `pxdesign/utils/inputs.py`
- `pxdesign/utils/infer.py` (`convert_to_bioassembly_dict`, PDB/CIF mapping helpers)
- `pxdesign/data/parser.py`, `pxdesign/data/json_parser.py`

Julia deliverables:
- Full YAML schema normalization into canonical task JSON.
- Chain/residue remapping parity for PDB input.
- mmCIF/PDB ingestion abstraction and task validation errors matching Python behavior.

Success gate:
- Reproduce canonicalized task JSON for provided examples.

### Phase 2: Feature Pipeline Port

Python sources:
- `pxdesign/data/infer_data_pipeline.py`
- `pxdesign/data/json_to_feature.py`
- `pxdesign/data/featurizer.py`
- `pxdesign/data/tokenizer.py`
- `pxdesign/data/utils.py`
- `pxdesign/data/constants.py`
- `pxdesign/data/ccd.py`

Julia deliverables:
- Tokenization and atom/token indexing.
- Feature tensor assembly (`restype`, template placeholders, mask features, hotspot features).
- CCD-backed chemistry helpers and cache readers.

Success gate:
- Feature tensor shapes and dtypes match Python for representative test structures.

Progress:
- Initial feature bundle path exists and is wired into both dry-run and non-dry-run infer.
- Native mmCIF/PDB parser now feeds condition chains with crop filtering.
- Feature bundle now merges condition+binder atoms and produces:
  - token metadata (`token_index`, `residue_index`, `asym_id`, `entity_id`, `sym_id`)
  - masks (`condition_token_mask`, `design_token_mask`, `condition_atom_mask`)
  - template conditioning (`conditional_templ`, `conditional_templ_mask`)
  - frame metadata (`has_frame`, `frame_atom_index`)
- Remaining gap: this is still a reduced feature set versus Python `Featurizer` and lacks full CCD/ref-space/permutation features.

### Phase 3: Model Blocks and Sampling Loop

Python sources:
- `pxdesign/model/pxdesign.py`
- `pxdesign/model/embedders.py`
- `pxdesign/model/generator.py`

Julia deliverables:
- Diffusion scheduler and sampler parity.
- Condition embedders and model forward pass.
- Runtime knobs (`fast_ln`, deepspeed/cutlass env behavior compatibility shims).

Success gate:
- Forward pass executes on at least one sample and emits coordinate tensors with expected shape.

Progress:
- `InferenceNoiseScheduler` and `sample_diffusion` are implemented and tested.
- Infer-only scaffold model now runs diffusion and writes CIF outputs (condition atoms fixed, design atoms sampled).
- Added typed scaffolding for core model components:
  - `RelativePositionEncoding`, `ConditionTemplateEmbedder`
  - primitive layers (`LinearNoBias`, `LayerNormNoOffset`, `AdaptiveLayerNorm`, `Transition`)
  - transformer units (`AttentionPairBias`, `ConditionedTransitionBlock`, `DiffusionTransformer`)
  - `DiffusionConditioning` and `DiffusionModule` forward path
  - local-window atom attention trunks (no dense `N_atom × N_atom` pair materialization)
- Added checkpoint bridge components:
  - `scripts/export_checkpoint_raw.py` (`.pt` -> raw float32 bundle)
  - `src/model/raw_weights.jl` (raw bundle loader)
  - `src/model/state_load.jl` (state-dict key -> module-field assignment helpers + strict key coverage)

### Phase 4: Checkpoint Loader + Numeric Parity

Python sources:
- State dict loading in `pxdesign/runner/inference.py`
- Checkpoint path/URL mapping in `pxdesign/utils/infer.py`

Julia deliverables:
- PyTorch checkpoint reader bridge (or conversion pipeline) for `pxdesign_v0.1.0.pt`.
- Parameter mapping table from Python key names to Julia model fields.
- Deterministic parity harness for small test tasks.

Success gate:
- Same checkpoint loads and produces numerically close outputs under fixed seeds.

Blocker notes:
- Checkpoint index is now available in-repo at `docs/checkpoint_index.json` and fully covered by prefix-map assertions.
- Tensor-value load/mapping into Julia model structs is implemented for infer-only model trees (`732/732` keys covered in strict mode with current raw bundle).
- Remaining gap is collecting committed Python reference snapshot tensors for numeric forward parity CI gating.

### Phase 5: Output Serialization and UX

Python sources:
- `pxdesign/runner/dumper.py`

Julia deliverables:
- CIF/result writing parity.
- Stable output tree:
  - `global_run_*/<task>/seed_*/predictions/*`
  - config + input snapshots

Success gate:
- Existing downstream scripts can consume infer-only outputs without path changes.

## Reuse Opportunities from Existing Julia Ports

Potential reusable components from your AF2/ESMFold ports:
- mmCIF/PDB parsing and chain-index normalization.
- residue encoding tables and atom naming conventions.
- geometry helpers for coordinate transforms.
- batching and tensor utility layers.

Reuse criteria:
- Must preserve PXDesign feature semantics exactly.
- If a reused helper changes behavior, prefer explicit PXDesign-specific wrappers.

## Immediate Next Coding Slices

1. Expand feature parity from reduced set toward Python-complete `Featurizer` coverage (CCD/ref-space/permutation features).
2. Generate deterministic Python reference snapshot bundles for representative tasks.
3. Wire snapshot comparisons into CI using `tensor_parity_report` / `scripts/compare_parity_raw.jl`.
4. Port confidence heads/output channels and add parity assertions for those tensors.

## Known Deltas (Tracked)

1. YAML parsing is now native Julia via `YAML.jl`; parity checks against `python3 + PyYAML` are kept in tests when available.
2. Non-dry-run infer uses the typed diffusion forward path with condition atoms clamped to template coordinates to preserve scaffold constraints.
3. Feature generation consumes structure-derived atoms from mmCIF/PDB, but does not yet include full Python `Featurizer`/CCD feature coverage.
4. Numeric parity harness is implemented, but reference snapshot artifacts are not yet checked into this repo.
