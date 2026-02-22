# PXDesign Model Port Map

This file maps Python reference model blocks to checkpoint key namespaces and Julia targets.

## Checkpoint Shape

- Source file: `/Users/benmurrell/JuliaM3/PXDesign/PXDesign.jl/docs/checkpoint_index.json`
- Model tensors: `732`
- Top-level namespaces:
  - `design_condition_embedder` (`96` tensors)
  - `diffusion_module` (`636` tensors)

## Python -> Key Prefix -> Julia Target

1. `pxdesign/model/embedders.py::ConditionTemplateEmbedder`
   - Prefix: `design_condition_embedder.condition_template_embedder.embedder`
   - Julia target: `src/model/design_condition_embedder.jl` (implemented)

2. `pxdesign/model/embedders.py::InputFeatureEmbedderDesign`
   - Prefix: `design_condition_embedder.input_embedder`
   - Dominant subtree:
     - `design_condition_embedder.input_embedder.atom_attention_encoder.*`
     - `design_condition_embedder.input_embedder.input_map.*`
   - Julia target: `src/model/design_condition_embedder.jl` (implemented)

3. `protenix/model/modules/diffusion.py::DiffusionConditioning`
   - Prefix: `diffusion_module.diffusion_conditioning.*`
   - Julia target: `src/model/diffusion_conditioning.jl` (implemented)

4. `protenix/model/modules/transformer.py::AtomAttentionEncoder`
   - Prefix: `diffusion_module.atom_attention_encoder.*`
   - Julia target: `src/model/atom_attention.jl` (implemented)

5. `protenix/model/modules/transformer.py::DiffusionTransformer`
   - Prefix: `diffusion_module.diffusion_transformer.*`
   - Julia target: `src/model/transformer_blocks.jl` (implemented)

6. `protenix/model/modules/transformer.py::AtomAttentionDecoder`
   - Prefix: `diffusion_module.atom_attention_decoder.*`
   - Julia target: `src/model/atom_attention.jl` (implemented)

7. `protenix/model/modules/diffusion.py::DiffusionModule` top-level layers
   - Prefixes:
     - `diffusion_module.layernorm_a.*`
     - `diffusion_module.layernorm_s.*`
     - `diffusion_module.linear_no_bias_s.*`
   - Julia target: `src/model/diffusion_module.jl` (implemented)

## Current Julia Status

- Implemented:
  - `src/model/scheduler.jl`
  - `src/model/sampler.jl`
  - typed embedders/primitives/conditioning/transformer/diffusion-module stack
  - shared atom attention modules (`AtomAttentionEncoder`, `AtomAttentionDecoder`)
  - design condition embedder (`InputFeatureEmbedderDesign`, `DesignConditionEmbedder`)
  - checkpoint state loading for:
    - design condition embedder
    - diffusion conditioning
    - diffusion atom encoder/decoder
    - diffusion transformer + top-level layers
  - scaffold dim inference from checkpoints:
    - `infer_model_scaffold_dims`
    - `infer_design_condition_embedder_dims`
  - raw checkpoint bridge:
    - exporter: `scripts/export_checkpoint_raw.py`
    - loader: `src/model/raw_weights.jl`
- Missing for parity:
  - trunked local atom attention is implemented with loop-based kernels; optimized fused kernels from Python are not yet ported
  - memory-optimized chunk/checkpoint behavior from Python is not yet mirrored
  - parity snapshots are validated locally; committed snapshot artifacts are intentionally out of scope

## Implementation Order

1. Done: Replace dense atom-pair attention with local trunked attention parity (`rearrange_*` + local masks).
2. Done: Add strict key-coverage assertion: every checkpoint key consumed or allowlisted.
3. Done (tooling): Add reference-vs-Julia numeric parity harness for representative inference snapshots (`tensor_parity_report`, `scripts/compare_parity_raw.jl`).
4. Ongoing (local validation): generate representative Python snapshot bundles locally and compare with Julia parity harness.
