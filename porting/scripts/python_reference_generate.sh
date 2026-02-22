#!/usr/bin/env bash
set -euo pipefail

# Generate Python Protenix reference outputs for parity testing.
# Uses the CLI interface which handles config properly.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/.venv_pyref/bin/activate"
export PYTHONPATH="$ROOT/.external/Protenix:${PYTHONPATH:-}"
export PROTENIX_DATA_ROOT_DIR="$ROOT/release_data/ccd_cache"
# Force CPU — GB10 (sm_121) has no PyTorch CUDA kernels
export CUDA_VISIBLE_DEVICES=""

SEQUENCE="MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"
SEED=101
OUT_BASE="$ROOT/e2e_output/python_reference"
CKPT_DIR="$ROOT/release_data/checkpoint"

mkdir -p "$OUT_BASE" "$CKPT_DIR"

# Create input JSON for protein-only folding
TMPJSON=$(mktemp /tmp/pyref_input_XXXXX.json)
cat > "$TMPJSON" <<ENDJSON
[{
    "sequences": [{
        "proteinChain": {
            "sequence": "$SEQUENCE",
            "count": 1
        }
    }],
    "name": "hemoglobin_alpha_51aa"
}]
ENDJSON

echo "Input JSON: $TMPJSON"
echo "Output dir: $OUT_BASE"
echo ""

# Run mini model (200 steps)
echo "============================================================"
echo "Running protenix_mini_default_v0.5.0 (200 steps)"
echo "============================================================"
python3 "$ROOT/.external/Protenix/runner/inference.py" \
    --model_name protenix_mini_default_v0.5.0 \
    --seeds "$SEED" \
    --dump_dir "$OUT_BASE/pyref_mini_200" \
    --input_json_path "$TMPJSON" \
    --model.N_cycle 4 \
    --sample_diffusion.N_sample 1 \
    --sample_diffusion.N_step 200 \
    --use_msa false \
    --load_checkpoint_dir "$CKPT_DIR" || echo "FAILED: mini"

# Run tiny model (200 steps)
echo "============================================================"
echo "Running protenix_tiny_default_v0.5.0 (200 steps)"
echo "============================================================"
python3 "$ROOT/.external/Protenix/runner/inference.py" \
    --model_name protenix_tiny_default_v0.5.0 \
    --seeds "$SEED" \
    --dump_dir "$OUT_BASE/pyref_tiny_200" \
    --input_json_path "$TMPJSON" \
    --model.N_cycle 4 \
    --sample_diffusion.N_sample 1 \
    --sample_diffusion.N_step 200 \
    --use_msa false \
    --load_checkpoint_dir "$CKPT_DIR" || echo "FAILED: tiny"

# Run base model (200 steps)
echo "============================================================"
echo "Running protenix_base_default_v0.5.0 (200 steps)"
echo "============================================================"
python3 "$ROOT/.external/Protenix/runner/inference.py" \
    --model_name protenix_base_default_v0.5.0 \
    --seeds "$SEED" \
    --dump_dir "$OUT_BASE/pyref_base_200" \
    --input_json_path "$TMPJSON" \
    --model.N_cycle 10 \
    --sample_diffusion.N_sample 1 \
    --sample_diffusion.N_step 200 \
    --use_msa false \
    --load_checkpoint_dir "$CKPT_DIR" || echo "FAILED: base"

rm -f "$TMPJSON"

echo ""
echo "============================================================"
echo "Python reference generation complete"
echo "Output: $OUT_BASE"
echo "============================================================"
