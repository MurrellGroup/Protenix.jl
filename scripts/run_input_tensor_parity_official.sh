#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck source=/dev/null
source "$ROOT/scripts/python_reference_env.sh"

JULIA_BIN="${JULIA_BIN:-$HOME/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia}"

INPUT_JSON="${1:-$ROOT/output/input_tensor_parity_full/input_single_chain.json}"
OUT_BASE="${2:-$ROOT/output/input_tensor_parity_official}"
SEED="${3:-101}"

PY_DUMP_DIR="$OUT_BASE/python_dumps"
RAW_REPORT_DIR="$OUT_BASE/reports_raw"
RIGID_REPORT_DIR="$OUT_BASE/reports_rigid"
LOG_DIR="$OUT_BASE/logs"

mkdir -p "$PY_DUMP_DIR" "$RAW_REPORT_DIR" "$RIGID_REPORT_DIR" "$LOG_DIR"

# Guardrail: keep Python reference implementation pristine except dump hook.
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  case "$line" in
    " M runner/inference.py") ;;
    "?? esm_embeddings/") ;;
    *)
      echo "[error] .external/Protenix has unexpected working-tree change: $line" >&2
      echo "[error] keep reference code unmodified except runner/inference.py dump instrumentation." >&2
      exit 2
      ;;
  esac
done < <(git -C "$ROOT/.external/Protenix" status --short)

MODELS=(
  protenix_base_default_v0.5.0
  protenix_base_constraint_v0.5.0
  protenix_mini_default_v0.5.0
  protenix_mini_tmpl_v0.5.0
  protenix_tiny_default_v0.5.0
  protenix_mini_esm_v0.5.0
  protenix_mini_ism_v0.5.0
)

echo "[parity] input_json=$INPUT_JSON"
echo "[parity] out_base=$OUT_BASE"
echo "[parity] seed=$SEED"

for model in "${MODELS[@]}"; do
  echo "[python-dump] $model"
  ckpt="$ROOT/release_data/checkpoint/$model.pt"
  if [[ ! -f "$ckpt" ]]; then
    echo "[error] missing checkpoint: $ckpt" >&2
    exit 1
  fi

  MODEL_OUT="$OUT_BASE/model_runs/$model"
  mkdir -p "$MODEL_OUT"

  if ls "$PY_DUMP_DIR/${model}"__*__seed"${SEED}".json >/dev/null 2>&1; then
    echo "[python-dump] skip existing dump for $model"
    continue
  fi

  PROTENIX_DUMP_INPUT_FEATURES_DIR="$PY_DUMP_DIR" \
  PROTENIX_DUMP_INPUT_FEATURES_ONCE=1 \
  python "$ROOT/.external/Protenix/runner/inference.py" \
    --input_json_path "$INPUT_JSON" \
    --dump_dir "$MODEL_OUT" \
    --load_checkpoint_path "$ckpt" \
    --model_name "$model" \
    --seeds "$SEED" \
    --num_workers 0 \
    --use_msa false \
    --sample_diffusion.N_step 1 \
    --sample_diffusion.N_sample 1 \
    --model.N_cycle 1 \
    >"$LOG_DIR/python_${model}.log" 2>&1
done

for model in "${MODELS[@]}"; do
  echo "[compare-raw] $model"
  if [[ "$model" == "protenix_mini_esm_v0.5.0" || "$model" == "protenix_mini_ism_v0.5.0" ]]; then
    JULIA_DEPOT_PATH="$ROOT/.julia_depot:$HOME/.julia" \
    JULIAUP_DEPOT_PATH="$ROOT/.julia_depot" \
    "$JULIA_BIN" --project="$ROOT" \
      "$ROOT/scripts/compare_python_input_tensors.jl" \
      --input-json "$INPUT_JSON" \
      --python-dump-dir "$PY_DUMP_DIR" \
      --model-name "$model" \
      --seed "$SEED" \
      --use-default-params true \
      --use-msa false \
      --ref-pos-augment true \
      --allow-ref-pos-rigid-equiv false \
      --strict-keyset false \
      --atol 1e-5 \
      --rtol 1e-5 \
      --inject-python-esm true \
      --report "$RAW_REPORT_DIR/$model.json" \
      >"$LOG_DIR/raw_${model}.log" 2>&1 || true
  else
    JULIA_DEPOT_PATH="$ROOT/.julia_depot:$HOME/.julia" \
    JULIAUP_DEPOT_PATH="$ROOT/.julia_depot" \
    "$JULIA_BIN" --project="$ROOT" \
      "$ROOT/scripts/compare_python_input_tensors.jl" \
      --input-json "$INPUT_JSON" \
      --python-dump-dir "$PY_DUMP_DIR" \
      --model-name "$model" \
      --seed "$SEED" \
      --use-default-params true \
      --use-msa false \
      --ref-pos-augment true \
      --allow-ref-pos-rigid-equiv false \
      --strict-keyset false \
      --atol 1e-5 \
      --rtol 1e-5 \
      --report "$RAW_REPORT_DIR/$model.json" \
      >"$LOG_DIR/raw_${model}.log" 2>&1 || true
  fi

  echo "[compare-rigid] $model"
  if [[ "$model" == "protenix_mini_esm_v0.5.0" || "$model" == "protenix_mini_ism_v0.5.0" ]]; then
    JULIA_DEPOT_PATH="$ROOT/.julia_depot:$HOME/.julia" \
    JULIAUP_DEPOT_PATH="$ROOT/.julia_depot" \
    "$JULIA_BIN" --project="$ROOT" \
      "$ROOT/scripts/compare_python_input_tensors.jl" \
      --input-json "$INPUT_JSON" \
      --python-dump-dir "$PY_DUMP_DIR" \
      --model-name "$model" \
      --seed "$SEED" \
      --use-default-params true \
      --use-msa false \
      --ref-pos-augment true \
      --allow-ref-pos-rigid-equiv true \
      --strict-keyset false \
      --atol 1e-5 \
      --rtol 1e-5 \
      --inject-python-esm true \
      --report "$RIGID_REPORT_DIR/$model.json" \
      >"$LOG_DIR/rigid_${model}.log" 2>&1 || true
  else
    JULIA_DEPOT_PATH="$ROOT/.julia_depot:$HOME/.julia" \
    JULIAUP_DEPOT_PATH="$ROOT/.julia_depot" \
    "$JULIA_BIN" --project="$ROOT" \
      "$ROOT/scripts/compare_python_input_tensors.jl" \
      --input-json "$INPUT_JSON" \
      --python-dump-dir "$PY_DUMP_DIR" \
      --model-name "$model" \
      --seed "$SEED" \
      --use-default-params true \
      --use-msa false \
      --ref-pos-augment true \
      --allow-ref-pos-rigid-equiv true \
      --strict-keyset false \
      --atol 1e-5 \
      --rtol 1e-5 \
      --report "$RIGID_REPORT_DIR/$model.json" \
      >"$LOG_DIR/rigid_${model}.log" 2>&1 || true
  fi
done

export OUT_BASE
python - <<'PY'
import glob, json, os
base = os.environ["OUT_BASE"]
for mode in ("reports_raw", "reports_rigid"):
    print(f"[summary] {mode}")
    for p in sorted(glob.glob(os.path.join(base, mode, "*.json"))):
        with open(p) as f:
            d = json.load(f)
        print(
            os.path.basename(p),
            "failed_keys=",
            d.get("failed_keys"),
            "shared=",
            d.get("shared_keys"),
        )
PY

echo "[done] logs at $LOG_DIR"
