#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/python_reference_env.sh"

JULIA_BIN="${JULIA_BIN:-$HOME/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia}"
JULIA_ENV_PREFIX=(env JULIA_DEPOT_PATH="$ROOT_DIR/.julia_depot" JULIAUP_DEPOT_PATH="$ROOT_DIR/.julia_depot")

run_model() {
  local model="$1"
  local ckpt="$ROOT_DIR/release_data/checkpoint/${model}.pt"
  local raw_dir="$ROOT_DIR/weights_raw_${model}"
  local st_dir="$ROOT_DIR/weights_safetensors_${model}"
  local report_dir="$ROOT_DIR/output/protenix_mini_audit"
  local parity_report="$report_dir/${model}_parity.toml"
  local audit_report="$report_dir/${model}_audit.toml"

  if [[ ! -s "$ckpt" ]]; then
    echo "checkpoint missing or empty: $ckpt" >&2
    return 1
  fi

  mkdir -p "$report_dir"

  python "$ROOT_DIR/scripts/export_checkpoint_raw.py" \
    --checkpoint "$ckpt" \
    --outdir "$raw_dir" \
    --cast-float32

  python "$ROOT_DIR/scripts/convert_raw_to_safetensors.py" \
    --raw-dir "$raw_dir" \
    --out-dir "$st_dir"

  "${JULIA_ENV_PREFIX[@]}" "$JULIA_BIN" --project=. "$ROOT_DIR/scripts/check_raw_vs_safetensors_parity.jl" \
    "$raw_dir" \
    "$st_dir" \
    --report "$parity_report"

  "${JULIA_ENV_PREFIX[@]}" "$JULIA_BIN" --project=. "$ROOT_DIR/scripts/audit_protenix_mini_port.jl" \
    "$st_dir" \
    --raw-dir "$raw_dir" \
    --report "$audit_report"
}

run_model "protenix_mini_default_v0.5.0"
run_model "protenix_mini_tmpl_v0.5.0"

echo "done: Protenix-Mini safetensors prepared and audited."
