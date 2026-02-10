#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/python_reference_env.sh"

JULIA_BIN="${JULIA_BIN:-$HOME/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia}"
JULIA_ENV_PREFIX=(env JULIA_DEPOT_PATH="$ROOT_DIR/.julia_depot" JULIAUP_DEPOT_PATH="$ROOT_DIR/.julia_depot")

MODEL="protenix_base_default_v0.5.0"
CKPT="$ROOT_DIR/release_data/checkpoint/${MODEL}.pt"
RAW_DIR="$ROOT_DIR/weights_raw_${MODEL}"
ST_DIR="$ROOT_DIR/weights_safetensors_${MODEL}"
REPORT_DIR="$ROOT_DIR/output/protenix_base_audit"

PARITY_REPORT="$REPORT_DIR/${MODEL}_parity.toml"
DIFFUSION_AUDIT_REPORT="$REPORT_DIR/${MODEL}_diffusion_audit.toml"
PORT_AUDIT_REPORT="$REPORT_DIR/${MODEL}_port_audit.toml"

if [[ ! -s "$CKPT" ]]; then
  echo "checkpoint missing or empty: $CKPT" >&2
  exit 1
fi

mkdir -p "$REPORT_DIR"

python "$ROOT_DIR/scripts/export_checkpoint_raw.py" \
  --checkpoint "$CKPT" \
  --outdir "$RAW_DIR" \
  --cast-float32

python "$ROOT_DIR/scripts/convert_raw_to_safetensors.py" \
  --raw-dir "$RAW_DIR" \
  --out-dir "$ST_DIR"

"${JULIA_ENV_PREFIX[@]}" "$JULIA_BIN" --project=. "$ROOT_DIR/scripts/check_raw_vs_safetensors_parity.jl" \
  "$RAW_DIR" \
  "$ST_DIR" \
  --report "$PARITY_REPORT"

"${JULIA_ENV_PREFIX[@]}" "$JULIA_BIN" --project=. "$ROOT_DIR/scripts/audit_protenix_mini_port.jl" \
  "$ST_DIR" \
  --raw-dir "$RAW_DIR" \
  --report "$DIFFUSION_AUDIT_REPORT"

"${JULIA_ENV_PREFIX[@]}" "$JULIA_BIN" --project=. "$ROOT_DIR/scripts/audit_protenix_v05_port.jl" \
  "$ST_DIR" \
  --raw-dir "$RAW_DIR" \
  --report "$PORT_AUDIT_REPORT"

echo "done: Protenix-base v0.5.0 safetensors prepared and audited."
