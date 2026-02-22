#!/usr/bin/env bash
set -euo pipefail

# Timed Python reference run — matches Julia run_all_julia.jl targets exactly.
# Records per-target wall-clock times for comparison with Julia.
#
# Usage:
#   cd /home/claudey/FixingKAFA/PXDesign.jl
#   bash clean_targets/scripts/run_all_python_timed.sh [target_number]

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="/home/claudey/venvs/python_esmfold"
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$ROOT/.external/Protenix:${PYTHONPATH:-}"
export PROTENIX_DATA_ROOT_DIR="$ROOT/release_data/ccd_cache"

INPUTS="$ROOT/clean_targets/inputs"
OUTBASE="$ROOT/clean_targets/python_outputs_timed"
CKPT_DIR="$ROOT/release_data/checkpoint"
SEED=101
TIMINGS="$ROOT/clean_targets/python_gpu_timings.txt"

mkdir -p "$OUTBASE" "$CKPT_DIR"
: > "$TIMINGS"

FILTER="${1:-}"
GLOBAL_START=$(date +%s)

should_run() {
    local num="$1"
    [[ -z "$FILTER" ]] || [[ "$num" == "$FILTER" ]]
}

log_timing() {
    local name="$1"
    local now=$(date +%s)
    local elapsed=$((now - GLOBAL_START))
    local mins=$((elapsed / 60))
    local secs=$((elapsed % 60))
    printf "%3d:%02d  %s\n" "$mins" "$secs" "$name" >> "$TIMINGS"
    printf "%3d:%02d  %s\n" "$mins" "$secs" "$name"
}

run_protenix() {
    local json="$1"
    local model="$2"
    local step="$3"
    local cycle="$4"
    local name
    name=$(basename "$json" .json)
    local target_name="${name}__${model}"
    local outdir="$OUTBASE/${target_name}"

    echo "============================================================"
    echo "  Target: $name"
    echo "  Model:  $model"
    echo "  Steps:  $step   Cycles: $cycle"
    echo "============================================================"

    python3 "$ROOT/.external/Protenix/runner/inference.py" \
        --model_name "$model" \
        --seeds "$SEED" \
        --dump_dir "$outdir" \
        --input_json_path "$json" \
        --model.N_cycle "$cycle" \
        --sample_diffusion.N_sample 1 \
        --sample_diffusion.N_step "$step" \
        --use_msa false \
        --load_checkpoint_dir "$CKPT_DIR" \
        || echo "FAILED: $name / $model"

    log_timing "$target_name"
    echo ""
}

run_protenix_msa() {
    local json="$1"
    local model="$2"
    local step="$3"
    local cycle="$4"
    local name
    name=$(basename "$json" .json)
    local target_name="${name}__${model}"
    local outdir="$OUTBASE/${target_name}"

    echo "============================================================"
    echo "  Target: $name (MSA enabled)"
    echo "  Model:  $model"
    echo "  Steps:  $step   Cycles: $cycle"
    echo "============================================================"

    python3 "$ROOT/.external/Protenix/runner/inference.py" \
        --model_name "$model" \
        --seeds "$SEED" \
        --dump_dir "$outdir" \
        --input_json_path "$json" \
        --model.N_cycle "$cycle" \
        --sample_diffusion.N_sample 1 \
        --sample_diffusion.N_step "$step" \
        --use_msa true \
        --load_checkpoint_dir "$CKPT_DIR" \
        || echo "FAILED: $name / $model"

    log_timing "$target_name"
    echo ""
}

echo "============================================================"
echo "  Python Protenix Timed Run (GPU)"
echo "  Start: $(date)"
echo "============================================================"
echo ""

# ── Category 1: Protein-Only (01-03) ──────────────────────────────────────────
for n in 01 02 03; do
    should_run "$n" || continue
    json="$INPUTS/${n}_*.json"
    for f in $json; do
        [ -f "$f" ] || continue
        run_protenix "$f" protenix_base_default_v0.5.0 200 10
        run_protenix "$f" protenix_mini_default_v0.5.0 20 4
    done
done

# Target 01 also runs on tiny
if should_run "01"; then
    run_protenix "$INPUTS/01_protein_monomer.json" protenix_tiny_default_v0.5.0 20 4
fi

# ── Category 2: Nucleic Acids (04-05) ─────────────────────────────────────────
for n in 04 05; do
    should_run "$n" || continue
    json="$INPUTS/${n}_*.json"
    for f in $json; do
        [ -f "$f" ] || continue
        run_protenix "$f" protenix_base_default_v0.5.0 200 10
        run_protenix "$f" protenix_mini_default_v0.5.0 20 4
    done
done

# ── Category 3: Small Molecules (06-08) ───────────────────────────────────────
for n in 06 07 08; do
    should_run "$n" || continue
    json="$INPUTS/${n}_*.json"
    for f in $json; do
        [ -f "$f" ] || continue
        run_protenix "$f" protenix_base_default_v0.5.0 200 10
        run_protenix "$f" protenix_mini_default_v0.5.0 20 4
    done
done

# ── Category 4: Multi-Entity (09-10) ──────────────────────────────────────────
for n in 09 10; do
    should_run "$n" || continue
    json="$INPUTS/${n}_*.json"
    for f in $json; do
        [ -f "$f" ] || continue
        run_protenix "$f" protenix_base_default_v0.5.0 200 10
        run_protenix "$f" protenix_mini_default_v0.5.0 20 4
    done
done

# ── Category 5: Modifications (11-13) ─────────────────────────────────────────
for n in 11 12 13; do
    should_run "$n" || continue
    json="$INPUTS/${n}_*.json"
    for f in $json; do
        [ -f "$f" ] || continue
        run_protenix "$f" protenix_base_default_v0.5.0 200 10
        run_protenix "$f" protenix_mini_default_v0.5.0 20 4
    done
done

# ── Category 6: Constraints (14-16) — base_constraint model ──────────────────
for n in 14 15 16; do
    should_run "$n" || continue
    json="$INPUTS/${n}_*.json"
    for f in $json; do
        [ -f "$f" ] || continue
        run_protenix "$f" protenix_base_constraint_v0.5.0 200 10
    done
done

# ── Category 7: Input Modalities (17-19) ──────────────────────────────────────
# Target 17: MSA
if should_run "17"; then
    run_protenix_msa "$INPUTS/17_protein_msa.json" protenix_base_default_v0.5.0 200 10
    run_protenix_msa "$INPUTS/17_protein_msa.json" protenix_mini_default_v0.5.0 20 4
fi

# Target 18: ESM/ISM
if should_run "18"; then
    run_protenix "$INPUTS/18_protein_esm.json" protenix_mini_esm_v0.5.0 20 4
    run_protenix "$INPUTS/18_protein_esm.json" protenix_mini_ism_v0.5.0 20 4
fi

# Target 19: Template
if should_run "19"; then
    run_protenix "$INPUTS/19_protein_template.json" protenix_mini_tmpl_v0.5.0 20 4
fi

# ── Category 8: Complex Assembly (20) ─────────────────────────────────────────
# Commented out: 1190 tokens OOMs flash attention bias tensor (~27 GB)
#if should_run "20"; then
#    run_protenix "$INPUTS/20_complex_multichain.json" protenix_base_default_v0.5.0 200 10
#    run_protenix "$INPUTS/20_complex_multichain.json" protenix_mini_default_v0.5.0 20 4
#fi

# ── Category 9: File-Based Ligand (21) ────────────────────────────────────────
if should_run "21"; then
    run_protenix "$INPUTS/21_ligand_file_sdf.json" protenix_base_default_v0.5.0 200 10
    run_protenix "$INPUTS/21_ligand_file_sdf.json" protenix_mini_default_v0.5.0 20 4
fi

echo "============================================================"
echo "  Python timed reference generation complete"
echo "  End: $(date)"
echo "  Timings: $TIMINGS"
echo "  Output: $OUTBASE"
echo "============================================================"
