#!/usr/bin/env bash
set -euo pipefail

# Run all clean_targets through Python Protenix CLI.
#
# Usage:
#   cd /home/claudey/FixingKAFA/PXDesign.jl
#   bash clean_targets/scripts/run_all_python.sh [target_number]
#
# If a target number is given (e.g. "01"), only that target is run.
# Otherwise, all targets are attempted.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="/home/claudey/venvs/python_esmfold"
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$ROOT/.external/Protenix:${PYTHONPATH:-}"
export PROTENIX_DATA_ROOT_DIR="$ROOT/release_data/ccd_cache"
# GPU enabled — PyTorch 2.10+cu130 supports GB10 (sm_121)

INPUTS="$ROOT/clean_targets/inputs"
OUTBASE="$ROOT/clean_targets/python_outputs"
CKPT_DIR="$ROOT/release_data/checkpoint"
SEED=101

mkdir -p "$OUTBASE" "$CKPT_DIR"

FILTER="${1:-}"

run_protenix() {
    local json="$1"
    local model="$2"
    local step="$3"
    local cycle="$4"
    local name
    name=$(basename "$json" .json)
    local outdir="$OUTBASE/${name}__${model}"

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
    echo ""
}

run_protenix_msa() {
    local json="$1"
    local model="$2"
    local step="$3"
    local cycle="$4"
    local name
    name=$(basename "$json" .json)
    local outdir="$OUTBASE/${name}__${model}"

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
    echo ""
}

should_run() {
    local num="$1"
    [[ -z "$FILTER" ]] || [[ "$num" == "$FILTER" ]]
}

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

# ── Category 10: Design (22-32) ── PXDesign CLI ──────────────────────────────
# Design targets use YAML format and PXDesign CLI, not Protenix.
# These are handled separately via PXDesign's `pxdesign infer` command.
for n in 22 23 24 25 26 27 28 29 30 31 32; do
    should_run "$n" || continue
    yaml="$INPUTS/${n}_*.yaml"
    for f in $yaml; do
        [ -f "$f" ] || continue
        name=$(basename "$f" .yaml)
        outdir="$OUTBASE/${name}__pxdesign"
        echo "============================================================"
        echo "  Design Target: $name"
        echo "  CLI: pxdesign infer"
        echo "============================================================"
        # PXDesign generation-only mode (no evaluation)
        # --num_workers 0 avoids multiprocessing deadlock on this machine
        # --seeds 101 for reproducibility / parity with Julia
        # --load_checkpoint_dir points to shared checkpoint cache
        # Note: Target 22 (unconditional) has no target.file; Python YAML parser
        # requires it, so unconditional generation must use JSON format or be skipped.
        python3 -m pxdesign.runner.cli infer \
            -i "$f" \
            -o "$outdir" \
            --N_sample 1 \
            --N_step 200 \
            --dtype bf16 \
            --seeds 101 \
            --num_workers 0 \
            --load_checkpoint_dir "$CKPT_DIR" \
            || echo "FAILED: $name / pxdesign infer"
        echo ""
    done
done

echo "============================================================"
echo "Python reference generation complete"
echo "Output: $OUTBASE"
echo "============================================================"
