#!/usr/bin/env bash
set -euo pipefail

# Run all clean_targets through Python Protenix/PXDesign on GPU.
# Usage: bash clean_targets/scripts/run_python_gpu.sh [target_number]

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PXDESIGN_PYTHON_VENV:?Set PXDESIGN_PYTHON_VENV}/bin/python3"
export PYTHONPATH="$ROOT/.external/Protenix:${PYTHONPATH:-}"

INPUTS="$ROOT/clean_targets/inputs"
OUTBASE="$ROOT/clean_targets/python_outputs"
SEED=101

mkdir -p "$OUTBASE"

FILTER="${1:-}"

run_protenix() {
    local json="$1"
    local model="$2"
    local use_msa="${3:-false}"
    local name
    name=$(/usr/bin/basename "$json" .json)
    local outdir="$OUTBASE/${name}__${model}"

    # Skip if already completed
    if [ -f "$outdir/$(echo $name | /usr/bin/sed 's/^[0-9]*_//')/seed_${SEED}/SUCCESS_FILE" ]; then
        echo "SKIP (already done): $name / $model"
        return 0
    fi

    echo "============================================================"
    echo "  Target: $name  Model: $model  MSA: $use_msa"
    echo "============================================================"

    $PYTHON "$ROOT/.external/Protenix/runner/inference.py" \
        --model_name "$model" \
        --seeds "$SEED" \
        --dump_dir "$outdir" \
        --input_json_path "$json" \
        --use_msa "$use_msa" \
        --sample_diffusion.N_sample 1 \
        || echo "FAILED: $name / $model"
    echo ""
}

should_run() {
    local num="$1"
    [[ -z "$FILTER" ]] || [[ "$num" == "$FILTER" ]]
}

# ── Targets 01-03: Protein-Only ──
for n in 01 02 03; do
    should_run "$n" || continue
    for f in $INPUTS/${n}_*.json; do
        [ -f "$f" ] || continue
        run_protenix "$f" protenix_mini_default_v0.5.0
    done
done
# Target 01 also on tiny
should_run "01" && run_protenix "$INPUTS/01_protein_monomer.json" protenix_tiny_default_v0.5.0

# ── Targets 04-05: Nucleic Acids ──
for n in 04 05; do
    should_run "$n" || continue
    for f in $INPUTS/${n}_*.json; do
        [ -f "$f" ] || continue
        run_protenix "$f" protenix_mini_default_v0.5.0
    done
done

# ── Targets 06-08: Small Molecules ──
for n in 06 07 08; do
    should_run "$n" || continue
    for f in $INPUTS/${n}_*.json; do
        [ -f "$f" ] || continue
        run_protenix "$f" protenix_mini_default_v0.5.0
    done
done

# ── Targets 09-10: Multi-Entity ──
for n in 09 10; do
    should_run "$n" || continue
    for f in $INPUTS/${n}_*.json; do
        [ -f "$f" ] || continue
        run_protenix "$f" protenix_mini_default_v0.5.0
    done
done

# ── Targets 11-13: Modifications ──
for n in 11 12 13; do
    should_run "$n" || continue
    for f in $INPUTS/${n}_*.json; do
        [ -f "$f" ] || continue
        run_protenix "$f" protenix_mini_default_v0.5.0
    done
done

# ── Targets 14-16: Constraints (base_constraint model) ──
for n in 14 15 16; do
    should_run "$n" || continue
    for f in $INPUTS/${n}_*.json; do
        [ -f "$f" ] || continue
        run_protenix "$f" protenix_base_constraint_v0.5.0
    done
done

# ── Target 17: MSA ──
should_run "17" && run_protenix "$INPUTS/17_protein_msa.json" protenix_mini_default_v0.5.0 true

# ── Target 18: ESM/ISM ──
if should_run "18"; then
    run_protenix "$INPUTS/18_protein_esm.json" protenix_mini_esm_v0.5.0
    run_protenix "$INPUTS/18_protein_esm.json" protenix_mini_ism_v0.5.0
fi

# ── Target 19: Template ──
should_run "19" && run_protenix "$INPUTS/19_protein_template.json" protenix_mini_tmpl_v0.5.0

# ── Target 20: Complex Assembly ──
should_run "20" && run_protenix "$INPUTS/20_complex_multichain.json" protenix_mini_default_v0.5.0

# ── Target 21: File-Based Ligand ──
should_run "21" && run_protenix "$INPUTS/21_ligand_file_sdf.json" protenix_mini_default_v0.5.0

echo "============================================================"
echo "Python reference generation complete"
echo "Output: $OUTBASE"
echo "============================================================"
