#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck source=/dev/null
source "$ROOT/scripts/python_reference_env.sh"

TMPDIR="$(mktemp -d /tmp/pyref_smoke_XXXXXX)"
mkdir -p "$TMPDIR/out" "$TMPDIR/mpl"

cat > "$TMPDIR/target.cif" <<'EOF'
data_demo
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_formal_charge
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
ATOM 1 N N . ALA A 1 1 ? 0.000 0.000 0.000 1.00 10.00 ? 1 ALA A N 1
ATOM 2 C CA . ALA A 1 1 ? 1.200 0.000 0.000 1.00 10.00 ? 1 ALA A CA 1
ATOM 3 C C . ALA A 1 1 ? 2.200 0.100 0.000 1.00 10.00 ? 1 ALA A C 1
ATOM 4 O O . ALA A 1 1 ? 3.000 0.200 0.000 1.00 10.00 ? 1 ALA A O 1
#
EOF

cat > "$TMPDIR/input.json" <<EOF
[
  {
    "name": "py_ref",
    "condition": {
      "structure_file": "$TMPDIR/target.cif",
      "filter": {"chain_id": ["A"], "crop": {}},
      "msa": {}
    },
    "hotspot": {"A": [1]},
    "generation": [{"type": "protein", "length": 2, "count": 1}]
  }
]
EOF

MPLCONFIGDIR="$TMPDIR/mpl" python -m pxdesign.runner.inference \
  --input_json_path "$TMPDIR/input.json" \
  --dump_dir "$TMPDIR/out" \
  --seeds 7 \
  --load_checkpoint_dir "$ROOT/release_data/checkpoint" \
  --model_name pxdesign_v0.1.0 \
  --num_workers 0 \
  --use_msa false \
  --sample_diffusion.N_sample 1 \
  --sample_diffusion.N_step 2 \
  --sample_diffusion.eta_schedule.type const \
  --sample_diffusion.eta_schedule.min 1.0 \
  --sample_diffusion.eta_schedule.max 1.0 \
  --infer_setting.sample_diffusion_chunk_size 1

PRED_DIR="$(find "$TMPDIR/out" -type d -path '*predictions' | head -n 1)"
PRED_CIF="$(find "$PRED_DIR" -type f -name '*.cif' | head -n 1)"
echo "PYREF_TMPDIR=$TMPDIR"
echo "PYREF_PRED_CIF=$PRED_CIF"
