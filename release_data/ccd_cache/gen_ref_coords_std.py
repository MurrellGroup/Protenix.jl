#!/usr/bin/env python3
"""
Extract CCD reference coordinates from the RDKit pickle into ref_coords_std.json.

This produces the same coordinates that Python Protenix uses at runtime via
get_ccd_ref_info(), ensuring Julia feature parity.

Usage:
    python3 gen_ref_coords_std.py
"""
import json
import pickle
import sys
import os
import numpy as np


def main():
    pkl_path = os.path.join(os.path.dirname(__file__), "components.v20240608.cif.rdkit_mol.pkl")
    out_path = os.path.join(os.path.dirname(__file__), "ref_coords_std.json")

    print(f"Loading RDKit pickle: {pkl_path}")
    with open(pkl_path, "rb") as f:
        mols = pickle.load(f)
    print(f"Loaded {len(mols)} CCD entries")

    result = {}
    skipped = 0
    for code, mol in mols.items():
        if mol is None or mol.GetNumAtoms() == 0:
            skipped += 1
            continue

        try:
            conf = mol.GetConformer(mol.ref_conf_id)
        except Exception:
            skipped += 1
            continue

        coord = conf.GetPositions()  # (n_atom, 3)
        charge = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
        # atom_map: dict str->int (0-indexed)
        atom_map = mol.atom_map
        # ref_mask: numpy bool array
        mask = mol.ref_mask.tolist() if hasattr(mol.ref_mask, 'tolist') else list(mol.ref_mask)

        result[code] = {
            "atom_map": {k: int(v) for k, v in atom_map.items()},
            "coord": coord.tolist(),
            "charge": [int(c) for c in charge],
            "mask": [int(m) for m in mask],
        }

    print(f"Extracted {len(result)} entries, skipped {skipped}")
    print(f"Writing: {out_path}")
    with open(out_path, "w") as f:
        json.dump(result, f, separators=(",", ":"))
    print(f"Done. File size: {os.path.getsize(out_path) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
