#!/usr/bin/env python3
import math
import statistics
import sys
from collections import defaultdict

BACKBONE = {"N", "CA", "C", "O", "OXT"}


def _split_cif_row(line: str):
    out = []
    i = 0
    n = len(line)
    while i < n:
        while i < n and line[i].isspace():
            i += 1
        if i >= n:
            break
        if line[i] in ("'", '"'):
            quote = line[i]
            i += 1
            start = i
            while i < n and line[i] != quote:
                i += 1
            out.append(line[start:i])
            if i < n:
                i += 1
            continue
        start = i
        while i < n and not line[i].isspace():
            i += 1
        out.append(line[start:i])
    return out


def _find_mmcif_atom_loop(lines):
    i = 0
    while i < len(lines):
        if lines[i].strip() != "loop_":
            i += 1
            continue
        j = i + 1
        fields = []
        while j < len(lines):
            s = lines[j].strip()
            if not s.startswith("_"):
                break
            fields.append(s)
            j += 1
        if fields and all(f.startswith("_atom_site.") for f in fields):
            return fields, j
        i = j
    return None, None


def _parse_cif_atoms(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    fields, data_start = _find_mmcif_atom_loop(lines)
    if fields is None:
        return []

    def col_index(*names):
        for name in names:
            if name in fields:
                return fields.index(name)
        return None

    idx_group = col_index("_atom_site.group_PDB")
    idx_atom = col_index("_atom_site.label_atom_id", "_atom_site.auth_atom_id")
    idx_res = col_index("_atom_site.label_comp_id", "_atom_site.auth_comp_id")
    idx_chain = col_index("_atom_site.auth_asym_id", "_atom_site.label_asym_id")
    idx_resid = col_index("_atom_site.auth_seq_id", "_atom_site.label_seq_id")
    idx_x = col_index("_atom_site.Cartn_x")
    idx_y = col_index("_atom_site.Cartn_y")
    idx_z = col_index("_atom_site.Cartn_z")

    required = [idx_group, idx_atom, idx_res, idx_chain, idx_resid, idx_x, idx_y, idx_z]
    if any(idx is None for idx in required):
        return []

    atoms = []
    k = data_start
    n_fields = len(fields)
    while k < len(lines):
        s = lines[k].strip()
        if not s:
            k += 1
            continue
        if s == "#" or s == "loop_" or s.startswith("_") or s.startswith("data_"):
            break
        cols = _split_cif_row(s)
        if len(cols) < n_fields:
            k += 1
            continue
        row = cols[:n_fields]
        if row[idx_group] not in ("ATOM", "HETATM"):
            k += 1
            continue
        try:
            atom_name = row[idx_atom]
            res_name = row[idx_res]
            chain_id = row[idx_chain]
            res_id = int(float(row[idx_resid]))
            x = float(row[idx_x])
            y = float(row[idx_y])
            z = float(row[idx_z])
        except (ValueError, IndexError):
            k += 1
            continue
        atoms.append((chain_id, res_id, res_name, atom_name, (x, y, z)))
        k += 1
    return atoms


def _dist(a, b):
    return math.sqrt(
        (a[0] - b[0]) * (a[0] - b[0])
        + (a[1] - b[1]) * (a[1] - b[1])
        + (a[2] - b[2]) * (a[2] - b[2])
    )


def _summarize(name, values):
    if not values:
        return f"{name}: n=0"
    return (
        f"{name}: n={len(values)} mean={statistics.mean(values):.3f} "
        f"min={min(values):.3f} max={max(values):.3f}"
    )


def main():
    if len(sys.argv) != 2:
        print("usage: geometry_check_sidechain.py <pred.cif>", file=sys.stderr)
        return 2

    path = sys.argv[1]
    atoms = _parse_cif_atoms(path)
    if not atoms:
        print(f"No atoms parsed from {path}", file=sys.stderr)
        return 3

    residues = defaultdict(dict)
    for chain_id, res_id, res_name, atom_name, coord in atoms:
        key = (chain_id, res_id)
        residues[key]["res_name"] = res_name
        residues[key][atom_name] = coord

    cb_ca = []
    side_min = []
    detached = 0
    clashes = 0
    side_atom_count = 0

    for rec in residues.values():
        if "CA" in rec and "CB" in rec:
            cb_ca.append(_dist(rec["CA"], rec["CB"]))

        atom_items = [(k, v) for (k, v) in rec.items() if k != "res_name"]
        for atom_name, coord in atom_items:
            if atom_name in BACKBONE:
                continue
            # sidechain/extra atom: require one nearby in-residue heavy atom
            side_atom_count += 1
            best = None
            for other_name, other_coord in atom_items:
                if other_name == atom_name:
                    continue
                d = _dist(coord, other_coord)
                best = d if best is None else min(best, d)
            if best is None:
                continue
            side_min.append(best)
            if best > 2.2:
                detached += 1
            if best < 0.9:
                clashes += 1

    print(f"file={path}")
    print(_summarize("CA-CB", cb_ca))
    print(_summarize("side_min_neighbor", side_min))
    print(f"side_atom_count={side_atom_count}")
    print(f"side_detached_gt2.2={detached}")
    print(f"side_clash_lt0.9={clashes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

