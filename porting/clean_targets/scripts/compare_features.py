#!/usr/bin/env python3
"""
Compare Python .pt vs Julia .npy feature tensors for parity.

Usage:
    python3 clean_targets/scripts/compare_features.py \
        clean_targets/feature_dumps/python/protein_ligand_ccd_features.pt \
        clean_targets/feature_dumps/julia/protein_ligand_ccd_features/
"""
import os
import sys
import numpy as np
import torch


def load_python_features(pt_path):
    feat = torch.load(pt_path, map_location="cpu", weights_only=False)
    result = {}
    for key in feat:
        val = feat[key]
        if isinstance(val, torch.Tensor):
            result[key] = val.detach().numpy()
        elif isinstance(val, dict):
            sub = {}
            for subkey, subval in val.items():
                if isinstance(subval, torch.Tensor):
                    sub[subkey] = subval.detach().numpy()
            result[key] = sub
    return result


def load_julia_features(npy_dir):
    """Load Julia features from a directory of .npy files."""
    result = {}
    for entry in sorted(os.listdir(npy_dir)):
        full_path = os.path.join(npy_dir, entry)
        if entry.endswith(".npy"):
            key = entry[:-4]
            result[key] = np.load(full_path)
        elif os.path.isdir(full_path):
            # Nested dict (e.g. constraint_feature/)
            sub = {}
            for subentry in sorted(os.listdir(full_path)):
                if subentry.endswith(".npy"):
                    subkey = subentry[:-4]
                    sub[subkey] = np.load(os.path.join(full_path, subentry))
            if sub:
                result[entry] = sub
    return result


def compare_tensor(key, py_val, jl_val, atol=1e-5, subkey=""):
    prefix = f"{key}.{subkey}" if subkey else key

    if not isinstance(py_val, np.ndarray) or not isinstance(jl_val, np.ndarray):
        print(f"  SKIP          {prefix}  (not both arrays)")
        return "skip"

    if py_val.shape != jl_val.shape:
        print(f"  SHAPE MISMATCH  {prefix}:  python={py_val.shape}  julia={jl_val.shape}")
        return "fail"

    if np.issubdtype(py_val.dtype, np.floating) or np.issubdtype(jl_val.dtype, np.floating):
        py_f = py_val.astype(np.float64)
        jl_f = jl_val.astype(np.float64)
        diffs = np.abs(py_f - jl_f)
        maxdiff = float(np.max(diffs))
        n_total = py_f.size
        n_diff = int(np.sum(diffs > atol))

        if maxdiff <= atol:
            print(f"  OK            {prefix:35s}  shape={list(py_val.shape)}  maxdiff={maxdiff:.2e}")
            return "ok"
        else:
            pct = round(100.0 * n_diff / n_total, 1)
            print(f"  MISMATCH      {prefix:35s}  shape={list(py_val.shape)}  maxdiff={maxdiff:.2e}  n_diff={n_diff}/{n_total} ({pct}%)")
            flat_idx = np.argsort(diffs.ravel())[::-1]
            for i in range(min(5, len(flat_idx))):
                idx = np.unravel_index(flat_idx[i], py_f.shape)
                print(f"    {idx}: py={py_f[idx]:.6f}  jl={jl_f[idx]:.6f}  diff={diffs[idx]:.6f}")
            return "fail"
    else:
        py_i = py_val.astype(np.int64)
        jl_i = jl_val.astype(np.int64)
        n_diff = int(np.sum(py_i != jl_i))

        if n_diff == 0:
            print(f"  OK            {prefix:35s}  shape={list(py_val.shape)}  (integer exact)")
            return "ok"
        else:
            n_total = py_i.size
            pct = round(100.0 * n_diff / n_total, 1)
            print(f"  MISMATCH      {prefix:35s}  shape={list(py_val.shape)}  n_diff={n_diff}/{n_total} ({pct}%)")
            diff_mask = py_i != jl_i
            shown = 0
            it = np.nditer(diff_mask, flags=['multi_index'])
            while not it.finished and shown < 10:
                if it[0]:
                    idx = it.multi_index
                    print(f"    {idx}: py={py_i[idx]}  jl={jl_i[idx]}")
                    shown += 1
                it.iternext()
            return "fail"


def main():
    if len(sys.argv) < 3:
        print("Usage: compare_features.py <python.pt> <julia_dir/>")
        sys.exit(1)

    py_path, jl_path = sys.argv[1], sys.argv[2]
    print(f"Loading Python features from: {py_path}")
    py = load_python_features(py_path)
    print(f"Loading Julia features from: {jl_path}")
    jl = load_julia_features(jl_path)

    py_keys = set(py.keys())
    jl_keys = set(jl.keys())

    print(f"\n{'='*70}")
    print(f"Python-only keys: {sorted(py_keys - jl_keys)}")
    print(f"Julia-only keys:  {sorted(jl_keys - py_keys)}")
    print(f"Common keys:      {len(py_keys & jl_keys)}")
    print(f"{'='*70}\n")

    n_ok = n_fail = n_skip = 0

    for key in sorted(py_keys & jl_keys):
        py_val = py[key]
        jl_val = jl[key]

        if isinstance(py_val, dict) and isinstance(jl_val, dict):
            for subkey in sorted(set(py_val.keys()) & set(jl_val.keys())):
                r = compare_tensor(key, py_val[subkey], jl_val[subkey], subkey=subkey)
                if r == "ok": n_ok += 1
                elif r == "fail": n_fail += 1
                else: n_skip += 1
        elif isinstance(py_val, np.ndarray) and isinstance(jl_val, np.ndarray):
            r = compare_tensor(key, py_val, jl_val)
            if r == "ok": n_ok += 1
            elif r == "fail": n_fail += 1
            else: n_skip += 1
        else:
            print(f"  SKIP          {key}  (types differ: py={type(py_val).__name__} jl={type(jl_val).__name__})")
            n_skip += 1

    print(f"\n{'='*70}")
    print(f"Summary: {n_ok} OK, {n_fail} MISMATCH, {n_skip} SKIP")
    print(f"{'='*70}")
    sys.exit(1 if n_fail > 0 else 0)


if __name__ == "__main__":
    main()
