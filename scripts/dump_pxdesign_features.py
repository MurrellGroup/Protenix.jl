#!/usr/bin/env python3
"""
Dump PXDesign input features from the Python pipeline.

Monkeypatches biotite 1.4 for compatibility with Protenix v0.5's
MMCIFParser, then runs PXDesign InferenceDataset to extract features.

Usage:
    /home/claudey/venvs/python_esmfold/bin/python3 \
        /home/claudey/FixingKAFA/PXDesign.jl/scripts/dump_pxdesign_features.py

Output: /tmp/pxdesign_parity/py_dumps/<case_name>.json
"""

import sys
import os
import json
import glob
import gzip
import pickle
import numpy as np

# =============================================================================
# Biotite 1.4 compatibility shim for Protenix v0.5
# =============================================================================
import biotite.structure.io.pdbx.convert as pdbx_convert

# Fix 1: PDBX_COVALENT_TYPES removed in biotite 1.4
if not hasattr(pdbx_convert, 'PDBX_COVALENT_TYPES'):
    pdbx_convert.PDBX_COVALENT_TYPES = set()

# Fix 2: _get_model_starts removed in biotite 1.4
# Protenix v0.5 calls: model_starts = pdbx_convert._get_model_starts(models)
# then: _filter_model(atom_site, model_starts, model)
# Biotite 1.4 uses: _filter_model(atom_site, model) directly
if not hasattr(pdbx_convert, '_get_model_starts'):
    def _get_model_starts(models):
        """Compatibility shim: find start indices of each model."""
        starts = [0]
        for i in range(1, len(models)):
            if models[i] != models[i-1]:
                starts.append(i)
        return starts
    pdbx_convert._get_model_starts = _get_model_starts

    # Also need to wrap _filter_model to accept old 3-arg signature
    _orig_filter_model = pdbx_convert._filter_model
    def _compat_filter_model(atom_site, model_starts_or_model, model=None):
        """Compatibility wrapper: accept old 3-arg or new 2-arg signature."""
        if model is not None:
            # Old signature: (atom_site, model_starts, model_number)
            return _orig_filter_model(atom_site, model)
        else:
            # New signature: (atom_site, model)
            return _orig_filter_model(atom_site, model_starts_or_model)
    pdbx_convert._filter_model = _compat_filter_model

# Fix 3: _get_box may also have changed
if not hasattr(pdbx_convert, '_get_box'):
    def _get_box(block):
        return None
    pdbx_convert._get_box = _get_box

# =============================================================================

# Add PXDesign and Protenix to path
sys.path.insert(0, '/home/claudey/FixingKAFA/PXDesign.jl/.external/PXDesign')
sys.path.insert(0, '/home/claudey/FixingKAFA/PXDesign.jl/.external/Protenix')

# Set PROTENIX_DATA_ROOT_DIR to point to the actual CCD cache location
CCD_CACHE_DIR = '/home/claudey/FixingKAFA/PXDesign.jl/.external/Protenix/release_data/ccd_cache'
os.environ.setdefault('PROTENIX_DATA_ROOT_DIR', CCD_CACHE_DIR)

import torch

INPUTS_DIR = "/home/claudey/FixingKAFA/PXDesign.jl/clean_targets/inputs"
PKL_CACHE_DIR = "/tmp/pxdesign_parity/pkl_cache"
DUMP_DIR = "/tmp/pxdesign_parity/py_dumps"

os.makedirs(PKL_CACHE_DIR, exist_ok=True)
os.makedirs(DUMP_DIR, exist_ok=True)


def cif_to_pkl_gz(cif_path: str) -> str:
    """Convert CIF file to .pkl.gz bioassembly dict using Protenix's parser."""
    name = os.path.splitext(os.path.basename(cif_path))[0]
    pkl_path = os.path.join(PKL_CACHE_DIR, f"{name}.pkl.gz")
    if os.path.exists(pkl_path):
        print(f"  Using cached {pkl_path}")
        return pkl_path

    print(f"  Converting {cif_path} → {pkl_path}")
    from protenix.data.parser import MMCIFParser
    parser = MMCIFParser(cif_path)
    bioassembly_dict = parser.get_bioassembly(assembly_id="1")

    if bioassembly_dict["atom_array"] is None:
        raise ValueError(f"Failed to parse {cif_path} - atom_array is None")

    with gzip.open(pkl_path, "wb") as f:
        pickle.dump(bioassembly_dict, f)
    n_atoms = len(bioassembly_dict["atom_array"])
    print(f"  Created {pkl_path} ({n_atoms} atoms)")
    return pkl_path


def rewrite_json_for_python(json_path: str) -> str:
    """Create a modified JSON file with .pkl.gz structure paths."""
    with open(json_path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        data = [data]

    json_dir = os.path.dirname(os.path.abspath(json_path))

    for task in data:
        if "condition" not in task:
            continue
        sf = task["condition"].get("structure_file", "")
        if not sf or sf.endswith(".pkl.gz"):
            continue

        # Resolve relative path
        if not os.path.isabs(sf):
            sf = os.path.normpath(os.path.join(json_dir, sf))

        pkl_path = cif_to_pkl_gz(sf)
        task["condition"]["structure_file"] = pkl_path

    name = os.path.splitext(os.path.basename(json_path))[0]
    out_path = os.path.join(PKL_CACHE_DIR, f"{name}_py.json")
    with open(out_path, "w") as f:
        json.dump(data, f)
    return out_path


def tensor_to_serializable(t):
    """Convert tensor/array to JSON-serializable format."""
    if isinstance(t, torch.Tensor):
        arr = t.detach().cpu().numpy()
    elif isinstance(t, np.ndarray):
        arr = t
    elif isinstance(t, (int, float, bool)):
        return {"data": t, "dtype": type(t).__name__}
    else:
        return {"data": str(t), "dtype": str(type(t))}

    return {
        "data": arr.tolist(),
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
    }


def dump_features(json_path: str, case_name: str):
    """Run Python PXDesign pipeline and dump input_feature_dict."""
    print(f"\n=== {case_name} ===")

    py_json = rewrite_json_for_python(json_path)
    print(f"  Python JSON: {py_json}")

    from pxdesign.data.infer_data_pipeline import InferenceDataset

    try:
        dataset = InferenceDataset(input_json_path=py_json, use_msa=False)
        data, atom_array, error_msg = dataset[0]
    except Exception as e:
        print(f"  ERROR in dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

    if error_msg:
        print(f"  Warning: {error_msg}")

    if "input_feature_dict" not in data:
        print(f"  ERROR: no input_feature_dict in data (keys: {list(data.keys())})")
        return False

    feat = data["input_feature_dict"]
    n_token = data.get("N_token", 0)
    n_atom = data.get("N_atom", 0)
    if isinstance(n_token, torch.Tensor):
        n_token = n_token.item()
    if isinstance(n_atom, torch.Tensor):
        n_atom = n_atom.item()

    print(f"  Features: {len(feat)} keys, N_token={n_token}, N_atom={n_atom}")

    serialized = {}
    for k in sorted(feat.keys()):
        v = feat[k]
        try:
            serialized[k] = tensor_to_serializable(v)
            if isinstance(v, torch.Tensor):
                print(f"    {k}: {list(v.shape)} {v.dtype}")
        except Exception as e:
            print(f"    Warning: could not serialize {k}: {e}")

    out = {
        "case_name": case_name,
        "N_token": int(n_token),
        "N_atom": int(n_atom),
        "input_feature_dict": serialized,
    }

    out_path = os.path.join(DUMP_DIR, f"{case_name}.json")
    with open(out_path, "w") as f:
        json.dump(out, f)
    fsize = os.path.getsize(out_path)
    print(f"  Dumped to {out_path} ({fsize // 1024} KB)")
    return True


def main():
    test_cases = []

    # Existing design targets (22-32): use input_source_snapshot.json
    run_dir = "/home/claudey/FixingKAFA/PXDesign.jl/clean_targets/run_20260224/clean_targets"
    for i in range(22, 33):
        yamls = glob.glob(os.path.join(INPUTS_DIR, f"{i}_*.yaml"))
        if yamls:
            name = os.path.splitext(os.path.basename(yamls[0]))[0]
            snap = os.path.join(run_dir, f"{name}__pxdesign_v0.1.0", "input_source_snapshot.json")
            if os.path.exists(snap):
                test_cases.append((snap, name))
            else:
                print(f"  Skipping {name}: no snapshot JSON found")

    # New design targets (38-45): direct JSON files
    for i in range(38, 46):
        jsons = glob.glob(os.path.join(INPUTS_DIR, f"{i}_*.json"))
        if jsons:
            name = os.path.splitext(os.path.basename(jsons[0]))[0]
            test_cases.append((jsons[0], name))

    print(f"Found {len(test_cases)} test cases:")
    for path, name in test_cases:
        print(f"  {name}")

    n_pass = 0
    n_fail = 0
    failures = []
    for path, name in test_cases:
        try:
            ok = dump_features(path, name)
            if ok:
                n_pass += 1
            else:
                n_fail += 1
                failures.append(name)
        except Exception as e:
            print(f"  EXCEPTION for {name}: {e}")
            import traceback
            traceback.print_exc()
            n_fail += 1
            failures.append(name)

    print(f"\n{'='*60}")
    print(f"PXDesign Feature Dump Summary")
    print(f"{'='*60}")
    print(f"  Pass: {n_pass}")
    print(f"  Fail: {n_fail}")
    print(f"  Total: {n_pass + n_fail}")
    if failures:
        print(f"  Failures: {', '.join(failures)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
