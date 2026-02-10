#!/usr/bin/env python3
"""
Convert PXDesign raw-weight bundles (manifest + float32 bin files) to safetensors.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file


def _read_manifest(raw_dir: Path) -> list[dict]:
    manifest_path = raw_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.json missing: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as fh:
        obj = json.load(fh)
    rows = obj.get("tensors")
    if not isinstance(rows, list):
        raise ValueError("manifest.json must contain a list at key 'tensors'")
    return rows


def _load_tensor(raw_dir: Path, row: dict) -> np.ndarray:
    key = row["key"]
    dtype = str(row["dtype"]).lower()
    if dtype != "float32":
        raise ValueError(f"Unsupported dtype for {key}: {dtype} (expected float32)")
    shape = [int(x) for x in row["shape"]]
    fpath = raw_dir / row["file"]
    if not fpath.is_file():
        raise FileNotFoundError(f"Tensor file missing for {key}: {fpath}")

    expected = int(math.prod(shape)) if shape else 1
    flat = np.fromfile(fpath, dtype="<f4")
    if flat.size != expected:
        raise ValueError(
            f"Tensor size mismatch for {key}: got {flat.size}, expected {expected}"
        )

    arr = flat.reshape(shape, order="C") if shape else flat.reshape(())
    # Keep contiguous host arrays for safetensors.numpy
    return np.array(arr, copy=True, dtype=np.float32)


def convert(raw_dir: Path, out_dir: Path, max_shard_bytes: int) -> None:
    rows = _read_manifest(raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shards: list[dict[str, np.ndarray]] = []
    current: dict[str, np.ndarray] = {}
    current_bytes = 0
    total_bytes = 0

    for row in rows:
        key = str(row["key"])
        tensor = _load_tensor(raw_dir, row)
        t_bytes = tensor.size * tensor.dtype.itemsize
        if current and current_bytes + t_bytes > max_shard_bytes:
            shards.append(current)
            current = {}
            current_bytes = 0
        current[key] = tensor
        current_bytes += t_bytes
        total_bytes += t_bytes

    if current:
        shards.append(current)

    if len(shards) == 1:
        out_file = out_dir / "model.safetensors"
        save_file(
            shards[0],
            str(out_file),
            metadata={"source_format": "pxdesign_raw_manifest_v1"},
        )
        print(f"wrote {out_file}")
        print(f"tensors={len(shards[0])} total_bytes={total_bytes}")
        return

    weight_map: dict[str, str] = {}
    nshard = len(shards)
    for i, shard in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{nshard:05d}.safetensors"
        save_file(
            shard,
            str(out_dir / fname),
            metadata={"source_format": "pxdesign_raw_manifest_v1"},
        )
        for key in shard:
            weight_map[key] = fname
        print(f"wrote {fname} tensors={len(shard)}")

    index_obj = {
        "metadata": {"total_size": total_bytes},
        "weight_map": weight_map,
    }
    index_file = out_dir / "model.safetensors.index.json"
    with index_file.open("w", encoding="utf-8") as fh:
        json.dump(index_obj, fh, indent=2, sort_keys=True)
    print(f"wrote {index_file}")
    print(f"shards={nshard} tensors={len(weight_map)} total_bytes={total_bytes}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, help="Directory with manifest.json and tensor_*.bin files")
    parser.add_argument("--out-dir", required=True, help="Destination directory for safetensors")
    parser.add_argument(
        "--max-shard-bytes",
        type=int,
        default=1_073_741_824,
        help="Maximum shard size in bytes before starting a new safetensors shard (default: 1 GiB)",
    )
    args = parser.parse_args()

    convert(Path(args.raw_dir), Path(args.out_dir), args.max_shard_bytes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
