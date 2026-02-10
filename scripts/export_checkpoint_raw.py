#!/usr/bin/env python3
"""
Export a PyTorch checkpoint into a Julia-friendly raw tensor bundle.

Output:
- manifest.json
- tensor_000000.bin, tensor_000001.bin, ...

Manifest row format:
{
  "key": "diffusion_module....weight",
  "dtype": "float32",
  "shape": [128, 256],
  "file": "tensor_000123.bin"
}
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def normalize_key(key: str) -> str:
    return key[7:] if key.startswith("module.") else key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument(
        "--cast-float32",
        action="store_true",
        default=False,
        help="Cast all floating tensors to float32 before export",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    obj = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(obj, dict) or "model" not in obj or not isinstance(obj["model"], dict):
        raise ValueError("Unexpected checkpoint format: expected dict with `model` state-dict")

    state = obj["model"]
    keys = sorted(state.keys())
    manifest = []

    for idx, raw_key in enumerate(keys):
        tensor = state[raw_key]
        if not torch.is_tensor(tensor):
            continue

        t = tensor.detach().cpu().contiguous()
        if args.cast_float32 and t.is_floating_point():
            t = t.to(dtype=torch.float32)

        arr = t.numpy()
        if arr.dtype != np.float32:
            raise ValueError(
                f"Tensor {raw_key} has dtype {arr.dtype}; rerun with --cast-float32 "
                "or extend loader/exporter dtype support."
            )

        fname = f"tensor_{idx:06d}.bin"
        fpath = outdir / fname
        arr.tofile(fpath)

        manifest.append(
            {
                "key": normalize_key(raw_key),
                "dtype": "float32",
                "shape": list(arr.shape),
                "file": fname,
            }
        )

    with open(outdir / "manifest.json", "w") as f:
        json.dump({"num_tensors": len(manifest), "tensors": manifest}, f, indent=2)

    print(f"Exported {len(manifest)} tensors to {outdir}")


if __name__ == "__main__":
    main()
