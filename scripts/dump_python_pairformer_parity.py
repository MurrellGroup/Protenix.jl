#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch

from protenix.model.modules.pairformer import PairformerStack


def _normalize_key(key: str) -> str:
    return key[7:] if key.startswith("module.") else key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-token", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["model"]
    state = {_normalize_key(k): v for k, v in state.items() if torch.is_tensor(v)}

    stack = PairformerStack(n_blocks=16, n_heads=16, c_z=128, c_s=384, dropout=0.25)
    sub = {
        k[len("pairformer_stack.") :]: v
        for k, v in state.items()
        if k.startswith("pairformer_stack.")
    }
    stack.load_state_dict(sub, strict=True)
    stack.eval()

    n = args.n_token
    s = torch.randn(n, 384)
    z = torch.randn(n, n, 128)

    with torch.no_grad():
        s_out, z_out = stack(
            s,
            z,
            pair_mask=None,
            use_deepspeed_evo_attention=False,
            use_lma=False,
            inplace_safe=False,
            chunk_size=None,
        )

    payload = {
        "n_token": n,
        "s_in": s.tolist(),
        "z_in": z.tolist(),
        "s_out": s_out.tolist(),
        "z_out": z_out.tolist(),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
