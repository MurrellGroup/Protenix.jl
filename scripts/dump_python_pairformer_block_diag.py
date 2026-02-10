#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch

from protenix.model.modules.pairformer import PairformerBlock


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

    blk = PairformerBlock(n_heads=16, c_z=128, c_s=384, dropout=0.25)
    sub = {
        k[len("pairformer_stack.blocks.0.") :]: v
        for k, v in state.items()
        if k.startswith("pairformer_stack.blocks.0.")
    }
    blk.load_state_dict(sub, strict=True)
    blk.eval()

    n = args.n_token
    s = torch.randn(n, 384)
    z = torch.randn(n, n, 128)

    with torch.no_grad():
        tmu_out = blk.tri_mul_out(z, mask=None, inplace_safe=False, _add_with_inplace=False)
        z1 = z + tmu_out

        tmu_in = blk.tri_mul_in(z1, mask=None, inplace_safe=False, _add_with_inplace=False)
        z2 = z1 + tmu_in

        ta_start = blk.tri_att_start(
            z2,
            mask=None,
            use_deepspeed_evo_attention=False,
            use_lma=False,
            inplace_safe=False,
            chunk_size=None,
        )
        z3 = z2 + ta_start

        z4 = z3.transpose(-2, -3)
        ta_end = blk.tri_att_end(
            z4,
            mask=None,
            use_deepspeed_evo_attention=False,
            use_lma=False,
            inplace_safe=False,
            chunk_size=None,
        )
        z5 = z4 + ta_end
        z6 = z5.transpose(-2, -3)

        pair_tr = blk.pair_transition(z6)
        z7 = z6 + pair_tr

        apb = blk.attention_pair_bias(a=s, s=None, z=z7)
        s1 = s + apb

        s_tr = blk.single_transition(s1)
        s2 = s1 + s_tr

    payload = {
        "s_in": s.tolist(),
        "z_in": z.tolist(),
        "tmu_out": tmu_out.tolist(),
        "z1": z1.tolist(),
        "tmu_in": tmu_in.tolist(),
        "z2": z2.tolist(),
        "ta_start": ta_start.tolist(),
        "z3": z3.tolist(),
        "ta_end": ta_end.tolist(),
        "z6": z6.tolist(),
        "pair_tr": pair_tr.tolist(),
        "z7": z7.tolist(),
        "apb": apb.tolist(),
        "s1": s1.tolist(),
        "s_tr": s_tr.tolist(),
        "s2": s2.tolist(),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
