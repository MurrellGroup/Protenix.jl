#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from protenix.model.modules.pairformer import MSAModule


def _normalize_key(key: str) -> str:
    return key[7:] if key.startswith("module.") else key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-token", type=int, default=6)
    parser.add_argument("--n-blocks", type=int, default=1)
    parser.add_argument("--c-z", type=int, default=128)
    parser.add_argument("--c-s-inputs", type=int, default=449)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["model"]
    state = {_normalize_key(k): v for k, v in state.items() if torch.is_tensor(v)}

    module = MSAModule(
        n_blocks=args.n_blocks,
        c_m=64,
        c_z=args.c_z,
        c_s_inputs=args.c_s_inputs,
        msa_configs={
            "enable": True,
            "strategy": "topk",
            "sample_cutoff": {"train": 1, "test": 1},
            "min_size": {"train": 1, "test": 1},
        },
    )
    sub = {
        k[len("msa_module.") :]: v
        for k, v in state.items()
        if k.startswith("msa_module.")
    }
    module.load_state_dict(sub, strict=True)
    module.eval()

    n = args.n_token
    n_msa = 1
    feat = {
        "msa": torch.randint(low=0, high=32, size=(n_msa, n), dtype=torch.long),
        "has_deletion": torch.zeros(n_msa, n),
        "deletion_value": torch.zeros(n_msa, n),
    }
    z = torch.randn(n, n, 128)
    s_inputs = torch.randn(n, 449)

    with torch.no_grad():
        msa_oh = F.one_hot(feat["msa"], num_classes=32)
        target_shape = msa_oh.shape[:-1]
        msa_sample = torch.cat(
            [
                msa_oh.reshape(*target_shape, 32),
                feat["has_deletion"].reshape(*target_shape, 1),
                feat["deletion_value"].reshape(*target_shape, 1),
            ],
            dim=-1,
        )
        m0 = module.linear_no_bias_m(msa_sample)
        m0 = m0 + module.linear_no_bias_s(s_inputs)
        opm = module.blocks[0].outer_product_mean_msa(m0, inplace_safe=False, chunk_size=None)
        z1 = z + opm
        _, z2 = module.blocks[0].pair_stack(
            s=None,
            z=z1,
            pair_mask=None,
            use_deepspeed_evo_attention=False,
            use_lma=False,
            inplace_safe=False,
            chunk_size=None,
        )
        m_cur = m0
        z_cur = z
        z_blocks = []
        m_blocks = []
        block_debug = []
        for blk in module.blocks:
            dbg = {}
            opm_blk = blk.outer_product_mean_msa(
                m_cur, inplace_safe=False, chunk_size=None
            )
            z_pre_pair = z_cur + opm_blk
            dbg["opm"] = opm_blk.tolist()
            dbg["z_pre_pair"] = z_pre_pair.tolist()

            m_next = m_cur
            if not blk.is_last_block:
                m_pair = blk.msa_stack.msa_pair_weighted_averaging(m_cur, z_pre_pair)
                m_after_pair = m_cur + m_pair
                m_trans = blk.msa_stack.transition_m(m_after_pair)
                m_next = m_after_pair + m_trans
                dbg["m_pair"] = m_pair.tolist()
                dbg["m_after_pair"] = m_after_pair.tolist()
                dbg["m_trans"] = m_trans.tolist()
                dbg["m_after_trans"] = m_next.tolist()
            else:
                dbg["m_pair"] = None
                dbg["m_after_pair"] = None
                dbg["m_trans"] = None
                dbg["m_after_trans"] = None

            m_cur, z_cur = blk(
                m_cur,
                z_cur,
                pair_mask=None,
                use_deepspeed_evo_attention=False,
                use_lma=False,
                inplace_safe=False,
                chunk_size=None,
            )
            z_blocks.append(z_cur.tolist())
            m_blocks.append(None if m_cur is None else m_cur.tolist())
            dbg["z_post_pair"] = z_cur.tolist()
            block_debug.append(dbg)
        z_out = z_cur
        z_out_module = module(
            feat,
            z,
            s_inputs,
            pair_mask=None,
            use_deepspeed_evo_attention=False,
            use_lma=False,
            inplace_safe=False,
            chunk_size=None,
        )

    payload = {
        "z_in": z.tolist(),
        "s_inputs": s_inputs.tolist(),
        "msa": feat["msa"].tolist(),
        "has_deletion": feat["has_deletion"].tolist(),
        "deletion_value": feat["deletion_value"].tolist(),
        "m0": m0.tolist(),
        "opm": opm.tolist(),
        "z1": z1.tolist(),
        "z2": z2.tolist(),
        "z_out": z_out.tolist(),
        "z_out_module": z_out_module.tolist(),
        "z_blocks": z_blocks,
        "m_blocks": m_blocks,
        "block_debug": block_debug,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
