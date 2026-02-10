#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch

from protenix.model.modules.confidence import ConfidenceHead


def _normalize_key(key: str) -> str:
    return key[7:] if key.startswith("module.") else key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-token", type=int, default=6)
    parser.add_argument("--n-atom", type=int, default=12)
    parser.add_argument("--n-sample", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["model"]
    state = {_normalize_key(k): v for k, v in state.items() if torch.is_tensor(v)}

    conf = ConfidenceHead(
        n_blocks=4,
        c_s=384,
        c_z=128,
        c_s_inputs=449,
        b_pae=64,
        b_pde=64,
        b_plddt=50,
        b_resolved=2,
        max_atoms_per_token=24,
        pairformer_dropout=0.0,
        distance_bin_start=3.25,
        distance_bin_end=52.0,
        distance_bin_step=1.25,
        stop_gradient=True,
    )
    sub = {
        k[len("confidence_head.") :]: v
        for k, v in state.items()
        if k.startswith("confidence_head.")
    }
    conf.load_state_dict(sub, strict=True)
    conf.eval()

    n_tok = args.n_token
    n_atom = args.n_atom
    n_sample = args.n_sample

    atom_to_token_idx = torch.arange(n_atom) % n_tok
    atom_to_tokatom_idx = torch.zeros(n_atom, dtype=torch.long)
    rep_mask = torch.zeros(n_atom, dtype=torch.bool)
    rep_mask[:n_tok] = True

    inp = {
        "distogram_rep_atom_mask": rep_mask,
        "atom_to_token_idx": atom_to_token_idx,
        "atom_to_tokatom_idx": atom_to_tokatom_idx,
    }

    s_inputs = torch.randn(n_tok, 449)
    s_trunk = torch.randn(n_tok, 384)
    z_trunk = torch.randn(n_tok, n_tok, 128)
    x_pred_coords = torch.randn(n_sample, n_atom, 3)

    with torch.no_grad():
        plddt, pae, pde, resolved = conf(
            input_feature_dict=inp,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            pair_mask=None,
            x_pred_coords=x_pred_coords,
            use_embedding=True,
            use_deepspeed_evo_attention=False,
            use_lma=False,
            inplace_safe=False,
            chunk_size=None,
        )

    payload = {
        "s_inputs": s_inputs.tolist(),
        "s_trunk": s_trunk.tolist(),
        "z_trunk": z_trunk.tolist(),
        "x_pred_coords": x_pred_coords.tolist(),
        "atom_to_token_idx": atom_to_token_idx.tolist(),
        "atom_to_tokatom_idx": atom_to_tokatom_idx.tolist(),
        "distogram_rep_atom_mask": rep_mask.tolist(),
        "plddt": plddt.tolist(),
        "pae": pae.tolist(),
        "pde": pde.tolist(),
        "resolved": resolved.tolist(),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
