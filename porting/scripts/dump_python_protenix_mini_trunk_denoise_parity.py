#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from protenix.model.modules.confidence import ConfidenceHead
from protenix.model.modules.diffusion import DiffusionModule
from protenix.model.modules.embedders import InputFeatureEmbedder, RelativePositionEncoding
from protenix.model.modules.head import DistogramHead
from protenix.model.modules.pairformer import MSAModule, PairformerStack, TemplateEmbedder
from protenix.model.modules.primitives import LayerNorm, LinearNoBias


def _normalize_key(key: str) -> str:
    return key[7:] if key.startswith("module.") else key


def _sub_state(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    out = {}
    p = prefix + "."
    for k, v in state.items():
        if k.startswith(p):
            out[k[len(p) :]] = v
    return out


def _load_linear_nobias_weight(mod: LinearNoBias, state: dict[str, torch.Tensor], key: str) -> None:
    with torch.no_grad():
        mod.weight.copy_(state[key])


def _load_layernorm_weight_bias(mod: LayerNorm, state: dict[str, torch.Tensor], prefix: str) -> None:
    with torch.no_grad():
        mod.weight.copy_(state[f"{prefix}.weight"])
        if hasattr(mod, "bias") and mod.bias is not None and f"{prefix}.bias" in state:
            mod.bias.copy_(state[f"{prefix}.bias"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-token", type=int, default=8)
    parser.add_argument("--n-atom", type=int, default=24)
    parser.add_argument("--n-cycle", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["model"]
    state = {_normalize_key(k): v for k, v in state.items() if torch.is_tensor(v)}

    input_embedder = InputFeatureEmbedder(c_atom=128, c_atompair=16, c_token=384, esm_configs={})
    relpos = RelativePositionEncoding(r_max=32, s_max=2, c_z=128)
    template_embedder = TemplateEmbedder(n_blocks=0, c=64, c_z=128, dropout=0.25)
    msa_module = MSAModule(
        c_m=64,
        c_z=128,
        c_s_inputs=449,
        n_blocks=1,
        msa_configs={
            "enable": True,
            "strategy": "topk",
            "sample_cutoff": {"train": 1, "test": 1},
            "min_size": {"train": 1, "test": 1},
        },
    )
    pairformer_stack = PairformerStack(c_s=384, c_z=128, n_blocks=16, n_heads=16, dropout=0.25)
    diffusion_module = DiffusionModule(
        sigma_data=16.0,
        c_atom=128,
        c_atompair=16,
        c_token=768,
        c_s=384,
        c_z=128,
        c_s_inputs=449,
        atom_encoder={"n_blocks": 1, "n_heads": 4},
        transformer={"n_blocks": 8, "n_heads": 16, "drop_path_rate": 0},
        atom_decoder={"n_blocks": 1, "n_heads": 4},
    )
    distogram_head = DistogramHead(c_z=128, no_bins=64)
    confidence_head = ConfidenceHead(
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

    linear_no_bias_sinit = LinearNoBias(in_features=449, out_features=384)
    linear_no_bias_zinit1 = LinearNoBias(in_features=384, out_features=128)
    linear_no_bias_zinit2 = LinearNoBias(in_features=384, out_features=128)
    linear_no_bias_token_bond = LinearNoBias(in_features=1, out_features=128)
    linear_no_bias_z_cycle = LinearNoBias(in_features=128, out_features=128)
    linear_no_bias_s = LinearNoBias(in_features=384, out_features=384)
    layernorm_z_cycle = LayerNorm(128)
    layernorm_s = LayerNorm(384)

    input_embedder.load_state_dict(_sub_state(state, "input_embedder"), strict=True)
    relpos.load_state_dict(_sub_state(state, "relative_position_encoding"), strict=True)
    template_embedder.load_state_dict(_sub_state(state, "template_embedder"), strict=True)
    msa_module.load_state_dict(_sub_state(state, "msa_module"), strict=True)
    pairformer_stack.load_state_dict(_sub_state(state, "pairformer_stack"), strict=True)
    diffusion_module.load_state_dict(_sub_state(state, "diffusion_module"), strict=True)
    distogram_head.load_state_dict(_sub_state(state, "distogram_head"), strict=True)
    confidence_head.load_state_dict(_sub_state(state, "confidence_head"), strict=True)

    _load_linear_nobias_weight(linear_no_bias_sinit, state, "linear_no_bias_sinit.weight")
    _load_linear_nobias_weight(linear_no_bias_zinit1, state, "linear_no_bias_zinit1.weight")
    _load_linear_nobias_weight(linear_no_bias_zinit2, state, "linear_no_bias_zinit2.weight")
    _load_linear_nobias_weight(linear_no_bias_token_bond, state, "linear_no_bias_token_bond.weight")
    _load_linear_nobias_weight(linear_no_bias_z_cycle, state, "linear_no_bias_z_cycle.weight")
    _load_linear_nobias_weight(linear_no_bias_s, state, "linear_no_bias_s.weight")
    _load_layernorm_weight_bias(layernorm_z_cycle, state, "layernorm_z_cycle")
    _load_layernorm_weight_bias(layernorm_s, state, "layernorm_s")

    for m in (
        input_embedder,
        relpos,
        template_embedder,
        msa_module,
        pairformer_stack,
        diffusion_module,
        distogram_head,
        confidence_head,
        linear_no_bias_sinit,
        linear_no_bias_zinit1,
        linear_no_bias_zinit2,
        linear_no_bias_token_bond,
        linear_no_bias_z_cycle,
        linear_no_bias_s,
        layernorm_z_cycle,
        layernorm_s,
    ):
        m.eval()

    n_tok = args.n_token
    n_atom = args.n_atom
    n_msa = 1

    restype_idx = torch.randint(low=0, high=21, size=(n_tok,), dtype=torch.long)
    restype = F.one_hot(restype_idx, num_classes=32).float()
    profile = restype.clone()
    deletion_mean = torch.zeros(n_tok)
    msa = restype_idx.reshape(1, n_tok).clone()
    has_deletion = torch.zeros(n_msa, n_tok)
    deletion_value = torch.zeros(n_msa, n_tok)
    token_bonds = torch.zeros(n_tok, n_tok)

    token_index = torch.arange(n_tok, dtype=torch.long)
    residue_index = torch.arange(n_tok, dtype=torch.long)
    asym_id = torch.zeros(n_tok, dtype=torch.long)
    entity_id = torch.zeros(n_tok, dtype=torch.long)
    sym_id = torch.zeros(n_tok, dtype=torch.long)

    atom_to_token_idx = torch.arange(n_atom, dtype=torch.long) % n_tok
    atom_to_tokatom_idx = torch.arange(n_atom, dtype=torch.long) % 24
    distogram_rep_atom_mask = torch.zeros(n_atom, dtype=torch.bool)
    distogram_rep_atom_mask[:n_tok] = True

    ref_pos = torch.randn(n_atom, 3)
    ref_charge = torch.randn(n_atom)
    ref_mask = torch.ones(n_atom)
    ref_element = F.one_hot(torch.randint(0, 128, size=(n_atom,)), num_classes=128).float()
    ref_atom_name_chars = F.one_hot(torch.randint(0, 256, size=(n_atom,)), num_classes=256).float()
    ref_space_uid = atom_to_token_idx.clone()

    feat = {
        "token_index": token_index,
        "residue_index": residue_index,
        "asym_id": asym_id,
        "entity_id": entity_id,
        "sym_id": sym_id,
        "token_bonds": token_bonds,
        "restype": restype,
        "profile": profile,
        "deletion_mean": deletion_mean,
        "msa": msa,
        "has_deletion": has_deletion,
        "deletion_value": deletion_value,
        "atom_to_token_idx": atom_to_token_idx,
        "atom_to_tokatom_idx": atom_to_tokatom_idx,
        "distogram_rep_atom_mask": distogram_rep_atom_mask,
        "ref_pos": ref_pos,
        "ref_charge": ref_charge,
        "ref_mask": ref_mask,
        "ref_element": ref_element,
        "ref_atom_name_chars": ref_atom_name_chars,
        "ref_space_uid": ref_space_uid,
    }

    with torch.no_grad():
        s_inputs = input_embedder(feat, inplace_safe=False, chunk_size=None)
        s_init = linear_no_bias_sinit(s_inputs)
        z_init = linear_no_bias_zinit1(s_init)[:, None, :] + linear_no_bias_zinit2(s_init)[None, :, :]
        z_init = z_init + relpos(feat)
        z_init = z_init + linear_no_bias_token_bond(token_bonds.unsqueeze(-1))

        z = torch.zeros_like(z_init)
        s = torch.zeros_like(s_init)
        for _ in range(args.n_cycle):
            z = z_init + linear_no_bias_z_cycle(layernorm_z_cycle(z))
            z = z + template_embedder(
                feat,
                z,
                pair_mask=None,
                use_deepspeed_evo_attention=False,
                use_lma=False,
                inplace_safe=False,
                chunk_size=None,
            )
            z = msa_module(
                feat,
                z,
                s_inputs,
                pair_mask=None,
                use_deepspeed_evo_attention=False,
                use_lma=False,
                inplace_safe=False,
                chunk_size=None,
            )
            s = s_init + linear_no_bias_s(layernorm_s(s))
            s, z = pairformer_stack(
                s,
                z,
                pair_mask=None,
                use_deepspeed_evo_attention=False,
                use_lma=False,
                inplace_safe=False,
                chunk_size=None,
            )

        x_noisy = torch.randn(1, n_atom, 3)
        t_hat = torch.full((1,), 1.5)
        x_denoised = diffusion_module(
            x_noisy,
            t_hat,
            input_feature_dict=feat,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=z,
            inplace_safe=False,
            chunk_size=None,
            use_conditioning=True,
        )

        distogram_logits = distogram_head(z)
        plddt, pae, pde, resolved = confidence_head(
            input_feature_dict=feat,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=z,
            pair_mask=None,
            x_pred_coords=x_denoised,
            use_embedding=True,
            use_deepspeed_evo_attention=False,
            use_lma=False,
            inplace_safe=False,
            chunk_size=None,
        )

    payload = {
        "n_cycle": args.n_cycle,
        "feat": {k: v.tolist() for k, v in feat.items()},
        "s_inputs": s_inputs.tolist(),
        "s_trunk": s.tolist(),
        "z_trunk": z.tolist(),
        "x_noisy": x_noisy.tolist(),
        "t_hat": t_hat.tolist(),
        "x_denoised": x_denoised.tolist(),
        "distogram_logits": distogram_logits.tolist(),
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
