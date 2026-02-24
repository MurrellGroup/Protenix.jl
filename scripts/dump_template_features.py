#!/usr/bin/env python3
"""
Extract template features from a user-provided CIF file for parity comparison.

This bypasses the template search (hmmsearch/hhblits) and directly processes
a CIF file as a template for a given query sequence. It uses the same
TemplateParser, TemplateHitProcessor, and TemplateFeatures code as Protenix
but skips the hit search/filter steps.

Usage:
    cd /tmp/protenix_v1
    export PATH=/home/claudey/.local/bin:$PATH
    /home/claudey/venvs/protenix_v1/bin/python \
        /home/claudey/FixingKAFA/PXDesign.jl/scripts/dump_template_features.py \
        --query-sequence "MQIFVKTLTGKTITLEVEPS..." \
        --template-cif /path/to/template.cif \
        --template-chain A \
        --dump-dir /tmp/template_parity \
        --name test_case_1

Output:
    <dump_dir>/<name>_template_features.npz — numpy archive with all template tensors
    <dump_dir>/<name>_template_features.json — JSON with shapes and metadata
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

from protenix.data.constants import (
    ATOM37_NUM,
    ATOM37_ORDER,
    PROTEIN_AATYPE_DENSE_ATOM_TO_ATOM37,
    PROTEIN_CHAIN,
)
from protenix.data.template.template_featurizer import (
    DistogramFeaturesConfig,
    Templates,
    TemplateFeatureAssemblyLine,
    TemplateFeatures,
)
from protenix.data.template.template_parser import (
    TemplateParser,
    encode_template_restype,
)
from protenix.data.tools.kalign import Kalign


def extract_template_from_cif(
    query_sequence: str,
    cif_path: str,
    chain_id: str,
    kalign_binary: str = "kalign",
    zero_center: bool = True,
    max_ca_dist: float = 150.0,
):
    """
    Extract template features from a CIF file for a query sequence.

    This replicates what TemplateHitProcessor does but takes a CIF file directly
    instead of going through the hit search pipeline.

    Returns:
        dict with keys:
            template_all_atom_positions: (num_query, 37, 3)
            template_all_atom_masks: (num_query, 37)
            template_aatype: (num_query,)
            template_sequence: bytes
    """
    # 1. Parse the CIF file
    with open(cif_path, "r") as f:
        cif_string = f.read()

    pdb_id = os.path.splitext(os.path.basename(cif_path))[0]
    result = TemplateParser.parse(
        file_id=pdb_id, mmcif_string=cif_string, auth_chain_id=chain_id
    )
    if not result.mmcif_object:
        raise RuntimeError(f"Failed to parse CIF: {result.errors}")

    mmcif_obj = result.mmcif_object

    # 2. Get the template chain sequence
    actual_chain_id = chain_id
    target_seq = mmcif_obj.chain_to_seqres.get(chain_id, "")
    if not target_seq:
        if len(mmcif_obj.chain_to_seqres) == 1:
            actual_chain_id = list(mmcif_obj.chain_to_seqres.keys())[0]
            target_seq = mmcif_obj.chain_to_seqres[actual_chain_id]
        else:
            available = list(mmcif_obj.chain_to_seqres.keys())
            raise RuntimeError(
                f"Chain {chain_id} not found. Available chains: {available}"
            )

    print(f"  Template chain {actual_chain_id}: {len(target_seq)} residues")
    print(f"  Query sequence: {len(query_sequence)} residues")

    # 3. Align query to template using Kalign
    aligner = Kalign(binary_path=kalign_binary)

    # Kalign requires >= 6 residues
    if len(query_sequence) < 6 or len(target_seq) < 6:
        raise ValueError("Both query and template must be >= 6 residues for Kalign")

    q_aln, t_aln = aligner.align([query_sequence, target_seq])

    # Build mapping: query_idx -> template_idx
    mapping = {}
    q_idx, t_idx, same, count = -1, -1, 0, 0
    for qa, ta in zip(q_aln, t_aln):
        if qa != "-":
            q_idx += 1
        if ta != "-":
            t_idx += 1
        if qa != "-" and ta != "-":
            mapping[q_idx] = t_idx
            count += 1
            if qa == ta:
                same += 1

    identity = same / count if count > 0 else 0.0
    print(f"  Alignment: {count} aligned positions, {same} identical ({identity:.1%})")

    # 4. Extract atom coordinates from the template CIF
    chains = list(mmcif_obj.structure.get_chains())
    relevant_chains = [c for c in chains if c.id == actual_chain_id]
    if len(relevant_chains) != 1:
        raise RuntimeError(
            f"Expected 1 chain {actual_chain_id}, found {len(relevant_chains)}"
        )
    chain = relevant_chains[0]

    num_res_template = len(mmcif_obj.chain_to_seqres[actual_chain_id])
    all_pos = np.zeros((num_res_template, ATOM37_NUM, 3), dtype=np.float32)
    all_mask = np.zeros((num_res_template, ATOM37_NUM), dtype=np.float32)

    for i in range(num_res_template):
        res_info = mmcif_obj.seqres_to_structure[actual_chain_id][i]
        if not res_info.is_missing:
            try:
                res = chain[
                    (
                        res_info.hetflag,
                        res_info.position.residue_number,
                        res_info.position.insertion_code,
                    )
                ]
                for atom in res.get_atoms():
                    name = atom.get_name()
                    coord = atom.get_coord()
                    if name in ATOM37_ORDER:
                        idx = ATOM37_ORDER[name]
                        all_pos[i, idx] = coord
                        all_mask[i, idx] = 1.0
                    elif name.upper() == "SE" and res.get_resname() == "MSE":
                        idx = ATOM37_ORDER["SD"]
                        all_pos[i, idx] = coord
                        all_mask[i, idx] = 1.0

                # Correct Arginine NH1/NH2 if swapped
                cd = ATOM37_ORDER["CD"]
                nh1 = ATOM37_ORDER["NH1"]
                nh2 = ATOM37_ORDER["NH2"]
                if res.get_resname() == "ARG" and all(all_mask[i, [cd, nh1, nh2]]):
                    if np.linalg.norm(
                        all_pos[i, nh1] - all_pos[i, cd]
                    ) > np.linalg.norm(all_pos[i, nh2] - all_pos[i, cd]):
                        all_pos[i, [nh1, nh2]] = all_pos[i, [nh2, nh1]]
                        all_mask[i, [nh1, nh2]] = all_mask[i, [nh2, nh1]]
            except KeyError:
                continue

    # Check CA distances
    ca_idx = ATOM37_ORDER["CA"]
    prev_pos = None
    for i in range(num_res_template):
        if all_mask[i, ca_idx]:
            curr = all_pos[i, ca_idx]
            if prev_pos is not None:
                dist = np.linalg.norm(curr - prev_pos)
                if dist > max_ca_dist:
                    print(
                        f"  WARNING: CA distance {dist:.1f} > {max_ca_dist} at residue {i}"
                    )
            prev_pos = curr

    # Zero-center positions
    if zero_center:
        mask_bool = all_mask.astype(bool)
        if np.any(mask_bool):
            center = all_pos[mask_bool].mean(axis=0)
            all_pos[mask_bool] -= center

    # 5. Map template atoms to query positions
    num_query = len(query_sequence)
    out_pos = np.zeros((num_query, ATOM37_NUM, 3), dtype=np.float32)
    out_mask = np.zeros((num_query, ATOM37_NUM), dtype=np.float32)
    out_seq = ["-"] * num_query

    for q_idx, t_idx in mapping.items():
        if t_idx != -1:
            out_pos[q_idx] = all_pos[t_idx]
            out_mask[q_idx] = all_mask[t_idx]
            out_seq[q_idx] = target_seq[t_idx]

    out_seq_str = "".join(out_seq)
    aatype = encode_template_restype(PROTEIN_CHAIN, out_seq_str)

    n_atoms_mapped = int(np.sum(out_mask))
    print(f"  Mapped atoms: {n_atoms_mapped}")

    features = {
        "template_all_atom_positions": out_pos,  # (num_query, 37, 3)
        "template_all_atom_masks": out_mask,  # (num_query, 37)
        "template_aatype": np.array(aatype, dtype=np.int32),  # (num_query,)
        "template_sequence": out_seq_str.encode(),
    }
    return features


def fix_and_derive_template_features(raw_features, num_query):
    """
    Convert raw 37-atom features to dense 24-atom and compute derived features.

    This replicates TemplateFeatures.fix_template_features() + Templates.as_protenix_dict().
    """
    # Stack into template batch dimension (N_template=1)
    aatype = raw_features["template_aatype"][None, :]  # (1, N)
    all_pos_37 = raw_features["template_all_atom_positions"][None, :]  # (1, N, 37, 3)
    all_mask_37 = raw_features["template_all_atom_masks"][None, :]  # (1, N, 37)

    # Convert from atom37 to dense 24-atom representation
    dense_atom_indices = np.take(
        PROTEIN_AATYPE_DENSE_ATOM_TO_ATOM37, aatype, axis=0
    )  # (1, N, 24)

    atom_mask = np.take_along_axis(all_mask_37, dense_atom_indices, axis=2)  # (1, N, 24)
    atom_positions = np.take_along_axis(
        all_pos_37, dense_atom_indices[..., None], axis=2
    )  # (1, N, 24, 3)
    atom_positions *= atom_mask[..., None]

    # Build Templates object and compute derived features
    templates = Templates(
        aatype=aatype,
        atom_positions=atom_positions.astype(np.float32),
        atom_mask=atom_mask.astype(bool),
    )
    features = templates.as_protenix_dict()

    return features


def main():
    parser = argparse.ArgumentParser(
        description="Extract template features from CIF for parity testing"
    )
    parser.add_argument(
        "--query-sequence", required=True, help="Query protein sequence (1-letter)"
    )
    parser.add_argument(
        "--template-cif", required=True, help="Path to template CIF file"
    )
    parser.add_argument(
        "--template-chain", required=True, help="Chain ID in the template CIF"
    )
    parser.add_argument("--dump-dir", required=True, help="Output directory")
    parser.add_argument("--name", default="template_test", help="Test case name")
    parser.add_argument(
        "--kalign-binary",
        default="kalign",
        help="Path to kalign binary",
    )
    parser.add_argument(
        "--no-zero-center",
        action="store_true",
        help="Disable zero-centering of template positions",
    )
    args = parser.parse_args()

    os.makedirs(args.dump_dir, exist_ok=True)

    print(f"=== Template Feature Extraction: {args.name} ===")
    print(f"  CIF: {args.template_cif}")
    print(f"  Chain: {args.template_chain}")

    # Extract raw template features (37-atom)
    raw_features = extract_template_from_cif(
        query_sequence=args.query_sequence,
        cif_path=args.template_cif,
        chain_id=args.template_chain,
        kalign_binary=args.kalign_binary,
        zero_center=not args.no_zero_center,
    )

    num_query = len(args.query_sequence)

    # Convert to dense 24-atom and derive additional features
    features = fix_and_derive_template_features(raw_features, num_query)

    print(f"\n  Output tensors:")
    for k, v in sorted(features.items()):
        if isinstance(v, np.ndarray):
            print(f"    {k}: shape={v.shape} dtype={v.dtype}")

    # Save as numpy archive
    npz_path = os.path.join(args.dump_dir, f"{args.name}_template_features.npz")
    np.savez(npz_path, **{k: v for k, v in features.items() if isinstance(v, np.ndarray)})
    print(f"\n  Saved: {npz_path}")

    # Also save raw 37-atom features for debugging
    raw_npz_path = os.path.join(
        args.dump_dir, f"{args.name}_template_raw37.npz"
    )
    np.savez(
        raw_npz_path,
        template_all_atom_positions_37=raw_features["template_all_atom_positions"],
        template_all_atom_masks_37=raw_features["template_all_atom_masks"],
        template_aatype_raw=raw_features["template_aatype"],
    )
    print(f"  Saved raw 37-atom: {raw_npz_path}")

    # Save metadata as JSON
    meta = {
        "name": args.name,
        "query_sequence": args.query_sequence,
        "query_length": num_query,
        "template_cif": args.template_cif,
        "template_chain": args.template_chain,
        "zero_center": not args.no_zero_center,
        "shapes": {
            k: list(v.shape) for k, v in features.items() if isinstance(v, np.ndarray)
        },
    }
    meta_path = os.path.join(args.dump_dir, f"{args.name}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved metadata: {meta_path}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
