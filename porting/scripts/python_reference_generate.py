#!/usr/bin/env python3
"""Generate Python Protenix reference outputs for parity testing.

Usage:
    cd PXDesign.jl
    source .venv_pyref/bin/activate
    PYTHONPATH=.external/Protenix python3 scripts/python_reference_generate.py
"""
import json
import logging
import os
import sys
import tempfile

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '.external', 'Protenix'))

SEQUENCE = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"
ROOT = os.path.join(os.path.dirname(__file__), '..')
OUT_BASE = os.path.join(ROOT, 'e2e_output', 'python_reference')
CHECKPOINT_DIR = os.path.join(ROOT, 'release_data', 'checkpoint')
CCD_DIR = os.path.join(ROOT, 'release_data', 'ccd_cache')

os.environ.setdefault('PROTENIX_DATA_ROOT_DIR', CCD_DIR)

# Models to test: (name, n_step, n_sample, n_cycle)
MODELS = [
    ("protenix_mini_default_v0.5.0", 200, 1, 4),
    ("protenix_tiny_default_v0.5.0", 200, 1, 4),
    ("protenix_base_default_v0.5.0", 200, 1, 10),
]


def create_input_json(sequence, name, tmpdir):
    data = [{
        "sequences": [{
            "proteinChain": {
                "sequence": sequence,
                "count": 1
            }
        }],
        "name": name,
    }]
    path = os.path.join(tmpdir, f"{name}.json")
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    return path


def run_one(model_name, n_step, n_sample, n_cycle, seed=101):
    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from configs.configs_model_type import model_configs
    from protenix.config import parse_configs
    from runner.inference import InferenceRunner, download_infercence_cache
    import ml_collections
    import torch

    name = f"pyref_{model_name}_step{n_step}"
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"  model={model_name} n_step={n_step} n_sample={n_sample} n_cycle={n_cycle} seed={seed}")
    print(f"{'='*60}")

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = create_input_json(SEQUENCE, name, tmpdir)

        # Build full config from all defaults
        configs = ml_collections.ConfigDict(configs_base)
        configs.update(ml_collections.ConfigDict(data_configs))
        configs.update(ml_collections.ConfigDict(inference_configs))

        # Apply model-specific overrides
        if model_name in model_configs:
            for k, v in model_configs[model_name].items():
                if isinstance(v, dict):
                    if k not in configs:
                        configs[k] = ml_collections.ConfigDict(v)
                    else:
                        configs[k].update(v)
                else:
                    configs[k] = v

        # Set our overrides
        configs.model_name = model_name
        configs.seeds = [seed]
        configs.dump_dir = os.path.join(OUT_BASE, name)
        configs.input_json_path = json_path
        configs.use_msa = False  # no precomputed MSA
        configs.load_checkpoint_dir = CHECKPOINT_DIR
        configs.sample_diffusion.N_step = n_step
        configs.sample_diffusion.N_sample = n_sample
        configs.model.N_cycle = n_cycle
        configs.use_deepspeed_evo_attention = False
        configs.load_strict = True
        configs.need_atom_confidence = False
        configs.sorted_by_ranking_score = True
        configs.dtype = "fp32"

        # Download CCD cache + checkpoint
        download_infercence_cache(configs)

        runner = InferenceRunner(configs)

        # Get the data loader
        from protenix.data.infer_data_pipeline import get_inference_dataloader
        dataloader = get_inference_dataloader(
            configs,
            json_path=json_path,
        )

        for batch in dataloader:
            for seed_val in configs.seeds:
                from protenix.utils.seed import seed_everything
                seed_everything(seed_val)

                prediction = runner.predict(batch)
                runner.dumper.dump(
                    batch,
                    prediction,
                    cur_seed=seed_val,
                )

        print(f"  Output: {configs.dump_dir}")
        return configs.dump_dir


if __name__ == '__main__':
    os.makedirs(OUT_BASE, exist_ok=True)

    for model_name, n_step, n_sample, n_cycle in MODELS:
        try:
            out_dir = run_one(model_name, n_step, n_sample, n_cycle)
        except Exception as e:
            print(f"FAILED {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print(f"Reference outputs in: {OUT_BASE}")
    print(f"{'='*60}")
