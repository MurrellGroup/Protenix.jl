#!/usr/bin/env julia

using PXDesign

function write_tiny_cif(path::AbstractString)
    write(
        path,
        """
data_demo
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_formal_charge
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
ATOM 1 N N . ALA A 1 1 ? 0.000 0.000 0.000 1.00 10.00 ? 1 ALA A N 1
ATOM 2 C CA . ALA A 1 1 ? 1.200 0.000 0.000 1.00 10.00 ? 1 ALA A CA 1
ATOM 3 C C . ALA A 1 1 ? 2.200 0.100 0.000 1.00 10.00 ? 1 ALA A C 1
ATOM 4 O O . ALA A 1 1 ? 3.000 0.200 0.000 1.00 10.00 ? 1 ALA A O 1
#
""",
    )
end

function write_tiny_input(path::AbstractString, target_cif::AbstractString)
    write(
        path,
        """
[
  {
    "name": "e2e_cpu_smoke",
    "condition": {
      "structure_file": "$target_cif",
      "filter": {"chain_id": ["A"], "crop": {}},
      "msa": {}
    },
    "hotspot": {"A": [1]},
    "generation": [{"type": "protein", "length": 2, "count": 1}]
  }
]
""",
    )
end

function main()
    project_root = normpath(joinpath(@__DIR__, ".."))
    raw_dir = joinpath(project_root, "weights_raw")
    isdir(raw_dir) || error("weights_raw directory is required for strict e2e smoke: $raw_dir")
    isfile(joinpath(raw_dir, "manifest.json")) || error("weights_raw/manifest.json missing: $raw_dir")

    mktempdir() do d
        target_cif = joinpath(d, "target.cif")
        input_json = joinpath(d, "input.json")
        write_tiny_cif(target_cif)
        write_tiny_input(input_json, target_cif)

        cfg = PXDesign.Config.default_config(project_root = project_root)
        cfg["input_json_path"] = input_json
        cfg["dump_dir"] = joinpath(d, "out")
        cfg["download_cache"] = false
        cfg["seeds"] = [7]
        cfg["sample_diffusion"]["N_sample"] = 1
        cfg["sample_diffusion"]["N_step"] = 2
        cfg["sample_diffusion"]["eta_schedule"] = Dict("type" => "const", "min" => 1.0, "max" => 1.0)
        cfg["infer_setting"]["sample_diffusion_chunk_size"] = 1
        cfg["raw_weights_dir"] = raw_dir
        cfg["strict_weight_load"] = true
        cfg["model_scaffold"]["enabled"] = true
        cfg["model_scaffold"]["auto_dims_from_weights"] = true
        cfg["model_scaffold"]["use_design_condition_embedder"] = true

        result = PXDesign.Infer.run_infer(cfg; dry_run = false, io = stdout)
        result["status"] == "ok_scaffold_model" || error("Unexpected run status: $(result["status"])")

        pred_dir = joinpath(cfg["dump_dir"], "global_run_0", "e2e_cpu_smoke", "seed_7", "predictions")
        isdir(pred_dir) || error("Missing predictions directory: $pred_dir")
        cif_files = filter(f -> endswith(f, ".cif"), readdir(pred_dir))
        length(cif_files) == 1 || error("Expected exactly one CIF prediction, got $(length(cif_files)) in $pred_dir")

        println("e2e_cpu_smoke_ok")
        println(pred_dir)
    end
end

main()
