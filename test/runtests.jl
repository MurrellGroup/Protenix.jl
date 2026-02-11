using Test
using Statistics
using Random
using PXDesign

function _as_string_dict_test(x)
    if x isa AbstractDict
        out = Dict{String, Any}()
        for (k, v) in x
            out[String(k)] = _as_string_dict_test(v)
        end
        return out
    elseif x isa AbstractVector
        return Any[_as_string_dict_test(v) for v in x]
    end
    return x
end

include("layer_regression.jl")

@testset "Protenix API surface" begin
    pmini = PXDesign.recommended_params("protenix_mini_default_v0.5.0")
    @test pmini.cycle == 4
    @test pmini.step == 5
    @test pmini.sample == 5
    @test pmini.use_msa == true

    pover = PXDesign.recommended_params(
        "protenix_base_default_v0.5.0";
        use_default_params = false,
        cycle = 3,
        step = 11,
        sample = 2,
        use_msa = false,
    )
    @test pover.cycle == 3
    @test pover.step == 11
    @test pover.sample == 2
    @test pover.use_msa == false

    @test_throws ErrorException PXDesign.resolve_model_spec("not_a_model")

    mktempdir() do d
        pdb_path = joinpath(d, "tiny.pdb")
        write(
            pdb_path,
            join(
                [
                    "ATOM      1  N   ALA A   1      11.104  13.207   9.474  1.00 20.00           N",
                    "ATOM      2  CA  ALA A   1      12.447  13.702   9.934  1.00 20.00           C",
                    "ATOM      3  C   ALA A   1      13.472  12.557   9.750  1.00 20.00           C",
                    "ATOM      4  O   ALA A   1      13.180  11.383   9.969  1.00 20.00           O",
                    "ATOM      5  N   GLY A   2      14.690  12.880   9.336  1.00 20.00           N",
                    "ATOM      6  CA  GLY A   2      15.786  11.944   9.169  1.00 20.00           C",
                    "ATOM      7  C   GLY A   2      16.026  11.393   7.773  1.00 20.00           C",
                    "ATOM      8  O   GLY A   2      16.001  12.141   6.794  1.00 20.00           O",
                    "TER",
                    "END",
                    "",
                ],
                "\n",
            ),
        )

        json_out_dir = joinpath(d, "tojson_out")
        out_paths = PXDesign.convert_structure_to_infer_json(pdb_path; out_dir = json_out_dir)
        @test length(out_paths) == 1
        @test isfile(out_paths[1])

        parsed = PXDesign.JSONLite.parse_json(read(out_paths[1], String))
        @test parsed isa AbstractVector
        @test length(parsed) == 1
        @test parsed[1]["name"] == "tiny"
        seqs = parsed[1]["sequences"]
        @test length(seqs) == 1
        @test seqs[1]["proteinChain"]["sequence"] == "AG"
        @test seqs[1]["proteinChain"]["count"] == 1

        in_json = joinpath(d, "input.json")
        PXDesign.JSONLite.write_json(
            in_json,
            Any[
                Dict(
                    "name" => "msa_demo",
                    "sequences" => Any[
                        Dict("proteinChain" => Dict("sequence" => "ACD", "count" => 1)),
                        Dict("ligand" => Dict("ligand" => "CCD_ATP", "count" => 1)),
                    ],
                ),
            ],
        )
        msa_out = PXDesign.add_precomputed_msa_to_json(
            in_json;
            out_dir = joinpath(d, "msa_out"),
            precomputed_msa_dir = "/tmp/precomp_msa",
        )
        @test isfile(msa_out)
        parsed_msa = PXDesign.JSONLite.parse_json(read(msa_out, String))
        @test parsed_msa[1]["sequences"][1]["proteinChain"]["msa"]["precomputed_msa_dir"] == "/tmp/precomp_msa"
        @test !haskey(parsed_msa[1]["sequences"][2], "proteinChain")

        @test PXDesign.main(["tojson", "--input", pdb_path, "--out_dir", joinpath(d, "cli_tojson")]) == 0
        @test PXDesign.main([
            "msa",
            "--input",
            in_json,
            "--precomputed_msa_dir",
            "/tmp/precomp_msa",
            "--out_dir",
            joinpath(d, "cli_msa"),
        ]) == 0
        @test PXDesign.main(["predict", "--help"]) == 0
    end
end

@testset "Protenix precomputed MSA ingestion" begin
    mktempdir() do d
        msa_dir = joinpath(d, "msa", "0")
        mkpath(msa_dir)
        write(
            joinpath(msa_dir, "non_pairing.a3m"),
            """
>query
ACD
>hit1
AFdD
>hit2
A-D
""",
        )
        write(joinpath(msa_dir, "pairing.a3m"), ">query\nACD\n")

        input_json = joinpath(d, "input.json")
        task = Dict{String, Any}(
            "name" => "msa_ingest_smoke",
            "sequences" => Any[
                Dict(
                    "proteinChain" => Dict(
                        "sequence" => "ACD",
                        "count" => 1,
                        "msa" => Dict(
                            "precomputed_msa_dir" => "msa/0",
                            "pairing_db" => "uniref100",
                        ),
                    ),
                ),
            ],
        )
        PXDesign.JSONLite.write_json(input_json, Any[task])

        atoms = PXDesign.ProtenixAPI._build_atoms_from_infer_task(task)
        bundle = PXDesign.Data.build_feature_bundle_from_atoms(atoms; task_name = "msa_ingest_smoke")
        feat = bundle["input_feature_dict"]
        PXDesign.ProtenixAPI._normalize_protenix_feature_dict!(feat)
        PXDesign.ProtenixAPI._inject_task_msa_features!(feat, task, input_json; use_msa = true)

        @test size(feat["msa"], 2) == 3
        @test size(feat["msa"], 1) >= 2
        @test maximum(feat["has_deletion"]) > 0f0
        @test maximum(feat["deletion_value"]) > 0f0
        @test size(feat["profile"]) == (3, 32)
    end
end

@testset "Protenix template feature ingestion" begin
    mktempdir() do d
        input_json = joinpath(d, "input_template.json")
        task = Dict{String, Any}(
            "name" => "template_ingest_smoke",
            "sequences" => Any[
                Dict(
                    "proteinChain" => Dict(
                        "sequence" => "ACD",
                        "count" => 1,
                    ),
                ),
            ],
            "template_features" => Dict(
                "template_restype" => Any[Any[0, 1, 2]],
                "template_all_atom_mask" => Any[
                    Any[
                        Any[ones(Int, 37)...],
                        Any[ones(Int, 37)...],
                        Any[ones(Int, 37)...],
                    ],
                ],
                "template_all_atom_positions" => Any[
                    Any[
                        [Any[Any[0.0, 0.0, 0.0] for _ in 1:37]...],
                        [Any[Any[0.0, 0.0, 0.0] for _ in 1:37]...],
                        [Any[Any[0.0, 0.0, 0.0] for _ in 1:37]...],
                    ],
                ],
            ),
        )
        PXDesign.JSONLite.write_json(input_json, Any[task])
        parsed_task = PXDesign.JSONLite.parse_json(read(input_json, String))[1]

        atoms = PXDesign.ProtenixAPI._build_atoms_from_infer_task(parsed_task)
        bundle = PXDesign.Data.build_feature_bundle_from_atoms(atoms; task_name = "template_ingest_smoke")
        feat = bundle["input_feature_dict"]
        PXDesign.ProtenixAPI._normalize_protenix_feature_dict!(feat)
        PXDesign.ProtenixAPI._inject_task_template_features!(feat, parsed_task)

        @test size(feat["template_restype"]) == (1, 3)
        @test size(feat["template_all_atom_mask"]) == (1, 3, 37)
        @test size(feat["template_all_atom_positions"]) == (1, 3, 37, 3)
    end
end

@testset "Protenix ESM token embedding injection" begin
    task = Dict{String, Any}(
        "name" => "esm_inject",
        "sequences" => Any[
            Dict(
                "proteinChain" => Dict(
                    "sequence" => "ACD",
                    "count" => 1,
                ),
            ),
        ],
        "esm_token_embedding" => Any[
            Any[0.1, 0.2, 0.3, 0.4],
            Any[0.5, 0.6, 0.7, 0.8],
            Any[0.9, 1.0, 1.1, 1.2],
        ],
    )
    atoms = PXDesign.ProtenixAPI._build_atoms_from_infer_task(task)
    bundle = PXDesign.Data.build_feature_bundle_from_atoms(atoms; task_name = "esm_inject")
    feat = bundle["input_feature_dict"]
    PXDesign.ProtenixAPI._normalize_protenix_feature_dict!(feat)
    PXDesign.ProtenixAPI._inject_task_esm_token_embedding!(feat, task)
    @test haskey(feat, "esm_token_embedding")
    @test size(feat["esm_token_embedding"]) == (3, 4)
    params = (model_name = "protenix_mini_esm_v0.5.0", needs_esm_embedding = true)
    @test PXDesign.ProtenixAPI._validate_required_model_inputs!(params, feat, "esm task") === feat

    feat_missing = Dict{String, Any}("restype" => zeros(Float32, 3, 32))
    @test_throws ErrorException PXDesign.ProtenixAPI._validate_required_model_inputs!(
        params,
        feat_missing,
        "missing esm",
    )
end

@testset "ProtenixBase sequence wrappers" begin
    atoms = PXDesign.ProtenixBase.build_sequence_atoms("ACD")
    @test !isempty(atoms)
    @test atoms[1].chain_id == "A0"

    bundle = PXDesign.ProtenixBase.build_sequence_feature_bundle("ACD")
    @test bundle["task_name"] == "protenix_base_sequence"
    @test haskey(bundle, "input_feature_dict")
    @test haskey(bundle, "dims")
    @test bundle["dims"]["N_token"] == 3
end

@testset "Cache zero-byte checkpoint refresh" begin
    mktempdir() do d
        src_dir = joinpath(d, "src")
        mkpath(src_dir)
        src_components = joinpath(src_dir, "components.v20240608.cif")
        src_rdkit = joinpath(src_dir, "components.v20240608.cif.rdkit_mol.pkl")
        src_cluster = joinpath(src_dir, "clusters-by-entity-40.txt")
        src_ckpt = joinpath(src_dir, "mock_model.pt")
        write(src_components, "components")
        write(src_rdkit, "rdkit")
        write(src_cluster, "cluster")
        write(src_ckpt, "checkpoint_payload")

        data_dir = joinpath(d, "data")
        ckpt_dir = joinpath(d, "checkpoint")
        mkpath(data_dir)
        mkpath(ckpt_dir)
        target_components = joinpath(data_dir, "components.v20240608.cif")
        target_rdkit = joinpath(data_dir, "components.v20240608.cif.rdkit_mol.pkl")
        target_cluster = joinpath(data_dir, "clusters-by-entity-40.txt")
        target_ckpt = joinpath(ckpt_dir, "mock_model.pt")

        write(target_components, "")
        write(target_rdkit, "")
        write(target_cluster, "")
        write(target_ckpt, "")

        cfg = Dict{String, Any}(
            "data" => Dict(
                "ccd_components_file" => target_components,
                "ccd_components_rdkit_mol_file" => target_rdkit,
                "pdb_cluster_file" => target_cluster,
            ),
            "load_checkpoint_dir" => ckpt_dir,
            "model_name" => "mock_model",
        )

        urls = Dict(
            "ccd_components_file" => "file://$(abspath(src_components))",
            "ccd_components_rdkit_mol_file" => "file://$(abspath(src_rdkit))",
            "pdb_cluster_file" => "file://$(abspath(src_cluster))",
            "mock_model" => "file://$(abspath(src_ckpt))",
        )

        PXDesign.Cache.ensure_inference_cache!(
            cfg;
            urls = urls,
            include_protenix_checkpoints = false,
            dry_run = false,
            io = devnull,
        )

        @test read(target_components, String) == "components"
        @test read(target_rdkit, String) == "rdkit"
        @test read(target_cluster, String) == "cluster"
        @test read(target_ckpt, String) == "checkpoint_payload"
    end
end

@testset "JSONLite" begin
    payload = """
    {
      "name": "demo",
      "n": 3,
      "ok": true,
      "arr": [1, 2.5, "x"],
      "obj": {"k": "v"}
    }
    """
    parsed = PXDesign.JSONLite.parse_json(payload)
    @test parsed["name"] == "demo"
    @test parsed["n"] == 3
    @test parsed["ok"] == true
    @test parsed["arr"][2] == 2.5
    @test parsed["obj"]["k"] == "v"
end

@testset "Range utils" begin
    @test PXDesign.parse_ranges("1-3,6,8-9") == [(1, 3), (6, 6), (8, 9)]
    @test PXDesign.parse_ranges(" 1-1 , 5 ") == [(1, 1), (5, 5)]
    @test PXDesign.format_ranges([1, 2, 3, 6, 8, 9]) == "1-3,6,8-9"
    @test PXDesign.format_ranges([7]) == "7"
    @test PXDesign.format_ranges(Int[]) == ""
end

@testset "Inputs JSON" begin
    mktempdir() do d
        json_path = joinpath(d, "task.json")
        target_cif = joinpath(d, "target.cif")
        write(target_cif, "data_demo\n#\n")
        write(
            json_path,
            """
[
  {
    "name": "t1",
    "condition": {
      "structure_file": "$target_cif",
      "filter": {"chain_id": ["A"], "crop": {}},
      "msa": {}
    },
    "hotspot": {},
    "generation": [{"type":"protein","length":80,"count":1}]
  },
  {
    "name": "t2",
    "condition": {
      "structure_file": "$target_cif",
      "filter": {"chain_id": ["A"], "crop": {}},
      "msa": {}
    },
    "hotspot": {},
    "generation": [{"type":"protein","length":40,"count":1}]
  }
]
""",
        )
        tasks = PXDesign.Inputs.load_input_tasks(json_path)
        @test length(tasks) == 2
        @test tasks[1]["name"] == "t1"

        typed = PXDesign.Schema.parse_tasks(tasks)
        @test length(typed) == 2
        @test typed[1].name == "t1"
    end
end

@testset "Data tokenizer/features" begin
    atoms = PXDesign.Data.build_design_backbone_atoms(3)
    ta = PXDesign.Data.tokenize_atoms(atoms)
    @test length(ta) == 3
    @test ta[1].value == PXDesign.Data.STD_RESIDUES["xpb"]
    @test ta[1].centre_atom_index == 2
    @test ta[2].centre_atom_index == 6
    @test count(a -> a.atom_name == "OXT", atoms) == 1
    @test atoms[end].atom_name == "OXT"

    mktempdir() do d
        target_cif = joinpath(d, "target.cif")
        write(
            target_cif,
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
ATOM 5 N N . TYR A 1 2 ? 2.300 1.300 0.100 1.00 11.00 ? 2 TYR A N 1
ATOM 6 C CA . TYR A 1 2 ? 3.200 1.700 0.100 1.00 11.00 ? 2 TYR A CA 1
ATOM 7 C C . TYR A 1 2 ? 4.000 2.100 0.200 1.00 11.00 ? 2 TYR A C 1
ATOM 8 O O . TYR A 1 2 ? 4.900 2.300 0.200 1.00 11.00 ? 2 TYR A O 1
#
""",
        )

        task = PXDesign.Schema.InputTask(
            "demo",
            target_cif,
            ["A"],
            Dict{String, String}("A" => "1-2"),
            Dict{String, Vector{Int}}("A" => [2]),
            Dict{String, Dict{String, Any}}(),
            [PXDesign.Schema.GenerationSpec("protein", 3, 1)],
        )
        bundle = PXDesign.Data.build_basic_feature_bundle(task)
        dims = bundle["dims"]
        feat = bundle["input_feature_dict"]
        @test dims["N_token"] == 5
        @test dims["N_atom"] == 30
        @test size(feat["restype"]) == (5, 36)
        @test size(feat["msa"]) == (1, 5)
        @test count(feat["condition_token_mask"]) == 2
        @test feat["hotspot"][2] == 1f0
        @test feat["conditional_templ_mask"][1, 1] == 1
        @test feat["conditional_templ_mask"][3, 3] == 0
        @test length(feat["atom_to_tokatom_idx"]) == dims["N_atom"]
        @test minimum(feat["atom_to_tokatom_idx"]) == 0
    end
end

@testset "Data design encoders" begin
    atoms = PXDesign.Data.build_design_backbone_atoms(2)
    cano = PXDesign.Data.cano_seq_resname_with_mask(atoms)
    @test cano == ["xpb", "xpb"]
    onehot = PXDesign.Data.restype_onehot_encoded(cano)
    @test size(onehot) == (2, 36)
    @test onehot[1, PXDesign.Data.STD_RESIDUES_WITH_GAP["xpb"] + 1] == 1f0
end

@testset "ProtenixMini sequence features" begin
    atoms = PXDesign.build_sequence_atoms("ACD"; chain_id = "A0")
    @test !isempty(atoms)
    @test count(a -> a.atom_name == "CA", atoms) == 3
    @test count(a -> a.atom_name == "OXT", atoms) == 1

    bundle = PXDesign.build_sequence_feature_bundle("ACD"; chain_id = "A0")
    feat = bundle["input_feature_dict"]
    dims = bundle["dims"]
    @test dims["N_token"] == 3
    @test size(feat["restype"]) == (3, 32)
    @test size(feat["profile"]) == (3, 32)
    @test size(feat["msa"]) == (1, 3)
    @test size(feat["has_deletion"]) == (1, 3)
    @test size(feat["deletion_value"]) == (1, 3)
    @test length(feat["deletion_mean"]) == 3
    @test length(feat["atom_to_tokatom_idx"]) == dims["N_atom"]
    @test length(feat["distogram_rep_atom_mask"]) == dims["N_atom"]
end

@testset "Inputs YAML native parser" begin
    mktempdir() do d
        target_cif = joinpath(d, "target.cif")
        write(target_cif, "data_demo\n#\n")

        msa_dir = joinpath(d, "msa")
        mkpath(msa_dir)
        write(joinpath(msa_dir, "pairing.a3m"), ">a\nAA\n")
        write(joinpath(msa_dir, "non_pairing.a3m"), ">a\nAA\n")

        yaml_path = joinpath(d, "input.yaml")
        write(
            yaml_path,
            """
task_name: demo_task
binder_length: 80
target:
  file: $target_cif
  chains:
    A:
      crop:
        - "1-10"
        - "20-30"
      hotspots: [2, 5]
      msa: $msa_dir
""",
        )

        cfg = PXDesign.Inputs.load_yaml_config(yaml_path)
        @test cfg["task_name"] == "demo_task"
        @test cfg["binder_length"] == 80
        @test cfg["target"]["chains"]["A"]["crop"] == Any["1-10", "20-30"]
        @test cfg["target"]["chains"]["A"]["hotspots"] == Any[2, 5]

        yaml_anchor_path = joinpath(d, "input_anchor.yaml")
        write(
            yaml_anchor_path,
            """
defaults: &chain_defaults
  crop: ["1-10", "20-30"]
  msa: $msa_dir
task_name: anchor_task
binder_length: 48
target:
  file: $target_cif
  chains:
    A:
      <<: *chain_defaults
      hotspots: [3, 9]
""",
        )
        cfg_anchor = PXDesign.Inputs.load_yaml_config(yaml_anchor_path)
        @test cfg_anchor["target"]["chains"]["A"]["crop"] == Any["1-10", "20-30"]
        @test cfg_anchor["target"]["chains"]["A"]["msa"] == msa_dir
        @test cfg_anchor["target"]["chains"]["A"]["hotspots"] == Any[3, 9]

        tasks = PXDesign.Inputs.parse_yaml_to_json(yaml_path)
        @test length(tasks) == 1
        t = tasks[1]
        @test t["name"] == "demo_task"
        @test t["generation"][1]["length"] == 80
        @test t["condition"]["filter"]["chain_id"][1] == "A"
        @test t["condition"]["filter"]["crop"]["A"] == "1-10,20-30"
        @test t["hotspot"]["A"][1] == 2

        out_dir = joinpath(d, "out")
        mkpath(out_dir)
        converted = PXDesign.Inputs.process_input_file(yaml_path; out_dir = out_dir)
        @test endswith(converted, ".json")
        @test isfile(converted)

        yml_path = joinpath(d, "input_nt.yml")
        write(
            yml_path,
            """
task_name: no_target_demo
binder_length: 12
""",
        )
        tasks_nt = PXDesign.Inputs.parse_yaml_to_json(yml_path)
        @test length(tasks_nt) == 1
        @test tasks_nt[1]["name"] == "no_target_demo"
        @test tasks_nt[1]["generation"][1]["length"] == 12
        @test !haskey(tasks_nt[1], "condition")

        msa_dir_nonpair = joinpath(d, "msa_nonpair")
        mkpath(msa_dir_nonpair)
        write(joinpath(msa_dir_nonpair, "non_pairing.a3m"), ">a\nAA\n")
        yaml_nonpair = joinpath(d, "input_nonpair.yaml")
        write(
            yaml_nonpair,
            """
task_name: nonpair_only
binder_length: 16
target:
  file: $target_cif
  chains:
    A:
      msa: $msa_dir_nonpair
""",
        )
        tasks_nonpair = PXDesign.Inputs.parse_yaml_to_json(yaml_nonpair)
        @test length(tasks_nonpair) == 1
        @test tasks_nonpair[1]["condition"]["msa"]["A"]["precomputed_msa_dir"] == msa_dir_nonpair

        yaml_bad_chain = joinpath(d, "input_bad_chain.yaml")
        write(
            yaml_bad_chain,
            """
task_name: bad_chain_cfg
binder_length: 16
target:
  file: $target_cif
  chains:
    A: nonsense
""",
        )
        @test_throws ErrorException PXDesign.Inputs.parse_yaml_to_json(yaml_bad_chain)
    end
end

@testset "Inputs YAML vs PyYAML parity (supported subset)" begin
    has_pyyaml = success(
        pipeline(`python3 -c "import yaml; print('ok')"`, stdout = devnull, stderr = devnull),
    )
    if !has_pyyaml
        @test_skip "Skipping YAML/PyYAML parity check: python3 + PyYAML unavailable in environment"
    else
        mktempdir() do d
            yaml_path = joinpath(d, "subset.yaml")
            write(
                yaml_path,
                """
task_name: parity_demo
binder_length: 64
target:
  file: /tmp/target.cif
  chains:
    A:
      crop: ["1-10", "20-30"]
      hotspots: [2, 5]
      msa: /tmp/msa
    B: all
flags:
  enabled: true
  weight: 1.5
  label: "demo"
""",
            )

            julia_cfg = PXDesign.Inputs.load_yaml_config(yaml_path)
            py_json = read(
                `python3 -c "import json, yaml, sys; print(json.dumps(yaml.safe_load(open(sys.argv[1], 'r')), sort_keys=True))" $yaml_path`,
                String,
            )
            py_cfg = _as_string_dict_test(PXDesign.JSONLite.parse_json(py_json))
            @test julia_cfg == py_cfg
        end
    end
end

@testset "Scheduler" begin
    sched = PXDesign.InferenceNoiseScheduler()
    ts = sched(10; dtype = Float32)
    @test length(ts) == 11
    @test ts[end] == 0f0
    @test ts[1] > ts[2]
end

@testset "Sampler" begin
    noise_schedule = Float32[8.0, 4.0, 2.0, 0.0]
    denoise_net(x_noisy, t_hat; kwargs...) = x_noisy .* 0.95f0
    x = PXDesign.sample_diffusion(
        denoise_net;
        noise_schedule = noise_schedule,
        N_sample = 2,
        N_atom = 5,
    )
    @test size(x) == (2, 5, 3)
    @test all(isfinite, x)

    x_chunk = PXDesign.sample_diffusion(
        denoise_net;
        noise_schedule = noise_schedule,
        N_sample = 5,
        N_atom = 4,
        diffusion_chunk_size = 2,
        rng = MersenneTwister(3),
    )
    @test size(x_chunk) == (5, 4, 3)
    @test all(isfinite, x_chunk)
end

@testset "Checkpoint map utilities" begin
    mktempdir() do d
        p = joinpath(d, "checkpoint_index.json")
        write(
            p,
            """
{
  "summary": {"num_model_tensors": 3},
  "tensors": [
    {"key": "module.design_condition_embedder.input_embedder.input_map.weight", "shape": [449, 449], "dtype": "torch.float32"},
    {"key": "module.diffusion_module.diffusion_transformer.blocks.0.attention_pair_bias.attention.linear_q.weight", "shape": [768, 768], "dtype": "torch.float32"},
    {"key": "module.unknown_block.foo", "shape": [1], "dtype": "torch.float32"}
  ]
}
""",
        )

        keys = PXDesign.Model.load_checkpoint_index(p)
        @test length(keys) == 3
        @test startswith(keys[1], "design_condition_embedder.")
        counts = PXDesign.Model.checkpoint_prefix_counts(keys)
        @test counts["design_condition_embedder.input_embedder"] == 1
        @test counts["diffusion_module.diffusion_transformer"] == 1
        @test counts["_unmatched"] == 1
    end

    real_idx = joinpath(dirname(@__DIR__), "docs", "checkpoint_index.json")
    if isfile(real_idx)
        keys = PXDesign.Model.load_checkpoint_index(real_idx)
        counts = PXDesign.Model.checkpoint_prefix_counts(keys)
        @test get(counts, "_unmatched", 0) == 0
        @test sum(values(counts)) == length(keys)
    else
        @test_skip "Skipping real checkpoint-index coverage assertion (docs/checkpoint_index.json not present)."
    end
end

@testset "Model embedders" begin
    rel = PXDesign.Model.relative_position_features(
        [1, 1, 2],
        [10, 11, 5],
        [1, 1, 2],
        [1, 1, 1],
        [0, 1, 2];
        r_max = 2,
        s_max = 1,
    )
    @test size(rel) == (3, 3, 17)
    # three one-hot blocks plus same_entity bit (0/1) => sum is 3 or 4
    @test all(sum(rel[i, j, :]) in (3f0, 4f0) for i in 1:3 for j in 1:3)
    # cross-entity pair should have same_entity bit = 0
    same_entity_idx = (2 * (2 + 1)) + (2 * (2 + 1)) + 1
    @test rel[1, 3, same_entity_idx] == 0f0
    @test rel[1, 2, same_entity_idx] == 1f0

    relpe = PXDesign.Model.RelativePositionEncoding(2, 1, 5)
    raw_relpos = Dict(
        "asym_id" => [1, 1, 2],
        "residue_index" => [10, 11, 5],
        "entity_id" => [1, 1, 2],
        "sym_id" => [1, 1, 1],
        "token_index" => [0, 1, 2],
    )
    z = relpe(raw_relpos)
    @test size(z) == (3, 3, 5)
    @test all(isfinite, z)
    relpos_input = PXDesign.Model.as_relpos_input(raw_relpos)
    z_nt = relpe(relpos_input)
    @test z_nt == z

    cte = PXDesign.Model.ConditionTemplateEmbedder(65, 7)
    templ = Int[
        0 12 0
        12 0 1
        0 1 0
    ]
    mask = Int[
        0 1 0
        1 0 1
        0 1 0
    ]
    ztempl = PXDesign.Model.condition_template_embedding(cte, templ, mask)
    @test size(ztempl) == (3, 3, 7)
    # masked-out entries use embedding index 1 (idx0=0 -> +1)
    @test ztempl[1, 1, :] == cte.weight[1, :]
    # masked-in entry with templ=12 uses index 14 (1 + templ then +1 for Julia indexing)
    @test ztempl[1, 2, :] == cte.weight[14, :]
    templ_input = PXDesign.Model.as_template_input(
        Dict("conditional_templ" => templ, "conditional_templ_mask" => mask),
    )
    @test PXDesign.Model.condition_template_embedding(cte, templ_input) == ztempl

    f = PXDesign.Model.fourier_embedding([0.0, 0.25], [1.0, 2.0], [0.0, 0.5])
    @test size(f) == (2, 2)
    @test all(isfinite, f)

    dce = PXDesign.Model.DesignConditionEmbedder(
        8;
        c_s_inputs = 16,
        c_z = 4,
        n_blocks = 1,
    )
    feat_dce = Dict{String, Any}(
        "ref_pos" => Float32[
            0 0 0
            1 0 0
            0 1 0
            1 1 0
        ],
        "ref_charge" => zeros(Float32, 4),
        "ref_mask" => ones(Float32, 4),
        "ref_element" => vcat(
            reshape(vcat(1f0, zeros(Float32, 127)), 1, :),
            reshape(vcat(1f0, zeros(Float32, 127)), 1, :),
            reshape(vcat(1f0, zeros(Float32, 127)), 1, :),
            reshape(vcat(1f0, zeros(Float32, 127)), 1, :),
        ),
        "ref_atom_name_chars" => vcat(
            reshape(vcat(1f0, zeros(Float32, 255)), 1, :),
            reshape(vcat(1f0, zeros(Float32, 255)), 1, :),
            reshape(vcat(1f0, zeros(Float32, 255)), 1, :),
            reshape(vcat(1f0, zeros(Float32, 255)), 1, :),
        ),
        "ref_space_uid" => Int[0, 0, 1, 1],
        "atom_to_token_idx" => Int[0, 0, 1, 1],
        "restype" => rand(Float32, 2, 36),
        "plddt" => Float32[0, 0],
        "hotspot" => Float32[0, 1],
        "conditional_templ" => Int[0 3; 3 0],
        "conditional_templ_mask" => Int[0 1; 1 0],
    )
    s_dce, z_dce = dce(feat_dce)
    @test size(s_dce) == (2, 16)
    @test size(z_dce) == (2, 2, 4)
    @test all(isfinite, s_dce)
    @test all(isfinite, z_dce)
end

@testset "Model primitives" begin
    lin = PXDesign.Model.LinearNoBias(4, 3)
    x = rand(Float32, 2, 5, 3)
    y = lin(x)
    @test size(y) == (2, 5, 4)
    @test all(isfinite, y)

    ln = PXDesign.Model.LayerNormNoOffset(3)
    z = ln(rand(Float32, 7, 3))
    @test size(z) == (7, 3)
    @test all(isfinite, z)
    # weighted LN keeps per-row mean near zero when weight is all ones
    @test all(abs.(vec(mean(z; dims = 2))) .< 1f-3)

    ada = PXDesign.Model.AdaptiveLayerNorm(6, 4)
    a = rand(Float32, 2, 3, 6)
    s = rand(Float32, 2, 3, 4)
    a2 = ada(a, s)
    @test size(a2) == (2, 3, 6)
    @test all(isfinite, a2)

    tr = PXDesign.Model.Transition(6, 2)
    t = tr(rand(Float32, 2, 3, 6))
    @test size(t) == (2, 3, 6)
    @test all(isfinite, t)
end

@testset "Diffusion conditioning" begin
    rng = MersenneTwister(7)
    cond = PXDesign.Model.DiffusionConditioning(
        16.0;
        c_z = 8,
        c_s = 12,
        c_s_inputs = 10,
        c_noise_embedding = 6,
        r_max = 2,
        s_max = 1,
        rng = rng,
    )

    relpos_input = (
        asym_id = [1, 1, 2, 2],
        residue_index = [10, 11, 3, 4],
        entity_id = [1, 1, 2, 2],
        sym_id = [1, 1, 1, 1],
        token_index = [0, 1, 2, 3],
    )
    z_trunk = rand(Float32, 4, 4, 8)
    pair = PXDesign.Model.prepare_pair_cache(cond, relpos_input, z_trunk)
    @test size(pair) == (4, 4, 8)
    @test all(isfinite, pair)

    s_inputs = rand(Float32, 4, 10)
    s_trunk = rand(Float32, 4, 12)
    t_hat = [8.0, 4.0, 2.0]
    single_s, pair2 = cond(t_hat, relpos_input, s_inputs, s_trunk, z_trunk)
    @test size(single_s) == (3, 4, 12)
    @test size(pair2) == (4, 4, 8)
    @test all(isfinite, single_s)
    @test all(isfinite, pair2)
end

@testset "Transformer blocks" begin
    blk = PXDesign.Model.ConditionedTransitionBlock(8, 6; n = 2)
    a = rand(Float32, 2, 4, 8)
    s = rand(Float32, 2, 4, 6)
    out = blk(a, s)
    @test size(out) == (2, 4, 8)
    @test all(isfinite, out)

    attn = PXDesign.Model.AttentionPairBias(8, 6, 4; n_heads = 2)
    a2 = rand(Float32, 5, 8)
    s2 = rand(Float32, 5, 6)
    z2 = rand(Float32, 5, 5, 4)
    attn_out = attn(a2, s2, z2)
    @test size(attn_out) == (5, 8)
    @test all(isfinite, attn_out)

    attn_cross = PXDesign.Model.AttentionPairBias(8, 6, 4; n_heads = 2, cross_attention_mode = true)
    attn_cross_out = attn_cross(a2, s2, z2)
    @test size(attn_cross_out) == (5, 8)
    @test all(isfinite, attn_cross_out)
    attn_local_out = attn_cross(a2, s2, z2, 2, 4)
    @test size(attn_local_out) == (5, 8)
    @test all(isfinite, attn_local_out)

    dblk = PXDesign.Model.DiffusionTransformerBlock(8, 6, 4; n_heads = 2)
    a3, s3, z3 = dblk(a2, s2, z2)
    @test size(a3) == (5, 8)
    @test size(s3) == (5, 6)
    @test size(z3) == (5, 5, 4)
    @test all(isfinite, a3)

    dtr = PXDesign.Model.DiffusionTransformer(8, 6, 4; n_blocks = 3, n_heads = 2)
    a4 = dtr(a2, s2, z2)
    @test size(a4) == (5, 8)
    @test all(isfinite, a4)
end

@testset "Atom attention modules" begin
    feat = Dict{String, Any}(
        "ref_pos" => Float32[
            0 0 0
            1 0 0
            0 1 0
            1 1 0
        ],
        "ref_charge" => zeros(Float32, 4),
        "ref_mask" => ones(Float32, 4),
        "ref_element" => vcat(
            reshape(vcat(1f0, zeros(Float32, 127)), 1, :),
            reshape(vcat(1f0, zeros(Float32, 127)), 1, :),
            reshape(vcat(1f0, zeros(Float32, 127)), 1, :),
            reshape(vcat(1f0, zeros(Float32, 127)), 1, :),
        ),
        "ref_atom_name_chars" => vcat(
            reshape(vcat(1f0, zeros(Float32, 255)), 1, :),
            reshape(vcat(1f0, zeros(Float32, 255)), 1, :),
            reshape(vcat(1f0, zeros(Float32, 255)), 1, :),
            reshape(vcat(1f0, zeros(Float32, 255)), 1, :),
        ),
        "ref_space_uid" => Int[0, 0, 1, 1],
        "atom_to_token_idx" => Int[0, 0, 1, 1],
    )

    enc0 = PXDesign.Model.AtomAttentionEncoder(
        8;
        has_coords = false,
        c_atom = 16,
        c_atompair = 4,
        c_s = 6,
        c_z = 4,
        n_blocks = 1,
        n_heads = 2,
        n_queries = 2,
        n_keys = 4,
    )
    a0, q0, c0, p0 = enc0(feat)
    @test size(a0) == (2, 8)
    @test size(q0) == (4, 16)
    @test size(c0) == (4, 16)
    @test size(p0) == (2, 2, 4, 4)
    @test all(isfinite, a0)
    atom_input = PXDesign.Model.as_atom_attention_input(feat)
    a0_nt, q0_nt, c0_nt, p0_nt = enc0(atom_input)
    @test size(a0_nt) == (2, 8)
    @test size(q0_nt) == (4, 16)
    @test size(c0_nt) == (4, 16)
    @test size(p0_nt) == (2, 2, 4, 4)
    @test all(isfinite, a0_nt)

    enc1 = PXDesign.Model.AtomAttentionEncoder(
        8;
        has_coords = true,
        c_atom = 16,
        c_atompair = 4,
        c_s = 6,
        c_z = 4,
        n_blocks = 1,
        n_heads = 2,
        n_queries = 2,
        n_keys = 4,
    )
    r = rand(Float32, 2, 4, 3)
    s = rand(Float32, 2, 2, 6)
    z = rand(Float32, 2, 2, 2, 4)
    a1, q1, c1, p1 = enc1(feat; r_l = r, s = s, z = z)
    @test size(a1) == (2, 2, 8)
    @test size(q1) == (2, 4, 16)
    @test size(c1) == (2, 4, 16)
    @test size(p1) == (2, 2, 2, 4, 4)
    @test all(isfinite, a1)

    dec = PXDesign.Model.AtomAttentionDecoder(
        8;
        c_atom = 16,
        c_atompair = 4,
        n_blocks = 1,
        n_heads = 2,
        n_queries = 2,
        n_keys = 4,
    )
    r_update = dec(feat, a1, q1, c1, p1)
    @test size(r_update) == (2, 4, 3)
    @test all(isfinite, r_update)
end

@testset "Diffusion module" begin
    dm = PXDesign.Model.DiffusionModule(8, 6, 4, 5; n_blocks = 2, n_heads = 2)
    relpos_input = (
        asym_id = [1, 1, 2],
        residue_index = [10, 11, 4],
        entity_id = [1, 1, 2],
        sym_id = [1, 1, 1],
        token_index = [0, 1, 2],
    )
    s_inputs = rand(Float32, 3, 5)
    s_trunk = rand(Float32, 3, 6)
    z_trunk = rand(Float32, 3, 3, 4)
    atom_to_token_idx = [0, 0, 1, 1, 2]
    x_noisy = randn(Float32, 2, 5, 3)
    out = dm(
        x_noisy,
        4.0f0;
        relpos_input = relpos_input,
        s_inputs = s_inputs,
        s_trunk = s_trunk,
        z_trunk = z_trunk,
        atom_to_token_idx = atom_to_token_idx,
    )
    @test size(out) == size(x_noisy)
    @test all(isfinite, out)
end

@testset "Raw weights loader" begin
    mktempdir() do d
        # Write two raw float32 tensors.
        t0 = Float32[1 2 3; 4 5 6]
        t1 = reshape(Float32[10, 20, 30, 40], 2, 2)
        open(joinpath(d, "tensor_000000.bin"), "w") do io
            # Raw checkpoint export uses NumPy/PyTorch C-order flattening.
            write(io, reinterpret(UInt8, vec(permutedims(t0))))
        end
        open(joinpath(d, "tensor_000001.bin"), "w") do io
            write(io, reinterpret(UInt8, vec(permutedims(t1))))
        end
        write(
            joinpath(d, "manifest.json"),
            """
{
  "num_tensors": 2,
  "tensors": [
    {"key":"a.weight","dtype":"float32","shape":[2,3],"file":"tensor_000000.bin"},
    {"key":"b.weight","dtype":"float32","shape":[2,2],"file":"tensor_000001.bin"}
  ]
}
""",
        )

        entries = PXDesign.Model.load_raw_manifest(joinpath(d, "manifest.json"))
        @test length(entries) == 2
        @test entries[1].key == "a.weight"
        @test entries[2].shape == [2, 2]

        wt = PXDesign.Model.load_raw_weights(d)
        @test haskey(wt, "a.weight")
        @test haskey(wt, "b.weight")
        @test size(wt["a.weight"]) == (2, 3)
        @test size(wt["b.weight"]) == (2, 2)
        @test wt["a.weight"][2, 3] == 6f0
        @test wt["b.weight"][1, 2] == 30f0
    end
end

@testset "Parity harness" begin
    ref = Dict{String, Any}(
        "a" => Float32[1, 2, 3],
        "b" => reshape(Float32[1, 2, 3, 4], 2, 2),
    )
    act = Dict{String, Any}(
        "a" => Float32[1, 2, 3.00001],
        "b" => reshape(Float32[1, 2, 3, 4], 2, 2),
    )
    report = PXDesign.Model.tensor_parity_report(ref, act; atol = 1f-4, rtol = 1f-4)
    @test isempty(report.missing_in_actual)
    @test isempty(report.missing_in_reference)
    @test isempty(report.failed)
    @test length(report.compared) == 2

    act_bad = Dict{String, Any}(
        "a" => Float32[1, 2, 4],
        "c" => Float32[0],
    )
    report_bad = PXDesign.Model.tensor_parity_report(ref, act_bad; atol = 1f-6, rtol = 1f-6)
    @test length(report_bad.missing_in_actual) == 1
    @test report_bad.missing_in_actual[1] == "b"
    @test length(report_bad.missing_in_reference) == 1
    @test report_bad.missing_in_reference[1] == "c"
    @test length(report_bad.failed) == 1
    @test report_bad.failed[1].key == "a"
end

@testset "CLI parity-check" begin
    mktempdir() do d
        ref_dir = joinpath(d, "ref")
        act_dir = joinpath(d, "act")
        mkpath(ref_dir)
        mkpath(act_dir)

        ref_t = Float32[1, 2, 3]
        act_t = Float32[1, 2, 3]
        open(joinpath(ref_dir, "tensor_000000.bin"), "w") do io
            write(io, reinterpret(UInt8, vec(ref_t)))
        end
        open(joinpath(act_dir, "tensor_000000.bin"), "w") do io
            write(io, reinterpret(UInt8, vec(act_t)))
        end
        manifest = """
{
  "num_tensors": 1,
  "tensors": [
    {"key":"x","dtype":"float32","shape":[3],"file":"tensor_000000.bin"}
  ]
}
"""
        write(joinpath(ref_dir, "manifest.json"), manifest)
        write(joinpath(act_dir, "manifest.json"), manifest)

        @test PXDesign.main(["parity-check", ref_dir, act_dir]) == 0

        act_bad_dir = joinpath(d, "act_bad")
        mkpath(act_bad_dir)
        open(joinpath(act_bad_dir, "tensor_000000.bin"), "w") do io
            write(io, reinterpret(UInt8, vec(Float32[1, 2, 4])))
        end
        write(joinpath(act_bad_dir, "manifest.json"), manifest)
        @test PXDesign.main([
            "parity-check",
            ref_dir,
            act_bad_dir,
            "--atol",
            "1e-8",
            "--rtol",
            "1e-8",
        ]) == 2
    end
end

@testset "State load mapping" begin
    cte = PXDesign.Model.ConditionTemplateEmbedder(5, 3)
    w_cte = fill(2f0, size(cte.weight))
    PXDesign.Model.load_condition_template_embedder!(
        cte,
        Dict("design_condition_embedder.condition_template_embedder.embedder.weight" => w_cte),
        "design_condition_embedder.condition_template_embedder";
        strict = true,
    )
    @test cte.weight == w_cte

    relpe = PXDesign.Model.RelativePositionEncoding(2, 1, 4)
    w_rel = fill(3f0, size(relpe.weight))
    PXDesign.Model.load_relative_position_encoding!(
        relpe,
        Dict("x.linear_no_bias.weight" => w_rel),
        "x";
        strict = true,
    )
    @test relpe.weight == w_rel

    dm = PXDesign.Model.DiffusionModule(8, 6, 4, 5; n_blocks = 1, n_heads = 2)
    w = Dict{String, Any}()
    w["diffusion_module.diffusion_conditioning.layernorm_z.weight"] = fill(
        4f0,
        size(dm.diffusion_conditioning.layernorm_z.weight),
    )
    w["diffusion_module.diffusion_conditioning.linear_no_bias_z.weight"] = fill(
        5f0,
        size(dm.diffusion_conditioning.linear_no_bias_z.weight),
    )
    w["diffusion_module.layernorm_s.weight"] = fill(6f0, size(dm.layernorm_s.weight))
    w["diffusion_module.linear_no_bias_s.weight"] = fill(7f0, size(dm.linear_no_bias_s.weight))
    w["diffusion_module.layernorm_a.weight"] = fill(8f0, size(dm.layernorm_a.weight))
    w[
        "diffusion_module.diffusion_transformer.blocks.0.conditioned_transition_block.linear_nobias_a1.weight"
    ] = fill(
        9f0,
        size(dm.diffusion_transformer.blocks[1].conditioned_transition_block.linear_a1.weight),
    )
    w["diffusion_module.atom_attention_encoder.linear_no_bias_ref_pos.weight"] = fill(
        10f0,
        size(dm.atom_attention_encoder.linear_no_bias_ref_pos.weight),
    )
    w["diffusion_module.atom_attention_decoder.linear_no_bias_out.weight"] = fill(
        11f0,
        size(dm.atom_attention_decoder.linear_no_bias_out.weight),
    )

    PXDesign.Model.load_diffusion_module!(dm, w; strict = false)
    @test all(dm.diffusion_conditioning.layernorm_z.weight .== 4f0)
    @test all(dm.diffusion_conditioning.linear_no_bias_z.weight .== 5f0)
    @test all(dm.layernorm_s.weight .== 6f0)
    @test all(dm.linear_no_bias_s.weight .== 7f0)
    @test all(dm.layernorm_a.weight .== 8f0)
    @test all(dm.diffusion_transformer.blocks[1].conditioned_transition_block.linear_a1.weight .== 9f0)
    @test all(dm.atom_attention_encoder.linear_no_bias_ref_pos.weight .== 10f0)
    @test all(dm.atom_attention_decoder.linear_no_bias_out.weight .== 11f0)

    inferred = PXDesign.Model.infer_model_scaffold_dims(
        Dict(
            "diffusion_module.diffusion_conditioning.relpe.linear_no_bias.weight" => zeros(Float32, 4, 139),
            "diffusion_module.linear_no_bias_s.weight" => zeros(Float32, 8, 6),
            "diffusion_module.diffusion_conditioning.linear_no_bias_s.weight" => zeros(Float32, 6, 16),
            "diffusion_module.atom_attention_encoder.linear_no_bias_ref_pos.weight" => zeros(Float32, 16, 3),
            "diffusion_module.atom_attention_encoder.linear_no_bias_d.weight" => zeros(Float32, 4, 3),
            "diffusion_module.diffusion_transformer.blocks.0.attention_pair_bias.linear_nobias_z.weight" => zeros(
                Float32,
                2,
                4,
            ),
            "diffusion_module.atom_attention_encoder.atom_transformer.diffusion_transformer.blocks.0.attention_pair_bias.linear_nobias_z.weight" => zeros(
                Float32,
                3,
                4,
            ),
            "diffusion_module.atom_attention_decoder.atom_transformer.diffusion_transformer.blocks.0.attention_pair_bias.linear_nobias_z.weight" => zeros(
                Float32,
                5,
                4,
            ),
            "diffusion_module.diffusion_transformer.blocks.2.conditioned_transition_block.linear_nobias_a1.weight" => zeros(
                Float32,
                16,
                8,
            ),
            "diffusion_module.atom_attention_encoder.atom_transformer.diffusion_transformer.blocks.3.conditioned_transition_block.linear_nobias_a1.weight" => zeros(
                Float32,
                32,
                16,
            ),
            "diffusion_module.atom_attention_decoder.atom_transformer.diffusion_transformer.blocks.4.conditioned_transition_block.linear_nobias_a1.weight" => zeros(
                Float32,
                32,
                16,
            ),
        ),
    )
    @test inferred.c_token == 8
    @test inferred.c_s == 6
    @test inferred.c_z == 4
    @test inferred.c_s_inputs == 10
    @test inferred.n_heads == 2
    @test inferred.n_blocks == 3
    @test inferred.c_atom == 16
    @test inferred.c_atompair == 4
    @test inferred.atom_encoder_heads == 3
    @test inferred.atom_encoder_blocks == 4
    @test inferred.atom_decoder_heads == 5
    @test inferred.atom_decoder_blocks == 5

    inferred_design = PXDesign.Model.infer_design_condition_embedder_dims(
        Dict(
            "design_condition_embedder.input_embedder.input_map.weight" => zeros(Float32, 16, 54),
            "design_condition_embedder.condition_template_embedder.embedder.weight" => zeros(Float32, 65, 4),
            "design_condition_embedder.input_embedder.atom_attention_encoder.atom_transformer.diffusion_transformer.blocks.0.attention_pair_bias.linear_nobias_z.weight" => zeros(
                Float32,
                2,
                4,
            ),
            "design_condition_embedder.input_embedder.atom_attention_encoder.atom_transformer.diffusion_transformer.blocks.2.conditioned_transition_block.linear_nobias_a1.weight" => zeros(
                Float32,
                32,
                16,
            ),
        ),
    )
    @test inferred_design.c_token == 8
    @test inferred_design.c_s_inputs == 16
    @test inferred_design.c_z == 4
    @test inferred_design.n_heads == 2
    @test inferred_design.n_blocks == 3

    dce = PXDesign.Model.DesignConditionEmbedder(
        8;
        c_s_inputs = 16,
        c_z = 4,
        n_blocks = 1,
    )
    w_dce = Dict{String, Any}(
        "design_condition_embedder.condition_template_embedder.embedder.weight" => fill(
            2f0,
            size(dce.condition_template_embedder.weight),
        ),
        "design_condition_embedder.input_embedder.input_map.weight" => fill(
            3f0,
            size(dce.input_embedder.input_map_weight),
        ),
        "design_condition_embedder.input_embedder.input_map.bias" => fill(
            4f0,
            size(dce.input_embedder.input_map_bias),
        ),
        "design_condition_embedder.input_embedder.atom_attention_encoder.linear_no_bias_ref_pos.weight" => fill(
            5f0,
            size(dce.input_embedder.atom_attention_encoder.linear_no_bias_ref_pos.weight),
        ),
        "design_condition_embedder.input_embedder.atom_attention_encoder.atom_transformer.diffusion_transformer.blocks.0.attention_pair_bias.layernorm_kv.linear_s.weight" => fill(
            6f0,
            size(
                dce.input_embedder.atom_attention_encoder.atom_transformer.blocks[1].attention_pair_bias.adaln_kv.linear_s.weight,
            ),
        ),
    )
    PXDesign.Model.load_design_condition_embedder!(dce, w_dce; strict = false)
    @test all(dce.condition_template_embedder.weight .== 2f0)
    @test all(dce.input_embedder.input_map_weight .== 3f0)
    @test all(dce.input_embedder.input_map_bias .== 4f0)
    @test all(dce.input_embedder.atom_attention_encoder.linear_no_bias_ref_pos.weight .== 5f0)
    @test all(
        dce.input_embedder.atom_attention_encoder.atom_transformer.blocks[1].attention_pair_bias.adaln_kv.linear_s.weight .== 6f0,
    )

    dm_cov = PXDesign.Model.DiffusionModule(8, 6, 4, 16; n_blocks = 1, n_heads = 2)
    dce_cov = PXDesign.Model.DesignConditionEmbedder(8; c_s_inputs = 16, c_z = 4, n_blocks = 1, n_heads = 2)
    expected_dm = PXDesign.Model.expected_diffusion_module_keys(dm_cov)
    expected_dce = PXDesign.Model.expected_design_condition_embedder_keys(dce_cov)
    w_cov = Dict{String, Any}()
    for k in vcat(expected_dm, expected_dce)
        w_cov[k] = zeros(Float32, 1)
    end
    report_ok = PXDesign.Model.checkpoint_coverage_report(dm_cov, dce_cov, w_cov)
    @test isempty(report_ok.missing)
    @test isempty(report_ok.unused)
    @test report_ok.n_expected == length(Set(vcat(expected_dm, expected_dce)))

    w_extra = copy(w_cov)
    w_extra["diffusion_module.__extra.unused"] = zeros(Float32, 1)
    report_extra = PXDesign.Model.checkpoint_coverage_report(dm_cov, dce_cov, w_extra)
    @test length(report_extra.unused) == 1

    w_missing = copy(w_cov)
    delete!(w_missing, expected_dm[1])
    report_missing = PXDesign.Model.checkpoint_coverage_report(dm_cov, dce_cov, w_missing)
    @test length(report_missing.missing) == 1

    @test_throws Exception PXDesign.Model.load_diffusion_module!(
        PXDesign.Model.DiffusionModule(8, 6, 4, 5; n_blocks = 1, n_heads = 2),
        Dict{String, Any}();
        strict = true,
    )
end

@testset "Real checkpoint coverage (optional)" begin
    raw_dir = joinpath(dirname(@__DIR__), "weights_raw")
    if isdir(raw_dir) && isfile(joinpath(raw_dir, "manifest.json"))
        w = PXDesign.Model.load_raw_weights(raw_dir)
        d = PXDesign.Model.infer_model_scaffold_dims(w)
        dd = PXDesign.Model.infer_design_condition_embedder_dims(w)
        dm = PXDesign.Model.DiffusionModule(
            d.c_token,
            d.c_s,
            d.c_z,
            d.c_s_inputs;
            c_atom = d.c_atom,
            c_atompair = d.c_atompair,
            atom_encoder_blocks = d.atom_encoder_blocks,
            atom_encoder_heads = d.atom_encoder_heads,
            n_blocks = d.n_blocks,
            n_heads = d.n_heads,
            atom_decoder_blocks = d.atom_decoder_blocks,
            atom_decoder_heads = d.atom_decoder_heads,
        )
        dce = PXDesign.Model.DesignConditionEmbedder(
            dd.c_token;
            c_s_inputs = dd.c_s_inputs,
            c_z = dd.c_z,
            c_atom = d.c_atom,
            c_atompair = d.c_atompair,
            n_blocks = dd.n_blocks,
            n_heads = dd.n_heads,
        )
        report = PXDesign.Model.checkpoint_coverage_report(dm, dce, w)
        @test isempty(report.missing)
        @test isempty(report.unused)
        @test report.n_expected == 732
        @test report.n_present == 732
    else
        @test_skip "Skipping real checkpoint coverage: weights_raw/manifest.json not present."
    end
end

@testset "Infer scaffold non-dry-run" begin
    mktempdir() do d
        target_cif = joinpath(d, "target.cif")
        write(
            target_cif,
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

        input_json = joinpath(d, "input.json")
        write(
            input_json,
            """
[
  {
    "name": "demo_infer",
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

        cfg = PXDesign.Config.default_config(project_root = d)
        cfg["input_json_path"] = input_json
        cfg["dump_dir"] = joinpath(d, "out")
        cfg["download_cache"] = false
        cfg["seeds"] = [12345]
        cfg["sample_diffusion"]["N_sample"] = 2
        cfg["sample_diffusion"]["N_step"] = 4
        cfg["sample_diffusion"]["eta_schedule"] = Dict("type" => "const", "min" => 1.0, "max" => 1.0)

        result = PXDesign.Infer.run_infer(cfg; dry_run = false, io = devnull)
        @test result["status"] == "ok_scaffold_model"

        pred_dir = joinpath(
            cfg["dump_dir"],
            "global_run_0",
            "demo_infer",
            "seed_12345",
            "predictions",
        )
        @test isdir(pred_dir)
        cif_files = filter(f -> endswith(f, ".cif"), readdir(pred_dir))
        @test length(cif_files) == 2
        cif_text = read(joinpath(pred_dir, first(sort(cif_files))), String)
        @test occursin("_entity_poly.", cif_text)
        @test occursin("_struct_conn.", cif_text)
        @test occursin(" A0 ", cif_text)
        @test occursin(" B0 ", cif_text)
        @test isfile(joinpath(pred_dir, "sample_level_output.csv"))
        @test isfile(joinpath(cfg["dump_dir"], "global_run_0", "demo_infer", "seed_12345", "SUCCESS_FILE"))

        # Also validate typed model-scaffold path.
        cfg2 = PXDesign.Config.default_config(project_root = d)
        cfg2["input_json_path"] = input_json
        cfg2["dump_dir"] = joinpath(d, "out_model_scaffold")
        cfg2["download_cache"] = false
        cfg2["seeds"] = [7]
        cfg2["sample_diffusion"]["N_sample"] = 1
        cfg2["sample_diffusion"]["N_step"] = 2
        cfg2["sample_diffusion"]["eta_schedule"] = Dict("type" => "const", "min" => 1.0, "max" => 1.0)
        cfg2["model_scaffold"]["enabled"] = true
        cfg2["model_scaffold"]["c_token"] = 8
        cfg2["model_scaffold"]["c_s"] = 8
        cfg2["model_scaffold"]["c_z"] = 4
        cfg2["model_scaffold"]["c_s_inputs"] = 16
        cfg2["model_scaffold"]["n_blocks"] = 1
        cfg2["model_scaffold"]["n_heads"] = 2

        result2 = PXDesign.Infer.run_infer(cfg2; dry_run = false, io = devnull)
        @test result2["status"] == "ok_scaffold_model"
        pred_dir2 = joinpath(
            cfg2["dump_dir"],
            "global_run_0",
            "demo_infer",
            "seed_7",
            "predictions",
        )
        @test isdir(pred_dir2)
        cif_files2 = filter(f -> endswith(f, ".cif"), readdir(pred_dir2))
        @test length(cif_files2) == 1

        # Model scaffold + raw weight directory path with auto dim inference.
        rwdir = joinpath(d, "rw")
        mkpath(rwdir)
        wt = fill(1f0, 8) # diffusion_module.layernorm_s.weight (c_s=8)
        open(joinpath(rwdir, "tensor_000000.bin"), "w") do io
            write(io, reinterpret(UInt8, wt))
        end
        open(joinpath(rwdir, "tensor_000001.bin"), "w") do io
            write(io, reinterpret(UInt8, vec(fill(0.1f0, 4, 139))))
        end
        open(joinpath(rwdir, "tensor_000002.bin"), "w") do io
            write(io, reinterpret(UInt8, vec(fill(0.2f0, 8, 8))))
        end
        open(joinpath(rwdir, "tensor_000003.bin"), "w") do io
            write(io, reinterpret(UInt8, vec(fill(0.3f0, 8, 16))))
        end
        open(joinpath(rwdir, "tensor_000004.bin"), "w") do io
            write(io, reinterpret(UInt8, vec(fill(0.4f0, 2, 4))))
        end
        open(joinpath(rwdir, "tensor_000005.bin"), "w") do io
            write(io, reinterpret(UInt8, vec(fill(0.5f0, 8, 3))))
        end
        open(joinpath(rwdir, "tensor_000006.bin"), "w") do io
            write(io, reinterpret(UInt8, vec(fill(0.6f0, 4, 3))))
        end
        open(joinpath(rwdir, "tensor_000007.bin"), "w") do io
            write(io, reinterpret(UInt8, vec(fill(0.7f0, 2, 4))))
        end
        open(joinpath(rwdir, "tensor_000008.bin"), "w") do io
            write(io, reinterpret(UInt8, vec(fill(0.8f0, 2, 4))))
        end
        write(
            joinpath(rwdir, "manifest.json"),
            """
{
  "num_tensors": 9,
  "tensors": [
    {"key":"diffusion_module.layernorm_s.weight","dtype":"float32","shape":[8],"file":"tensor_000000.bin"},
    {"key":"diffusion_module.diffusion_conditioning.relpe.linear_no_bias.weight","dtype":"float32","shape":[4,139],"file":"tensor_000001.bin"},
    {"key":"diffusion_module.linear_no_bias_s.weight","dtype":"float32","shape":[8,8],"file":"tensor_000002.bin"},
    {"key":"diffusion_module.diffusion_conditioning.linear_no_bias_s.weight","dtype":"float32","shape":[8,16],"file":"tensor_000003.bin"},
    {"key":"diffusion_module.diffusion_transformer.blocks.0.attention_pair_bias.linear_nobias_z.weight","dtype":"float32","shape":[2,4],"file":"tensor_000004.bin"},
    {"key":"diffusion_module.atom_attention_encoder.linear_no_bias_ref_pos.weight","dtype":"float32","shape":[8,3],"file":"tensor_000005.bin"},
    {"key":"diffusion_module.atom_attention_encoder.linear_no_bias_d.weight","dtype":"float32","shape":[4,3],"file":"tensor_000006.bin"},
    {"key":"diffusion_module.atom_attention_encoder.atom_transformer.diffusion_transformer.blocks.0.attention_pair_bias.linear_nobias_z.weight","dtype":"float32","shape":[2,4],"file":"tensor_000007.bin"},
    {"key":"diffusion_module.atom_attention_decoder.atom_transformer.diffusion_transformer.blocks.0.attention_pair_bias.linear_nobias_z.weight","dtype":"float32","shape":[2,4],"file":"tensor_000008.bin"}
  ]
}
""",
        )

        cfg3 = PXDesign.Config.default_config(project_root = d)
        cfg3["input_json_path"] = input_json
        cfg3["dump_dir"] = joinpath(d, "out_model_scaffold_rw")
        cfg3["download_cache"] = false
        cfg3["seeds"] = [9]
        cfg3["sample_diffusion"]["N_sample"] = 1
        cfg3["sample_diffusion"]["N_step"] = 2
        cfg3["sample_diffusion"]["eta_schedule"] = Dict("type" => "const", "min" => 1.0, "max" => 1.0)
        cfg3["model_scaffold"]["enabled"] = true
        cfg3["model_scaffold"]["c_token"] = 12
        cfg3["model_scaffold"]["c_s"] = 10
        cfg3["model_scaffold"]["c_z"] = 6
        cfg3["model_scaffold"]["c_s_inputs"] = 20
        cfg3["model_scaffold"]["n_blocks"] = 3
        cfg3["model_scaffold"]["n_heads"] = 3
        cfg3["model_scaffold"]["auto_dims_from_weights"] = true
        cfg3["raw_weights_dir"] = rwdir
        cfg3["strict_weight_load"] = false

        result3 = PXDesign.Infer.run_infer(cfg3; dry_run = false, io = devnull)
        @test result3["status"] == "ok_scaffold_model"
        pred_dir3 = joinpath(
            cfg3["dump_dir"],
            "global_run_0",
            "demo_infer",
            "seed_9",
            "predictions",
        )
        @test isdir(pred_dir3)
        @test length(filter(f -> endswith(f, ".cif"), readdir(pred_dir3))) == 1

        # Strict real-weight tiny CPU smoke (optional).
        repo_root = dirname(@__DIR__)
        real_raw = joinpath(repo_root, "weights_raw")
        if isdir(real_raw) && isfile(joinpath(real_raw, "manifest.json"))
            cfg4 = PXDesign.Config.default_config(project_root = d)
            cfg4["input_json_path"] = input_json
            cfg4["dump_dir"] = joinpath(d, "out_model_scaffold_real_strict")
            cfg4["download_cache"] = false
            cfg4["seeds"] = [13]
            cfg4["sample_diffusion"]["N_sample"] = 1
            cfg4["sample_diffusion"]["N_step"] = 2
            cfg4["sample_diffusion"]["eta_schedule"] = Dict("type" => "const", "min" => 1.0, "max" => 1.0)
            cfg4["infer_setting"]["sample_diffusion_chunk_size"] = 1
            cfg4["model_scaffold"]["enabled"] = true
            cfg4["model_scaffold"]["auto_dims_from_weights"] = true
            cfg4["model_scaffold"]["use_design_condition_embedder"] = true
            cfg4["raw_weights_dir"] = real_raw
            cfg4["strict_weight_load"] = true

            result4 = PXDesign.Infer.run_infer(cfg4; dry_run = false, io = devnull)
            @test result4["status"] == "ok_scaffold_model"
            pred_dir4 = joinpath(
                cfg4["dump_dir"],
                "global_run_0",
                "demo_infer",
                "seed_13",
                "predictions",
            )
            @test isdir(pred_dir4)
            @test length(filter(f -> endswith(f, ".cif"), readdir(pred_dir4))) == 1
        else
            @test_skip "Skipping strict real-weight e2e smoke: weights_raw/manifest.json not present."
        end
    end
end
