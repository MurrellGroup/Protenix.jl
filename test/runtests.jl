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

@testset "Config defaults and overrides" begin
    mktempdir() do d
        data_root = joinpath(d, "data_root")
        mkpath(data_root)
        comp = joinpath(data_root, "components.cif")
        rdkit = joinpath(data_root, "components.cif.rdkit_mol.pkl")
        write(comp, "cif")
        write(rdkit, "rdkit")

        old_root = get(ENV, "PROTENIX_DATA_ROOT_DIR", nothing)
        ENV["PROTENIX_DATA_ROOT_DIR"] = data_root
        try
            cfg = PXDesign.Config.default_config(project_root = d)
            @test cfg["dump_dir"] == joinpath(d, "output")
            @test cfg["load_checkpoint_dir"] == joinpath(d, "release_data", "checkpoint")
            @test cfg["model_name"] == "pxdesign_v0.1.0"
            @test cfg["data"]["ccd_components_file"] == comp
            @test cfg["data"]["ccd_components_rdkit_mol_file"] == rdkit
            @test cfg["data"]["pdb_cluster_file"] == joinpath(data_root, "clusters-by-entity-40.txt")
            @test cfg["sample_diffusion"]["N_step"] == 400
            @test cfg["sample_diffusion"]["eta_schedule"]["type"] == "piecewise_65"
            @test cfg["inference_noise_scheduler"]["sigma_data"] == 16.0
            @test cfg["model_scaffold"]["c_s_inputs"] == 128

            push!(cfg["seeds"], 7)
            cfg2 = PXDesign.Config.default_config(project_root = d)
            @test isempty(cfg2["seeds"])
        finally
            if old_root === nothing
                delete!(ENV, "PROTENIX_DATA_ROOT_DIR")
            else
                ENV["PROTENIX_DATA_ROOT_DIR"] = old_root
            end
        end
    end

    mktempdir() do d
        cfg = PXDesign.Config.default_config(project_root = d)
        PXDesign.Config.set_by_key!(cfg, "N_sample", "9")
        PXDesign.Config.set_by_key!(cfg, "eta_type", "const")
        PXDesign.Config.set_by_key!(cfg, "eta_min", "1.25")
        PXDesign.Config.set_by_key!(cfg, "eta_max", "2.75")
        PXDesign.Config.set_by_key!(cfg, "sample_diffusion_chunk_size", "3")
        PXDesign.Config.set_nested!(cfg, "model_scaffold.enabled", true)

        @test cfg["sample_diffusion"]["N_sample"] == 9
        @test cfg["sample_diffusion"]["eta_schedule"]["type"] == "const"
        @test cfg["sample_diffusion"]["eta_schedule"]["min"] == 1.25
        @test cfg["sample_diffusion"]["eta_schedule"]["max"] == 2.75
        @test cfg["infer_setting"]["sample_diffusion_chunk_size"] == 3
        @test cfg["model_scaffold"]["enabled"] == true
    end

    cfg_preserve = Dict{String, Any}(
        "sample_diffusion" => Dict("existing_key" => 11),
    )
    PXDesign.Config.set_nested!(cfg_preserve, "sample_diffusion.N_step", 17)
    @test cfg_preserve["sample_diffusion"]["existing_key"] == 11
    @test cfg_preserve["sample_diffusion"]["N_step"] == 17

    @test PXDesign.Config.parse_override_value("true") == true
    @test PXDesign.Config.parse_override_value("False") == false
    @test PXDesign.Config.parse_override_value("none") === nothing
    @test PXDesign.Config.parse_override_value("17") == 17
    @test PXDesign.Config.parse_override_value("3.5") == 3.5
    @test PXDesign.Config.parse_override_value("abc") == "abc"
end

@testset "Protenix API surface" begin
    models = PXDesign.list_supported_models()
    @test length(models) >= 5
    @test issorted(getindex.(models, :model_name))
    @test any(m -> m.model_name == "protenix_base_default_v0.5.0", models)
    @test any(m -> m.model_name == "protenix_mini_default_v0.5.0", models)

    opts = PXDesign.ProtenixPredictOptions(
        model_name = "protenix_mini_default_v0.5.0",
        seeds = [3, 5],
        cycle = 2,
        step = 7,
        sample = 1,
    )
    @test opts.model_name == "protenix_mini_default_v0.5.0"
    @test opts.seeds == [3, 5]
    @test opts.cycle == 2
    @test opts.step == 7
    @test opts.sample == 1

    seq_opts = PXDesign.ProtenixSequenceOptions(common = opts, task_name = "demo", chain_id = "A0")
    @test seq_opts.common === opts
    @test seq_opts.task_name == "demo"
    @test seq_opts.chain_id == "A0"
    @test seq_opts.esm_token_embedding === nothing

    @test_throws ErrorException PXDesign.ProtenixPredictOptions(seeds = Int[])
    @test PXDesign.main(["predict", "--list-models"]) == 0

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

    tiny_src = PXDesign.resolve_weight_source("protenix_tiny_default_v0.5.0")
    @test tiny_src.layout == :single
    @test tiny_src.filename == "weights_safetensors_protenix_tiny_default_v0.5.0/protenix_tiny_default_v0.5.0.safetensors"
    @test tiny_src.repo_id == get(ENV, "PXDESIGN_WEIGHTS_REPO_ID", "MurrellLab/PXDesign.jl")
    mini_esm_src = PXDesign.resolve_weight_source("protenix_mini_esm_v0.5.0")
    @test mini_esm_src.layout == :single
    @test mini_esm_src.filename == "weights_safetensors_protenix_mini_esm_v0.5.0/protenix_mini_esm_v0.5.0.safetensors"
    mini_ism_src = PXDesign.resolve_weight_source("protenix_mini_ism_v0.5.0")
    @test mini_ism_src.layout == :single
    @test mini_ism_src.filename == "weights_safetensors_protenix_mini_ism_v0.5.0/protenix_mini_ism_v0.5.0.safetensors"
    @test_throws ErrorException PXDesign.resolve_weight_source("not_a_model_for_hf")

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

        mmcif_path = joinpath(d, "tiny_assembly.cif")
        write(
            mmcif_path,
            """
data_tiny
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.auth_asym_id
_atom_site.label_seq_id
_atom_site.auth_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.pdbx_PDB_model_num
ATOM 1 N N ALA A A 1 1 0.0 0.0 0.0 1
ATOM 2 C CA ALA A A 1 1 1.0 0.0 0.0 1
ATOM 3 C C ALA A A 1 1 2.0 0.0 0.0 1
ATOM 4 O O ALA A A 1 1 3.0 0.0 0.0 1
ATOM 5 N N GLY B B 1 1 0.0 1.0 0.0 1
ATOM 6 C CA GLY B B 1 1 1.0 1.0 0.0 1
ATOM 7 C C GLY B B 1 1 2.0 1.0 0.0 1
ATOM 8 O O GLY B B 1 1 3.0 1.0 0.0 1
#
loop_
_pdbx_struct_assembly_gen.assembly_id
_pdbx_struct_assembly_gen.oper_expression
_pdbx_struct_assembly_gen.asym_id_list
1 "1,2" A
2 1 B
#
""",
        )
        assembly_out_paths = PXDesign.convert_structure_to_infer_json(
            mmcif_path;
            out_dir = json_out_dir,
            assembly_id = "1",
        )
        @test length(assembly_out_paths) == 1
        parsed_assembly = PXDesign.JSONLite.parse_json(read(assembly_out_paths[1], String))
        @test parsed_assembly[1]["assembly_id"] == "1"
        seqs_assembly = parsed_assembly[1]["sequences"]
        @test length(seqs_assembly) == 1
        @test seqs_assembly[1]["proteinChain"]["sequence"] == "A"
        @test seqs_assembly[1]["proteinChain"]["count"] == 2

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

        in_json_object = joinpath(d, "input_object.json")
        PXDesign.JSONLite.write_json(
            in_json_object,
            Dict(
                "name" => "msa_demo_object",
                "sequences" => Any[
                    Dict("proteinChain" => Dict("sequence" => "ACD", "count" => 1)),
                    Dict("ligand" => Dict("ligand" => "CCD_ATP", "count" => 1)),
                ],
            ),
        )
        msa_out_object = PXDesign.add_precomputed_msa_to_json(
            in_json_object;
            out_dir = joinpath(d, "msa_out_object"),
            precomputed_msa_dir = "/tmp/precomp_msa",
        )
        parsed_msa_object = PXDesign.JSONLite.parse_json(read(msa_out_object, String))
        @test parsed_msa_object isa AbstractDict
        @test parsed_msa_object["sequences"][1]["proteinChain"]["msa"]["precomputed_msa_dir"] == "/tmp/precomp_msa"

        wrapper_json = joinpath(d, "input_wrapper.json")
        PXDesign.JSONLite.write_json(
            wrapper_json,
            Dict(
                "name" => "wrapper_payload",
                "tasks" => Any[
                    Dict(
                        "name" => "msa_demo_wrapped",
                        "sequences" => Any[
                            Dict("proteinChain" => Dict("sequence" => "ACD", "count" => 1)),
                            Dict("ligand" => Dict("ligand" => "CCD_ATP", "count" => 1)),
                        ],
                    ),
                ],
            ),
        )
        wrapped_tasks = PXDesign.ProtenixAPI._ensure_json_tasks(wrapper_json)
        @test length(wrapped_tasks) == 1
        @test wrapped_tasks[1]["name"] == "msa_demo_wrapped"
        msa_out_wrapped = PXDesign.add_precomputed_msa_to_json(
            wrapper_json;
            out_dir = joinpath(d, "msa_out_wrapped"),
            precomputed_msa_dir = "/tmp/precomp_msa",
        )
        @test isfile(msa_out_wrapped)
        parsed_msa_wrapped = PXDesign.JSONLite.parse_json(read(msa_out_wrapped, String))
        @test parsed_msa_wrapped isa AbstractDict
        @test haskey(parsed_msa_wrapped, "tasks")
        @test parsed_msa_wrapped["name"] == "wrapper_payload"
        @test parsed_msa_wrapped["tasks"][1]["sequences"][1]["proteinChain"]["msa"]["precomputed_msa_dir"] == "/tmp/precomp_msa"

        bad_wrapper_json = joinpath(d, "input_wrapper_bad.json")
        PXDesign.JSONLite.write_json(
            bad_wrapper_json,
            Dict(
                "name" => "wrapper_payload_bad",
                "tasks" => Dict("name" => "not_an_array"),
            ),
        )
        @test_throws ErrorException PXDesign.ProtenixAPI._ensure_json_tasks(bad_wrapper_json)

        _has_ccd_data = PXDesign.ProtenixAPI._default_ccd_components_path() != ""
        mini_weights = try
            PXDesign.default_weights_path("protenix_mini_default_v0.5.0")
        catch
            nothing
        end
        if mini_weights === nothing || !_has_ccd_data
            @test_skip "Skipping predict_json wrapper smoke: requires protenix_mini weights and CCD data."
        else
            pred_opts = PXDesign.ProtenixPredictOptions(
                out_dir = joinpath(d, "predict_out_wrapped"),
                model_name = "protenix_mini_default_v0.5.0",
                seeds = [7],
                use_default_params = false,
                cycle = 1,
                step = 1,
                sample = 1,
                use_msa = false,
                strict = true,
            )
            pred_records = PXDesign.predict_json(wrapper_json, pred_opts)
            @test pred_records isa Vector{PXDesign.ProtenixAPI.PredictJSONRecord}
            @test length(pred_records) == 1
            @test pred_records[1].task_name == "msa_demo_wrapped"
            @test !isempty(pred_records[1].cif_paths)
            @test isfile(pred_records[1].cif_paths[1])

            seq_records = PXDesign.predict_sequence(
                "ACD";
                out_dir = joinpath(d, "predict_seq_out"),
                model_name = "protenix_mini_default_v0.5.0",
                seeds = [9],
                use_default_params = false,
                cycle = 1,
                step = 1,
                sample = 1,
                use_msa = false,
                strict = true,
                task_name = "seq_wrapper_smoke",
                chain_id = "A0",
            )
            @test seq_records isa Vector{PXDesign.ProtenixAPI.PredictSequenceRecord}
            @test length(seq_records) == 1
            @test seq_records[1].task_name == "seq_wrapper_smoke"
            @test !isempty(seq_records[1].cif_paths)
            @test isfile(seq_records[1].cif_paths[1])
        end

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

@testset "Protenix mixed-entity parsing and covalent bonds" begin
    if PXDesign.ProtenixAPI._default_ccd_components_path() == ""
        @test_skip "Skipping mixed-entity CCD ligand test: CCD components file not available."
    else
        task = Dict{String, Any}(
            "name" => "mixed_entities_smoke",
            "sequences" => Any[
                Dict("proteinChain" => Dict("sequence" => "AC", "count" => 1)),
                Dict("dnaSequence" => Dict("sequence" => "AT", "count" => 1)),
                Dict("rnaSequence" => Dict("sequence" => "GU", "count" => 1)),
                Dict("ligand" => Dict("ligand" => "CCD_ATP", "count" => 1)),
                Dict("condition_ligand" => Dict("ligand" => "CCD_FAD", "count" => 1)),
                Dict("ion" => Dict("ion" => "MG", "count" => 1)),
            ],
        )
        parsed = PXDesign.ProtenixAPI._parse_task_entities(task)
        @test !isempty(parsed.atoms)
        @test length(parsed.protein_specs) == 1
        @test length(parsed.entity_chain_ids) == 6
        @test any(a -> a.mol_type == "protein", parsed.atoms)
        @test any(a -> a.mol_type == "dna", parsed.atoms)
        @test any(a -> a.mol_type == "rna", parsed.atoms)
        @test any(a -> a.mol_type == "ligand", parsed.atoms)
    end

    bond_task = Dict{String, Any}(
        "name" => "bond_smoke",
        "sequences" => Any[
            Dict("proteinChain" => Dict("sequence" => "AC", "count" => 1)),
            Dict("proteinChain" => Dict("sequence" => "GG", "count" => 1)),
        ],
        "covalent_bonds" => Any[
            Dict(
                "entity1" => "1",
                "position1" => "1",
                "atom1" => "N",
                "entity2" => "2",
                "position2" => "1",
                "atom2" => "N",
            ),
        ],
    )
    parsed_bond = PXDesign.ProtenixAPI._parse_task_entities(bond_task)
    bundle = PXDesign.Data.build_feature_bundle_from_atoms(parsed_bond.atoms; task_name = "bond_smoke")
    feat = bundle["input_feature_dict"]
    PXDesign.ProtenixAPI._normalize_protenix_feature_dict!(feat)
    PXDesign.ProtenixAPI._inject_task_covalent_token_bonds!(feat, bundle["atoms"], bond_task, parsed_bond.entity_chain_ids)

    token_chain_ids = [bundle["atoms"][tok.centre_atom_index].chain_id for tok in bundle["tokens"]]
    tok_a = findfirst(==("A0"), token_chain_ids)
    tok_b = findfirst(==("B0"), token_chain_ids)
    @test tok_a !== nothing
    @test tok_b !== nothing
    @test feat["token_bonds"][tok_a, tok_b] == 1
    @test feat["token_bonds"][tok_b, tok_a] == 1

    smiles_task = Dict{String, Any}(
        "name" => "smiles_bond_smoke",
        "sequences" => Any[
            Dict("proteinChain" => Dict("sequence" => "AC", "count" => 1)),
            Dict("ligand" => Dict("ligand" => "C[C:9]O", "count" => 1)),
        ],
        "covalent_bonds" => Any[
            Dict(
                "entity1" => 1,
                "position1" => 1,
                "atom1" => "CA",
                "entity2" => 2,
                "position2" => 1,
                "atom2" => 9,
            ),
        ],
    )
    parsed_smiles = PXDesign.ProtenixAPI._parse_task_entities(smiles_task)
    @test length(parsed_smiles.entity_atom_map) >= 2
    @test haskey(parsed_smiles.entity_atom_map[2], 9)

    bundle_smiles = PXDesign.Data.build_feature_bundle_from_atoms(parsed_smiles.atoms; task_name = "smiles_bond_smoke")
    feat_smiles = bundle_smiles["input_feature_dict"]
    PXDesign.ProtenixAPI._normalize_protenix_feature_dict!(feat_smiles)
    PXDesign.ProtenixAPI._inject_task_covalent_token_bonds!(
        feat_smiles,
        bundle_smiles["atoms"],
        smiles_task,
        parsed_smiles.entity_chain_ids,
        parsed_smiles.entity_atom_map,
    )
    token_chain_ids_smiles = [bundle_smiles["atoms"][tok.centre_atom_index].chain_id for tok in bundle_smiles["tokens"]]
    prot_cols = findall(==("A0"), token_chain_ids_smiles)
    lig_cols = findall(==("B0"), token_chain_ids_smiles)
    @test !isempty(prot_cols)
    @test !isempty(lig_cols)
    @test any(feat_smiles["token_bonds"][i, j] == 1 for i in prot_cols, j in lig_cols)

    smiles_prefix_task = Dict{String, Any}(
        "name" => "smiles_prefix_smoke",
        "sequences" => Any[
            Dict("ligand" => Dict("ligand" => "SMILES_CCO", "count" => 1)),
        ],
    )
    parsed_smiles_prefix = PXDesign.ProtenixAPI._parse_task_entities(smiles_prefix_task)
    @test any(a -> a.element == "C", parsed_smiles_prefix.atoms)
    @test any(a -> a.element == "O", parsed_smiles_prefix.atoms)
    @test_throws Exception PXDesign.ProtenixAPI._parse_task_entities(
        Dict{String, Any}(
            "name" => "smiles_prefix_empty",
            "sequences" => Any[Dict("ligand" => Dict("ligand" => "SMILES_", "count" => 1))],
        ),
    )
    parsed_condition_smiles = PXDesign.ProtenixAPI._parse_task_entities(
        Dict{String, Any}(
            "name" => "condition_smiles_prefix_smoke",
            "sequences" => Any[
                Dict("condition_ligand" => Dict("ligand" => "SMILES_CCO", "count" => 1)),
            ],
        ),
    )
    @test any(a -> a.element == "C", parsed_condition_smiles.atoms)
    @test any(a -> a.element == "O", parsed_condition_smiles.atoms)

    parsed_ion_ccd = PXDesign.ProtenixAPI._parse_task_entities(
        Dict{String, Any}(
            "name" => "ion_ccd_prefix_smoke",
            "sequences" => Any[
                Dict("ion" => Dict("ion" => "CCD_MG", "count" => 1)),
            ],
        ),
    )
    @test any(a -> uppercase(a.element) == "MG", parsed_ion_ccd.atoms)
    @test any(a -> uppercase(a.atom_name) == "MG", parsed_ion_ccd.atoms)

    mixed_runtime_task = Dict{String, Any}(
        "name" => "mixed_runtime_smoke",
        "sequences" => Any[
            Dict("proteinChain" => Dict("sequence" => "ACD", "count" => 1)),
            Dict("dnaSequence" => Dict("sequence" => "AG", "count" => 1)),
            Dict("rnaSequence" => Dict("sequence" => "CU", "count" => 1)),
            Dict("ligand" => Dict("ligand" => "SMILES_CCO", "count" => 1)),
            Dict("ion" => Dict("ion" => "MG", "count" => 1)),
        ],
    )
    parsed_mixed = PXDesign.ProtenixAPI._parse_task_entities(mixed_runtime_task)
    bundle_mixed = PXDesign.Data.build_feature_bundle_from_atoms(parsed_mixed.atoms; task_name = "mixed_runtime_smoke")
    feat_mixed = bundle_mixed["input_feature_dict"]
    PXDesign.ProtenixAPI._normalize_protenix_feature_dict!(feat_mixed)
    typed_mixed = PXDesign.ProtenixMini.as_protenix_features(feat_mixed)
    @test typed_mixed.constraint_feature === nothing

    mixed_model = PXDesign.ProtenixMini.ProtenixMiniModel(
        32,
        32,
        16,
        8,
        97;
        c_atom = 16,
        c_atompair = 8,
        n_cycle = 1,
        pairformer_blocks = 1,
        msa_blocks = 1,
        diffusion_transformer_blocks = 1,
        diffusion_atom_encoder_blocks = 1,
        diffusion_atom_decoder_blocks = 1,
        confidence_max_atoms_per_token = 128,
        sample_gamma0 = 0.0,
        sample_gamma_min = 1.0,
        sample_noise_scale_lambda = 1.0,
        sample_step_scale_eta = 0.0,
        sample_n_step = 1,
        sample_n_sample = 1,
        rng = MersenneTwister(4444),
    )
    pred_mixed = PXDesign.ProtenixMini.run_inference(
        mixed_model,
        typed_mixed;
        n_cycle = 1,
        n_step = 1,
        n_sample = 1,
        rng = MersenneTwister(5555),
    )
    @test size(pred_mixed.coordinate) == (1, length(bundle_mixed["atoms"]), 3)
    @test all(isfinite, pred_mixed.coordinate)

    mktempdir() do d
        lig_pdb = joinpath(d, "ligand.pdb")
        write(
            lig_pdb,
            """
HETATM    1  C1  UNL A   1       0.000   0.000   0.000  1.00 20.00           C
HETATM    2  O1  UNL A   1       1.200   0.000   0.000  1.00 20.00           O
END
""",
        )
        file_task = Dict{String, Any}(
            "name" => "file_ligand_smoke",
            "sequences" => Any[Dict("ligand" => Dict("ligand" => "FILE_ligand.pdb", "count" => 1))],
        )
        parsed_file = PXDesign.ProtenixAPI._parse_task_entities(file_task; json_dir = d)
        @test any(a -> a.mol_type == "ligand", parsed_file.atoms)
        @test length(parsed_file.entity_atom_map) >= 1
        @test !isempty(parsed_file.entity_atom_map[1])
    end

    constraint_task = Dict{String, Any}(
        "name" => "constraint_smoke",
        "sequences" => Any[
            Dict("proteinChain" => Dict("sequence" => "AC", "count" => 1)),
            Dict("proteinChain" => Dict("sequence" => "GG", "count" => 1)),
        ],
        "constraint" => Dict(
            "contact" => Any[
                Dict("residue1" => Any[1, 1, 1], "residue2" => Any[2, 1, 1], "max_distance" => 10.0),
            ],
            "pocket" => Dict(
                "binder_chain" => Any[1, 1],
                "contact_residues" => Any[Any[2, 1, 1]],
                "max_distance" => 12.0,
            ),
            "structure" => Dict("token_indices" => Any[1, 2]),
        ),
    )
    parsed_constraint = PXDesign.ProtenixAPI._parse_task_entities(constraint_task)
    bundle_constraint = PXDesign.Data.build_feature_bundle_from_atoms(parsed_constraint.atoms; task_name = "constraint_smoke")
    feat_constraint = bundle_constraint["input_feature_dict"]
    PXDesign.ProtenixAPI._normalize_protenix_feature_dict!(feat_constraint)
    PXDesign.ProtenixAPI._inject_task_constraint_feature!(
        feat_constraint,
        constraint_task,
        bundle_constraint["atoms"],
        parsed_constraint.entity_chain_ids,
        parsed_constraint.entity_atom_map,
        "constraint_smoke",
    )
    @test haskey(feat_constraint, "constraint_feature")
    cf = feat_constraint["constraint_feature"]
    contact = cf isa NamedTuple ? cf.contact : cf["contact"]
    substructure = cf isa NamedTuple ? cf.substructure : cf["substructure"]
    @test size(contact) == (size(feat_constraint["restype"], 1), size(feat_constraint["restype"], 1), 2)
    @test sum(abs, substructure) == 0f0

    bad_contact_same_chain = Dict{String, Any}(
        "name" => "constraint_bad_contact_same_chain",
        "sequences" => Any[
            Dict("proteinChain" => Dict("sequence" => "AC", "count" => 1)),
        ],
        "constraint" => Dict(
            "contact" => Any[
                Dict("residue1" => Any[1, 1, 1], "residue2" => Any[1, 1, 2], "max_distance" => 10.0),
            ],
        ),
    )
    parsed_bad_contact = PXDesign.ProtenixAPI._parse_task_entities(bad_contact_same_chain)
    bundle_bad_contact = PXDesign.Data.build_feature_bundle_from_atoms(parsed_bad_contact.atoms; task_name = "constraint_bad_contact_same_chain")
    feat_bad_contact = bundle_bad_contact["input_feature_dict"]
    PXDesign.ProtenixAPI._normalize_protenix_feature_dict!(feat_bad_contact)
    @test_throws ErrorException PXDesign.ProtenixAPI._inject_task_constraint_feature!(
        feat_bad_contact,
        bad_contact_same_chain,
        bundle_bad_contact["atoms"],
        parsed_bad_contact.entity_chain_ids,
        parsed_bad_contact.entity_atom_map,
        "constraint_bad_contact_same_chain",
    )

    bad_contact_dist = Dict{String, Any}(
        "name" => "constraint_bad_contact_dist",
        "sequences" => Any[
            Dict("proteinChain" => Dict("sequence" => "AC", "count" => 1)),
            Dict("proteinChain" => Dict("sequence" => "GG", "count" => 1)),
        ],
        "constraint" => Dict(
            "contact" => Any[
                Dict("residue1" => Any[1, 1, 1], "residue2" => Any[2, 1, 1], "min_distance" => 12.0, "max_distance" => 10.0),
            ],
        ),
    )
    parsed_bad_dist = PXDesign.ProtenixAPI._parse_task_entities(bad_contact_dist)
    bundle_bad_dist = PXDesign.Data.build_feature_bundle_from_atoms(parsed_bad_dist.atoms; task_name = "constraint_bad_contact_dist")
    feat_bad_dist = bundle_bad_dist["input_feature_dict"]
    PXDesign.ProtenixAPI._normalize_protenix_feature_dict!(feat_bad_dist)
    @test_throws ErrorException PXDesign.ProtenixAPI._inject_task_constraint_feature!(
        feat_bad_dist,
        bad_contact_dist,
        bundle_bad_dist["atoms"],
        parsed_bad_dist.entity_chain_ids,
        parsed_bad_dist.entity_atom_map,
        "constraint_bad_contact_dist",
    )

    bad_pocket_same_chain = Dict{String, Any}(
        "name" => "constraint_bad_pocket_same_chain",
        "sequences" => Any[
            Dict("proteinChain" => Dict("sequence" => "AC", "count" => 1)),
        ],
        "constraint" => Dict(
            "pocket" => Dict(
                "binder_chain" => Any[1, 1],
                "contact_residues" => Any[Any[1, 1, 2]],
                "max_distance" => 10.0,
            ),
        ),
    )
    parsed_bad_pocket = PXDesign.ProtenixAPI._parse_task_entities(bad_pocket_same_chain)
    bundle_bad_pocket = PXDesign.Data.build_feature_bundle_from_atoms(parsed_bad_pocket.atoms; task_name = "constraint_bad_pocket_same_chain")
    feat_bad_pocket = bundle_bad_pocket["input_feature_dict"]
    PXDesign.ProtenixAPI._normalize_protenix_feature_dict!(feat_bad_pocket)
    @test_throws ErrorException PXDesign.ProtenixAPI._inject_task_constraint_feature!(
        feat_bad_pocket,
        bad_pocket_same_chain,
        bundle_bad_pocket["atoms"],
        parsed_bad_pocket.entity_chain_ids,
        parsed_bad_pocket.entity_atom_map,
        "constraint_bad_pocket_same_chain",
    )
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

    mktempdir() do d
        msa_a = joinpath(d, "msa", "0")
        msa_b = joinpath(d, "msa", "1")
        mkpath(msa_a)
        mkpath(msa_b)
        write(joinpath(msa_a, "pairing.a3m"), ">query\nAC\n>p1\nAA\n>p2\nCC\n")
        write(joinpath(msa_a, "non_pairing.a3m"), ">query\nAC\n>n1\nAG\n")
        write(joinpath(msa_b, "pairing.a3m"), ">query\nGG\n>p1\nTT\n>p2\nAA\n")
        write(joinpath(msa_b, "non_pairing.a3m"), ">query\nGG\n>n1\nGT\n")

        task = Dict{String, Any}(
            "name" => "msa_pair_merge_smoke",
            "sequences" => Any[
                Dict("proteinChain" => Dict("sequence" => "AC", "count" => 1, "msa" => Dict("precomputed_msa_dir" => "msa/0"))),
                Dict("proteinChain" => Dict("sequence" => "GG", "count" => 1, "msa" => Dict("precomputed_msa_dir" => "msa/1"))),
            ],
        )
        input_json = joinpath(d, "input_pair_merge.json")
        PXDesign.JSONLite.write_json(input_json, Any[task])

        parsed = PXDesign.ProtenixAPI._parse_task_entities(task; json_dir = d)
        bundle = PXDesign.Data.build_feature_bundle_from_atoms(parsed.atoms; task_name = "msa_pair_merge_smoke")
        feat = bundle["input_feature_dict"]
        token_chain_ids = [bundle["atoms"][tok.centre_atom_index].chain_id for tok in bundle["tokens"]]
        PXDesign.ProtenixAPI._normalize_protenix_feature_dict!(feat)
        PXDesign.ProtenixAPI._inject_task_msa_features!(
            feat,
            task,
            input_json;
            use_msa = true,
            chain_specs = parsed.protein_specs,
            token_chain_ids = token_chain_ids,
        )

        msa = Int.(feat["msa"])
        a_cols = findall(==("A0"), token_chain_ids)
        b_cols = findall(==("B0"), token_chain_ids)
        q_a = PXDesign.ProtenixAPI._sequence_to_protenix_indices("AC")
        q_b = PXDesign.ProtenixAPI._sequence_to_protenix_indices("GG")
        pair1_a = PXDesign.ProtenixAPI._sequence_to_protenix_indices("AA")
        pair1_b = PXDesign.ProtenixAPI._sequence_to_protenix_indices("TT")
        pair2_a = PXDesign.ProtenixAPI._sequence_to_protenix_indices("CC")
        pair2_b = PXDesign.ProtenixAPI._sequence_to_protenix_indices("AA")
        nonpair_a = PXDesign.ProtenixAPI._sequence_to_protenix_indices("AG")
        nonpair_b = PXDesign.ProtenixAPI._sequence_to_protenix_indices("GT")

        @test size(msa) == (5, 4)
        @test msa[1, a_cols] == q_a
        @test msa[1, b_cols] == q_b
        @test msa[2, a_cols] == pair1_a
        @test msa[2, b_cols] == pair1_b
        @test msa[3, a_cols] == pair2_a
        @test msa[3, b_cols] == pair2_b
        @test msa[4, a_cols] == nonpair_a
        @test msa[4, b_cols] == q_b
        @test msa[5, a_cols] == q_a
        @test msa[5, b_cols] == nonpair_b
    end

    mktempdir() do d
        msa_a = joinpath(d, "msa", "0")
        msa_b = joinpath(d, "msa", "1")
        mkpath(msa_a)
        mkpath(msa_b)
        write(joinpath(msa_a, "pairing.a3m"), ">query\nAC\n>h1 TaxID=111\nAA\n>h2 TaxID=222\nCC\n")
        write(joinpath(msa_a, "non_pairing.a3m"), ">query\nAC\n>n1\nAG\n")
        write(joinpath(msa_b, "pairing.a3m"), ">query\nGG\n>x2 TaxID=222\nTT\n>x1 TaxID=111\nAA\n")
        write(joinpath(msa_b, "non_pairing.a3m"), ">query\nGG\n>n1\nGT\n")

        task = Dict{String, Any}(
            "name" => "msa_pair_merge_taxid",
            "sequences" => Any[
                Dict("proteinChain" => Dict("sequence" => "AC", "count" => 1, "msa" => Dict("precomputed_msa_dir" => "msa/0"))),
                Dict("proteinChain" => Dict("sequence" => "GG", "count" => 1, "msa" => Dict("precomputed_msa_dir" => "msa/1"))),
            ],
        )
        input_json = joinpath(d, "input_pair_merge_taxid.json")
        PXDesign.JSONLite.write_json(input_json, Any[task])

        parsed = PXDesign.ProtenixAPI._parse_task_entities(task; json_dir = d)
        bundle = PXDesign.Data.build_feature_bundle_from_atoms(parsed.atoms; task_name = "msa_pair_merge_taxid")
        feat = bundle["input_feature_dict"]
        token_chain_ids = [bundle["atoms"][tok.centre_atom_index].chain_id for tok in bundle["tokens"]]
        PXDesign.ProtenixAPI._normalize_protenix_feature_dict!(feat)
        PXDesign.ProtenixAPI._inject_task_msa_features!(
            feat,
            task,
            input_json;
            use_msa = true,
            chain_specs = parsed.protein_specs,
            token_chain_ids = token_chain_ids,
        )

        msa = Int.(feat["msa"])
        a_cols = findall(==("A0"), token_chain_ids)
        b_cols = findall(==("B0"), token_chain_ids)
        q_a = PXDesign.ProtenixAPI._sequence_to_protenix_indices("AC")
        q_b = PXDesign.ProtenixAPI._sequence_to_protenix_indices("GG")
        tax111_a = PXDesign.ProtenixAPI._sequence_to_protenix_indices("AA")
        tax111_b = PXDesign.ProtenixAPI._sequence_to_protenix_indices("AA")
        tax222_a = PXDesign.ProtenixAPI._sequence_to_protenix_indices("CC")
        tax222_b = PXDesign.ProtenixAPI._sequence_to_protenix_indices("TT")
        nonpair_a = PXDesign.ProtenixAPI._sequence_to_protenix_indices("AG")
        nonpair_b = PXDesign.ProtenixAPI._sequence_to_protenix_indices("GT")

        @test size(msa) == (5, 4)
        @test msa[1, a_cols] == q_a
        @test msa[1, b_cols] == q_b
        @test msa[2, a_cols] == tax111_a
        @test msa[2, b_cols] == tax111_b
        @test msa[3, a_cols] == tax222_a
        @test msa[3, b_cols] == tax222_b
        @test msa[4, a_cols] == nonpair_a
        @test msa[4, b_cols] == q_b
        @test msa[5, a_cols] == q_a
        @test msa[5, b_cols] == nonpair_b
    end

    mktempdir() do d
        msa_a = joinpath(d, "msa", "0")
        msa_b = joinpath(d, "msa", "1")
        mkpath(msa_a)
        mkpath(msa_b)
        write(joinpath(msa_a, "pairing.a3m"), ">query\nAC\n>a1 OS=Homo sapiens\nAA\n>a2 OS=Mus musculus\nCC\n")
        write(joinpath(msa_a, "non_pairing.a3m"), ">query\nAC\n>n1\nAG\n")
        write(joinpath(msa_b, "pairing.a3m"), ">query\nGG\n>b2 OS=Mus musculus\nTT\n>b1 OS=Homo sapiens\nAA\n")
        write(joinpath(msa_b, "non_pairing.a3m"), ">query\nGG\n>n1\nGT\n")

        task = Dict{String, Any}(
            "name" => "msa_pair_merge_species_name",
            "sequences" => Any[
                Dict("proteinChain" => Dict("sequence" => "AC", "count" => 1, "msa" => Dict("precomputed_msa_dir" => "msa/0"))),
                Dict("proteinChain" => Dict("sequence" => "GG", "count" => 1, "msa" => Dict("precomputed_msa_dir" => "msa/1"))),
            ],
        )
        input_json = joinpath(d, "input_pair_merge_species_name.json")
        PXDesign.JSONLite.write_json(input_json, Any[task])

        parsed = PXDesign.ProtenixAPI._parse_task_entities(task; json_dir = d)
        bundle = PXDesign.Data.build_feature_bundle_from_atoms(parsed.atoms; task_name = "msa_pair_merge_species_name")
        feat = bundle["input_feature_dict"]
        token_chain_ids = [bundle["atoms"][tok.centre_atom_index].chain_id for tok in bundle["tokens"]]
        PXDesign.ProtenixAPI._normalize_protenix_feature_dict!(feat)
        PXDesign.ProtenixAPI._inject_task_msa_features!(
            feat,
            task,
            input_json;
            use_msa = true,
            chain_specs = parsed.protein_specs,
            token_chain_ids = token_chain_ids,
        )

        msa = Int.(feat["msa"])
        a_cols = findall(==("A0"), token_chain_ids)
        b_cols = findall(==("B0"), token_chain_ids)
        q_a = PXDesign.ProtenixAPI._sequence_to_protenix_indices("AC")
        q_b = PXDesign.ProtenixAPI._sequence_to_protenix_indices("GG")
        homo_a = PXDesign.ProtenixAPI._sequence_to_protenix_indices("AA")
        homo_b = PXDesign.ProtenixAPI._sequence_to_protenix_indices("AA")
        mus_a = PXDesign.ProtenixAPI._sequence_to_protenix_indices("CC")
        mus_b = PXDesign.ProtenixAPI._sequence_to_protenix_indices("TT")
        nonpair_a = PXDesign.ProtenixAPI._sequence_to_protenix_indices("AG")
        nonpair_b = PXDesign.ProtenixAPI._sequence_to_protenix_indices("GT")

        @test size(msa) == (5, 4)
        @test msa[1, a_cols] == q_a
        @test msa[1, b_cols] == q_b
        @test msa[2, a_cols] == homo_a
        @test msa[2, b_cols] == homo_b
        @test msa[3, a_cols] == mus_a
        @test msa[3, b_cols] == mus_b
        @test msa[4, a_cols] == nonpair_a
        @test msa[4, b_cols] == q_b
        @test msa[5, a_cols] == q_a
        @test msa[5, b_cols] == nonpair_b
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
    old_ism_repo = get(ENV, "PXDESIGN_ESM_ISM_REPO_ID", nothing)
    old_ism_file = get(ENV, "PXDESIGN_ESM_ISM_FILENAME", nothing)
    old_ism_rev = get(ENV, "PXDESIGN_ESM_ISM_REVISION", nothing)
    old_ism_loader = get(ENV, "PXDESIGN_ESM_ISM_LOADER", nothing)
    old_weights_repo = get(ENV, "PXDESIGN_WEIGHTS_REPO_ID", nothing)
    old_weights_rev = get(ENV, "PXDESIGN_WEIGHTS_REVISION", nothing)
    try
        pop!(ENV, "PXDESIGN_ESM_ISM_REPO_ID", nothing)
        pop!(ENV, "PXDESIGN_ESM_ISM_FILENAME", nothing)
        pop!(ENV, "PXDESIGN_ESM_ISM_REVISION", nothing)
        pop!(ENV, "PXDESIGN_ESM_ISM_LOADER", nothing)
        ENV["PXDESIGN_WEIGHTS_REPO_ID"] = "MurrellLab/PXDesign.jl"
        ENV["PXDESIGN_WEIGHTS_REVISION"] = "main"

        ism_src = PXDesign.ESMProvider._resolve_source(:esm2_3b_ism)
        @test ism_src.repo_id == "MurrellLab/PXDesign.jl"
        @test ism_src.revision == "main"
        @test ism_src.filename ==
              "weights_safetensors_esm2_t36_3B_UR50D_ism/esm2_t36_3B_UR50D_ism.safetensors"
        @test ism_src.loader_kind == :fair_esm2
    finally
        old_ism_repo === nothing ? pop!(ENV, "PXDESIGN_ESM_ISM_REPO_ID", nothing) : (ENV["PXDESIGN_ESM_ISM_REPO_ID"] = old_ism_repo)
        old_ism_file === nothing ? pop!(ENV, "PXDESIGN_ESM_ISM_FILENAME", nothing) : (ENV["PXDESIGN_ESM_ISM_FILENAME"] = old_ism_file)
        old_ism_rev === nothing ? pop!(ENV, "PXDESIGN_ESM_ISM_REVISION", nothing) : (ENV["PXDESIGN_ESM_ISM_REVISION"] = old_ism_rev)
        old_ism_loader === nothing ? pop!(ENV, "PXDESIGN_ESM_ISM_LOADER", nothing) : (ENV["PXDESIGN_ESM_ISM_LOADER"] = old_ism_loader)
        old_weights_repo === nothing ? pop!(ENV, "PXDESIGN_WEIGHTS_REPO_ID", nothing) : (ENV["PXDESIGN_WEIGHTS_REPO_ID"] = old_weights_repo)
        old_weights_rev === nothing ? pop!(ENV, "PXDESIGN_WEIGHTS_REVISION", nothing) : (ENV["PXDESIGN_WEIGHTS_REVISION"] = old_weights_rev)
    end

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

    if PXDesign.ProtenixAPI._default_ccd_components_path() == ""
        @test_skip "Skipping ESM auto-inject with CCD ligand: CCD components file not available."
    else
        auto_task = Dict{String, Any}(
            "name" => "esm_auto_inject",
            "sequences" => Any[
                Dict("proteinChain" => Dict("sequence" => "AC", "count" => 1)),
                Dict("ligand" => Dict("ligand" => "CCD_ATP", "count" => 1)),
            ],
        )
        parsed_auto = PXDesign.ProtenixAPI._parse_task_entities(auto_task)
        auto_bundle = PXDesign.Data.build_feature_bundle_from_atoms(parsed_auto.atoms; task_name = "esm_auto_inject")
        auto_feat = auto_bundle["input_feature_dict"]
        PXDesign.ProtenixAPI._normalize_protenix_feature_dict!(auto_feat)

        PXDesign.ESMProvider.set_sequence_embedder_override!((sequence, variant) -> begin
            n = length(sequence)
            out = zeros(Float32, n, 2)
            bias = variant == :esm2_3b ? 10f0 : 20f0
            for i in 1:n
                out[i, 1] = Float32(i)
                out[i, 2] = bias + Float32(i)
            end
            out
        end)
        try
            PXDesign.ProtenixAPI._inject_auto_esm_token_embedding!(
                auto_feat,
                auto_bundle["atoms"],
                auto_bundle["tokens"],
                Dict("A0" => "AC"),
                params,
                "auto esm test",
            )
            @test haskey(auto_feat, "esm_token_embedding")
            @test size(auto_feat["esm_token_embedding"], 1) == size(auto_feat["restype"], 1)
            @test size(auto_feat["esm_token_embedding"], 2) == 2

            centre_atoms = [auto_bundle["atoms"][tok.centre_atom_index] for tok in auto_bundle["tokens"]]
            for (i, atom) in enumerate(centre_atoms)
                if atom.mol_type == "protein"
                    @test auto_feat["esm_token_embedding"][i, 1] == Float32(atom.res_id)
                else
                    @test all(auto_feat["esm_token_embedding"][i, :] .== 0f0)
                end
            end

            frozen = copy(auto_feat["esm_token_embedding"])
            PXDesign.ProtenixAPI._inject_auto_esm_token_embedding!(
                auto_feat,
                auto_bundle["atoms"],
                auto_bundle["tokens"],
                Dict("A0" => "AC"),
                params,
                "auto esm no-overwrite",
            )
            @test auto_feat["esm_token_embedding"] == frozen
        finally
            PXDesign.ESMProvider.set_sequence_embedder_override!(nothing)
            PXDesign.ESMProvider.clear_esm_cache!()
        end
    end
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

@testset "Protenix mini/base end-to-end smoke" begin
    seq = "ACDE"
    mini_model = PXDesign.ProtenixMini.ProtenixMiniModel(
        32,
        32,
        16,
        8,
        97;
        c_atom = 16,
        c_atompair = 8,
        n_cycle = 1,
        pairformer_blocks = 1,
        msa_blocks = 1,
        diffusion_transformer_blocks = 1,
        diffusion_atom_encoder_blocks = 1,
        diffusion_atom_decoder_blocks = 1,
        confidence_max_atoms_per_token = 20,
        sample_gamma0 = 0.0,
        sample_gamma_min = 1.0,
        sample_noise_scale_lambda = 1.0,
        sample_step_scale_eta = 0.0,
        sample_n_step = 1,
        sample_n_sample = 1,
        rng = MersenneTwister(1234),
    )

    folded_mini = PXDesign.ProtenixMini.fold_sequence(
        mini_model,
        seq;
        n_cycle = 1,
        n_step = 1,
        n_sample = 1,
        rng = MersenneTwister(11),
    )
    @test size(folded_mini.prediction.coordinate) == (1, length(folded_mini.atoms), 3)
    @test all(isfinite, folded_mini.prediction.coordinate)

    mktempdir() do d
        pred_dir = PXDesign.Output.dump_prediction_bundle(
            joinpath(d, "mini"),
            "protenix_mini_e2e",
            folded_mini.atoms,
            folded_mini.prediction.coordinate,
        )
        cif_path = joinpath(pred_dir, "protenix_mini_e2e_sample_0.cif")
        @test isfile(cif_path)
        @test filesize(cif_path) > 0
        cif_text = read(cif_path, String)
        @test occursin("_entity_poly.", cif_text)
        @test occursin("_struct_conn.", cif_text)
    end

    folded_base = PXDesign.ProtenixBase.fold_sequence(
        mini_model,
        seq;
        n_cycle = 1,
        n_step = 1,
        n_sample = 1,
        rng = MersenneTwister(22),
    )
    @test size(folded_base.prediction.coordinate) == (1, length(folded_base.atoms), 3)
    @test all(isfinite, folded_base.prediction.coordinate)

    mktempdir() do d
        pred_dir = PXDesign.Output.dump_prediction_bundle(
            joinpath(d, "base"),
            "protenix_base_e2e",
            folded_base.atoms,
            folded_base.prediction.coordinate,
        )
        cif_path = joinpath(pred_dir, "protenix_base_e2e_sample_0.cif")
        @test isfile(cif_path)
        @test filesize(cif_path) > 0
        cif_text = read(cif_path, String)
        @test occursin("_entity_poly.", cif_text)
        @test occursin("_struct_conn.", cif_text)
    end
end

@testset "Protenix base-constraint end-to-end smoke" begin
    task = Dict{String, Any}(
        "name" => "constraint_e2e",
        "sequences" => Any[
            Dict("proteinChain" => Dict("sequence" => "AC", "count" => 1)),
            Dict("proteinChain" => Dict("sequence" => "GG", "count" => 1)),
        ],
        "constraint" => Dict(
            "contact" => Any[
                Dict("residue1" => Any[1, 1, 1], "residue2" => Any[2, 1, 1], "max_distance" => 10.0),
            ],
            "pocket" => Dict(
                "binder_chain" => Any[1, 1],
                "contact_residues" => Any[Any[2, 1, 1]],
                "max_distance" => 12.0,
            ),
        ),
    )
    parsed = PXDesign.ProtenixAPI._parse_task_entities(task)
    bundle = PXDesign.Data.build_feature_bundle_from_atoms(parsed.atoms; task_name = "constraint_e2e")
    feat = bundle["input_feature_dict"]
    PXDesign.ProtenixAPI._normalize_protenix_feature_dict!(feat)
    PXDesign.ProtenixAPI._inject_task_constraint_feature!(
        feat,
        task,
        bundle["atoms"],
        parsed.entity_chain_ids,
        parsed.entity_atom_map,
        "constraint_e2e",
    )
    typed = PXDesign.ProtenixMini.as_protenix_features(feat)
    @test typed.constraint_feature !== nothing

    model = PXDesign.ProtenixMini.ProtenixMiniModel(
        32,
        32,
        16,
        8,
        97;
        c_atom = 16,
        c_atompair = 8,
        n_cycle = 1,
        pairformer_blocks = 1,
        msa_blocks = 1,
        diffusion_transformer_blocks = 1,
        diffusion_atom_encoder_blocks = 1,
        diffusion_atom_decoder_blocks = 1,
        confidence_max_atoms_per_token = 20,
        sample_gamma0 = 0.0,
        sample_gamma_min = 1.0,
        sample_noise_scale_lambda = 1.0,
        sample_step_scale_eta = 0.0,
        sample_n_step = 1,
        sample_n_sample = 1,
        constraint_enable = true,
        constraint_substructure_enable = true,
        constraint_substructure_architecture = :linear,
        rng = MersenneTwister(222),
    )
    pred = PXDesign.ProtenixBase.run_inference(
        model,
        typed;
        n_cycle = 1,
        n_step = 1,
        n_sample = 1,
        rng = MersenneTwister(333),
    )
    @test size(pred.coordinate) == (1, length(bundle["atoms"]), 3)
    @test all(isfinite, pred.coordinate)

    mktempdir() do d
        pred_dir = PXDesign.Output.dump_prediction_bundle(
            joinpath(d, "constraint"),
            "protenix_constraint_e2e",
            bundle["atoms"],
            pred.coordinate,
        )
        cif_path = joinpath(pred_dir, "protenix_constraint_e2e_sample_0.cif")
        @test isfile(cif_path)
        @test filesize(cif_path) > 0
    end
end

@testset "Protenix typed feature path parity" begin
    bundle = PXDesign.ProtenixMini.build_sequence_feature_bundle(
        "ACDE";
        chain_id = "A0",
        task_name = "typed_feature_smoke",
        rng = MersenneTwister(123),
    )
    feat_dict = bundle["input_feature_dict"]
    feat_typed = PXDesign.ProtenixMini.as_protenix_features(feat_dict)

    @test feat_typed.restype isa Matrix{Float32}
    @test feat_typed.profile isa Matrix{Float32}
    @test feat_typed.token_bonds isa Matrix{Float32}
    @test feat_typed.atom_to_token_idx isa Vector{Int}

    pm = PXDesign.ProtenixMini.ProtenixMiniModel(
        32,
        32,
        16,
        8,
        97;
        c_atom = 16,
        c_atompair = 8,
        n_cycle = 1,
        pairformer_blocks = 1,
        msa_blocks = 1,
        diffusion_transformer_blocks = 1,
        diffusion_atom_encoder_blocks = 1,
        diffusion_atom_decoder_blocks = 1,
        confidence_max_atoms_per_token = 20,
        sample_gamma0 = 0.0,
        sample_gamma_min = 1.0,
        sample_noise_scale_lambda = 1.0,
        sample_step_scale_eta = 0.0,
        sample_n_step = 1,
        sample_n_sample = 1,
        rng = MersenneTwister(456),
    )

    trunk_dict = PXDesign.ProtenixMini.get_pairformer_output(pm, feat_dict; n_cycle = 1, rng = MersenneTwister(789))
    trunk_typed = PXDesign.ProtenixMini.get_pairformer_output(pm, feat_typed; n_cycle = 1, rng = MersenneTwister(789))

    @test isapprox(trunk_typed.s_inputs, trunk_dict.s_inputs; atol = 1e-6, rtol = 1e-6)
    @test isapprox(trunk_typed.s, trunk_dict.s; atol = 1e-6, rtol = 1e-6)
    @test isapprox(trunk_typed.z, trunk_dict.z; atol = 1e-6, rtol = 1e-6)
end

@testset "Protenix constraint embedder plumbing" begin
    ce = PXDesign.ProtenixMini.ConstraintEmbedder(
        8;
        pocket_enable = true,
        contact_enable = true,
        contact_atom_enable = true,
        substructure_enable = true,
        initialize_method = :zero,
        rng = MersenneTwister(1),
    )
    ce.pocket_z_embedder !== nothing && (ce.pocket_z_embedder.weight .= 1f0)
    ce.contact_z_embedder !== nothing && (ce.contact_z_embedder.weight .= 0f0)
    ce.contact_atom_z_embedder !== nothing && (ce.contact_atom_z_embedder.weight .= 0f0)
    if ce.substructure_z_embedder isa PXDesign.ProtenixMini.SubstructureLinearEmbedder
        ce.substructure_z_embedder.proj.weight .= 0f0
    end

    cf = Dict(
        "pocket" => ones(Float32, 3, 3, 1),
        "contact" => zeros(Float32, 3, 3, 2),
        "contact_atom" => zeros(Float32, 3, 3, 2),
        "substructure" => zeros(Float32, 3, 3, 4),
    )
    z = ce(cf)
    @test z !== nothing
    @test size(z) == (3, 3, 8)
    @test all(z .≈ 1f0)

    ce_mlp = PXDesign.ProtenixMini.ConstraintEmbedder(
        8;
        substructure_enable = true,
        substructure_architecture = :mlp,
        substructure_hidden_dim = 16,
        substructure_n_layers = 3,
        initialize_method = :zero,
        rng = MersenneTwister(2),
    )
    z_mlp = ce_mlp(Dict("substructure" => zeros(Float32, 3, 3, 4)))
    @test z_mlp !== nothing
    @test size(z_mlp) == (3, 3, 8)

    ce_tr = PXDesign.ProtenixMini.ConstraintEmbedder(
        8;
        substructure_enable = true,
        substructure_architecture = :transformer,
        substructure_hidden_dim = 8,
        substructure_n_layers = 1,
        substructure_n_heads = 4,
        initialize_method = :zero,
        rng = MersenneTwister(3),
    )
    z_tr = ce_tr(Dict("substructure" => zeros(Float32, 3, 3, 4)))
    @test z_tr !== nothing
    @test size(z_tr) == (3, 3, 8)

    bundle = PXDesign.ProtenixMini.build_sequence_feature_bundle("ACDE"; task_name = "constraint_typed")
    feat = copy(bundle["input_feature_dict"])
    n_tok = size(feat["restype"], 1)
    feat["constraint_feature"] = Dict(
        "contact" => zeros(Float32, n_tok, n_tok, 2),
        "pocket" => zeros(Float32, n_tok, n_tok, 1),
        "contact_atom" => zeros(Float32, n_tok, n_tok, 2),
        "substructure" => zeros(Float32, n_tok, n_tok, 4),
    )
    typed = PXDesign.ProtenixMini.as_protenix_features(feat)
    @test typed.constraint_feature !== nothing
    z_typed = ce(typed.constraint_feature)
    @test z_typed !== nothing
    @test size(z_typed) == (n_tok, n_tok, 8)
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

    io = IOBuffer()
    PXDesign.JSONLite.write_json(
        io,
        (name = "nt", n = 7, ok = true, arr = [1, 2], obj = (k = "v",)),
    )
    encoded = String(take!(io))
    parsed_nt = PXDesign.JSONLite.parse_json(encoded)
    @test parsed_nt["name"] == "nt"
    @test parsed_nt["n"] == 7
    @test parsed_nt["ok"] == true
    @test parsed_nt["arr"][2] == 2
    @test parsed_nt["obj"]["k"] == "v"
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
    if get(ENV, "PXDESIGN_ENABLE_PYTHON_PARITY_TESTS", "0") != "1"
        @test true
        return
    end

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
    @test size(x) == (3, 5, 2)
    @test all(isfinite, x)

    x_chunk = PXDesign.sample_diffusion(
        denoise_net;
        noise_schedule = noise_schedule,
        N_sample = 5,
        N_atom = 4,
        diffusion_chunk_size = 2,
        rng = MersenneTwister(3),
    )
    @test size(x_chunk) == (3, 4, 5)
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
    @test size(rel) == (17, 3, 3)
    # three one-hot blocks plus same_entity bit (0/1) => sum is 3 or 4
    @test all(sum(rel[:, i, j]) in (3f0, 4f0) for i in 1:3 for j in 1:3)
    # cross-entity pair should have same_entity bit = 0
    same_entity_idx = (2 * (2 + 1)) + (2 * (2 + 1)) + 1
    @test rel[same_entity_idx, 1, 3] == 0f0
    @test rel[same_entity_idx, 1, 2] == 1f0

    relpe = PXDesign.Model.RelativePositionEncoding(2, 1, 5)
    raw_relpos = Dict(
        "asym_id" => [1, 1, 2],
        "residue_index" => [10, 11, 5],
        "entity_id" => [1, 1, 2],
        "sym_id" => [1, 1, 1],
        "token_index" => [0, 1, 2],
    )
    z = relpe(raw_relpos)
    @test size(z) == (5, 3, 3)
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
    @test size(ztempl) == (7, 3, 3)
    # masked-out entries use embedding index 1 (idx0=0 -> +1)
    @test ztempl[:, 1, 1] == cte.weight[1, :]
    # masked-in entry with templ=12 uses index 14 (1 + templ then +1 for Julia indexing)
    @test ztempl[:, 1, 2] == cte.weight[14, :]
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
    @test size(s_dce) == (16, 2)
    @test size(z_dce) == (4, 2, 2)
    @test all(isfinite, s_dce)
    @test all(isfinite, z_dce)
end

@testset "Model primitives" begin
    lin = PXDesign.Model.LinearNoBias(3, 4)
    x = rand(Float32, 3, 5)
    y = lin(x)
    @test size(y) == (4, 5)
    @test all(isfinite, y)

    ln = PXDesign.Model.LayerNormFirst(3)
    z = ln(rand(Float32, 3, 7))
    @test size(z) == (3, 7)
    @test all(isfinite, z)
    # weighted LN keeps per-column mean near zero when weight is all ones
    @test all(abs.(vec(mean(z; dims = 1))) .< 1f-3)

    ada = PXDesign.Model.AdaptiveLayerNorm(6, 4)
    a = rand(Float32, 6, 3, 2)
    s = rand(Float32, 4, 3, 2)
    a2 = ada(a, s)
    @test size(a2) == (6, 3, 2)
    @test all(isfinite, a2)

    tr = PXDesign.Model.Transition(6, 12)
    t = tr(rand(Float32, 6, 3, 2))
    @test size(t) == (6, 3, 2)
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
    z_trunk = rand(Float32, 8, 4, 4)
    pair = PXDesign.Model.prepare_pair_cache(cond, relpos_input, z_trunk)
    @test size(pair) == (8, 4, 4)
    @test all(isfinite, pair)

    s_inputs = rand(Float32, 10, 4)
    s_trunk = rand(Float32, 12, 4)
    t_hat = [8.0, 4.0, 2.0]
    single_s, pair2 = cond(t_hat, relpos_input, s_inputs, s_trunk, z_trunk)
    @test size(single_s) == (12, 4, 3)
    @test size(pair2) == (8, 4, 4)
    @test all(isfinite, single_s)
    @test all(isfinite, pair2)
end

@testset "Transformer blocks" begin
    blk = PXDesign.Model.ConditionedTransitionBlock(8, 6; n = 2)
    a = rand(Float32, 8, 4, 2)
    s = rand(Float32, 6, 4, 2)
    out = blk(a, s)
    @test size(out) == (8, 4, 2)
    @test all(isfinite, out)

    attn = PXDesign.Model.AttentionPairBias(8, 6, 4; n_heads = 2)
    a2 = rand(Float32, 8, 5)
    s2 = rand(Float32, 6, 5)
    z2 = rand(Float32, 4, 5, 5)
    attn_out = attn(a2, s2, z2)
    @test size(attn_out) == (8, 5)
    @test all(isfinite, attn_out)

    attn_cross = PXDesign.Model.AttentionPairBias(8, 6, 4; n_heads = 2, cross_attention_mode = true)
    attn_cross_out = attn_cross(a2, s2, z2)
    @test size(attn_cross_out) == (8, 5)
    @test all(isfinite, attn_cross_out)
    attn_local_out = attn_cross(a2, s2, z2; n_queries = 2, n_keys = 4)
    @test size(attn_local_out) == (8, 5)
    @test all(isfinite, attn_local_out)

    dblk = PXDesign.Model.DiffusionTransformerBlock(8, 6, 4; n_heads = 2)
    mask2 = ones(Float32, 5, 1)
    a3 = dblk(a2, s2, z2, mask2)
    @test size(a3) == (8, 5, 1)  # mask (N, 1) → batch=1 output
    @test all(isfinite, a3)

    dtr = PXDesign.Model.DiffusionTransformer(8, 6, 4; n_blocks = 3, n_heads = 2)
    a4 = dtr(a2, s2, z2)
    @test size(a4) == (8, 5)
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
    @test size(a0) == (8, 2)
    @test size(q0) == (16, 4)
    @test size(c0) == (16, 4)
    @test size(p0) == (4, 2, 4, 2)
    @test all(isfinite, a0)
    atom_input = PXDesign.Model.as_atom_attention_input(feat)
    a0_nt, q0_nt, c0_nt, p0_nt = enc0(atom_input)
    @test size(a0_nt) == (8, 2)
    @test size(q0_nt) == (16, 4)
    @test size(c0_nt) == (16, 4)
    @test size(p0_nt) == (4, 2, 4, 2)
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
    r = rand(Float32, 3, 4, 2)
    s = rand(Float32, 6, 2, 2)
    z = rand(Float32, 4, 2, 2, 2)
    a1, q1, c1, p1 = enc1(feat; r_l = r, s = s, z = z)
    @test size(a1) == (8, 2, 2)
    @test size(q1) == (16, 4, 2)
    @test size(c1) == (16, 4, 2)
    @test size(p1) == (4, 2, 4, 2, 2)
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
    @test size(r_update) == (3, 4, 2)
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
    s_inputs = rand(Float32, 5, 3)
    s_trunk = rand(Float32, 6, 3)
    z_trunk = rand(Float32, 4, 3, 3)
    atom_to_token_idx = [0, 0, 1, 1, 2]
    x_noisy = randn(Float32, 3, 5, 2)
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
        size(dm.diffusion_conditioning.layernorm_z.w),
    )
    w["diffusion_module.diffusion_conditioning.linear_no_bias_z.weight"] = fill(
        5f0,
        size(dm.diffusion_conditioning.linear_no_bias_z.weight),
    )
    w["diffusion_module.layernorm_s.weight"] = fill(6f0, size(dm.layernorm_s.w))
    w["diffusion_module.linear_no_bias_s.weight"] = fill(7f0, size(dm.linear_no_bias_s.weight))
    w["diffusion_module.layernorm_a.weight"] = fill(8f0, size(dm.layernorm_a.w))
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
    @test all(dm.diffusion_conditioning.layernorm_z.w .== 4f0)
    @test all(dm.diffusion_conditioning.linear_no_bias_z.weight .== 5f0)
    @test all(dm.layernorm_s.w .== 6f0)
    @test all(dm.linear_no_bias_s.weight .== 7f0)
    @test all(dm.layernorm_a.w .== 8f0)
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
        cfg2["model_scaffold"]["auto_dims_from_weights"] = true

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

        # Local raw/safetensors paths are intentionally disabled in runtime.
        cfg3 = PXDesign.Config.default_config(project_root = d)
        cfg3["input_json_path"] = input_json
        cfg3["dump_dir"] = joinpath(d, "out_model_scaffold_raw_blocked")
        cfg3["download_cache"] = false
        cfg3["seeds"] = [9]
        cfg3["model_scaffold"]["enabled"] = true
        cfg3["model_scaffold"]["auto_dims_from_weights"] = true
        cfg3["raw_weights_dir"] = joinpath(d, "rw_disabled")
        cfg3["strict_weight_load"] = false
        @test_throws ErrorException PXDesign.Infer.run_infer(cfg3; dry_run = false, io = devnull)

        cfg4 = PXDesign.Config.default_config(project_root = d)
        cfg4["input_json_path"] = input_json
        cfg4["dump_dir"] = joinpath(d, "out_model_scaffold_st_blocked")
        cfg4["download_cache"] = false
        cfg4["seeds"] = [13]
        cfg4["model_scaffold"]["enabled"] = true
        cfg4["model_scaffold"]["auto_dims_from_weights"] = true
        cfg4["safetensors_weights_path"] = joinpath(d, "weights_disabled.safetensors")
        @test_throws ErrorException PXDesign.Infer.run_infer(cfg4; dry_run = false, io = devnull)
    end
end
