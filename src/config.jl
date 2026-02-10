module Config

export default_config, default_urls, set_by_key!, set_nested!, parse_override_value

const ALIASES = Dict(
    "N_sample" => "sample_diffusion.N_sample",
    "N_step" => "sample_diffusion.N_step",
    "eta_type" => "sample_diffusion.eta_schedule.type",
    "eta_min" => "sample_diffusion.eta_schedule.min",
    "eta_max" => "sample_diffusion.eta_schedule.max",
    "gamma0" => "sample_diffusion.gamma0",
    "gamma_min" => "sample_diffusion.gamma_min",
    "sample_diffusion_chunk_size" => "infer_setting.sample_diffusion_chunk_size",
)

function default_urls()
    return Dict(
        "pxdesign_v0.1.0" => "https://pxdesign.tos-cn-beijing.volces.com/release_model/pxdesign_v0.1.0.pt",
        "protenix_base_default_v0.5.0" => "https://pxdesign.tos-cn-beijing.volces.com/release_model/protenix_base_default_v0.5.0.pt",
        "protenix_mini_default_v0.5.0" => "https://pxdesign.tos-cn-beijing.volces.com/release_model/protenix_mini_default_v0.5.0.pt",
        "protenix_mini_tmpl_v0.5.0" => "https://pxdesign.tos-cn-beijing.volces.com/release_model/protenix_mini_tmpl_v0.5.0.pt",
        "ccd_components_file" => "https://pxdesign.tos-cn-beijing.volces.com/release_data/components.v20240608.cif",
        "ccd_components_rdkit_mol_file" => "https://pxdesign.tos-cn-beijing.volces.com/release_data/components.v20240608.cif.rdkit_mol.pkl",
        "pdb_cluster_file" => "https://pxdesign.tos-cn-beijing.volces.com/release_data/clusters-by-entity-40.txt",
    )
end

function resolve_data_root(project_root::AbstractString)
    if haskey(ENV, "PROTENIX_DATA_ROOT_DIR")
        return ENV["PROTENIX_DATA_ROOT_DIR"]
    end
    return joinpath(project_root, "release_data", "ccd_cache")
end

function resolve_ccd_paths(data_root::AbstractString)
    components = joinpath(data_root, "components.cif")
    rdkit = joinpath(data_root, "components.cif.rdkit_mol.pkl")
    if isfile(components) && isfile(rdkit)
        return components, rdkit
    end
    return (
        joinpath(data_root, "components.v20240608.cif"),
        joinpath(data_root, "components.v20240608.cif.rdkit_mol.pkl"),
    )
end

function default_config(; project_root::AbstractString = pwd())
    data_root = resolve_data_root(project_root)
    ccd_components_file, ccd_rdkit_file = resolve_ccd_paths(data_root)
    return Dict{String, Any}(
        "model_name" => "pxdesign_v0.1.0",
        "dump_dir" => joinpath(project_root, "output"),
        "input_json_path" => "",
        "load_checkpoint_dir" => joinpath(project_root, "release_data", "checkpoint"),
        "num_workers" => 16,
        "dtype" => "bf16",
        "seeds" => Int[],
        "use_msa" => true,
        "download_cache" => false,
        "raw_weights_dir" => "",
        "strict_weight_load" => false,
        "use_fast_ln" => true,
        "use_deepspeed_evo_attention" => false,
        "include_protenix_checkpoints" => false,
        "data" => Dict(
            "ccd_components_file" => ccd_components_file,
            "ccd_components_rdkit_mol_file" => ccd_rdkit_file,
            "pdb_cluster_file" => joinpath(data_root, "clusters-by-entity-40.txt"),
        ),
        "infer_setting" => Dict(
            "sample_diffusion_chunk_size" => 10,
        ),
        "sample_diffusion" => Dict(
            "gamma0" => 1.0,
            "gamma_min" => 0.01,
            "noise_scale_lambda" => 1.003,
            "N_step" => 400,
            "N_sample" => 100,
            "eta_schedule" => Dict(
                "type" => "piecewise_65",
                "min" => 1.0,
                "max" => 2.5,
            ),
        ),
        "inference_noise_scheduler" => Dict(
            "s_max" => 160.0,
            "s_min" => 4e-4,
            "rho" => 7.0,
            "sigma_data" => 16.0,
        ),
        "model_scaffold" => Dict(
            "enabled" => false,
            "auto_dims_from_weights" => false,
            "use_design_condition_embedder" => true,
            "c_atom" => 128,
            "c_atompair" => 16,
            "c_token" => 64,
            "c_s" => 64,
            "c_z" => 32,
            "c_s_inputs" => 128,
            "atom_encoder_blocks" => 3,
            "atom_encoder_heads" => 4,
            "n_blocks" => 2,
            "n_heads" => 4,
            "atom_decoder_blocks" => 3,
            "atom_decoder_heads" => 4,
        ),
    )
end

function parse_override_value(raw::AbstractString)
    s = strip(raw)
    lower = lowercase(s)
    if lower == "true"
        return true
    elseif lower == "false"
        return false
    elseif lower == "null" || lower == "none"
        return nothing
    end

    v_int = tryparse(Int, s)
    if v_int !== nothing
        return v_int
    end

    v_float = tryparse(Float64, s)
    if v_float !== nothing
        return v_float
    end

    return s
end

function set_nested!(cfg::Dict{String, Any}, dotted_key::AbstractString, value)
    parts = split(dotted_key, '.')
    isempty(parts) && error("Empty override key is invalid.")

    cursor = cfg
    for part in parts[1:end-1]
        if !haskey(cursor, part) || !(cursor[part] isa Dict{String, Any})
            cursor[part] = Dict{String, Any}()
        end
        cursor = cursor[part]
    end
    cursor[parts[end]] = value
    return cfg
end

function set_by_key!(cfg::Dict{String, Any}, raw_key::AbstractString, raw_value)
    mapped = get(ALIASES, raw_key, raw_key)
    value = raw_value isa AbstractString ? parse_override_value(raw_value) : raw_value
    return set_nested!(cfg, mapped, value)
end

end
