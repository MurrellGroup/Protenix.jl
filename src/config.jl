module Config

export default_config, default_urls, set_by_key!, set_nested!, parse_override_value

struct DataConfigPaths
    ccd_components_file::String
    ccd_components_rdkit_mol_file::String
    pdb_cluster_file::String
end

struct InferSettingConfig
    sample_diffusion_chunk_size::Int
end

struct EtaScheduleConfig
    schedule_type::String
    min::Float64
    max::Float64
end

struct SampleDiffusionConfig
    gamma0::Float64
    gamma_min::Float64
    noise_scale_lambda::Float64
    n_step::Int
    n_sample::Int
    eta_schedule::EtaScheduleConfig
end

struct InferenceNoiseSchedulerConfig
    s_max::Float64
    s_min::Float64
    rho::Float64
    sigma_data::Float64
end

struct ModelScaffoldConfig
    enabled::Bool
    auto_dims_from_weights::Bool
    use_design_condition_embedder::Bool
    c_atom::Int
    c_atompair::Int
    c_token::Int
    c_s::Int
    c_z::Int
    c_s_inputs::Int
    atom_encoder_blocks::Int
    atom_encoder_heads::Int
    n_blocks::Int
    n_heads::Int
    atom_decoder_blocks::Int
    atom_decoder_heads::Int
end

struct DefaultConfigSpec
    model_name::String
    dump_dir::String
    input_json_path::String
    load_checkpoint_dir::String
    num_workers::Int
    dtype::String
    seeds::Vector{Int}
    use_msa::Bool
    download_cache::Bool
    raw_weights_dir::String
    safetensors_weights_path::String
    strict_weight_load::Bool
    use_fast_ln::Bool
    use_deepspeed_evo_attention::Bool
    include_protenix_checkpoints::Bool
    data::DataConfigPaths
    infer_setting::InferSettingConfig
    sample_diffusion::SampleDiffusionConfig
    inference_noise_scheduler::InferenceNoiseSchedulerConfig
    model_scaffold::ModelScaffoldConfig
end

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
        "protenix_base_constraint_v0.5.0" => "https://af3-dev.tos-cn-beijing.volces.com/release_model/protenix_base_constraint_v0.5.0.pt",
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

function _as_dict(cfg::DataConfigPaths)
    return Dict{String, Any}(
        "ccd_components_file" => cfg.ccd_components_file,
        "ccd_components_rdkit_mol_file" => cfg.ccd_components_rdkit_mol_file,
        "pdb_cluster_file" => cfg.pdb_cluster_file,
    )
end

function _as_dict(cfg::InferSettingConfig)
    return Dict{String, Any}(
        "sample_diffusion_chunk_size" => cfg.sample_diffusion_chunk_size,
    )
end

function _as_dict(cfg::EtaScheduleConfig)
    return Dict{String, Any}(
        "type" => cfg.schedule_type,
        "min" => cfg.min,
        "max" => cfg.max,
    )
end

function _as_dict(cfg::SampleDiffusionConfig)
    return Dict{String, Any}(
        "gamma0" => cfg.gamma0,
        "gamma_min" => cfg.gamma_min,
        "noise_scale_lambda" => cfg.noise_scale_lambda,
        "N_step" => cfg.n_step,
        "N_sample" => cfg.n_sample,
        "eta_schedule" => _as_dict(cfg.eta_schedule),
    )
end

function _as_dict(cfg::InferenceNoiseSchedulerConfig)
    return Dict{String, Any}(
        "s_max" => cfg.s_max,
        "s_min" => cfg.s_min,
        "rho" => cfg.rho,
        "sigma_data" => cfg.sigma_data,
    )
end

function _as_dict(cfg::ModelScaffoldConfig)
    return Dict{String, Any}(
        "enabled" => cfg.enabled,
        "auto_dims_from_weights" => cfg.auto_dims_from_weights,
        "use_design_condition_embedder" => cfg.use_design_condition_embedder,
        "c_atom" => cfg.c_atom,
        "c_atompair" => cfg.c_atompair,
        "c_token" => cfg.c_token,
        "c_s" => cfg.c_s,
        "c_z" => cfg.c_z,
        "c_s_inputs" => cfg.c_s_inputs,
        "atom_encoder_blocks" => cfg.atom_encoder_blocks,
        "atom_encoder_heads" => cfg.atom_encoder_heads,
        "n_blocks" => cfg.n_blocks,
        "n_heads" => cfg.n_heads,
        "atom_decoder_blocks" => cfg.atom_decoder_blocks,
        "atom_decoder_heads" => cfg.atom_decoder_heads,
    )
end

function _as_dict(cfg::DefaultConfigSpec)
    return Dict{String, Any}(
        "model_name" => cfg.model_name,
        "dump_dir" => cfg.dump_dir,
        "input_json_path" => cfg.input_json_path,
        "load_checkpoint_dir" => cfg.load_checkpoint_dir,
        "num_workers" => cfg.num_workers,
        "dtype" => cfg.dtype,
        "seeds" => copy(cfg.seeds),
        "use_msa" => cfg.use_msa,
        "download_cache" => cfg.download_cache,
        "raw_weights_dir" => cfg.raw_weights_dir,
        "safetensors_weights_path" => cfg.safetensors_weights_path,
        "strict_weight_load" => cfg.strict_weight_load,
        "use_fast_ln" => cfg.use_fast_ln,
        "use_deepspeed_evo_attention" => cfg.use_deepspeed_evo_attention,
        "include_protenix_checkpoints" => cfg.include_protenix_checkpoints,
        "data" => _as_dict(cfg.data),
        "infer_setting" => _as_dict(cfg.infer_setting),
        "sample_diffusion" => _as_dict(cfg.sample_diffusion),
        "inference_noise_scheduler" => _as_dict(cfg.inference_noise_scheduler),
        "model_scaffold" => _as_dict(cfg.model_scaffold),
    )
end

function _default_config_spec(project_root::AbstractString)
    data_root = resolve_data_root(project_root)
    ccd_components_file, ccd_rdkit_file = resolve_ccd_paths(data_root)
    data_cfg = DataConfigPaths(
        ccd_components_file,
        ccd_rdkit_file,
        joinpath(data_root, "clusters-by-entity-40.txt"),
    )
    infer_setting = InferSettingConfig(1)
    eta_schedule = EtaScheduleConfig("piecewise_65", 1.0, 2.5)
    sample_diffusion = SampleDiffusionConfig(1.0, 0.01, 1.003, 400, 100, eta_schedule)
    noise_scheduler = InferenceNoiseSchedulerConfig(160.0, 4e-4, 7.0, 16.0)
    model_scaffold = ModelScaffoldConfig(
        true,
        true,
        true,
        128,
        16,
        64,
        64,
        32,
        128,
        3,
        4,
        2,
        4,
        3,
        4,
    )
    return DefaultConfigSpec(
        "pxdesign_v0.1.0",
        joinpath(project_root, "output"),
        "",
        joinpath(project_root, "release_data", "checkpoint"),
        16,
        "bf16",
        Int[],
        true,
        false,
        "",
        "",
        true,
        true,
        false,
        false,
        data_cfg,
        infer_setting,
        sample_diffusion,
        noise_scheduler,
        model_scaffold,
    )
end

function default_config(; project_root::AbstractString = pwd())
    return _as_dict(_default_config_spec(project_root))
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

function _to_string_any_dict(x)::Dict{String, Any}
    x isa AbstractDict || error("Expected dictionary-like override subtree, got $(typeof(x)).")
    out = Dict{String, Any}()
    for (k, v) in x
        out[String(k)] = v
    end
    return out
end

function set_nested!(cfg::Dict{String, Any}, dotted_key::AbstractString, value)
    parts = split(dotted_key, '.')
    isempty(parts) && error("Empty override key is invalid.")

    cursor = cfg
    for part in parts[1:end-1]
        if !haskey(cursor, part)
            child = Dict{String, Any}()
            cursor[part] = child
            cursor = child
            continue
        end
        child_any = cursor[part]
        if child_any isa Dict{String, Any}
            cursor = child_any
        elseif child_any isa AbstractDict
            # Preserve existing nested keys even when the dict is not concretely Dict{String,Any}.
            child = _to_string_any_dict(child_any)
            cursor[part] = child
            cursor = child
        else
            child = Dict{String, Any}()
            cursor[part] = child
            cursor = child
        end
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
