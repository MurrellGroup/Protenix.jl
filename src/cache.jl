module Cache

using Downloads

import ..Config: default_urls

export ensure_inference_cache!

struct InferenceCacheOptions
    ccd_components_file::String
    ccd_components_rdkit_mol_file::String
    pdb_cluster_file::String
    checkpoint_dir::String
    model_name::String
    include_protenix_checkpoints::Bool
end

function ensure_parent_dir(path::AbstractString)
    mkpath(dirname(path))
end

function maybe_download(url::AbstractString, target::AbstractString; dry_run::Bool = false, io::IO = stdout)
    if isfile(target)
        sz = filesize(target)
        if sz > 0
            println(io, "[cache] hit: $(abspath(target))")
            return target
        end
        println(io, "[cache] stale zero-byte file, re-downloading: $(abspath(target))")
        dry_run || rm(target; force = true)
    end

    ensure_parent_dir(target)
    println(io, "[cache] download: $url")
    println(io, "[cache]   -> $(abspath(target))")
    if dry_run
        return target
    end

    Downloads.download(url, target)
    return target
end

function InferenceCacheOptions(cfg::Dict{String, Any}; include_protenix_checkpoints::Bool)
    data_cfg_raw = get(cfg, "data", nothing)
    data_cfg_raw isa AbstractDict || error("cfg[\"data\"] must be a dictionary.")
    data_cfg = Dict{String, Any}(String(k) => v for (k, v) in data_cfg_raw)
    return InferenceCacheOptions(
        String(data_cfg["ccd_components_file"]),
        String(data_cfg["ccd_components_rdkit_mol_file"]),
        String(data_cfg["pdb_cluster_file"]),
        String(cfg["load_checkpoint_dir"]),
        String(cfg["model_name"]),
        include_protenix_checkpoints,
    )
end

function _ensure_inference_cache!(
    opts::InferenceCacheOptions;
    urls::Dict{String, String},
    dry_run::Bool,
    io::IO,
)
    for (cache_name, cache_path) in (
        ("ccd_components_file", opts.ccd_components_file),
        ("ccd_components_rdkit_mol_file", opts.ccd_components_rdkit_mol_file),
        ("pdb_cluster_file", opts.pdb_cluster_file),
    )
        url = get(urls, cache_name, nothing)
        url === nothing && error("Missing URL for $cache_name")
        maybe_download(url, cache_path; dry_run = dry_run, io = io)
    end

    main_ckpt = joinpath(opts.checkpoint_dir, "$(opts.model_name).pt")
    maybe_download(urls[opts.model_name], main_ckpt; dry_run = dry_run, io = io)

    if opts.include_protenix_checkpoints
        for model in (
            "protenix_base_default_v0.5.0",
            "protenix_base_constraint_v0.5.0",
            "protenix_mini_default_v0.5.0",
            "protenix_mini_tmpl_v0.5.0",
        )
            target = joinpath(opts.checkpoint_dir, "$model.pt")
            maybe_download(urls[model], target; dry_run = dry_run, io = io)
        end
    end
end

function ensure_inference_cache!(
    cfg::Dict{String, Any};
    urls::Dict{String, String} = default_urls(),
    include_protenix_checkpoints::Bool = get(cfg, "include_protenix_checkpoints", false),
    dry_run::Bool = false,
    io::IO = stdout,
)
    opts = InferenceCacheOptions(cfg; include_protenix_checkpoints = include_protenix_checkpoints)
    _ensure_inference_cache!(opts; urls = urls, dry_run = dry_run, io = io)

    return cfg
end

end
