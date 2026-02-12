module Cache

using Downloads

import ..Config: default_urls

export ensure_inference_cache!

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

function ensure_inference_cache!(
    cfg::Dict{String, Any};
    urls::Dict{String, String} = default_urls(),
    include_protenix_checkpoints::Bool = get(cfg, "include_protenix_checkpoints", false),
    dry_run::Bool = false,
    io::IO = stdout,
)
    data_cfg = cfg["data"]
    data_cfg isa AbstractDict || error("cfg[\"data\"] must be a dictionary.")

    for cache_name in ("ccd_components_file", "ccd_components_rdkit_mol_file", "pdb_cluster_file")
        cache_path = String(data_cfg[cache_name])
        url = get(urls, cache_name, nothing)
        url === nothing && error("Missing URL for $cache_name")
        maybe_download(url, cache_path; dry_run = dry_run, io = io)
    end

    checkpoint_dir = String(cfg["load_checkpoint_dir"])
    model_name = String(cfg["model_name"])
    main_ckpt = joinpath(checkpoint_dir, "$model_name.pt")
    maybe_download(urls[model_name], main_ckpt; dry_run = dry_run, io = io)

    if include_protenix_checkpoints
        for model in (
            "protenix_base_default_v0.5.0",
            "protenix_base_constraint_v0.5.0",
            "protenix_mini_default_v0.5.0",
            "protenix_mini_tmpl_v0.5.0",
        )
            target = joinpath(checkpoint_dir, "$model.pt")
            maybe_download(urls[model], target; dry_run = dry_run, io = io)
        end
    end

    return cfg
end

end
