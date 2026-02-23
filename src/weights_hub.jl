module WeightsHub

using HuggingFaceApi: hf_hub_download
import ..JSONLite: parse_json

export resolve_weight_source, download_model_weights

const _DEFAULT_REPO_ID = "MurrellLab/PXDesign.jl"
const _DEFAULT_REVISION = "main"

const _WEIGHT_LAYOUTS = Dict{String, NamedTuple{(:layout, :filename, :index_filename), Tuple{Symbol, String, String}}}(
    "pxdesign_v0.1.0" => (layout = :single, filename = "weights_safetensors_PXDesign/PGXDesign.safetensors", index_filename = ""),
    "protenix_base_default_v0.5.0" => (
        layout = :sharded,
        filename = "",
        index_filename = "weights_safetensors_protenix_base_default_v0.5.0/protenix_base_default_v0.5.0.safetensors.index.json",
    ),
    "protenix_base_constraint_v0.5.0" => (
        layout = :sharded,
        filename = "",
        index_filename = "weights_safetensors_protenix_base_constraint_v0.5.0/protenix_base_constraint_v0.5.0.safetensors.index.json",
    ),
    "protenix_mini_default_v0.5.0" => (
        layout = :single,
        filename = "weights_safetensors_protenix_mini_default_v0.5.0/protenix_mini_default_v0.5.0.safetensors",
        index_filename = "",
    ),
    "protenix_mini_tmpl_v0.5.0" => (
        layout = :single,
        filename = "weights_safetensors_protenix_mini_tmpl_v0.5.0/protenix_mini_tmpl_v0.5.0.safetensors",
        index_filename = "",
    ),
    "protenix_mini_default" => (
        layout = :single,
        filename = "weights_safetensors_protenix_mini_default/protenix_mini_default.safetensors",
        index_filename = "",
    ),
    "protenix_mini_tmpl" => (
        layout = :single,
        filename = "weights_safetensors_protenix_mini_tmpl/protenix_mini_tmpl.safetensors",
        index_filename = "",
    ),
    "protenix_tiny_default_v0.5.0" => (
        layout = :single,
        filename = "weights_safetensors_protenix_tiny_default_v0.5.0/protenix_tiny_default_v0.5.0.safetensors",
        index_filename = "",
    ),
    "protenix_mini_esm_v0.5.0" => (
        layout = :single,
        filename = "weights_safetensors_protenix_mini_esm_v0.5.0/protenix_mini_esm_v0.5.0.safetensors",
        index_filename = "",
    ),
    "protenix_mini_ism_v0.5.0" => (
        layout = :single,
        filename = "weights_safetensors_protenix_mini_ism_v0.5.0/protenix_mini_ism_v0.5.0.safetensors",
        index_filename = "",
    ),
    "protenix_base_default_v1.0.0" => (
        layout = :sharded,
        filename = "",
        index_filename = "weights_safetensors_protenix_base_default_v1.0.0/protenix_base_default_v1.0.0.safetensors.index.json",
    ),
    "protenix_base_20250630_v1.0.0" => (
        layout = :sharded,
        filename = "",
        index_filename = "weights_safetensors_protenix_base_20250630_v1.0.0/protenix_base_20250630_v1.0.0.safetensors.index.json",
    ),
)

function _env_bool(key::AbstractString, default::Bool = false)
    raw = get(ENV, String(key), "")
    isempty(raw) && return default
    s = lowercase(strip(raw))
    return s in ("1", "true", "yes", "y", "on")
end

function _normalized_model_key(model_name::AbstractString)
    return lowercase(strip(String(model_name)))
end

function _lookup_model_layout(model_name::AbstractString)
    key = _normalized_model_key(model_name)
    haskey(_WEIGHT_LAYOUTS, key) && return _WEIGHT_LAYOUTS[key]
    supported = join(sort!(collect(keys(_WEIGHT_LAYOUTS))), ", ")
    error(
        "No HuggingFace weights mapping found for model '$model_name'. " *
        "Supported keys: $supported",
    )
end

function _repo_join(dir::AbstractString, rel::AbstractString)
    isempty(dir) && return String(rel)
    startswith(String(rel), String(dir) * "/") && return String(rel)
    return String(dir) * "/" * String(rel)
end

function _download_one(
    repo_id::AbstractString,
    filename::AbstractString,
    revision::AbstractString,
    local_files_only::Bool;
    cache::Bool = true,
)
    return hf_hub_download(
        repo_id,
        filename;
        revision = revision,
        cache = cache,
        local_files_only = local_files_only,
    )
end

function _download_sharded(
    repo_id::AbstractString,
    index_filename::AbstractString,
    revision::AbstractString,
    local_files_only::Bool;
    cache::Bool = true,
)
    index_path = _download_one(
        repo_id,
        index_filename,
        revision,
        local_files_only;
        cache = cache,
    )
    parsed = parse_json(read(index_path, String))
    parsed isa AbstractDict || error("Invalid safetensors index JSON at '$index_filename'.")
    weight_map = get(parsed, "weight_map", nothing)
    weight_map isa AbstractDict || error("Missing `weight_map` in safetensors index '$index_filename'.")
    repo_dir = dirname(String(index_filename))
    if repo_dir == "."
        repo_dir = ""
    end
    shard_names = sort!(unique(String(v) for v in values(weight_map)))
    # Derive the model prefix from the index filename to fix shard name mismatches.
    # Some index JSONs reference shards as "model-NNNNN-of-MMMMM.safetensors" while
    # the actual HF files are named "{model_name}-NNNNN-of-MMMMM.safetensors".
    idx_base = basename(String(index_filename))
    model_prefix = replace(idx_base, ".safetensors.index.json" => "")

    # HuggingFaceApi caches each file in a content-addressed hash directory, so shard
    # files end up in separate directories.  Stage symlinks into a single directory so
    # that load_safetensors_weights can discover them.
    stage_dir = dirname(index_path) * "_shards"
    mkpath(stage_dir)

    for shard in shard_names
        actual_shard = shard
        if startswith(shard, "model-") && !isempty(model_prefix)
            actual_shard = replace(shard, "model-" => model_prefix * "-"; count = 1)
        end
        shard_remote = _repo_join(repo_dir, actual_shard)
        cached_path = _download_one(repo_id, shard_remote, revision, local_files_only; cache = cache)
        link_name = basename(actual_shard)
        link_path = joinpath(stage_dir, link_name)
        if !isfile(link_path)
            symlink(cached_path, link_path)
        end
    end
    return stage_dir
end

function resolve_weight_source(model_name::AbstractString)
    layout_spec = _lookup_model_layout(model_name)
    repo_id = get(ENV, "PXDESIGN_WEIGHTS_REPO_ID", _DEFAULT_REPO_ID)
    revision = get(ENV, "PXDESIGN_WEIGHTS_REVISION", _DEFAULT_REVISION)
    local_files_only = _env_bool("PXDESIGN_WEIGHTS_LOCAL_FILES_ONLY", false)
    return (
        model_name = _normalized_model_key(model_name),
        layout = layout_spec.layout,
        filename = layout_spec.filename,
        index_filename = layout_spec.index_filename,
        repo_id = repo_id,
        revision = revision,
        local_files_only = local_files_only,
    )
end

function download_model_weights(model_name::AbstractString; cache::Bool = true)
    src = resolve_weight_source(model_name)
    if src.layout == :single
        return _download_one(
            src.repo_id,
            src.filename,
            src.revision,
            src.local_files_only;
            cache = cache,
        )
    elseif src.layout == :sharded
        return _download_sharded(
            src.repo_id,
            src.index_filename,
            src.revision,
            src.local_files_only;
            cache = cache,
        )
    end
    error("Unsupported safetensors layout for model '$(src.model_name)': $(src.layout)")
end

end # module WeightsHub
