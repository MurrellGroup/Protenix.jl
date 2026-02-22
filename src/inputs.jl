module Inputs

using YAML

import ..JSONLite: parse_json, write_json

export parse_yaml_to_json, load_input_tasks, process_input_file, snapshot_input, load_yaml_config

function _as_string_dict(x)
    if x isa AbstractDict
        out = Dict{String, Any}()
        for (k, v) in x
            out[String(k)] = _as_string_dict(v)
        end
        return out
    elseif x isa AbstractVector
        return [_as_string_dict(v) for v in x]
    else
        return x
    end
end

function load_yaml_config(yaml_path::AbstractString)
    yaml_abspath = abspath(yaml_path)
    isfile(yaml_abspath) || error("YAML config file not found: $yaml_abspath")
    parsed = YAML.load_file(yaml_abspath)
    parsed isa AbstractDict || error("YAML top-level must be a mapping/object.")
    return _as_string_dict(parsed)
end

function _normalize_chain_props(props)
    if props === nothing
        return Dict{String, Any}()
    elseif props isa AbstractString
        t = lowercase(strip(props))
        t in ("all", "full") && return Dict{String, Any}()
        error("Invalid chain config string '$props': expected 'all'/'full' or a mapping")
    elseif props isa AbstractDict
        return _as_string_dict(props)
    end
    error("Invalid chain config value: expected mapping/null/'all', got $(typeof(props))")
end

function _parse_yaml_cfg_to_tasks(cfg::Dict{String, Any}, yaml_path::AbstractString)
    default_name = splitext(basename(yaml_path))[1]
    task_name = haskey(cfg, "task_name") ? String(cfg["task_name"]) : default_name

    haskey(cfg, "binder_length") || error("Missing required field: 'binder_length'")
    binder_length = Int(cfg["binder_length"])

    target_cfg = get(cfg, "target", nothing)
    if target_cfg === nothing
        return Any[
            Dict{String, Any}(
                "name" => task_name,
                "generation" => Any[
                    Dict{String, Any}(
                        "type" => "protein",
                        "length" => binder_length,
                        "count" => 1,
                    ),
                ],
            ),
        ]
    end
    target_cfg isa AbstractDict || error("Field 'target' must be a mapping")
    haskey(target_cfg, "file") || error("Missing required field: 'target.file'")

    target_file_raw = String(target_cfg["file"])
    target_file_path = isabspath(target_file_raw) ? target_file_raw : normpath(joinpath(dirname(abspath(yaml_path)), target_file_raw))
    isfile(target_file_path) || error("Target structure file not found: $target_file_path")

    chains_cfg = get(target_cfg, "chains", nothing)
    chains_cfg isa AbstractDict || error("Missing required field: 'target.chains'")
    isempty(chains_cfg) && error("Missing required field: 'target.chains'")

    chain_ids = String[]
    crop_dict = Dict{String, Any}()
    hotspot_dict = Dict{String, Any}()
    msa_dict_per_chain = Dict{String, Any}()

    for (chain_id_raw, props_raw) in chains_cfg
        chain_id = String(chain_id_raw)
        push!(chain_ids, chain_id)
        props = _normalize_chain_props(props_raw)

        if haskey(props, "crop")
            raw_crop = props["crop"]
            crop_val = nothing
            if raw_crop isa AbstractVector
                crop_val = join(string.(raw_crop), ",")
            elseif raw_crop isa AbstractString
                crop_val = lowercase(raw_crop) in ("all", "full") ? nothing : raw_crop
            elseif raw_crop !== nothing
                crop_val = string(raw_crop)
            end
            if crop_val !== nothing && !isempty(crop_val)
                crop_dict[chain_id] = crop_val
            end
        end

        if haskey(props, "hotspots")
            hotspot_dict[chain_id] = props["hotspots"]
        end

        if haskey(props, "msa") && props["msa"] !== nothing && !isempty(string(props["msa"]))
            msa_path_raw = String(props["msa"])
            msa_path = isabspath(msa_path_raw) ? msa_path_raw : normpath(joinpath(dirname(abspath(yaml_path)), msa_path_raw))
            non_pair = joinpath(msa_path, "non_pairing.a3m")
            isfile(non_pair) || error("MSA file not found: $non_pair")
            msa_dict_per_chain[chain_id] = Dict(
                "precomputed_msa_dir" => msa_path,
                "pairing_db" => "uniref100",
            )
        end
    end

    json_task = Dict{String, Any}(
        "name" => task_name,
        "condition" => Dict{String, Any}(
            "structure_file" => target_file_path,
            "filter" => Dict{String, Any}(
                "chain_id" => chain_ids,
                "crop" => crop_dict,
            ),
            "msa" => msa_dict_per_chain,
        ),
        "hotspot" => hotspot_dict,
        "generation" => Any[
            Dict{String, Any}(
                "type" => "protein",
                "length" => binder_length,
                "count" => 1,
            ),
        ],
    )

    return Any[json_task]
end

function parse_yaml_to_json(yaml_path::AbstractString, json_path::Union{Nothing, AbstractString} = nothing)
    cfg = load_yaml_config(yaml_path)
    tasks = _parse_yaml_cfg_to_tasks(cfg, yaml_path)

    if json_path !== nothing
        mkpath(dirname(json_path))
        write_json(json_path, tasks)
    end
    return tasks
end

function _normalize_loaded_json(value, input_path::AbstractString)
    if value isa AbstractVector
        return Any[_as_string_dict(x) for x in value]
    elseif value isa Dict
        return Any[_as_string_dict(value)]
    end
    error("Expected JSON object or array in $input_path")
end

function load_input_tasks(input_path::AbstractString)
    path = abspath(input_path)
    isfile(path) || error("Input file not found: $path")
    ext = lowercase(splitext(path)[2])

    if ext == ".json"
        parsed = parse_json(read(path, String))
        return _normalize_loaded_json(parsed, path)
    elseif ext == ".yaml" || ext == ".yml"
        return parse_yaml_to_json(path, nothing)
    end

    error("Unsupported input file format: $ext. Supported formats are: JSON, YAML (.yaml/.yml).")
end

function process_input_file(input_path::AbstractString; out_dir::Union{Nothing, AbstractString} = nothing)
    path = abspath(input_path)
    isfile(path) || error("Input file not found: $path")
    ext = lowercase(splitext(path)[2])
    if !(ext in (".json", ".yaml", ".yml"))
        error("Unsupported input file format: $ext. Supported formats are: JSON, YAML (.yaml/.yml).")
    end

    if ext == ".yaml" || ext == ".yml"
        base = splitext(basename(path))[1]
        target_dir = out_dir === nothing ? dirname(path) : String(out_dir)
        json_path = joinpath(target_dir, "$base.json")
        parse_yaml_to_json(path, json_path)
        return json_path
    end
    return path
end

function snapshot_input(input_path::AbstractString, snapshot_path::AbstractString)
    mkpath(dirname(snapshot_path))
    cp(input_path, snapshot_path; force = true)
    return snapshot_path
end

end
