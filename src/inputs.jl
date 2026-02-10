module Inputs

import ..JSONLite: parse_json, write_json

export parse_yaml_to_json, load_input_tasks, process_input_file, snapshot_input

function _as_string_dict(x)
    if x isa Dict
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

function _load_yaml_via_python(yaml_path::AbstractString)
    # Temporary YAML compatibility shim:
    # keep Julia core self-contained while matching Python safe_load semantics.
    pycode = """
import json, sys
try:
    import yaml
except Exception as e:
    print(f"Unable to import PyYAML: {e}", file=sys.stderr)
    sys.exit(3)
path = sys.argv[1]
with open(path, "r") as f:
    try:
        cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"YAML parse failed: {e}", file=sys.stderr)
        sys.exit(4)
print(json.dumps(cfg))
"""
    cmd = `python3 -c $pycode $yaml_path`
    json_text = try
        read(cmd, String)
    catch e
        error(
            "Failed to parse YAML via python3/PyYAML for $yaml_path. " *
            "Provide JSON input or ensure python3 + PyYAML are available. Original error: $e",
        )
    end
    parsed = parse_json(json_text)
    parsed isa Dict || error("YAML top-level must be a mapping/object.")
    return _as_string_dict(parsed)
end

function _normalize_chain_props(props)
    if props === nothing
        return Dict{String, Any}()
    elseif props isa AbstractString
        return lowercase(strip(props)) in ("all", "full") ? Dict{String, Any}() : Dict{String, Any}()
    elseif props isa Dict
        return _as_string_dict(props)
    end
    error("Invalid chain config value: expected mapping/null/'all', got $(typeof(props))")
end

function _parse_yaml_cfg_to_tasks(cfg::Dict{String, Any}, yaml_path::AbstractString)
    default_name = splitext(basename(yaml_path))[1]
    task_name = haskey(cfg, "task_name") ? String(cfg["task_name"]) : default_name

    haskey(cfg, "binder_length") || error("Missing required field: 'binder_length'")
    binder_length = Int(cfg["binder_length"])

    target_cfg = haskey(cfg, "target") ? cfg["target"] : Dict{String, Any}()
    target_cfg isa Dict || error("Field 'target' must be a mapping")
    haskey(target_cfg, "file") || error("Missing required field: 'target.file'")

    target_file_path = String(target_cfg["file"])
    isfile(target_file_path) || error("Target structure file not found: $target_file_path")

    chains_cfg = get(target_cfg, "chains", nothing)
    chains_cfg isa Dict || error("Missing required field: 'target.chains'")
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
            msa_path = String(props["msa"])
            for fname in ("pairing.a3m", "non_pairing.a3m")
                isfile(joinpath(msa_path, fname)) || error("MSA file not found: $(joinpath(msa_path, fname))")
            end
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
    yaml_abspath = abspath(yaml_path)
    isfile(yaml_abspath) || error("YAML config file not found: $yaml_abspath")

    cfg = _load_yaml_via_python(yaml_abspath)
    tasks = _parse_yaml_cfg_to_tasks(cfg, yaml_abspath)

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
    elseif ext == ".yaml"
        return parse_yaml_to_json(path, nothing)
    end

    error("Unsupported input file format: $ext. Supported formats are: JSON, YAML.")
end

function process_input_file(input_path::AbstractString; out_dir::Union{Nothing, AbstractString} = nothing)
    path = abspath(input_path)
    isfile(path) || error("Input file not found: $path")
    ext = lowercase(splitext(path)[2])
    if !(ext in (".json", ".yaml"))
        error("Unsupported input file format: $ext. Supported formats are: JSON, YAML.")
    end

    if ext == ".yaml"
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
