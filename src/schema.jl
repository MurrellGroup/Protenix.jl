module Schema

import ..Ranges: parse_ranges

export GenerationSpec, InputTask, parse_tasks

struct GenerationSpec
    type::String
    length::Int
    count::Int
end

struct InputTask
    name::String
    structure_file::String
    chain_ids::Vector{String}
    crop::Dict{String, String}
    hotspots::Dict{String, Vector{Int}}
    msa::Dict{String, Dict{String, Any}}
    generation::Vector{GenerationSpec}
end

function _as_dict(x, field::AbstractString)
    x isa AbstractDict || error("Field '$field' must be a mapping/object.")
    return x
end

function _as_vector(x, field::AbstractString)
    x isa AbstractVector || error("Field '$field' must be an array.")
    return x
end

function _to_string_vector(x, field::AbstractString)
    xs = _as_vector(x, field)
    return String.(xs)
end

function _to_int_vector(x, field::AbstractString)
    xs = _as_vector(x, field)
    return Int.(xs)
end

function _parse_crop(x)
    out = Dict{String, String}()
    if !(x isa AbstractDict)
        return out
    end
    for (k, v) in x
        crop_str = String(v)
        # Validate syntax/parsability on ingest.
        parse_ranges(crop_str)
        out[String(k)] = crop_str
    end
    return out
end

function _parse_hotspots(x)
    out = Dict{String, Vector{Int}}()
    if !(x isa AbstractDict)
        return out
    end
    for (k, v) in x
        out[String(k)] = _to_int_vector(v, "hotspot.$(k)")
    end
    return out
end

function _parse_msa(x)
    out = Dict{String, Dict{String, Any}}()
    if !(x isa AbstractDict)
        return out
    end
    for (chain, cfg_any) in x
        cfg = _as_dict(cfg_any, "condition.msa.$(chain)")
        chain_dict = Dict{String, Any}()
        for (k, v) in cfg
            chain_dict[String(k)] = v
        end
        out[String(chain)] = chain_dict
    end
    return out
end

function _parse_generation(x)
    arr = _as_vector(x, "generation")
    out = GenerationSpec[]
    for (idx, item_any) in enumerate(arr)
        item = _as_dict(item_any, "generation[$idx]")
        typ = haskey(item, "type") ? String(item["type"]) : error("Missing generation[$idx].type")
        len = haskey(item, "length") ? Int(item["length"]) : error("Missing generation[$idx].length")
        cnt = haskey(item, "count") ? Int(item["count"]) : 1
        push!(out, GenerationSpec(typ, len, cnt))
    end
    return out
end

function parse_task(raw)::InputTask
    d = _as_dict(raw, "task")
    haskey(d, "name") || error("Missing task.name")
    haskey(d, "generation") || error("Missing task.generation")

    name = String(d["name"])
    structure_file = ""
    chain_ids = String[]
    crop = Dict{String, String}()
    msa = Dict{String, Dict{String, Any}}()

    if haskey(d, "condition")
        cond = _as_dict(d["condition"], "condition")
        haskey(cond, "structure_file") || error("Missing condition.structure_file")
        haskey(cond, "filter") || error("Missing condition.filter")
        filt = _as_dict(cond["filter"], "condition.filter")
        haskey(filt, "chain_id") || error("Missing condition.filter.chain_id")

        structure_file = String(cond["structure_file"])
        chain_ids = _to_string_vector(filt["chain_id"], "condition.filter.chain_id")
        crop = _parse_crop(get(filt, "crop", Dict{String, Any}()))
        msa = _parse_msa(get(cond, "msa", Dict{String, Any}()))
    end

    hotspots = _parse_hotspots(get(d, "hotspot", Dict{String, Any}()))
    generation = _parse_generation(d["generation"])

    return InputTask(name, structure_file, chain_ids, crop, hotspots, msa, generation)
end

function parse_tasks(raw_tasks)::Vector{InputTask}
    raw_tasks isa AbstractVector || error("Tasks must be an array.")
    return [parse_task(task) for task in raw_tasks]
end

end
