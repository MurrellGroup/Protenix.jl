module SafeTensorWeights

using SafeTensors

export load_safetensors_weights

function _as_f32_weights(raw::AbstractDict)
    out = Dict{String, Any}()
    for (k, v) in raw
        v isa AbstractArray || error("SafeTensors entry '$k' is not an array.")
        out[String(k)] = Float32.(v)
    end
    return out
end

"""
Load safetensors weights from either:
- a single `.safetensors` file
- a directory with `model.safetensors.index.json` shards
- a directory containing one or more `.safetensors` files
"""
function load_safetensors_weights(path::AbstractString; mmap::Bool = true)
    p = abspath(path)
    if isfile(p)
        endswith(lowercase(p), ".safetensors") ||
            error("Expected a .safetensors file, got: $p")
        return _as_f32_weights(SafeTensors.load_safetensors(p; mmap = mmap))
    elseif isdir(p)
        index_file = joinpath(p, "model.safetensors.index.json")
        if isfile(index_file)
            return _as_f32_weights(SafeTensors.load_sharded_safetensors(p; mmap = mmap))
        end

        files = sort(filter(f -> endswith(lowercase(f), ".safetensors"), readdir(p; join = true)))
        isempty(files) && error("No .safetensors files found in directory: $p")
        if length(files) == 1
            return _as_f32_weights(SafeTensors.load_safetensors(files[1]; mmap = mmap))
        end

        merged = Dict{String, Any}()
        for f in files
            shard = SafeTensors.load_safetensors(f; mmap = mmap)
            for (k, v) in shard
                key = String(k)
                haskey(merged, key) && error("Duplicate tensor key '$key' across safetensor files in $p")
                merged[key] = Float32.(v)
            end
        end
        return merged
    end

    error("SafeTensors path not found: $p")
end

end
