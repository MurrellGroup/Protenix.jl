module CheckpointMap

import ...JSONLite: parse_json

export expected_checkpoint_prefixes, load_checkpoint_index, checkpoint_prefix_counts

const EXPECTED_PREFIXES = (
    "design_condition_embedder.condition_template_embedder.embedder",
    "design_condition_embedder.input_embedder",
    "diffusion_module.diffusion_conditioning",
    "diffusion_module.atom_attention_encoder",
    "diffusion_module.diffusion_transformer",
    "diffusion_module.atom_attention_decoder",
    "diffusion_module.layernorm_a",
    "diffusion_module.layernorm_s",
    "diffusion_module.linear_no_bias_s",
)

expected_checkpoint_prefixes() = EXPECTED_PREFIXES

function load_checkpoint_index(path::AbstractString)
    isfile(path) || error("Checkpoint index not found: $path")
    raw = parse_json(read(path, String))
    raw isa AbstractDict || error("Checkpoint index must be a JSON object.")
    haskey(raw, "tensors") || error("Checkpoint index missing `tensors`.")
    tensors = raw["tensors"]
    tensors isa AbstractVector || error("`tensors` must be an array.")

    keys = String[]
    for row in tensors
        row isa AbstractDict || error("Each tensor row must be an object.")
        haskey(row, "key") || error("Tensor row missing `key`.")
        key = String(row["key"])
        if startswith(key, "module.")
            key = key[8:end]
        end
        push!(keys, key)
    end
    return keys
end

function checkpoint_prefix_counts(keys::Vector{String})
    counts = Dict{String, Int}()
    for key in keys
        matched = false
        for pfx in EXPECTED_PREFIXES
            if startswith(key, pfx)
                counts[pfx] = get(counts, pfx, 0) + 1
                matched = true
                break
            end
        end
        if !matched
            counts["_unmatched"] = get(counts, "_unmatched", 0) + 1
        end
    end
    return counts
end

end
