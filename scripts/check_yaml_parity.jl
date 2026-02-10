#!/usr/bin/env julia

using PXDesign

function as_string_dict(x)
    if x isa AbstractDict
        out = Dict{String, Any}()
        for (k, v) in x
            out[String(k)] = as_string_dict(v)
        end
        return out
    elseif x isa AbstractVector
        return Any[as_string_dict(v) for v in x]
    end
    return x
end

function diff_values(a, b, path::String, out::Vector{String})
    if a isa AbstractDict
        b isa AbstractDict || (push!(out, "$path type mismatch: $(typeof(a)) vs $(typeof(b))"); return)
        ka = Set(String.(keys(a)))
        kb = Set(String.(keys(b)))
        for k in sort(collect(setdiff(ka, kb)))
            push!(out, "$path missing in python: $k")
        end
        for k in sort(collect(setdiff(kb, ka)))
            push!(out, "$path missing in julia: $k")
        end
        for k in sort(collect(intersect(ka, kb)))
            diff_values(a[k], b[k], "$path.$k", out)
        end
        return
    elseif a isa AbstractVector
        b isa AbstractVector || (push!(out, "$path type mismatch: $(typeof(a)) vs $(typeof(b))"); return)
        length(a) == length(b) || push!(out, "$path length mismatch: $(length(a)) vs $(length(b))")
        for i in 1:min(length(a), length(b))
            diff_values(a[i], b[i], "$path[$i]", out)
        end
        return
    elseif a isa AbstractString
        b isa AbstractString || (push!(out, "$path type mismatch: $(typeof(a)) vs $(typeof(b))"); return)
        String(a) == String(b) || push!(out, "$path value mismatch: $(repr(a)) vs $(repr(b))")
        return
    elseif a isa Number
        b isa Number || (push!(out, "$path type mismatch: $(typeof(a)) vs $(typeof(b))"); return)
        (Float64(a) == Float64(b)) || push!(out, "$path value mismatch: $(repr(a)) vs $(repr(b))")
        return
    end
    a == b || push!(out, "$path value mismatch: $(repr(a)) vs $(repr(b))")
end

function main(argv::Vector{String})
    length(argv) == 1 || error("Usage: check_yaml_parity.jl <input.yaml|input.yml>")
    yaml_path = abspath(argv[1])
    isfile(yaml_path) || error("YAML file not found: $yaml_path")

    julia_cfg = PXDesign.Inputs.load_yaml_config(yaml_path)
    py_ok = success(pipeline(`python3 -c "import yaml; print('ok')"`, stdout = devnull, stderr = devnull))
    py_ok || error("python3 + PyYAML unavailable; cannot run parity check.")

    py_json = read(
        `python3 -c "import json, yaml, sys; print(json.dumps(yaml.safe_load(open(sys.argv[1], 'r')), sort_keys=True))" $yaml_path`,
        String,
    )
    py_cfg = as_string_dict(PXDesign.JSONLite.parse_json(py_json))

    diffs = String[]
    diff_values(julia_cfg, py_cfg, "root", diffs)
    if isempty(diffs)
        println("yaml_parity_ok")
        return
    end

    println("yaml_parity_failed: $(length(diffs)) difference(s)")
    for d in diffs[1:min(end, 30)]
        println(" - $d")
    end
    error("YAML parity check failed.")
end

main(ARGS)
