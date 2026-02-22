#!/usr/bin/env julia
#
# Compare Python vs Julia feature tensors for parity.
#
# Usage:
#   cd /home/claudey/FixingKAFA/ka_run_env
#   julia --project=. ../PXDesign.jl/clean_targets/scripts/compare_features.jl \
#       ../PXDesign.jl/clean_targets/feature_dumps/python/protein_ligand_ccd_features.pt \
#       ../PXDesign.jl/clean_targets/feature_dumps/julia/protein_ligand_ccd_features.jld2

using JLD2
using PyCall

const torch = pyimport("torch")
const np = pyimport("numpy")

function load_python_features(pt_path::String)
    py_feat = torch.load(pt_path, map_location="cpu", weights_only=false)
    result = Dict{String, Any}()
    for key in py_feat.keys()
        val = py_feat[key]
        if PyCall.pyisinstance(val, torch.Tensor)
            # Convert to Julia array
            arr = val.detach().numpy()
            result[string(key)] = convert(Array, arr)
        elseif PyCall.pyisinstance(val, PyCall.pybuiltin("dict"))
            sub = Dict{String, Any}()
            for subkey in val.keys()
                subval = val[subkey]
                if PyCall.pyisinstance(subval, torch.Tensor)
                    sub[string(subkey)] = convert(Array, subval.detach().numpy())
                end
            end
            result[string(key)] = sub
        end
    end
    return result
end

function load_julia_features(jld2_path::String)
    data = load(jld2_path)
    return data["feat"]
end

# Python uses row-major (N_tok, C), Julia uses column-major (C, N_tok) for some tensors.
# For comparison, we keep Python layout (features-last) and check Julia features-last too,
# since dump_julia_features.jl dumps the raw dict BEFORE as_protenix_features transposition.

function compare_tensor(key::String, py_val, jl_val; atol=1e-5, subkey="")
    prefix = isempty(subkey) ? key : "$key.$subkey"

    if py_val isa AbstractArray && jl_val isa AbstractArray
        if size(py_val) != size(jl_val)
            println("  SHAPE MISMATCH  $prefix:  python=$(size(py_val))  julia=$(size(jl_val))")
            return false
        end

        if eltype(py_val) <: AbstractFloat || eltype(jl_val) <: AbstractFloat
            py_f = Float64.(py_val)
            jl_f = Float64.(jl_val)
            maxdiff = maximum(abs.(py_f .- jl_f))
            n_total = length(py_f)
            n_diff = count(abs.(py_f .- jl_f) .> atol)

            if maxdiff <= atol
                println("  OK            $prefix  shape=$(size(py_val))  maxdiff=$maxdiff")
                return true
            else
                pct = round(100.0 * n_diff / n_total; digits=1)
                println("  MISMATCH      $prefix  shape=$(size(py_val))  maxdiff=$maxdiff  n_diff=$n_diff/$n_total ($pct%)")

                # Show first few mismatches
                diffs = abs.(py_f .- jl_f)
                worst_idx = sortperm(vec(diffs); rev=true)
                for i in 1:min(5, length(worst_idx))
                    idx = worst_idx[i]
                    ci = CartesianIndices(py_f)[idx]
                    println("    [$ci]: py=$(py_f[idx])  jl=$(jl_f[idx])  diff=$(diffs[idx])")
                end
                return false
            end
        else
            # Integer comparison
            py_i = Int64.(py_val)
            jl_i = Int64.(jl_val)
            n_diff = count(py_i .!= jl_i)
            if n_diff == 0
                println("  OK            $prefix  shape=$(size(py_val))  (integer exact)")
                return true
            else
                n_total = length(py_i)
                pct = round(100.0 * n_diff / n_total; digits=1)
                println("  MISMATCH      $prefix  shape=$(size(py_val))  n_diff=$n_diff/$n_total ($pct%)")

                # Show first few mismatches
                for (k, ci) in enumerate(CartesianIndices(py_i))
                    py_i[ci] == jl_i[ci] && continue
                    println("    [$ci]: py=$(py_i[ci])  jl=$(jl_i[ci])")
                    k >= 10 && break
                end
                return false
            end
        end
    else
        println("  SKIP          $prefix  (not both arrays: py=$(typeof(py_val)) jl=$(typeof(jl_val)))")
        return true
    end
end

function main(py_path::String, jl_path::String)
    println("Loading Python features from: $py_path")
    py = load_python_features(py_path)
    println("Loading Julia features from: $jl_path")
    jl = load_julia_features(jl_path)

    py_keys = Set(keys(py))
    jl_keys = Set(keys(jl))

    println("\n", "="^70)
    println("Python-only keys: ", sort(collect(setdiff(py_keys, jl_keys))))
    println("Julia-only keys:  ", sort(collect(setdiff(jl_keys, py_keys))))
    println("Common keys:      ", length(intersect(py_keys, jl_keys)))
    println("="^70, "\n")

    n_ok = 0
    n_fail = 0
    n_skip = 0

    for key in sort(collect(intersect(py_keys, jl_keys)))
        py_val = py[key]
        jl_val = jl[key]

        if py_val isa Dict && jl_val isa Dict
            # Compare nested dict (constraint_feature)
            sub_py_keys = Set(keys(py_val))
            sub_jl_keys = Set(keys(jl_val))
            for subkey in sort(collect(intersect(sub_py_keys, sub_jl_keys)))
                ok = compare_tensor(key, py_val[subkey], jl_val[subkey]; subkey=subkey)
                if ok; n_ok += 1; else; n_fail += 1; end
            end
        elseif py_val isa AbstractArray && jl_val isa AbstractArray
            ok = compare_tensor(key, py_val, jl_val)
            if ok; n_ok += 1; else; n_fail += 1; end
        else
            println("  SKIP          $key  (types: py=$(typeof(py_val)) jl=$(typeof(jl_val)))")
            n_skip += 1
        end
    end

    println("\n", "="^70)
    println("Summary: $n_ok OK, $n_fail MISMATCH, $n_skip SKIP")
    println("="^70)
    return n_fail
end

if length(ARGS) < 2
    println("Usage: julia compare_features.jl <python.pt> <julia.jld2>")
    exit(1)
end

n_fail = main(ARGS[1], ARGS[2])
exit(n_fail > 0 ? 1 : 0)
