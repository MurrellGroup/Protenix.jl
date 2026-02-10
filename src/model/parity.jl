module ParityHarness

import ..RawWeights: load_raw_weights

export TensorParityStats,
    TensorParityReport,
    tensor_parity_report,
    compare_raw_weight_dirs

struct TensorParityStats
    key::String
    numel::Int
    max_abs::Float32
    max_rel::Float32
    mean_abs::Float32
    passed::Bool
end

struct TensorParityReport
    compared::Vector{TensorParityStats}
    failed::Vector{TensorParityStats}
    missing_in_actual::Vector{String}
    missing_in_reference::Vector{String}
    atol::Float32
    rtol::Float32
end

function _tensor_stats(
    key::String,
    ref::AbstractArray{<:Real},
    act::AbstractArray{<:Real},
    atol::Float32,
    rtol::Float32,
)
    if size(ref) != size(act)
        return TensorParityStats(key, 0, Inf32, Inf32, Inf32, false)
    end

    vref = vec(Float32.(ref))
    vact = vec(Float32.(act))
    n = length(vref)
    n == length(vact) || return TensorParityStats(key, 0, Inf32, Inf32, Inf32, false)

    max_abs = 0f0
    max_rel = 0f0
    sum_abs = 0f0
    passed = true
    for i in eachindex(vref, vact)
        r = vref[i]
        a = vact[i]
        d = abs(a - r)
        sum_abs += d
        if d > max_abs
            max_abs = d
        end
        rel = d / max(abs(r), atol)
        if rel > max_rel
            max_rel = rel
        end
        if d > atol + rtol * abs(r)
            passed = false
        end
    end
    mean_abs = n == 0 ? 0f0 : (sum_abs / n)
    return TensorParityStats(key, n, max_abs, max_rel, mean_abs, passed)
end

"""
Compare two tensor maps for numeric parity under allclose-style thresholds.

Returns a `TensorParityReport` with per-key stats and key-presence differences.
"""
function tensor_parity_report(
    reference::AbstractDict{<:AbstractString, <:Any},
    actual::AbstractDict{<:AbstractString, <:Any};
    atol::Real = 1f-5,
    rtol::Real = 1f-4,
)
    atol_f = Float32(atol)
    rtol_f = Float32(rtol)

    ref_keys = Set(String.(collect(keys(reference))))
    act_keys = Set(String.(collect(keys(actual))))
    missing_in_actual = sort(collect(setdiff(ref_keys, act_keys)))
    missing_in_reference = sort(collect(setdiff(act_keys, ref_keys)))
    common_keys = sort(collect(intersect(ref_keys, act_keys)))

    compared = TensorParityStats[]
    failed = TensorParityStats[]
    for key in common_keys
        ref_val = reference[key]
        act_val = actual[key]
        ref_val isa AbstractArray || continue
        act_val isa AbstractArray || continue
        stats = _tensor_stats(key, ref_val, act_val, atol_f, rtol_f)
        push!(compared, stats)
        stats.passed || push!(failed, stats)
    end

    return TensorParityReport(
        compared,
        failed,
        missing_in_actual,
        missing_in_reference,
        atol_f,
        rtol_f,
    )
end

"""
Load two raw-weight bundles and compare them with `tensor_parity_report`.
"""
function compare_raw_weight_dirs(
    reference_dir::AbstractString,
    actual_dir::AbstractString;
    atol::Real = 1f-5,
    rtol::Real = 1f-4,
)
    reference = load_raw_weights(reference_dir)
    actual = load_raw_weights(actual_dir)
    return tensor_parity_report(reference, actual; atol = atol, rtol = rtol)
end

end
