module RawWeights

import ...JSONLite: parse_json

export RawWeightEntry, load_raw_manifest, load_raw_weights

struct RawWeightEntry
    key::String
    dtype::String
    shape::Vector{Int}
    file::String
end

function _as_entry(row)::RawWeightEntry
    row isa AbstractDict || error("Manifest tensor row must be an object.")
    for req in ("key", "dtype", "shape", "file")
        haskey(row, req) || error("Manifest tensor row missing '$req'.")
    end
    shape_any = row["shape"]
    shape_any isa AbstractVector || error("Manifest `shape` must be an array.")
    return RawWeightEntry(
        String(row["key"]),
        lowercase(String(row["dtype"])),
        Int.(shape_any),
        String(row["file"]),
    )
end

function load_raw_manifest(manifest_path::AbstractString)
    isfile(manifest_path) || error("Raw-weight manifest not found: $manifest_path")
    raw = parse_json(read(manifest_path, String))
    raw isa AbstractDict || error("Raw-weight manifest must be an object.")
    haskey(raw, "tensors") || error("Raw-weight manifest missing `tensors`.")
    rows = raw["tensors"]
    rows isa AbstractVector || error("Raw-weight manifest `tensors` must be an array.")
    return [_as_entry(r) for r in rows]
end

function _read_float32_tensor(path::AbstractString, shape::Vector{Int})
    buf = read(path)
    n = isempty(shape) ? 1 : prod(shape)
    expected = 4 * n
    length(buf) == expected || error("Tensor byte-length mismatch at $path: expected $expected, got $(length(buf))")
    values = reinterpret(Float32, buf)
    data = copy(values)

    isempty(shape) && return data[1]
    length(shape) == 1 && return reshape(data, Tuple(shape))

    # Export path uses NumPy/PyTorch row-major flattening (C-order).
    # Reconstruct by reshaping reversed dims then permuting axes back.
    rev_shape = reverse(shape)
    arr_rev = reshape(data, Tuple(rev_shape))
    perm = ntuple(i -> length(shape) - i + 1, length(shape))
    return permutedims(arr_rev, perm)
end

"""
Load a raw-weight bundle into `Dict{String, Array{Float32}}`.

Bundle layout:
- `manifest.json`
- tensor binary files listed by manifest rows
"""
function load_raw_weights(bundle_dir::AbstractString)
    manifest_path = joinpath(bundle_dir, "manifest.json")
    entries = load_raw_manifest(manifest_path)
    out = Dict{String, Any}()
    for e in entries
        e.dtype == "float32" || error(
            "Unsupported dtype $(e.dtype) for key $(e.key). " *
            "Use scripts/export_checkpoint_raw.py --cast-float32.",
        )
        tpath = joinpath(bundle_dir, e.file)
        isfile(tpath) || error("Raw tensor file missing: $tpath")
        out[e.key] = _read_float32_tensor(tpath, e.shape)
    end
    return out
end

end
