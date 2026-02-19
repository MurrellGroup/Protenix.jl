module Device

using Flux: cpu, gpu

export to_device, zeros_like, ones_like, device_ref, feats_to_device, feats_to_cpu

"""
    zeros_like(x::AbstractArray, dims...)

Create a zero-filled array on the same device as `x`.
"""
zeros_like(x::AbstractArray, dims...) = fill!(similar(x, Float32, dims...), 0f0)

"""
    ones_like(x::AbstractArray, dims...)

Create a ones-filled array on the same device as `x`.
"""
ones_like(x::AbstractArray, dims...) = fill!(similar(x, Float32, dims...), 1f0)

"""
    to_device(src::AbstractArray, like::AbstractArray)

Copy `src` to the same device as `like`, preserving element type.
"""
function to_device(src::AbstractArray, like::AbstractArray)
    dst = similar(like, eltype(src), size(src))
    copyto!(dst, src)
    return dst
end

"""
    to_device(src::AbstractArray, like::AbstractArray, ::Type{T})

Copy `src` to the same device as `like`, converting to element type `T`.
"""
function to_device(src::AbstractArray, like::AbstractArray, ::Type{T}) where T
    dst = similar(like, T, size(src))
    copyto!(dst, T.(src))
    return dst
end

"""
    device_ref(model)

Get a reference array from a model to determine its device.
Traverses the model tree to find the first parameter array.
"""
function device_ref(model)
    ref = nothing
    Flux.fmap(model) do x
        if ref === nothing && x isa AbstractArray{Float32}
            ref = x
        end
        x
    end
    ref === nothing && error("No Float32 parameter arrays found in model.")
    return ref
end

"""
    feats_to_device(feats::AbstractDict, ref::AbstractArray)

Transfer all array values in a feature dict to the same device as `ref`.
Non-array values are passed through unchanged.
"""
function feats_to_device(feats::AbstractDict, ref::AbstractArray)
    out = Dict{String,Any}()
    for (k, v) in feats
        if v isa AbstractArray
            out[String(k)] = copyto!(similar(ref, eltype(v), size(v)), v)
        else
            out[String(k)] = v
        end
    end
    return out
end

"""
    feats_to_cpu(feats::AbstractDict)

Bring all array values in a feature dict back to CPU.
"""
function feats_to_cpu(feats::AbstractDict)
    out = Dict{String,Any}()
    for (k, v) in feats
        out[String(k)] = v isa AbstractArray ? Array(v) : v
    end
    return out
end

end
