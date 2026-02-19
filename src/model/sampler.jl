module Sampler

using Random

export sample_diffusion

function _eta_for_step(step_scale_eta, step_t::Int, total_steps::Int)
    if step_scale_eta isa Real
        return float(step_scale_eta)
    elseif step_scale_eta isa NamedTuple
        haskey(step_scale_eta, :type) || error("NamedTuple eta schedule missing :type.")
        haskey(step_scale_eta, :min) || error("NamedTuple eta schedule missing :min.")
        haskey(step_scale_eta, :max) || error("NamedTuple eta schedule missing :max.")
        typ = String(step_scale_eta.type)
        eta_min = float(step_scale_eta.min)
        eta_max = float(step_scale_eta.max)
        r = step_t / max(total_steps, 1)
        if typ == "const"
            return eta_min
        elseif typ == "linear"
            return eta_min + (eta_max - eta_min) * r
        elseif typ == "poly"
            return eta_min + (eta_max - eta_min) * (r^2)
        elseif typ == "cos"
            return eta_min + 0.5 * (eta_max - eta_min) * (1 - cos(pi * r))
        elseif typ == "piecewise"
            return r < 0.5 ? eta_min : eta_max
        elseif typ == "piecewise_65"
            return r < 0.65 ? eta_min : eta_max
        elseif typ == "piecewise_70"
            return r < 0.70 ? eta_min : eta_max
        else
            error("Unsupported eta schedule type: $typ")
        end
    elseif step_scale_eta isa AbstractDict
        typ = String(step_scale_eta["type"])
        eta_min = float(step_scale_eta["min"])
        eta_max = float(step_scale_eta["max"])
        r = step_t / max(total_steps, 1)
        if typ == "const"
            return eta_min
        elseif typ == "linear"
            return eta_min + (eta_max - eta_min) * r
        elseif typ == "poly"
            return eta_min + (eta_max - eta_min) * (r^2)
        elseif typ == "cos"
            return eta_min + 0.5 * (eta_max - eta_min) * (1 - cos(pi * r))
        elseif typ == "piecewise"
            return r < 0.5 ? eta_min : eta_max
        elseif typ == "piecewise_65"
            return r < 0.65 ? eta_min : eta_max
        elseif typ == "piecewise_70"
            return r < 0.70 ? eta_min : eta_max
        else
            error("Unsupported eta schedule type: $typ")
        end
    else
        error("step_scale_eta must be a Real, NamedTuple, or Dict")
    end
end

function _random_rotation_matrix(rng::AbstractRNG, ::Type{T}) where {T}
    q = randn(rng, T, 4)
    q ./= sqrt(sum(abs2, q))
    w, x, y, z = q
    return T[
        1 - 2 * (y * y + z * z) 2 * (x * y - z * w) 2 * (x * z + y * w)
        2 * (x * y + z * w) 1 - 2 * (x * x + z * z) 2 * (y * z - x * w)
        2 * (x * z - y * w) 2 * (y * z + x * w) 1 - 2 * (x * x + y * y)
    ]
end

# Features-first: x shape (3, N_atom, N_sample)
function _center_random_augmentation!(x::Array{T,3}, rng::AbstractRNG; s_trans::T = one(T)) where {T}
    n_atom = size(x, 2)
    n_sample = size(x, 3)
    for s in 1:n_sample
        mx = zero(T)
        my = zero(T)
        mz = zero(T)
        @inbounds for a in 1:n_atom
            mx += x[1, a, s]
            my += x[2, a, s]
            mz += x[3, a, s]
        end
        inv_n = inv(T(n_atom))
        mx *= inv_n
        my *= inv_n
        mz *= inv_n
        @inbounds for a in 1:n_atom
            x[1, a, s] -= mx
            x[2, a, s] -= my
            x[3, a, s] -= mz
        end

        rot = _random_rotation_matrix(rng, T)
        tx = s_trans * randn(rng, T)
        ty = s_trans * randn(rng, T)
        tz = s_trans * randn(rng, T)
        @inbounds for a in 1:n_atom
            x0 = x[1, a, s]
            y0 = x[2, a, s]
            z0 = x[3, a, s]
            x[1, a, s] = rot[1, 1] * x0 + rot[1, 2] * y0 + rot[1, 3] * z0 + tx
            x[2, a, s] = rot[2, 1] * x0 + rot[2, 2] * y0 + rot[2, 3] * z0 + ty
            x[3, a, s] = rot[3, 1] * x0 + rot[3, 2] * y0 + rot[3, 3] * z0 + tz
        end
    end
    return x
end

# Copy cpu_array to the same device as x_ref (no-op for CPU Arrays).
function _to_device(cpu_array::Array, x_ref::Array)
    return cpu_array
end
function _to_device(cpu_array::Array, x_ref::AbstractArray)
    dst = similar(x_ref, eltype(cpu_array), size(cpu_array))
    copyto!(dst, cpu_array)
    return dst
end

function sample_diffusion(
    denoise_net::Function;
    noise_schedule::AbstractVector{<:Real},
    N_sample::Int,
    N_atom::Int,
    gamma0::Real = 0.8,
    gamma_min::Real = 1.0,
    noise_scale_lambda::Real = 1.003,
    step_scale_eta = (type = "const", min = 1.5, max = 1.5),
    diffusion_chunk_size::Union{Nothing, Int} = nothing,
    rng::AbstractRNG = Random.default_rng(),
    device_ref::Union{Nothing, AbstractArray} = nothing,
    kwargs...,
)
    N_sample <= 0 && error("N_sample must be positive.")
    N_atom <= 0 && error("N_atom must be positive.")
    length(noise_schedule) >= 2 || error("noise_schedule must have at least two entries.")

    T = Float32
    function _chunk_sample(n_s::Int)
        sigma0 = T(noise_schedule[1])
        noise_cpu = sigma0 .* randn(rng, T, 3, N_atom, n_s)
        x_l = device_ref === nothing ? noise_cpu : _to_device(noise_cpu, device_ref)

        total = length(noise_schedule)
        for (step_t, (tau_last, tau)) in enumerate(zip(noise_schedule[1:end-1], noise_schedule[2:end]))
            # Center + augmentation on CPU (scalar loops), then copy back
            x_l_cpu = Array(x_l)
            _center_random_augmentation!(x_l_cpu, rng)
            if !(x_l isa Array)
                copyto!(x_l, x_l_cpu)
            end

            c_tau_last = T(tau_last)
            c_tau = T(tau)

            gamma = c_tau > T(gamma_min) ? T(gamma0) : zero(T)
            t_hat = c_tau_last * (gamma + one(T))
            delta_noise = sqrt(max(zero(T), t_hat^2 - c_tau_last^2))

            step_noise_cpu = T(noise_scale_lambda) * delta_noise .* randn(rng, T, size(x_l))
            step_noise = device_ref === nothing ? step_noise_cpu : _to_device(step_noise_cpu, x_l)
            x_noisy = x_l .+ step_noise
            x_denoised = denoise_net(x_noisy, t_hat; kwargs...)
            size(x_denoised) == size(x_noisy) || error("denoise_net must return same shape as input.")

            delta = (x_noisy .- x_denoised) ./ max(t_hat, eps(T))
            dt = c_tau - t_hat
            eta = T(_eta_for_step(step_scale_eta, step_t - 1, total))
            x_l = x_noisy .+ eta * dt .* delta
        end
        return x_l
    end

    if diffusion_chunk_size === nothing || diffusion_chunk_size <= 0 || diffusion_chunk_size >= N_sample
        return _chunk_sample(N_sample)
    end

    chunks = AbstractArray{T,3}[]
    remaining = N_sample
    while remaining > 0
        n_s = min(diffusion_chunk_size, remaining)
        push!(chunks, _chunk_sample(n_s))
        remaining -= n_s
    end
    return cat(chunks...; dims = 3)
end

end
