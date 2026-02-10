module Scheduler

export InferenceNoiseScheduler

"""
EDM-style inference noise scheduler used by PXDesign.
"""
Base.@kwdef struct InferenceNoiseScheduler
    s_max::Float64 = 160.0
    s_min::Float64 = 4e-4
    rho::Float64 = 7.0
    sigma_data::Float64 = 16.0
end

function (sched::InferenceNoiseScheduler)(N_step::Integer; dtype::Type{T} = Float32) where {T<:AbstractFloat}
    N_step <= 0 && error("N_step must be positive.")

    step_size = one(T) / T(N_step)
    ts = Vector{T}(undef, N_step + 1)

    smax = T(sched.s_max)
    smin = T(sched.s_min)
    rho = T(sched.rho)
    sigma_data = T(sched.sigma_data)
    one_over_rho = inv(rho)

    a = smax^one_over_rho
    b = smin^one_over_rho - a

    for i in 0:N_step
        step = T(i) * step_size
        ts[i + 1] = sigma_data * (a + step * b)^rho
    end
    ts[end] = zero(T)

    return ts
end

end
