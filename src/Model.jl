module Model

using Random, Statistics, LinearAlgebra
using ..Types: Params
using ..Hazard: ν

export euler_maruyama_step!, reset_step!

# One Euler–Maruyama step for dissonance u with mean-field coupling κ * ḡ
@inline function euler_maruyama_step!(u::Vector{Float64}, κ::Float64, gbar::Float64, p::Params, dt::Float64)
    sigma_sqrt_dt = p.σ * sqrt(dt)
    λ = p.λ
    for i in eachindex(u)
        du = (-λ * u[i] + κ * gbar) * dt + sigma_sqrt_dt * randn()
        u[i] += du
    end
    return nothing
end

# Apply reset with Poisson hazard ν(u) over dt; landing at c0 * u
@inline function reset_step!(u::Vector{Float64}, p::Params, dt::Float64)
    Θ = p.Θ
    c0 = p.c0
    for i in eachindex(u)
        rate = ν(p.hazard, u[i], Θ)
        if rate > 0.0
            # Poisson thinning: event occurs with prob 1 - exp(-rate*dt)
            if rand() < 1.0 - exp(-rate*dt)
                u[i] = c0 * u[i]
            end
        end
    end
    return nothing
end

end # module
