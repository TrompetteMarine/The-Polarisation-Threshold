module Simulate

using Statistics
using ..Types: Params
using ..Model: euler_maruyama_step!, reset_step!

export simulate_population, simulate_population_summary

struct SimulationResult
    u::Vector{Float64}
    steady_gbar_abs::Float64
end

function simulate_population(p::Params; N::Int=20_000, T::Float64=300.0, dt::Float64=0.01,
                           κ::Float64=0.0, burn_in::Float64=100.0, record::Bool=false, seed::Int=0)
    burn_in_idx = Int(floor(burn_in/dt))
    total_steps = Int(floor(T/dt))
    
    u = zeros(N)
    for step in 1:total_steps
        g = mean(u)
        euler_maruyama_step!(u, κ, g, p, dt)
        reset_step!(u, p, dt)
    end
    
    steady_gbar = mean(u[burn_in_idx+1:end])
    return SimulationResult(u, abs(steady_gbar))
end

function simulate_population_summary(p::Params; kwargs...)
    result = simulate_population(p; kwargs...)
    return result.steady_gbar_abs
end

end # module
