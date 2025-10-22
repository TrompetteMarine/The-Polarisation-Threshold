module Network

using Random, Statistics, LinearAlgebra, Graphs
using ..Types: Params, StepHazard
using ..Model: reset_step!
using ..Hazard: ν

export simulate_network

"""
Network simulation: social term is κ * mean of neighbors' u (including self if self-loops provided)
"""
function simulate_network(g::Graphs.AbstractGraph, p::Params; 
                         T::Float64=200.0, dt::Float64=0.01,
                         burn_in::Float64=50.0, κ::Float64=0.0, seed::Int=0)
    seed != 0 && Random.seed!(seed)
    N = nv(g)
    
    # Init from OU stationary
    σ2_over_2λ = p.σ^2 / (2.0 * p.λ)
    u = randn(N) * sqrt(σ2_over_2λ)

    steps = Int(round(T/dt))
    degs = degree.(Ref(g), 1:N)
    
    # Precompute neighbor lists
    nbrs = [neighbors(g, v) for v in 1:N]

    for t in 1:steps
        # Compute neighbor means
        gneigh = similar(u)
        for i in 1:N
            if degs[i] == 0
                gneigh[i] = 0.0
            else
                s = 0.0
                for j in nbrs[i]
                    s += u[j]
                end
                gneigh[i] = s / degs[i]
            end
        end
        
        # EM step + resets
        for i in 1:N
            du = (-p.λ * u[i] + κ * gneigh[i]) * dt + p.σ * sqrt(dt) * randn()
            u[i] += du
            
            # Apply reset using hazard
            rate = ν(p.hazard, u[i], p.Θ)
            if rate > 0.0 && rand() < 1.0 - exp(-rate * dt)
                u[i] = p.c0 * u[i]
            end
        end
    end

    # Return statistics after burn-in
    burn_in_idx = Int(floor(burn_in / dt))
    u_final = u
    
    return (
        u = u_final,
        gbar = mean(u_final),
        V = mean(u_final.^2),
        degs = degs
    )
end

end # module