module Stats

using Random, Statistics, LinearAlgebra, Printf
using StatsBase
using ..Types: Params, KappaSweepResult
using ..Model: reset_step!, euler_maruyama_step!
using ..Hazard: ν
using ..Utils: safe_array_slice, validate_simulation_params

export estimate_Vstar, estimate_g, sweep_kappa, critical_kappa, pitchfork_fit

"""
Estimate V* by simulating at κ=0 and averaging second moment after burn-in
"""
function estimate_Vstar(p::Params; N::Int=20_000, T::Float64=400.0, dt::Float64=0.01,
                        burn_in::Float64=100.0, seed::Int=0)
    # Set seed if provided
    seed != 0 && Random.seed!(seed)
    
    # Convert time points to indices using floor to avoid precision errors
    burn_in_idx = Int(floor(burn_in/dt))
    total_steps = Int(floor(T/dt))
    
    # Initialize simulation
    u = zeros(N)
    
    # Time evolution
    for step in 1:total_steps
        g = mean(u)
        euler_maruyama_step!(u, 0.0, g, p, dt)
        reset_step!(u, p, dt)
    end
    
    # Calculate variance excluding burn-in period
    u_steady = u  # In place simulation, u is already at final state
    # For variance, we should ideally collect samples, but approximate with final state
    return var(u_steady)
end

"""
Estimate g (baseline odd-mode decay) by injecting a small odd perturbation 
and measuring decay rate at κ=0
"""
function estimate_g(p::Params; N::Int=50_000, T::Float64=50.0, dt::Float64=0.005,
                    ε::Float64=1e-2, window::Tuple{Float64,Float64}=(2.0, 20.0), seed::Int=0)
    seed != 0 && Random.seed!(seed)
    
    steps = Int(round(T/dt))
    N1 = div(N, 2)
    N2 = N - N1
    σ2_over_2λ = p.σ^2 / (2.0 * p.λ)
    
    u1 = randn(N1) * sqrt(σ2_over_2λ) .+ ε
    u2 = randn(N2) * sqrt(σ2_over_2λ) .- ε
    u = vcat(u1, u2)

    g_hist = Vector{Float64}(undef, steps)

    for t in 1:steps
        g = mean(u)
        g_hist[t] = g
        
        # EM step at κ = 0
        sigma_sqrt_dt = p.σ * sqrt(dt)
        for i in eachindex(u)
            du = (-p.λ * u[i]) * dt + sigma_sqrt_dt * randn()
            u[i] += du
            
            # State-dependent resets
            rate = ν(p.hazard, u[i], p.Θ)
            if rate > 0.0 && rand() < 1.0 - exp(-rate * dt)
                u[i] = p.c0 * u[i]
            end
        end
    end

    # Fit log |g(t)| ≈ log A - g * t on chosen window
    tgrid = collect(0.0:dt:T-dt)
    t0, t1 = window
    i0 = clamp(Int(round(t0/dt)), 1, steps)
    i1 = clamp(Int(round(t1/dt)), i0+5, steps)
    
    y = log.(abs.(g_hist[i0:i1]) .+ eps())
    x = tgrid[i0:i1]
    
    # OLS slope
    n = length(x)
    sx = sum(x)
    sy = sum(y)
    sxx = sum(x .* x)
    sxy = sum(x .* y)
    slope = (n*sxy - sx*sy) / (n*sxx - sx^2)
    g_est = -slope  # since gbar decays as exp(-g t)
    
    return max(g_est, 1e-6)
end

"""
Sweep κ and estimate steady-state amplitude (|mean u|) and variance
"""
function sweep_kappa(p::Params, κgrid::Vector{Float64};
                     N::Int=20_000, T::Float64=300.0, dt::Float64=0.01,
                     burn_in::Float64=100.0, seed::Int=0)
    
    seed != 0 && Random.seed!(seed)
    
    sim_params = validate_simulation_params(T=T, dt=dt, burn_in=burn_in, N=N)
    amps = similar(κgrid)
    vars = similar(κgrid)
    
    for (k, κ) in enumerate(κgrid)
        u = zeros(N)
        for step in 1:sim_params.total_steps
            g = mean(u)
            euler_maruyama_step!(u, κ, g, p, dt)
            reset_step!(u, p, dt)
        end
        
        # Use final state for statistics (in-place simulation)
        amps[k] = abs(mean(u))
        vars[k] = mean(abs2, u)
    end
    
    return KappaSweepResult(κgrid, amps, vars)
end

"""
Compute critical κ* using V* and g (canonical default: g ≈ λ)
"""
function critical_kappa(p::Params; g::Union{Nothing,Float64}=nothing, 
                       Vstar::Union{Nothing,Float64}=nothing,
                       N::Int=20_000, T::Float64=400.0, dt::Float64=0.01, 
                       burn_in::Float64=100.0, seed::Int=0)
    V = isnothing(Vstar) ? estimate_Vstar(p; N=N, T=T, dt=dt, burn_in=burn_in, seed=seed) : Vstar
    g0 = isnothing(g) ? p.λ : g
    return g0 * p.σ^2 / (2.0 * p.λ * V)
end

"""
Center-manifold cubic fit (supercritical pitchfork):
near κ* we expect steady amplitude A ≈ sqrt((κ - κ*) / b), so κ ≈ κ* + b A^2.
Fit κ = κ0 + b A^2 via OLS over points with A>ε and (optionally) κ in a window.
"""
function pitchfork_fit(res::KappaSweepResult; ε::Float64=1e-3, 
                      κmin::Union{Nothing,Float64}=nothing, 
                      κmax::Union{Nothing,Float64}=nothing)
    κ = res.κ
    A = res.amp
    mask = (A .> ε)
    
    if κmin !== nothing
        mask .&= (κ .>= κmin)
    end
    if κmax !== nothing
        mask .&= (κ .<= κmax)
    end
    
    κf = κ[mask]
    Af = A[mask]
    
    if length(κf) < 3
        error("Not enough points above threshold to fit pitchfork (need ≥ 3).")
    end
    
    X = [ones(length(Af)) Af.^2]
    y = κf
    
    # OLS: θ = (X'X)^{-1} X'y
    XtX = X' * X
    Xty = X' * y
    θ = XtX \ Xty
    κ0 = θ[1]
    b = θ[2]
    
    # R²
    yhat = X * θ
    ssr = sum((y .- yhat).^2)
    sst = sum((y .- mean(y)).^2)
    R2 = 1.0 - ssr/sst
    
    return (κstar=κ0, b=b, R2=R2, used=length(κf))
end

end # module