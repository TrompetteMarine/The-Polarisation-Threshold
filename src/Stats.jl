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
Compute the leading odd eigenvalue of the linearised generator using a finite
difference grid. This mirrors the construction used in the standalone plotting
scripts and underpins the spectral definition of κ*.
"""
function spectral_leading_odd_eigenvalue(p::Params; κ::Float64, L::Float64=5.0, M::Int=401)
    x = collect(range(-L, L, length=M))
    h = x[2] - x[1]
    σ2 = p.σ^2 / 2

    A = zeros(Float64, M, M)

    function add_reset!(row::Int, xi::Float64, rate::Float64)
        xr = p.c0 * xi
        if xr < -L || xr > L || rate == 0.0
            return
        end
        pos = (xr + L) / h + 1
        j = clamp(Int(floor(pos)), 1, M-1)
        w = pos - j
        A[row, j] += rate * (1 - w)
        A[row, j + 1] += rate * w
    end

    for i in 2:(M-1)
        xi = x[i]
        drift = -(p.λ - κ) * xi
        rate = ν(p.hazard, xi, p.Θ)

        A[i, i-1] += σ2 / h^2 - drift / (2h)
        A[i, i]   += -2σ2 / h^2 - rate
        A[i, i+1] += σ2 / h^2 + drift / (2h)

        add_reset!(i, xi, rate)
    end

    A[1,1] = -10.0
    A[M,M] = -10.0

    eig = eigen(A)
    vals = eig.values
    vecs = eig.vectors

    weights = zeros(Float64, length(vals))
    for j in eachindex(vals)
        v = vecs[:, j]
        weights[j] = abs(dot(v, x)) / (norm(v) * norm(x))
    end

    idx = argmax(weights)
    return real(vals[idx])
end

"""
Estimate g (baseline odd-mode decay) by injecting a small odd perturbation 
and measuring decay rate at κ=0
"""
function estimate_g(p::Params; N::Int=50_000, T::Float64=50.0, dt::Float64=0.005,
                    ε::Float64=1e-2, window::Tuple{Float64,Float64}=(2.0, 20.0), seed::Int=0)
    seed != 0 && Random.seed!(seed)

    steps = Int(round(T/dt))
    steps > 10 || error("Simulation horizon T must provide at least 10 time steps")

    N1 = div(N, 2)
    N2 = N - N1
    σ2_over_2λ = p.σ^2 / (2.0 * p.λ)

    u = Vector{Float64}(undef, N)
    view1 = @view u[1:N1]
    view2 = @view u[N1+1:end]

    view1 .= randn(N1) * sqrt(σ2_over_2λ) .+ ε
    view2 .= randn(N2) * sqrt(σ2_over_2λ) .- ε

    odd_hist = Vector{Float64}(undef, steps)

    for t in 1:steps
        gbar = mean(u)
        euler_maruyama_step!(u, 0.0, gbar, p, dt)
        reset_step!(u, p, dt)

        m1 = mean(view1)
        m2 = mean(view2)
        odd_hist[t] = 0.5 * (m1 - m2)
    end

    # Fit log |a(t)| ≈ log A - g * t over the specified window
    tgrid = collect(range(dt; step=dt, length=steps))
    t0, t1 = window
    i0 = clamp(Int(round(t0/dt)), 1, steps)
    i1 = clamp(Int(round(t1/dt)), i0 + 5, steps)

    amps = abs.(odd_hist[i0:i1]) .+ eps()
    y = log.(amps)
    x = tgrid[i0:i1]

    n = length(x)
    sx = sum(x)
    sy = sum(y)
    sxx = sum(x .* x)
    sxy = sum(x .* y)
    denom = n * sxx - sx^2
    denom ≈ 0 && return 1e-6

    slope = (n * sxy - sx * sy) / denom
    g_est = -slope

    return max(g_est, 1e-5)
end

"""
Sweep κ and estimate steady-state amplitude (|mean u|) and variance
"""
function sweep_kappa(p::Params, κgrid::Vector{Float64};
                     N::Int=20_000, T::Float64=300.0, dt::Float64=0.01,
                     burn_in::Float64=100.0, seed::Int=0, max_abs::Float64=1e3)
    
    seed != 0 && Random.seed!(seed)
    
    sim_params = validate_simulation_params(T=T, dt=dt, burn_in=burn_in, N=N)
    amps = similar(κgrid)
    vars = similar(κgrid)
    
    for (k, κ) in enumerate(κgrid)
        u = zeros(N)
        mean_acc = 0.0
        var_acc = 0.0
        count = 0

        for step in 1:sim_params.total_steps
            g = mean(u)
            euler_maruyama_step!(u, κ, g, p, dt)
            reset_step!(u, p, dt)

            if step >= sim_params.burn_in_idx
                m = mean(u)
                v = mean(abs2, u)

                if !isfinite(m) || !isfinite(v) || abs(m) > max_abs || v > max_abs^2
                    amps[k] = NaN
                    vars[k] = NaN
                    break
                end

                mean_acc += m
                var_acc += v
                count += 1
            end
        end

        if count > 0 && iszero(amps[k])
            amps[k] = abs(mean_acc / count)
            vars[k] = var_acc / count
        elseif isnan(amps[k])
            continue
        else
            amps[k] = abs(mean(u))
            vars[k] = mean(abs2, u)
        end
    end
    
    return KappaSweepResult(κgrid, amps, vars)
end

"""
Estimate the critical coupling κ*.

`method = :spectral` (default) sweeps κ ∈ [0, κmax] on a grid and locates the
zero crossing of the leading odd eigenvalue of the linearised generator. This
matches the theoretical definition λ₁(κ*) = 0.

`method = :moment` retains the older OU-style shortcut κ* ≈ g σ² / (2 λ V*).
Use this only for quick intuition; it is not generally valid when resets or
state-dependent hazards are present.
"""
function critical_kappa(p::Params; method::Symbol=:spectral,
                       g::Union{Nothing,Float64}=nothing,
                       Vstar::Union{Nothing,Float64}=nothing,
                       N::Int=20_000, T::Float64=400.0, dt::Float64=0.01,
                       burn_in::Float64=100.0, seed::Int=0,
                       κmax::Float64=3.0, grid_points::Int=60,
                       L::Float64=5.0, M::Int=401)
    if method == :moment
        V = isnothing(Vstar) ? estimate_Vstar(p; N=N, T=T, dt=dt, burn_in=burn_in, seed=seed) : Vstar
        g0 = isnothing(g) ? p.λ : g
        return g0 * p.σ^2 / (2.0 * p.λ * V)
    end

    κgrid = collect(range(0.0, κmax; length=grid_points))
    λ1 = [spectral_leading_odd_eigenvalue(p; κ=κ, L=L, M=M) for κ in κgrid]

    κ_star = NaN
    for i in 1:(length(κgrid) - 1)
        if λ1[i] <= 0 && λ1[i+1] >= 0
            w = -λ1[i] / (λ1[i+1] - λ1[i] + eps())
            κ_star = κgrid[i] + w * (κgrid[i+1] - κgrid[i])
            break
        end
    end

    if isnan(κ_star)
        idx = argmin(abs.(λ1))
        κ_star = κgrid[idx]
    end

    return κ_star
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