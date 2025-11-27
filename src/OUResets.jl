module OUResets

using Random, LinearAlgebra, Statistics
using StatsBase
using ..Types: Params
using ..Hazard: ν
using ..Model: euler_maruyama_step!, reset_step!
using ..Stats: sweep_kappa

export simulate_single_path, stationary_density,
       leading_odd_eigenvalue, order_parameter,
       compute_welfare_curves

"""
    simulate_single_path(p; κ=0.0, T=200.0, dt=0.01, seed=0, with_resets=true)

Simulate a single-agent OU-with-resets trajectory. Returns a tuple `(t, x, reset_indices)`
where `t` is the time grid, `x` the path, and `reset_indices` the indices where a reset occurred.
Set `with_resets=false` to obtain the pure OU path.
"""
function simulate_single_path(p::Params; κ::Float64=0.0, T::Float64=200.0,
                              dt::Float64=0.01, seed::Int=0, with_resets::Bool=true)
    seed != 0 && Random.seed!(seed)

    steps = Int(floor(T / dt)) + 1
    t = collect(range(0.0, length=steps, step=dt))
    x = zeros(Float64, steps)
    reset_indices = Int[]

    σsqrt = p.σ * sqrt(dt)
    λ = p.λ

    for i in 2:steps
        drift = (-λ * x[i-1] + κ * x[i-1]) * dt
        shock = σsqrt * randn()
        x[i] = x[i-1] + drift + shock

        if with_resets
            rate = ν(p.hazard, x[i], p.Θ)
            if rate > 0.0 && rand() < 1.0 - exp(-rate * dt)
                x[i] = p.c0 * x[i]
                push!(reset_indices, i)
            end
        end
    end

    return t, x, reset_indices
end

"""
    stationary_density(p; κ=0.0, T=400.0, dt=0.01, burn_in=100.0, nbins=80,
                       seed=0, with_resets=true)

Estimate the stationary density by simulating a long single-agent path and
computing a histogram after discarding an initial `burn_in` period. Returns
`(centers, density)` where density integrates to one.
"""
function stationary_density(p::Params; κ::Float64=0.0, T::Float64=400.0,
                            dt::Float64=0.01, burn_in::Float64=100.0,
                            nbins::Int=80, seed::Int=0, with_resets::Bool=true)
    t, x, _ = simulate_single_path(p; κ=κ, T=T, dt=dt, seed=seed, with_resets=with_resets)
    start_idx = clamp(Int(floor(burn_in / dt)), 1, length(x))
    xs = @view x[start_idx:end]

    hist = fit(Histogram, xs, nbins; closed=:left)
    centers = midpoints(hist.edges[1])
    dens = normalize(hist.weights, 1)
    return centers, dens
end

"""
    leading_odd_eigenvalue(p; κ; L=5.0, M=401)

Build a finite-difference generator for the linearised Fokker–Planck operator
with resets and return the leading (largest real part) odd eigenvalue. The grid
spans `[-L, L]` with `M` points.
"""
function leading_odd_eigenvalue(p::Params; κ::Float64, L::Float64=5.0, M::Int=401)
    x = collect(range(-L, L, length=M))
    h = x[2] - x[1]
    σ2 = p.σ^2 / 2

    A = zeros(Float64, M, M)

    # Helper for interpolating reset landing position
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

    # Simple absorbing boundaries
    A[1,1] = -10.0
    A[M,M] = -10.0

    # Eigen decomposition
    eig = eigen(A)
    vals = eig.values
    vecs = eig.vectors

    # Pick eigenvalue most aligned with odd basis x
    weights = similar(vals)
    for j in eachindex(vals)
        v = vecs[:, j]
        weights[j] = abs(dot(v, x)) / (norm(v) * norm(x))
    end
    idx = argmax(weights)
    return real(vals[idx]), x
end

"""
    order_parameter(p; κ, N=10_000, T=200.0, dt=0.01, bias=1e-2, seed=0)

Simulate an N-agent system with a small mean-field bias (positive or negative)
so that the long-run sign is well defined. Returns the final mean belief.
"""
function order_parameter(p::Params; κ::Float64, N::Int=10_000, T::Float64=200.0,
                         dt::Float64=0.01, bias::Float64=1e-2, seed::Int=0)
    seed != 0 && Random.seed!(seed)

    steps = Int(floor(T / dt))
    u = randn(N) .* (p.σ / sqrt(2 * p.λ)) .+ bias

    for _ in 1:steps
        gbar = mean(u)
        euler_maruyama_step!(u, κ, gbar, p, dt)
        reset_step!(u, p, dt)
    end

    return mean(u)
end

"""
    compute_welfare_curves(p, κgrid; α=0.2, β=0.6, N=5_000, T=200.0,
                           dt=0.01, seed=0)

Compute simple decentralised and social welfare metrics based on simulated
order-parameter amplitudes and variances. Returns a named tuple with fields
`κ`, `W_dec`, `W_soc`, `amp`, and `var`.
"""
function compute_welfare_curves(p::Params, κgrid::Vector{Float64}; α::Float64=0.2,
                                β::Float64=0.6, N::Int=5_000, T::Float64=200.0,
                                dt::Float64=0.01, seed::Int=0)
    sweep = sweep_kappa(p, κgrid; N=N, T=T, dt=dt, burn_in=T/4, seed=seed)
    amp = sweep.amp
    var = sweep.V

    # Simple welfare: consensus utility minus dispersion and polarisation costs
    W_dec = @. -α * var - amp^2
    W_soc = @. W_dec - β * amp^2

    return (κ=κgrid, W_dec=W_dec, W_soc=W_soc, amp=amp, var=var)
end

end # module
