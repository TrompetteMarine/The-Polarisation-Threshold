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

# ----------------------------------------------------------------------
# 1. Single-agent simulation: OU with optional resets
# ----------------------------------------------------------------------

"""
    simulate_single_path(p; κ=0.0, T=200.0, dt=0.01, seed=0, with_resets=true)

Simulate a single-agent OU-with-resets trajectory with mean-field coupling κ.

Dynamics (discrete-time Euler–Maruyama approximation):
    x_{t+dt} = x_t + [-(λ - κ) x_t] dt + σ √dt ξ_t,
where ξ_t ~ N(0,1). When `with_resets=true`, at each step a reset event
occurs with hazard `ν(p.hazard, x, p.Θ)`; if a reset occurs, x is mapped
to `p.c0 * x`.

Returns a tuple `(t, x, reset_indices)` where:
  - `t` is the time grid,
  - `x` is the path,
  - `reset_indices` are the indices where a reset occurred.

Set `with_resets=false` to obtain the pure OU path.
"""
function simulate_single_path(p::Params;
                              κ::Float64 = 0.0,
                              T::Float64 = 200.0,
                              dt::Float64 = 0.01,
                              seed::Int   = 0,
                              with_resets::Bool = true)

    if seed != 0
        Random.seed!(seed)
    end

    steps = Int(floor(T / dt)) + 1
    t = collect(range(0.0, length = steps, step = dt))
    x = zeros(Float64, steps)
    reset_indices = Int[]

    λ = p.λ
    σsqrt = p.σ * sqrt(dt)

    @inbounds for i in 2:steps
        # Linearised drift around 0: −(λ − κ)x
        drift = (-(λ - κ) * x[i-1]) * dt
        shock = σsqrt * randn()
        x[i] = x[i-1] + drift + shock

        if with_resets
            rate = ν(p.hazard, x[i], p.Θ)
            if rate > 0.0
                # Poisson clock with intensity rate over dt
                if rand() < 1.0 - exp(-rate * dt)
                    x[i] = p.c0 * x[i]
                    push!(reset_indices, i)
                end
            end
        end
    end

    return t, x, reset_indices
end

# ----------------------------------------------------------------------
# 2. Stationary density estimator (histogram)
# ----------------------------------------------------------------------

"""
    stationary_density(p; κ=0.0, T=400.0, dt=0.01, burn_in=100.0,
                       nbins=80, seed=0, with_resets=true)

Estimate the stationary distribution of the single-agent process by simulating
a long path, discarding an initial `burn_in` period, and building a histogram.

Returns `(centers, probs)` where:
  - `centers` are the bin midpoints,
  - `probs` are the normalised bin masses (sum(probs) = 1).

Note: this is a discrete approximation to the stationary law, not a continuous
density. It is suitable for plotting and moment estimation.
"""
function stationary_density(p::Params;
                            κ::Float64 = 0.0,
                            T::Float64 = 400.0,
                            dt::Float64 = 0.01,
                            burn_in::Float64 = 100.0,
                            nbins::Int = 80,
                            seed::Int = 0,
                            with_resets::Bool = true)

    t, x, _ = simulate_single_path(p; κ = κ, T = T, dt = dt,
                                   seed = seed, with_resets = with_resets)

    start_idx = clamp(Int(floor(burn_in / dt)), 1, length(x))
    xs = @view x[start_idx:end]

    hist = fit(Histogram, xs; nbins = nbins, closed = :left)
    centers = midpoints(hist.edges[1])

    # Normalise weights to sum to 1 (discrete probability masses)
    weights = hist.weights
    total = sum(weights)
    probs = total > 0 ? weights ./ total : fill(0.0, length(weights))

    return centers, probs
end

# ----------------------------------------------------------------------
# 3. Leading odd eigenvalue of the linearised generator
# ----------------------------------------------------------------------

"""
    leading_odd_eigenvalue(p; κ; L=5.0, M=401, corr_tol=1e-3)

Construct a finite-difference approximation of the *forward* generator
for the linearised Fokker–Planck operator with resets, on a symmetric
grid `x ∈ [-L, L]` with `M` points. Then:

  1. Diagonalise the discretised operator `A`.
  2. Identify eigenvectors that are sufficiently "odd" (high correlation
     with the prototype odd vector x).
  3. Among those, pick the eigenvalue with the largest real part.

Returns `(λ_odd, xgrid)` where `λ_odd` is the **leading odd eigenvalue**
(closest to zero from the left in Re(λ)), and `xgrid` is the grid.

This is the quantity used to locate the critical κ* where λ_odd(κ*) = 0.
"""
function leading_odd_eigenvalue(p::Params;
                                κ::Float64,
                                L::Float64 = 5.0,
                                M::Int     = 401,
                                corr_tol::Float64 = 1e-3)

    @assert M ≥ 3 "Need at least 3 grid points"
    @assert isodd(M) "M should be odd so that x=0 is included in the grid"

    # Symmetric grid
    x = collect(range(-L, L, length = M))
    h = x[2] - x[1]

    # Diffusion coefficient in forward operator: (σ^2 / 2) ∂_xx
    σ2 = p.σ^2 / 2

    # Forward generator matrix A such that dρ/dt = A * ρ
    A = zeros(Float64, M, M)

    # -----------------------------
    # 1) DRIFT + DIFFUSION PART
    # -----------------------------
    @inbounds for i in 2:(M - 1)
        xi    = x[i]
        drift = (κ - p.λ) * xi       # b(x) in -∂_x(b ρ)

        # Diffusion: (σ^2/2) ∂_xx ρ
        A[i, i - 1] +=  σ2 / h^2
        A[i, i]     += -2σ2 / h^2
        A[i, i + 1] +=  σ2 / h^2

        # Drift: -∂_x(b ρ) with central differences
        # Approximate (b ρ)_x ≈ [b_{i+1} ρ_{i+1} - b_{i-1} ρ_{i-1}] / (2h)
        # In matrix form, that contributes:
        A[i, i - 1] +=  drift / (2h)
        A[i, i + 1] += -drift / (2h)
    end

    # Crude absorbing boundaries
    A[1, 1] = -10.0
    A[M, M] = -10.0

    # -----------------------------
    # 2) JUMP / RESET PART (FORWARD)
    # -----------------------------
    #
    # Continuous formula:
    #   (L^* ρ)(x)
    #     ... - ν(x) ρ(x)
    #         + (1/c0) ν(x/c0) ρ(x/c0)
    #
    # In discrete form: for each source y_j, we:
    #  - subtract ν_j ρ_j from row j (sink),
    #  - add ν_j ρ_j (scaled) into row i_dest, where x_i_dest ≈ c0*y_j.
    #
    @inbounds for j in 2:(M - 1)
        y   = x[j]                         # source location
        rate = ν(p.hazard, y, p.Θ)        # hazard at y

        if rate <= 0.0
            continue
        end

        # Loss at source y:  -rate * ρ_j
        A[j, j] += -rate

        # Destination of reset: x = c0 * y
        x_dest = p.c0 * y
        if x_dest < -L || x_dest > L
            continue   # landing point outside truncated domain
        end

        # Locate x_dest on the grid (linear interpolation between i and i+1)
        pos = (x_dest + L) / h + 1             # ∈ [1, M]
        i    = clamp(Int(floor(pos)), 1, M - 1)
        w    = pos - i                         # weight ∈ [0,1)

        # Optional: include 1/c0 Jacobian factor for exact mass conservation
        jac = 1.0 / p.c0

        # Gain at destination:  + jac * rate * ρ_j
        A[i,     j] += jac * rate * (1 - w)
        A[i + 1, j] += jac * rate * w
    end

    # -----------------------------
    # 3) EIGEN-DECOMPOSITION AND ODD MODE SELECTION
    # -----------------------------
    eig  = eigen(A)
    vals = eig.values
    vecs = eig.vectors

    # Prototype odd vector: x itself (odd on symmetric grid)
    x_vec = x ./ norm(x)

    # Oddness weight: |<v, x_vec>|
    weights = zeros(Float64, length(vals))
    @inbounds for k in eachindex(vals)
        v = vecs[:, k]
        weights[k] = abs(real(dot(v, x_vec)))
    end

    # Only keep eigenvectors that are sufficiently odd
    idxs = findall(w -> w > corr_tol, weights)
    if isempty(idxs)
        error("No sufficiently odd eigenvector found; max correlation = $(maximum(weights))")
    end

    # Among odd-ish modes, pick eigenvalue with largest real part
    real_vals = real.(vals[idxs])
    kmax      = idxs[argmax(real_vals)]

    λ_odd = real(vals[kmax])
    return λ_odd, x
end

# ----------------------------------------------------------------------
# 4. Order parameter from N-agent mean-field simulation
# ----------------------------------------------------------------------

"""
    order_parameter(p; κ, N=10_000, T=200.0, dt=0.01,
                    bias=1e-2, seed=0, max_abs=200.0)

Simulate an N-agent mean-field system with OU-with-resets dynamics and coupling κ.
Each agent state u_j follows:

    du_j = [-(λ - κ) u_j + κ (ḡ_t - u_j)] dt + σ dW_j + resets,

where ḡ_t is the empirical mean (implemented inside `euler_maruyama_step!`
and `reset_step!`).

We initialise with a small bias in the mean to break symmetry and obtain a
well-defined sign for the stationary branch. The order parameter is computed
as the time-average of the mean over the second half of the simulation.

Returns the estimated order parameter (long-run mean belief). If the process
blows up numerically (abs(mean) > max_abs), returns `NaN`.
"""
function order_parameter(p::Params;
                         κ::Float64,
                         N::Int = 10_000,
                         T::Float64 = 200.0,
                         dt::Float64 = 0.01,
                         bias::Float64 = 1e-2,
                         seed::Int = 0,
                         max_abs::Float64 = 200.0)

    if seed != 0
        Random.seed!(seed)
    end

    steps = Int(floor(T / dt))

    # Initialise near the OU stationary variance with a small bias
    u = randn(N) .* (p.σ / sqrt(2 * p.λ)) .+ bias

    window_start = max(1, Int(floor(0.5 * steps)))
    acc = 0.0
    count = 0

    @inbounds for s in 1:steps
        gbar = mean(u)
        euler_maruyama_step!(u, κ, gbar, p, dt)
        reset_step!(u, p, dt)

        if s >= window_start
            m = mean(u)
            # Basic blow-up safeguard
            if !isfinite(m) || abs(m) > max_abs
                return NaN
            end
            acc += m
            count += 1
        end
    end

    return count > 0 ? acc / count : mean(u)
end

# ----------------------------------------------------------------------
# 5. Welfare curves as simple functions of dispersion and amplitude
# ----------------------------------------------------------------------

"""
    compute_welfare_curves(p, κgrid; α=0.2, β=0.6, N=5_000, T=200.0,
                           dt=0.01, seed=0)

Compute simple decentralised and social welfare metrics over a grid of κ values
using summary statistics from `sweep_kappa`.

- `sweep_kappa(p, κgrid; ...)` is expected to return an object with fields:
    * `amp` : estimated order-parameter amplitude at each κ (e.g. |a(κ)|),
    * `V`   : estimated dispersion (variance) at each κ.

We define (elementwise):

    W_dec(κ) = -α * V(κ) - amp(κ)^2
    W_soc(κ) = W_dec(κ) - β * amp(κ)^2

where:
  - `α` weights dispersion costs,
  - `amp^2` captures the basic cost of polarisation,
  - `β * amp^2` is an additional externality term the planner internalises.

Returns a named tuple `(κ, W_dec, W_soc, amp, var)`.
"""
function compute_welfare_curves(p::Params,
                                κgrid::Vector{Float64};
                                α::Float64 = 0.2,
                                β::Float64 = 0.6,
                                N::Int = 5_000,
                                T::Float64 = 200.0,
                                dt::Float64 = 0.01,
                                seed::Int = 0)

    sweep = sweep_kappa(p, κgrid; N = N, T = T, dt = dt,
                        burn_in = T / 4, seed = seed)

    amp = sweep.amp             # assumed Vector{Float64}
    var = sweep.V               # assumed Vector{Float64}

    @assert length(amp) == length(κgrid)
    @assert length(var) == length(κgrid)

    # Decentralised welfare: penalise dispersion and polarisation
    W_dec = @. -α * var - amp^2

    # Social welfare: additional penalty on polarisation (externality)
    W_soc = @. W_dec - β * amp^2

    return (κ = κgrid, W_dec = W_dec, W_soc = W_soc, amp = amp, var = var)
end

end # module
