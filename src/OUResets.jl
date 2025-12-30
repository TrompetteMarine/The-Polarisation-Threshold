module OUResets

using Random, LinearAlgebra, Statistics
using StatsBase
using ..Types: Params
using ..Hazard: ν
using ..Model: euler_maruyama_step!, reset_step!
using ..Stats: sweep_kappa

const _BIF_STATS = Ref((spectral = 0, fallback = 0))

"""
    reset_bifurcation_stats!()

Reset counters tracking how many times `bifurcation_loss` used the spectral
formula versus the fallback proxy.
"""
function reset_bifurcation_stats!()
    _BIF_STATS[] = (spectral = 0, fallback = 0)
end

"""
    get_bifurcation_stats()

Return a named tuple with counts `(spectral, fallback)` accumulated by
`bifurcation_loss`.
"""
function get_bifurcation_stats()
    return _BIF_STATS[]
end

export simulate_single_path, stationary_density,
       leading_odd_eigenvalue, order_parameter,
       compute_welfare_curves,
       compute_stationary_variance, compute_lambda1_and_derivative,
       compute_b_cubic, private_cost, bifurcation_loss, welfare_loss,
       welfare_private, welfare_social,
       reset_bifurcation_stats!, get_bifurcation_stats


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
density. It is suitable for plotting and moment estimation. Larger T yields
more stable histograms; consider T=600 for smoother density plots.
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
    leading_odd_eigenvalue(p; κ; L=5.0, M=401, corr_tol=1e-3,
                           return_diag=false, return_mats=false, return_ops=false)

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
                                corr_tol::Float64 = 1e-3,
                                return_diag::Bool = false,
                                return_mats::Bool = false,
                                return_ops::Bool = false)

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
    jump_rates = zeros(Float64, M)
    jump_loss_diag = zeros(Float64, M)
    @inbounds for j in 2:(M - 1)
        y   = x[j]                         # source location
        rate = ν(p.hazard, y, p.Θ)        # hazard at y
        jump_rates[j] = rate

        if rate <= 0.0
            continue
        end

        # Loss at source y:  -rate * ρ_j
        A[j, j] += -rate
        jump_loss_diag[j] += -rate

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

    eigval = vals[kmax]
    v = vecs[:, kmax]
    λ_odd = real(eigval)

    if !return_diag
        return λ_odd, x
    end

    nu0_expected = NaN
    if hasproperty(p.hazard, :ν0)
        nu0_expected = getproperty(p.hazard, :ν0)
    elseif hasproperty(p.hazard, :nu0)
        nu0_expected = getproperty(p.hazard, :nu0)
    end
    nu0_used = maximum(jump_rates)
    jump_coeff_summary = (minimum(jump_rates), maximum(jump_rates))

    idx_out = findall(i -> (i > 1) && (i < M) && (abs(x[i]) >= p.Θ) && (jump_rates[i] > 0), eachindex(x))
    u_sample = isempty(idx_out) ? x[end - 1] : x[idx_out[1]]
    nu_sample = ν(p.hazard, u_sample, p.Θ)
    nu_grid_sample = (u_sample, nu_sample)

    A_jump_diag_sample = Tuple{Float64, Float64}[]
    if !isempty(idx_out)
        mid = idx_out[cld(length(idx_out), 2)]
        for i in unique([idx_out[1], mid, idx_out[end]])
            push!(A_jump_diag_sample, (x[i], jump_loss_diag[i]))
        end
    end

    # Dense eigen-decomposition always returns a full spectrum; mark as converged.
    solver_resid = norm(A * v - eigval * v) / max(norm(v), eps())
    diag = (
        converged = isfinite(real(eigval)) && isfinite(imag(eigval)),
        niter = 0,
        info = :dense_eigen,
        solver_resid = solver_resid,
        eigval = eigval,
        x = v,
        xgrid = x,
        nu0_expected = nu0_expected,
        nu0_used = nu0_used,
        nu_grid_sample = nu_grid_sample,
        jump_coeff_summary = jump_coeff_summary,
        A_jump_diag_sample = A_jump_diag_sample,
        A = return_mats ? A : nothing,
        Mmat = return_mats ? I : nothing,
        apply_A = return_ops ? (u -> A * u) : nothing,
        apply_M = return_ops ? (u -> u) : nothing
    )
    return λ_odd, diag
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
function compute_welfare_curves(
    p::Params,
    κgrid::Vector{Float64};
    # --- dispersion / information terms ---
    αV::Float64   = 0.5,   # direct loss from high dispersion V*
    γ1::Float64   = 2.0,   # linear benefit from variance reduction
    γ2::Float64   = 1.0,   # curvature of benefit in (ΔV)^2  (γ2 > 0)
    # --- polarisation costs (private vs social) ---
    c_priv::Float64 = 1.0, # private weight on amp^2
    β_ext::Float64  = 0.8, # extra social weight on amp^2 (externality)
    φA::Float64     = 0.2, # optional quartic cost in amp^4 (tail risk)
    # --- simulation controls ---
    N::Int = 5_000,
    T::Float64 = 200.0,
    dt::Float64 = 0.01,
    seed::Int = 0,
)
    # MC sweep over κ
    sweep = sweep_kappa(p, κgrid;
                        N      = N,
                        T      = T,
                        dt     = dt,
                        burn_in = T/4,
                        seed   = seed)

    amp = sweep.amp      # polarisation amplitude
    var = sweep.V        # stationary dispersion V*
    @assert length(amp) == length(κgrid)
    @assert length(var) == length(κgrid)

    # ------------------------------------------------------------------
    # 1. Information gain: benefit from variance reduction
    #    relative to κ = 0 for this parameter set.
    # ------------------------------------------------------------------
    V0 = var[1]                     # dispersion at κ = 0
    ΔV = @. V0 - var                # >0 when interaction lowers dispersion
    # concave benefit in variance reduction: γ1 ΔV − γ2 (ΔV)^2
    ΔV2 = @. ΔV^2
    benefit = @. γ1 * ΔV - γ2 * ΔV2

    # ------------------------------------------------------------------
    # 2. Losses: dispersion + polarisation (quadratic / quartic)
    # ------------------------------------------------------------------
    amp2 = @. amp^2
    amp4 = @. amp^4

    # dispersion loss (kept simple: linear in V*)
    loss_var = @. αV * var

    # private polarisation cost
    priv_pol_cost = @. c_priv * amp2 + φA * amp4

    # social polarisation cost = private + externality
    soc_pol_cost  = @. (c_priv + β_ext) * amp2 + φA * amp4

    # ------------------------------------------------------------------
    # 3. Welfare objects
    # ------------------------------------------------------------------
    W_dec = @. benefit - loss_var - priv_pol_cost
    W_soc = @. benefit - loss_var - soc_pol_cost

    return (
        κ        = κgrid,
        W_dec    = W_dec,
        W_soc    = W_soc,
        amp      = amp,
        var      = var,
        benefit  = benefit,
        loss_var = loss_var,
        priv_pol = priv_pol_cost,
        soc_pol  = soc_pol_cost,
    )
end

# ----------------------------------------------------------------------
# 6. Bifurcation-based welfare analysis (Section 6 & Theorem 4.8)
# ----------------------------------------------------------------------

"""
    compute_stationary_variance(theta, c0, params; kwargs...)

Compute the stationary dispersion V* for given reset policy (θ, c₀) by solving
the variance balance equation:

    2λ V* = σ² + Λ_reset(V*; θ, c₀)

where Λ_reset captures the variance contribution from the reset mechanism.

Currently implemented via simulation at κ=0 with modified parameters.

# Arguments
- `theta::Float64`: Dissonance tolerance threshold
- `c0::Float64`: Reset contraction factor ∈ (0,1)
- `params::Params`: Base parameters (λ, σ, hazard type)

# Keyword Arguments
- `N::Int=20_000`: Number of agents
- `T::Float64=400.0`: Simulation time
- `dt::Float64=0.01`: Time step
- `burn_in::Float64=100.0`: Burn-in period
- `seed::Int=0`: Random seed

# Returns
`Float64`: Stationary variance V* > 0, or `NaN` if computation fails

# Theory
See Section 2, equation (2.9) for the variance balance equation.
"""
function compute_stationary_variance(
    theta::Float64,
    c0::Float64,
    params::Params;
    N::Int = 20_000,
    T::Float64 = 400.0,
    dt::Float64 = 0.01,
    burn_in::Float64 = 100.0,
    seed::Int = 0
)
    # Create modified params with new (Θ, c0)
    p_modified = Params(
        λ = params.λ,
        σ = params.σ,
        Θ = theta,
        c0 = c0,
        hazard = params.hazard
    )

    # Simulate at κ=0 to estimate V*
    seed != 0 && Random.seed!(seed)

    burn_in_idx = Int(floor(burn_in / dt))
    total_steps = Int(floor(T / dt))

    u = zeros(N)

    for step in 1:total_steps
        g = mean(u)
        euler_maruyama_step!(u, 0.0, g, p_modified, dt)
        reset_step!(u, p_modified, dt)
    end

    V = var(u)
    return isfinite(V) && V > 0 ? V : NaN
end

"""
    compute_lambda1_and_derivative(V, params; δκ=0.01, kwargs...)

Compute the leading odd eigenvalue λ₁₀(V) = λ₁(0; V) and its derivative
λ̇₁₀(V) = ∂λ₁/∂κ|_{κ=0} using finite differences.

# Arguments
- `V::Float64`: Stationary dispersion level
- `params::Params`: Parameters with (λ, σ, Θ, c₀, hazard)

# Keyword Arguments
- `δκ::Float64=0.01`: Finite difference step size
- `L::Float64=5.0`: Spatial domain truncation
- `M::Int=301`: Number of grid points (should be odd)

# Returns
`(lambda10, lambda1_dot)` where:
- `lambda10::Float64`: Leading odd eigenvalue at κ=0
- `lambda1_dot::Float64`: Derivative ∂λ₁/∂κ at κ=0

Returns `(NaN, NaN)` if computation fails.

# Theory
Section 4 shows λ₁(κ; V) crosses zero at κ*(V). Near κ=0:
    λ₁(κ; V) ≈ λ₁₀(V) + λ̇₁₀(V) · κ
Thus κ*(V) ≈ -λ₁₀(V) / λ̇₁₀(V) to first order.
"""
function compute_lambda1_and_derivative(
    V::Float64,
    params::Params;
    δκ::Float64 = 0.01,
    L::Float64 = 5.0,
    M::Int = 301
)
    try
        # Compute λ₁ at κ = 0, +δκ, -δκ
        λ1_0, _ = leading_odd_eigenvalue(params; κ = 0.0, L = L, M = M)
        λ1_plus, _ = leading_odd_eigenvalue(params; κ = δκ, L = L, M = M)
        λ1_minus, _ = leading_odd_eigenvalue(params; κ = -δκ, L = L, M = M)

        lambda10 = λ1_0
        lambda1_dot = (λ1_plus - λ1_minus) / (2 * δκ)

        if !isfinite(lambda10) || !isfinite(lambda1_dot)
            return (NaN, NaN)
        end

        return (lambda10, lambda1_dot)
    catch
        return (NaN, NaN)
    end
end

"""
    compute_b_cubic(V, params; b_default=0.5)

Compute the cubic coefficient b(V) from the normal form reduction:

    ȧ = μ(κ; V) a - b(V) a³

Currently uses a calibrated default value. This can be refined via
Dirichlet-form calculations (Section 5, centre-manifold reduction).

# Arguments
- `V::Float64`: Stationary dispersion level
- `params::Params`: Parameters

# Keyword Arguments
- `b_default::Float64=0.5`: Default cubic coefficient

# Returns
`Float64`: Cubic coefficient b(V) > 0

# Theory
Section 5 derives b(V) from a Dirichlet-form expression involving the
eigenfunction and nonlinearity. For now we use a calibrated default.
"""
function compute_b_cubic(
    V::Float64,
    params::Params;
    b_default::Float64 = 0.5
)
    # For now, use calibrated default
    # TODO: Implement Dirichlet-form computation from PhylogeneticDiagram.jl
    return b_default
end
"""
    private_cost(V, theta; αV, K_reset)

Private (decentralised) cost functional

    J_ind(θ, c₀) = αV * V*(θ, c₀) + K_reset / θ².

- `V`     : stationary dispersion V*(θ,c₀)
- `theta` : tolerance parameter θ > 0
- `αV`    : weight on dispersion
- `K_reset`: reset-frequency cost scale (~ OU hitting-time ≈ 1/θ²)
"""
function private_cost(
    V::Float64,
    theta::Float64;
    αV::Float64 = 1.0,
    K_reset::Float64 = 0.5,
)
    theta <= 0 && return Inf
    return αV * V + K_reset / (theta^2)
end

"""
    bifurcation_loss(V, params; c_pol, φA, κ_ratio_max, δκ, L, M, b_default, use_fallback)

Bifurcation / polarisation externality

    Φ(V*) = c_pol * E[a²] + φA * E[a⁴],

where `a` is the normal-form amplitude for κ > κ*(V), under κ ~ Uniform[κ*, κ_max]
with κ_max = κ_ratio_max * κ*(V).

Uses the same normal-form logic as before:

    a²(κ,V) ≈ (α_V / b(V)) (κ - κ*),
    E[a²]   = (α_V / b(V)) Δκ / 2,
    E[a⁴]   = (α_V / b(V))² Δκ² / 3,

with α_V = λ̇₁₀(V), Δκ = κ_max − κ*.
"""
function bifurcation_loss(
    V::Float64,
    params::Params;
    c_pol::Float64 = 1.5,
    φA::Float64 = 0.2,
    κ_ratio_max::Float64 = 2.0,
    δκ::Float64 = 0.01,
    L::Float64 = 5.0,
    M::Int = 301,
    b_default::Float64 = 0.5,
    lambda10::Union{Nothing, Float64} = nothing,
    lambda1_dot::Union{Nothing, Float64} = nothing,
    use_fallback::Bool = true,
)
    # --- spectral objects (as in old welfare_loss) -----------------------
    λ10_val, λ1_dot_val =
        lambda10 === nothing || lambda1_dot === nothing ?
        compute_lambda1_and_derivative(V, params; δκ = δκ, L = L, M = M) :
        (lambda10, lambda1_dot)

    if !isfinite(λ10_val) || !isfinite(λ1_dot_val) || abs(λ1_dot_val) < 1e-12
        if use_fallback
            stats = _BIF_STATS[]
            _BIF_STATS[] = (spectral = stats.spectral, fallback = stats.fallback + 1)
            return c_pol * V
        else
            return NaN
        end
    end

    κ_star = -λ10_val / λ1_dot_val
    if κ_star <= 0.0
        if use_fallback
            stats = _BIF_STATS[]
            _BIF_STATS[] = (spectral = stats.spectral, fallback = stats.fallback + 1)
            return c_pol * V
        else
            return NaN
        end
    end

    α_V = λ1_dot_val
    if α_V <= 0.0
        if use_fallback
            stats = _BIF_STATS[]
            _BIF_STATS[] = (spectral = stats.spectral, fallback = stats.fallback + 1)
            return c_pol * V
        else
            return NaN
        end
    end

    b_V = compute_b_cubic(V, params; b_default = b_default)
    if b_V <= 0.0
        if use_fallback
            stats = _BIF_STATS[]
            _BIF_STATS[] = (spectral = stats.spectral, fallback = stats.fallback + 1)
            return c_pol * V
        else
            return NaN
        end
    end

    κ_max = κ_ratio_max * κ_star
    Δκ    = κ_max - κ_star
    if Δκ <= 0.0
        if use_fallback
            stats = _BIF_STATS[]
            _BIF_STATS[] = (spectral = stats.spectral, fallback = stats.fallback + 1)
            return c_pol * V
        else
            return NaN
        end
    end

    # Normal-form moments of a
    coeff = α_V / b_V
    E_a2  = coeff * Δκ / 2
    E_a4  = coeff^2 * Δκ^2 / 3

    stats = _BIF_STATS[]
    _BIF_STATS[] = (spectral = stats.spectral + 1, fallback = stats.fallback)

    return c_pol * E_a2 + φA * E_a4
end

"""
    welfare_loss(V, theta, params; regime=:dec, ...)

Regime-specific welfare:

- Decentralised (private costs only):
      L_dec(θ,c₀) = J_ind(θ,c₀)
                  = αV * V*(θ,c₀) + K_reset / θ².

- Social planner (private costs + bifurcation externality):
      L_soc(θ,c₀) = J_ind(θ,c₀) + Φ(V*(θ,c₀)),
where Φ(V*) is given by `bifurcation_loss`.

This matches Section 6:

- agents minimise J_ind and ignore Φ;
- the planner minimises J_ind + Φ.
"""
function welfare_loss(
    V::Float64,
    theta::Float64,
    params::Params;
    regime::Symbol = :dec,
    αV::Float64 = 1.0,
    K_reset::Float64 = 0.5,
    c_pol::Float64 = 1.5,
    φA::Float64 = 0.2,
    κ_ratio_max::Float64 = 2.0,
    δκ::Float64 = 0.01,
    L::Float64 = 5.0,
    M::Int = 301,
    b_default::Float64 = 0.5,
    Phi::Union{Nothing, Float64} = nothing,
    lambda10::Union{Nothing, Float64} = nothing,
    lambda1_dot::Union{Nothing, Float64} = nothing,
    use_fallback::Bool = true,
)
    # Private part, shared by both regimes
    J_ind = private_cost(V, theta; αV = αV, K_reset = K_reset)
    if !isfinite(J_ind)
        return NaN
    end

    if regime == :dec
        # Pure private problem: no polarisation term
        return J_ind
    elseif regime == :soc
        Φ_val = Phi === nothing ? bifurcation_loss(
            V, params;
            c_pol       = c_pol,
            φA          = φA,
            κ_ratio_max = κ_ratio_max,
            δκ          = δκ,
            L           = L,
            M           = M,
            b_default   = b_default,
            lambda10    = lambda10,
            lambda1_dot = lambda1_dot,
            use_fallback = use_fallback,
        ) : Phi
        return isfinite(Φ_val) ? J_ind + Φ_val : NaN
    else
        error("Unknown regime $regime; expected :dec or :soc")
    end
end

"""
    welfare_private(V, θ, params; kwargs...)

Decentralised **welfare level** corresponding to the loss
`welfare_loss(…; regime=:dec)`:

    W_dec(θ,c₀) = -L_dec(θ,c₀).

Use this when you want to *maximise* instead of minimise.
"""
function welfare_private(
    V::Float64,
    theta::Float64,
    params::Params;
    kwargs...
)
    return -welfare_loss(V, theta, params; regime = :dec, kwargs...)
end

"""
    welfare_social(V, θ, params; kwargs...)

Social planner **welfare level** corresponding to the loss
`welfare_loss(…; regime=:soc)`:

    W_soc(θ,c₀) = -L_soc(θ,c₀) = -(J_ind + Φ).
"""
function welfare_social(
    V::Float64,
    theta::Float64,
    params::Params;
    kwargs...
)
    return -welfare_loss(V, theta, params; regime = :soc, kwargs...)
end


end #module
