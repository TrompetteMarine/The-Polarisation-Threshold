module PhylogeneticDiagram

using Statistics, Random, LinearAlgebra
using StatsBase: median

export NormalFormParams, PhylogeneticResult
export μ, normal_form_rhs, equilibria, stability
export phylogenetic_sweep, cluster_attractors
export calibrate_normal_form, scaling_exponent, theoretical_envelope

"""
    NormalFormParams

Parameters for the reduced normal form dynamics: ȧ = μ(κ)·a - b·a³

At κ = κ*, μ crosses zero (supercritical pitchfork bifurcation).
For κ > κ*: stable equilibria at a = ±√(μ/b)

# Fields
- `κ_star::Float64`: Critical coupling strength
- `b::Float64`: Cubic coefficient (> 0 for supercritical bifurcation)
- `μ_slope::Float64`: Transversality parameter dμ/dκ at κ*
- `Vstar::Float64`: Stationary dispersion (for calibration)
"""
struct NormalFormParams
    κ_star::Float64
    b::Float64
    μ_slope::Float64
    Vstar::Float64

    function NormalFormParams(κ_star, b, μ_slope, Vstar)
        b > 0 || throw(ArgumentError("Cubic coefficient b must be positive for supercritical bifurcation"))
        new(κ_star, b, μ_slope, Vstar)
    end
end

"""
    PhylogeneticResult

Results from phylogenetic bifurcation sweep.

# Fields
- `κ_vals::Vector{Float64}`: Coupling values
- `a_vals::Vector{Float64}`: Polarisation amplitudes
- `stable_mask::Vector{Bool}`: Stability classification
- `n_initial_conditions::Int`: Number of ICs per κ
- `params::NormalFormParams`: Normal form parameters used
"""
struct PhylogeneticResult
    κ_vals::Vector{Float64}
    a_vals::Vector{Float64}
    stable_mask::Vector{Bool}
    n_initial_conditions::Int
    params::NormalFormParams
end

"""
    μ(κ, nf::NormalFormParams)

Linear growth rate μ(κ) = μ_slope * (κ - κ*)
"""
μ(κ, nf::NormalFormParams) = nf.μ_slope * (κ - nf.κ_star)

"""
    normal_form_rhs(a, κ, nf::NormalFormParams)

Right-hand side of the reduced normal form: ȧ = μ(κ)·a - b·a³
"""
function normal_form_rhs(a, κ, nf::NormalFormParams)
    return μ(κ, nf) * a - nf.b * a^3
end

"""
    equilibria(κ, nf::NormalFormParams)

Compute all equilibria of the normal form at given κ.

Returns a vector of equilibrium values:
- For κ < κ*: [0.0] (consensus only)
- For κ ≥ κ*: [-a*, 0.0, a*] where a* = √(μ/b)
"""
function equilibria(κ, nf::NormalFormParams)
    μ_val = μ(κ, nf)
    if μ_val ≤ 0
        return [0.0]
    else
        a_star = sqrt(μ_val / nf.b)
        return [-a_star, 0.0, a_star]
    end
end

"""
    stability(a_eq, κ, nf::NormalFormParams)

Check stability of equilibrium a_eq at coupling κ.

Returns true if stable, false if unstable.
Stability criterion: dF/da = μ - 3ba² < 0
"""
function stability(a_eq, κ, nf::NormalFormParams)
    μ_val = μ(κ, nf)
    jacobian = μ_val - 3 * nf.b * a_eq^2
    return jacobian < 0
end

"""
    cluster_attractors(values::Vector{Float64}, tol::Float64=0.05)

Cluster nearby attractor values into unique representatives.

Uses a simple greedy algorithm: sort values and merge those within tolerance.
"""
function cluster_attractors(values::Vector{Float64}, tol::Float64=0.05)
    isempty(values) && return Float64[]

    sorted = sort(values)
    clusters = Float64[]

    current_cluster = [sorted[1]]

    for i in 2:length(sorted)
        if sorted[i] - sorted[i-1] < tol
            push!(current_cluster, sorted[i])
        else
            # Finalize current cluster with median
            push!(clusters, median(current_cluster))
            current_cluster = [sorted[i]]
        end
    end

    # Don't forget last cluster
    push!(clusters, median(current_cluster))

    return clusters
end

"""
    phylogenetic_sweep(nf::NormalFormParams; kwargs...)

Perform dense parameter sweep to generate phylogenetic bifurcation diagram.

This function tracks ALL equilibria (stable and unstable) analytically, showing
the complete bifurcation structure as κ increases.

# Arguments
- `nf::NormalFormParams`: Normal form parameters

# Keyword Arguments
- `κ_points::Int=500`: Number of κ values to sweep
- `κ_max_factor::Float64=2.0`: Upper bound as κ_max = κ_max_factor * κ*
- `n_ics::Int=30`: Number of initial conditions per κ (for validation/display density)
- `a_range::Tuple{Float64,Float64}=(-3.0, 3.0)`: Range for IC sampling (unused in analytical version)
- `T_burn::Float64=500.0`: Base burn-in time (unused in analytical version)
- `T_sample::Float64=100.0`: Sampling time (unused, kept for API)
- `dt::Float64=0.01`: Integration time step (unused in analytical version)
- `cluster_tol::Float64=0.05`: Tolerance for attractor clustering (unused in analytical version)
- `adaptive_burn::Bool=true`: Use adaptive burn-in near critical point (unused in analytical version)
- `seed::Int=0`: Random seed (0 = no seeding)

# Returns
`PhylogeneticResult` containing κ values, amplitudes, and stability classifications
for ALL equilibria (both stable and unstable)
"""
function phylogenetic_sweep(nf::NormalFormParams;
                            κ_points::Int=500,
                            κ_max_factor::Float64=2.0,
                            n_ics::Int=30,
                            a_range::Tuple{Float64,Float64}=(-3.0, 3.0),
                            T_burn::Float64=500.0,
                            T_sample::Float64=100.0,
                            dt::Float64=0.01,
                            cluster_tol::Float64=0.05,
                            adaptive_burn::Bool=true,
                            seed::Int=0)

    seed != 0 && Random.seed!(seed)

    κ_grid = range(0.0, κ_max_factor * nf.κ_star, length=κ_points)

    κ_vals = Float64[]
    a_vals = Float64[]
    stable_mask = Bool[]

    # Track ALL equilibria analytically (not just attractors)
    for κ in κ_grid
        eq_list = equilibria(κ, nf)

        for a_eq in eq_list
            push!(κ_vals, κ)
            push!(a_vals, a_eq)
            push!(stable_mask, stability(a_eq, κ, nf))
        end
    end

    return PhylogeneticResult(κ_vals, a_vals, stable_mask, n_ics, nf)
end

"""
    calibrate_normal_form(κ_star, Vstar; λ=1.0, σ=0.8, b_default=0.5)

Calibrate normal form parameters from micro-level parameters.

# Arguments
- `κ_star::Float64`: Critical coupling from spectral analysis
- `Vstar::Float64`: Stationary dispersion

# Keyword Arguments
- `λ::Float64=1.0`: Individual relaxation rate
- `σ::Float64=0.8`: Noise intensity
- `b_default::Float64=0.5`: Default cubic coefficient

# Returns
`NormalFormParams` with calibrated coefficients

# Theory
The growth rate is μ(κ) = (2λV*/σ²)(κ - κ*), giving:
    μ_slope = 2λV*/σ²

The cubic coefficient b is either provided or estimated from simulations.
"""
function calibrate_normal_form(κ_star, Vstar;
                               λ::Float64=1.0,
                               σ::Float64=0.8,
                               b_default::Float64=0.5)

    # Transversality from center manifold reduction
    σ_sq = max(σ^2, eps())
    μ_slope = (2.0 * λ * Vstar) / σ_sq

    # Use default cubic coefficient (could be refined with simulations)
    b = b_default

    return NormalFormParams(κ_star, b, μ_slope, Vstar)
end

"""
    theoretical_envelope(κ_grid, nf::NormalFormParams)

Compute theoretical envelope ±√(μ/b) for κ > κ*.

Returns a tuple (κ_theory, a_pos, a_neg) where a_pos and a_neg are the
positive and negative branches.
"""
function theoretical_envelope(κ_grid, nf::NormalFormParams)
    # Only compute for κ > κ*
    κ_above = filter(κ -> κ > nf.κ_star, κ_grid)

    if isempty(κ_above)
        return (Float64[], Float64[], Float64[])
    end

    a_pos = [sqrt(μ(κ, nf) / nf.b) for κ in κ_above]
    a_neg = -a_pos

    return (collect(κ_above), a_pos, a_neg)
end

"""
    scaling_exponent(result::PhylogeneticResult; κ_range_factor=(1.0, 1.5))

Verify scaling exponent β ≈ 0.5 for supercritical pitchfork.

Fits |a*(κ)| = C(κ - κ*)^β for κ in specified range above threshold.

Returns (β, C, r²) where β is the exponent, C is the prefactor, and r² is
the coefficient of determination.
"""
function scaling_exponent(result::PhylogeneticResult;
                         κ_range_factor::Tuple{Float64,Float64}=(1.0, 1.5))

    κ_star = result.params.κ_star

    # Filter to stable branches above threshold in specified range
    κ_min = κ_star * κ_range_factor[1]
    κ_max = κ_star * κ_range_factor[2]

    valid_idx = findall(result.stable_mask .&&
                       (result.κ_vals .> κ_min) .&&
                       (result.κ_vals .< κ_max) .&&
                       (abs.(result.a_vals) .> 0.01))

    if length(valid_idx) < 10
        @warn "Insufficient data points for scaling analysis"
        return (NaN, NaN, NaN)
    end

    # Log-log regression: log|a| = β*log(κ-κ*) + log(C)
    x = log.(result.κ_vals[valid_idx] .- κ_star)
    y = log.(abs.(result.a_vals[valid_idx]))

    # Simple linear regression
    n = length(x)
    x_mean = mean(x)
    y_mean = mean(y)

    β = sum((x .- x_mean) .* (y .- y_mean)) / sum((x .- x_mean).^2)
    log_C = y_mean - β * x_mean
    C = exp(log_C)

    # R² calculation
    y_pred = β .* x .+ log_C
    ss_res = sum((y .- y_pred).^2)
    ss_tot = sum((y .- y_mean).^2)
    r_squared = 1.0 - ss_res / ss_tot

    return (β, C, r_squared)
end

end # module
