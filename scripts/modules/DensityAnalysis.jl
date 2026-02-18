module DensityAnalysis

using Statistics
using StatsBase
using Distributions

using ..EnsembleUtils: EnsembleResults
using ..BranchClassification: late_window_indices

export compute_edges_from_snapshots,
       histogram_density,
       summarize_density_stack,
       stack_vectors,
       compute_density_moments,
       run_density_sanity_checks

"""
    compute_edges_from_snapshots(results_dict; bins=240, symmetric=true)

Compute histogram bin edges from ensemble snapshot data.
"""
function compute_edges_from_snapshots(
    results_dict::Dict{String, EnsembleResults};
    bins::Int = 240,
    symmetric::Bool = true,
)
    umin = Inf
    umax = -Inf
    for res in values(results_dict)
        if isempty(res.snapshots)
            continue
        end
        for snaps in res.snapshots
            for u in snaps
                umin = min(umin, minimum(u))
                umax = max(umax, maximum(u))
            end
        end
    end
    if !isfinite(umin) || !isfinite(umax)
        error("Unable to compute histogram edges (no snapshots available).")
    end
    if symmetric
        maxabs = max(abs(umin), abs(umax))
        pad = 0.05 * max(1e-6, maxabs)
        return collect(range(-maxabs - pad, maxabs + pad; length=bins + 1))
    else
        pad = 0.05 * max(1e-6, umax - umin)
        return collect(range(umin - pad, umax + pad; length=bins + 1))
    end
end

"""
    histogram_density(data, edges) -> (centers, density)

Compute density histogram on fixed edges.
"""
function histogram_density(data::Vector{Float64}, edges::Vector{Float64})
    hist = fit(Histogram, data, edges; closed=:left)
    weights = hist.weights
    total = sum(weights)
    binwidth = edges[2] - edges[1]
    density = total > 0 ? (weights ./ (total * binwidth)) : fill(0.0, length(weights))
    centers = midpoints(hist.edges[1])
    return centers, density
end

"""
    summarize_density_stack(dens_stack; level=0.95)

Compute mean and CI bands for density stacks.
"""
function summarize_density_stack(dens_stack::Matrix{Float64}; level::Float64 = 0.95)
    n = size(dens_stack, 1)
    if n == 0
        n_bins = size(dens_stack, 2)
        return fill(NaN, n_bins), fill(NaN, n_bins), fill(NaN, n_bins)
    end
    mean_vec = vec(mean(dens_stack; dims=1))
    if n < 2
        return mean_vec, fill(NaN, length(mean_vec)), fill(NaN, length(mean_vec))
    end
    std_vec = vec(std(dens_stack; dims=1, corrected=true))
    se = std_vec ./ sqrt(n)
    tcrit = quantile(TDist(n - 1), 0.5 + level / 2)
    return mean_vec, mean_vec .- tcrit .* se, mean_vec .+ tcrit .* se
end

"""
    stack_vectors(vecs, n_bins) -> Matrix

Stack vectors into a matrix with shape (n, n_bins).
"""
function stack_vectors(vecs::Vector{Vector{Float64}}, n_bins::Int)
    if isempty(vecs)
        return zeros(0, n_bins)
    end
    mat = Matrix{Float64}(undef, length(vecs), n_bins)
    for (i, v) in enumerate(vecs)
        mat[i, :] .= v
    end
    return mat
end

"""
    compute_density_moments(x, density) -> NamedTuple

Compute mean, variance, skewness, kurtosis from density.
"""
function compute_density_moments(x::Vector{Float64}, density::Vector{Float64})
    dx = x[2] - x[1]
    mass = sum(density) * dx
    if mass <= 0
        return (mean=NaN, var=NaN, skew=NaN, kurt=NaN)
    end
    μ = sum(x .* density) * dx / mass
    centered = x .- μ
    var = sum((centered .^ 2) .* density) * dx / mass
    std = sqrt(var)
    if std == 0
        return (mean=μ, var=var, skew=0.0, kurt=0.0)
    end
    skew = sum((centered .^ 3) .* density) * dx / (mass * std^3)
    kurt = sum((centered .^ 4) .* density) * dx / (mass * std^4)
    return (mean=μ, var=var, skew=skew, kurt=kurt)
end

"""
    run_density_sanity_checks(centers, density; tol=1e-2) -> Bool

Basic checks: integral close to 1 and symmetry around zero.
"""
function run_density_sanity_checks(
    centers::Vector{Float64},
    density::Vector{Float64};
    tol::Float64 = 1e-2,
)
    dx = centers[2] - centers[1]
    mass = sum(density) * dx
    ok_mass = isfinite(mass) && abs(mass - 1.0) <= tol
    # symmetry check (coarse)
    n = length(centers)
    sym = true
    for i in 1:div(n, 2)
        if abs(density[i] - density[end - i + 1]) > 5 * tol
            sym = false
            break
        end
    end
    return ok_mass && sym
end

end
