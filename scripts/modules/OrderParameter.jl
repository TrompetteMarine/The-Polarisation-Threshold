module OrderParameter

using Statistics
using Distributions
using StatsBase

using ..BranchClassification: late_window_indices

export compute_m_abs_star,
       estimate_baseline_m0,
       correct_amplitude,
       correct_amplitude_vector,
       measure_polarization_variance,
       measure_polarization_bimodal,
       estimate_baseline_variance_from_sweep

"""
    compute_m_abs_star(mean_traj, time_grid) -> (mbar, m_abs_star, m_abs_ci, mean_abs_traj)

Compute late-window |m| averages per run and the ensemble mean.
Returns per-run averages, mean, CI, and the full mean-abs trajectory.
"""
function compute_m_abs_star(
    mean_traj::Matrix{Float64},
    time_grid::Vector{Float64},
)::Tuple{Vector{Float64}, Float64, Float64, Vector{Float64}}
    idx = late_window_indices(time_grid)
    n_runs = size(mean_traj, 1)
    mbar = [mean(abs.(mean_traj[i, idx])) for i in 1:n_runs]
    mean_abs_traj = vec(mean(abs.(mean_traj); dims=1))
    m_abs_star = mean(mbar)
    m_abs_ci = n_runs > 1 ? 1.96 * std(mbar; corrected=true) / sqrt(n_runs) : NaN
    return mbar, m_abs_star, m_abs_ci, mean_abs_traj
end

"""
    estimate_baseline_m0(kappas, m_abs_values, kappa_star; delta_range=(-0.30,-0.10), trim_fraction=0.10)

Estimate the finite-N baseline m₀ using trimmed mean well below κ*.
"""
function estimate_baseline_m0(
    kappas::Vector{Float64},
    m_abs_values::Vector{Float64},
    kappa_star::Float64;
    delta_range::Tuple{Float64,Float64} = (-0.30, -0.10),
    trim_fraction::Float64 = 0.10,
)::Tuple{Float64,Float64,Int}
    delta = (kappas .- kappa_star) ./ kappa_star
    mask = (delta .>= delta_range[1]) .& (delta .<= delta_range[2]) .&
           isfinite.(m_abs_values) .& (m_abs_values .>= 0)
    vals = m_abs_values[mask]
    n = length(vals)
    if n == 0
        @warn "No subcritical points in delta_range=$delta_range for m0 estimation"
        return (0.0, NaN, 0)
    end
    if n < 4
        return (median(vals), NaN, n)
    end
    sorted = sort(vals)
    trim_n = max(1, round(Int, trim_fraction * n))
    lo = trim_n + 1
    hi = n - trim_n
    if lo > hi
        return (median(vals), NaN, n)
    end
    trimmed = sorted[lo:hi]
    m0 = mean(trimmed)
    m0_se = std(trimmed; corrected=true) / sqrt(length(trimmed))
    return (m0, m0_se, length(trimmed))
end

"""
    correct_amplitude(m_abs, m_abs_se, m0, m0_se) -> (m_corr, m_corr_se)

Baseline correction with SE propagation.
"""
function correct_amplitude(
    m_abs::Float64,
    m_abs_se::Float64,
    m0::Float64,
    m0_se::Float64,
)::Tuple{Float64,Float64}
    excess = m_abs^2 - m0^2
    if excess <= 0.0
        return (0.0, NaN)
    end
    m_corr = sqrt(excess)
    se_from_abs = (m_abs / m_corr) * (isfinite(m_abs_se) ? m_abs_se : 0.0)
    se_from_m0 = (m0 / m_corr) * (isfinite(m0_se) ? m0_se : 0.0)
    m_corr_se = sqrt(se_from_abs^2 + se_from_m0^2)
    return (m_corr, m_corr_se)
end

"""
    correct_amplitude_vector(m_abs_vec, m_abs_se_vec, m0, m0_se) -> (m_corr_vec, m_corr_se_vec)

Vectorised baseline correction with SE propagation.
"""
function correct_amplitude_vector(
    m_abs_vec::Vector{Float64},
    m_abs_se_vec::Vector{Float64},
    m0::Float64,
    m0_se::Float64,
)::Tuple{Vector{Float64},Vector{Float64}}
    n = length(m_abs_vec)
    m_corr = Vector{Float64}(undef, n)
    m_corr_se = Vector{Float64}(undef, n)
    for i in 1:n
        m_corr[i], m_corr_se[i] = correct_amplitude(m_abs_vec[i], m_abs_se_vec[i], m0, m0_se)
    end
    return m_corr, m_corr_se
end

"""
    measure_polarization_variance(snapshots; baseline_var=0.0)

Measure polarization amplitude via excess variance.
"""
function measure_polarization_variance(
    snapshots::Vector{Vector{Float64}};
    baseline_var::Float64 = 0.0,
)
    if isempty(snapshots)
        return (
            a_star = NaN,
            a_star_se = NaN,
            variance_mean = NaN,
            variance_std = NaN,
            excess_variance = NaN,
            n_runs = 0,
            method = "variance",
        )
    end

    V_per_run = [var(snap) for snap in snapshots if !isempty(snap)]
    n_runs = length(V_per_run)
    if n_runs == 0
        return (
            a_star = NaN,
            a_star_se = NaN,
            variance_mean = NaN,
            variance_std = NaN,
            excess_variance = NaN,
            n_runs = 0,
            method = "variance",
        )
    end

    V_mean = mean(V_per_run)
    V_std = std(V_per_run)
    V_excess = max(0.0, V_mean - baseline_var)
    a_star = sqrt(V_excess)
    V_se = V_std / sqrt(n_runs)
    a_star_se = V_excess > 1e-10 ? V_se / (2 * sqrt(V_excess)) : NaN

    return (
        a_star = a_star,
        a_star_se = a_star_se,
        variance_mean = V_mean,
        variance_std = V_std,
        excess_variance = V_excess,
        n_runs = n_runs,
        method = "variance",
    )
end

"""
    measure_polarization_bimodal(samples)

Estimate polarization via symmetric Gaussian mixture.
"""
function measure_polarization_bimodal(samples::Vector{Float64})
    n = length(samples)
    if n < 10
        return (a_star = NaN, sigma = NaN, converged = false, method = "mixture")
    end

    m2 = mean(samples .^ 2)
    σ_init = std(samples) / sqrt(2)
    μ_init = sqrt(max(0.0, m2 - σ_init^2))

    μ_hat = μ_init
    σ_hat = σ_init
    for _ in 1:50
        w_plus = zeros(n)
        for i in 1:n
            x = samples[i]
            p_plus = exp(-0.5 * ((x - μ_hat) / σ_hat)^2)
            p_minus = exp(-0.5 * ((x + μ_hat) / σ_hat)^2)
            w_plus[i] = p_plus / (p_plus + p_minus + 1e-300)
        end
        μ_new = sum(w_plus .* samples .- (1 .- w_plus) .* samples) / n
        μ_new = abs(μ_new)
        residuals_sq = w_plus .* (samples .- μ_new).^2 .+
                       (1 .- w_plus) .* (samples .+ μ_new).^2
        σ_new = sqrt(sum(residuals_sq) / n)
        if abs(μ_new - μ_hat) < 1e-6 && abs(σ_new - σ_hat) < 1e-6
            μ_hat, σ_hat = μ_new, σ_new
            break
        end
        μ_hat, σ_hat = μ_new, σ_new
    end

    return (a_star = μ_hat, sigma = σ_hat, converged = true, method = "mixture")
end

"""
    estimate_baseline_variance_from_sweep(variances, kappas, kappa_star)

Estimate baseline variance as the minimum variance across the sweep.
"""
function estimate_baseline_variance_from_sweep(
    variances::Vector{Float64},
    kappas::Vector{Float64},
    kappa_star::Float64,
)
    if isempty(variances) || isempty(kappas)
        return (
            V_baseline = NaN,
            V_baseline_kappa = NaN,
            V_baseline_ratio = NaN,
            method = "minimum",
        )
    end

    finite_idx = findall(isfinite, variances)
    if isempty(finite_idx)
        return (
            V_baseline = NaN,
            V_baseline_kappa = NaN,
            V_baseline_ratio = NaN,
            method = "minimum",
        )
    end

    idx_min = finite_idx[argmin(variances[finite_idx])]
    V_min = variances[idx_min]
    kappa_at_min = kappas[idx_min]
    ratio_at_min = kappa_at_min / kappa_star

    return (
        V_baseline = V_min,
        V_baseline_kappa = kappa_at_min,
        V_baseline_ratio = ratio_at_min,
        method = "minimum",
    )
end

end # module
