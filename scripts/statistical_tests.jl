module StatisticalTests

using Statistics
using LinearAlgebra
using Distributions
using LsqFit
using StatsBase

using Main.EnsembleUtils: EnsembleResults

export EnsembleStatistics,
       GrowthRateResult,
       AlternativeObservables,
       compute_ensemble_statistics,
       fit_growth_rate_ensemble,
       fit_growth_rate_multiwindow,
       select_best_growth_result,
       histogram_density,
       compute_alternative_observables,
       compute_bimodality_coefficient,
       compute_overlap_integral

struct EnsembleStatistics
    mean_traj::Vector{Float64}            # signed mean across runs
    mean_ci_lower::Vector{Float64}
    mean_ci_upper::Vector{Float64}
    mean_abs::Vector{Float64}             # mean of |m_r(t)|
    mean_abs_ci_lower::Vector{Float64}
    mean_abs_ci_upper::Vector{Float64}
    mean_rms::Vector{Float64}             # sqrt(mean of m_r(t)^2)
    mean_rms_ci_lower::Vector{Float64}
    mean_rms_ci_upper::Vector{Float64}
    var_traj::Vector{Float64}
    var_ci_lower::Vector{Float64}
    var_ci_upper::Vector{Float64}
    std_error::Vector{Float64}
end

struct GrowthRateResult
    lambda_mean::Float64
    lambda_std::Float64
    lambda_ci_lower::Float64
    lambda_ci_upper::Float64
    r_squared::Float64
    p_value::Float64
    is_significant::Bool
    window_used::Tuple{Float64, Float64}
    method::Symbol
end

struct AlternativeObservables
    time_grid::Vector{Float64}
    kurtosis_mean::Vector{Float64}
    kurtosis_ci_lower::Vector{Float64}
    kurtosis_ci_upper::Vector{Float64}
    bimodality_mean::Vector{Float64}
    bimodality_ci_lower::Vector{Float64}
    bimodality_ci_upper::Vector{Float64}
    overlap_times::Vector{Float64}
    overlap_mean::Vector{Float64}
    overlap_ci_lower::Vector{Float64}
    overlap_ci_upper::Vector{Float64}
    variance_growth_rate::Float64
end

function t_ci(samples::Vector{Float64}; level::Float64 = 0.95)
    n = length(samples)
    if n < 2
        m = mean(samples)
        return m, m
    end
    m = mean(samples)
    s = std(samples; corrected=true)
    if s == 0.0
        return m, m
    end
    se = s / sqrt(n)
    tcrit = quantile(TDist(n - 1), 0.5 + level / 2)
    return m - tcrit * se, m + tcrit * se
end

function compute_ensemble_statistics(results::EnsembleResults; level::Float64 = 0.95)
    n = size(results.mean_trajectories, 1)
    mean_traj = vec(mean(results.mean_trajectories; dims=1))
    mean_abs = vec(mean(abs.(results.mean_trajectories); dims=1))
    mean_sq = vec(mean(results.mean_trajectories .^ 2; dims=1))
    mean_rms = sqrt.(mean_sq)
    var_traj = vec(mean(results.var_trajectories; dims=1))

    mean_std = vec(std(results.mean_trajectories; dims=1, corrected=true))
    mean_abs_std = vec(std(abs.(results.mean_trajectories); dims=1, corrected=true))
    mean_sq_std = vec(std(results.mean_trajectories .^ 2; dims=1, corrected=true))
    var_std = vec(std(results.var_trajectories; dims=1, corrected=true))

    if n < 2
        mean_ci_lower = mean_traj
        mean_ci_upper = mean_traj
        mean_abs_ci_lower = mean_abs
        mean_abs_ci_upper = mean_abs
        mean_rms_ci_lower = mean_rms
        mean_rms_ci_upper = mean_rms
        var_ci_lower = var_traj
        var_ci_upper = var_traj
        std_error = zeros(length(mean_traj))
    else
        se = mean_std ./ sqrt(n)
        tcrit = quantile(TDist(n - 1), 0.5 + level / 2)
        mean_ci_lower = mean_traj .- tcrit .* se
        mean_ci_upper = mean_traj .+ tcrit .* se

        se_abs = mean_abs_std ./ sqrt(n)
        mean_abs_ci_lower = mean_abs .- tcrit .* se_abs
        mean_abs_ci_upper = mean_abs .+ tcrit .* se_abs

        se_sq = mean_sq_std ./ sqrt(n)
        mean_sq_ci_lower = mean_sq .- tcrit .* se_sq
        mean_sq_ci_upper = mean_sq .+ tcrit .* se_sq
        mean_rms_ci_lower = sqrt.(max.(mean_sq_ci_lower, 0.0))
        mean_rms_ci_upper = sqrt.(max.(mean_sq_ci_upper, 0.0))

        se_var = var_std ./ sqrt(n)
        var_ci_lower = var_traj .- tcrit .* se_var
        var_ci_upper = var_traj .+ tcrit .* se_var

        std_error = se
    end

    return EnsembleStatistics(
        mean_traj,
        mean_ci_lower,
        mean_ci_upper,
        mean_abs,
        mean_abs_ci_lower,
        mean_abs_ci_upper,
        mean_rms,
        mean_rms_ci_lower,
        mean_rms_ci_upper,
        var_traj,
        var_ci_lower,
        var_ci_upper,
        std_error,
    )
end

function fit_log_linear(times::Vector{Float64}, values::Vector{Float64};
                        window::Tuple{Float64, Float64},
                        weights::Union{Nothing, Vector{Float64}} = nothing)
    t_start, t_end = window
    idx = findall(t -> t_start <= t <= t_end, times)
    if length(idx) < 3
        return NaN, NaN, NaN
    end

    t_fit = times[idx]
    m_fit = abs.(values[idx])
    valid = findall(m_fit .> 1e-10)
    if length(valid) < 3
        return NaN, NaN, NaN
    end

    t_fit = t_fit[valid]
    log_m = log.(m_fit[valid])

    A = hcat(ones(length(t_fit)), t_fit)
    if weights === nothing
        coeffs = A \ log_m
    else
        w = weights[idx][valid]
        W = Diagonal(w)
        coeffs = (A' * W * A) \ (A' * W * log_m)
    end

    intercept = coeffs[1]
    lambda = coeffs[2]
    predictions = A * coeffs
    ss_res = sum((log_m .- predictions).^2)
    ss_tot = sum((log_m .- mean(log_m)).^2)
    r_squared = ss_tot == 0.0 ? NaN : (1 - ss_res / ss_tot)

    return lambda, intercept, r_squared
end

function fit_nonlinear_exp(times::Vector{Float64}, values::Vector{Float64};
                           window::Tuple{Float64, Float64})
    t_start, t_end = window
    idx = findall(t -> t_start <= t <= t_end, times)
    if length(idx) < 3
        return NaN, NaN, NaN
    end
    t_fit = times[idx]
    m_fit = abs.(values[idx])
    valid = findall(m_fit .> 1e-10)
    if length(valid) < 3
        return NaN, NaN, NaN
    end
    t_fit = t_fit[valid]
    m_fit = m_fit[valid]

    # Initial guess from log-linear
    lambda0, intercept0, _ = fit_log_linear(t_fit, m_fit; window=(t_fit[1], t_fit[end]))
    p0 = [exp(intercept0), lambda0]
    model(t, p) = p[1] .* exp.(p[2] .* t)
    fit = curve_fit(model, t_fit, m_fit, p0)
    p = fit.param
    lambda = p[2]
    preds = model(t_fit, p)
    ss_res = sum((m_fit .- preds).^2)
    ss_tot = sum((m_fit .- mean(m_fit)).^2)
    r_squared = ss_tot == 0.0 ? NaN : (1 - ss_res / ss_tot)
    return lambda, p[1], r_squared
end

function one_sample_t_pvalue(samples::Vector{Float64}; direction::Symbol = :two_sided)
    n = length(samples)
    if n < 2
        return NaN
    end
    m = mean(samples)
    s = std(samples; corrected=true)
    if s == 0.0
        return m == 0.0 ? 1.0 : 0.0
    end
    tstat = m / (s / sqrt(n))
    dist = TDist(n - 1)
    if direction == :greater
        return 1 - cdf(dist, tstat)
    elseif direction == :less
        return cdf(dist, tstat)
    else
        return 2 * (1 - cdf(dist, abs(tstat)))
    end
end

function fit_growth_rate_ensemble(
    times::Vector{Float64},
    ensemble_means::Matrix{Float64};
    fitting_window::Tuple{Float64, Float64} = (10.0, 50.0),
    method::Symbol = :log_linear,
    direction::Symbol = :two_sided,
)
    n = size(ensemble_means, 1)
    lambdas = fill(NaN, n)
    for i in 1:n
        lambda_i, _, _ = fit_log_linear(times, vec(ensemble_means[i, :]); window=fitting_window)
        lambdas[i] = lambda_i
    end

    valid = filter(!isnan, lambdas)
    n_valid = length(valid)
    if n_valid == 0
        lambda_mean = NaN
        lambda_std = NaN
        lambda_ci_lower = NaN
        lambda_ci_upper = NaN
        p_value = NaN
        is_significant = false
    else
        lambda_mean = mean(valid)
        lambda_std = std(valid; corrected=true)
        if n_valid < 2 || isnan(lambda_std) || lambda_std == 0.0
            lambda_ci_lower = lambda_mean
            lambda_ci_upper = lambda_mean
        else
            se = lambda_std / sqrt(n_valid)
            tcrit = quantile(TDist(n_valid - 1), 0.5 + 0.95 / 2)
            lambda_ci_lower = lambda_mean - tcrit * se
            lambda_ci_upper = lambda_mean + tcrit * se
        end
        p_value = one_sample_t_pvalue(valid; direction=direction)
        is_significant = !isnan(p_value) && p_value < 0.05
    end

    # R^2 on ensemble mean
    ensemble_mean = vec(mean(ensemble_means; dims=1))
    if method == :nonlinear
        _, _, r_squared = fit_nonlinear_exp(times, ensemble_mean; window=fitting_window)
        return GrowthRateResult(
            lambda_mean,
            lambda_std,
            lambda_ci_lower,
            lambda_ci_upper,
            r_squared,
            p_value,
            is_significant,
            fitting_window,
            method,
        )
    elseif method == :weighted
        se = vec(std(ensemble_means; dims=1, corrected=true)) ./ sqrt(n)
        weights = 1.0 ./ (se .^ 2 .+ eps())
        _, _, r_squared = fit_log_linear(times, ensemble_mean; window=fitting_window, weights=weights)
        return GrowthRateResult(
            lambda_mean,
            lambda_std,
            lambda_ci_lower,
            lambda_ci_upper,
            r_squared,
            p_value,
            is_significant,
            fitting_window,
            method,
        )
    else
        lambda_fit, _, r_squared = fit_log_linear(times, ensemble_mean; window=fitting_window)
        return GrowthRateResult(
            lambda_mean,
            lambda_std,
            lambda_ci_lower,
            lambda_ci_upper,
            r_squared,
            p_value,
            is_significant,
            fitting_window,
            method,
        )
    end
end

function fit_growth_rate_multiwindow(
    times::Vector{Float64},
    ensemble_means::Matrix{Float64};
    windows::Vector{Tuple{Float64, Float64}},
    method::Symbol = :log_linear,
    direction::Symbol = :two_sided,
)
    results = Dict{Tuple{Float64, Float64}, GrowthRateResult}()
    for w in windows
        results[w] = fit_growth_rate_ensemble(
            times,
            ensemble_means;
            fitting_window=w,
            method=method,
            direction=direction,
        )
    end
    return results
end

function select_best_growth_result(results::Dict{Tuple{Float64, Float64}, GrowthRateResult})
    best_key = first(keys(results))
    best = results[best_key]
    for (k, res) in results
        if !isnan(res.r_squared) && (isnan(best.r_squared) || res.r_squared > best.r_squared)
            best = res
            best_key = k
        end
    end
    return best
end

function histogram_density(data::Vector{Float64}, edges::Vector{Float64})
    hist = fit(Histogram, data, edges; closed=:left)
    weights = hist.weights
    total = sum(weights)
    binwidth = edges[2] - edges[1]
    density = total > 0 ? (weights ./ (total * binwidth)) : fill(0.0, length(weights))
    centers = midpoints(hist.edges[1])
    return centers, density
end

function compute_bimodality_coefficient(skew::Float64, kurt::Float64, n::Int)
    if n < 4
        return NaN
    end
    correction = 3 * (n - 1)^2 / ((n - 2) * (n - 3))
    return (skew^2 + 1) / (kurt + correction)
end

function compute_overlap_integral(centers::Vector{Float64}, density::Vector{Float64})
    if length(centers) < 2
        return NaN
    end
    binwidth = centers[2] - centers[1]
    overlap = 0.0
    for i in eachindex(centers)
        if centers[i] >= 0
            target = -centers[i]
            j = round(Int, (target - centers[1]) / binwidth) + 1
            if 1 <= j <= length(centers)
                overlap += min(density[i], density[j]) * binwidth
            end
        end
    end
    return overlap
end

function compute_alternative_observables(
    results::EnsembleResults,
    edges::Vector{Float64};
    level::Float64 = 0.95,
    var_window::Tuple{Float64, Float64} = (50.0, 150.0),
)
    n = size(results.mean_trajectories, 1)
    t = results.time_grid

    # Kurtosis and bimodality coefficient
    kurt_mean = vec(mean(results.kurt_trajectories; dims=1))
    kurt_std = vec(std(results.kurt_trajectories; dims=1, corrected=true))
    if n < 2
        kurt_ci_lower = kurt_mean
        kurt_ci_upper = kurt_mean
    else
        tcrit = quantile(TDist(n - 1), 0.5 + level / 2)
        se = kurt_std ./ sqrt(n)
        kurt_ci_lower = kurt_mean .- tcrit .* se
        kurt_ci_upper = kurt_mean .+ tcrit .* se
    end

    bimodality = Matrix{Float64}(undef, size(results.skew_trajectories)...)
    for i in 1:size(results.skew_trajectories, 1)
        for j in 1:size(results.skew_trajectories, 2)
            bimodality[i, j] = compute_bimodality_coefficient(
                results.skew_trajectories[i, j],
                results.kurt_trajectories[i, j],
                results.n_agents,
            )
        end
    end

    bimod_mean = vec(mean(bimodality; dims=1))
    bimod_std = vec(std(bimodality; dims=1, corrected=true))
    if n < 2
        bimod_ci_lower = bimod_mean
        bimod_ci_upper = bimod_mean
    else
        tcrit = quantile(TDist(n - 1), 0.5 + level / 2)
        se = bimod_std ./ sqrt(n)
        bimod_ci_lower = bimod_mean .- tcrit .* se
        bimod_ci_upper = bimod_mean .+ tcrit .* se
    end

    # Overlap integral computed from snapshots (if available)
    overlap_times = results.snapshot_times
    overlap_mean = Float64[]
    overlap_ci_lower = Float64[]
    overlap_ci_upper = Float64[]
    if !isempty(results.snapshots)
        n_times = length(overlap_times)
        n_ensemble = length(results.snapshots)
        for t_idx in 1:n_times
            overlaps = Float64[]
            for e_idx in 1:n_ensemble
                centers, dens = histogram_density(results.snapshots[e_idx][t_idx], edges)
                push!(overlaps, compute_overlap_integral(centers, dens))
            end
            push!(overlap_mean, mean(overlaps))
            lo, hi = t_ci(overlaps; level=level)
            push!(overlap_ci_lower, lo)
            push!(overlap_ci_upper, hi)
        end
    end

    # Variance growth rate from ensemble mean variance trajectory
    var_mean = vec(mean(results.var_trajectories; dims=1))
    var_diff = abs.(var_mean .- var_mean[end])
    lambda_var, _, _ = fit_log_linear(t, var_diff; window=var_window)
    lambda_var = 0.5 * lambda_var

    return AlternativeObservables(
        t,
        kurt_mean,
        kurt_ci_lower,
        kurt_ci_upper,
        bimod_mean,
        bimod_ci_lower,
        bimod_ci_upper,
        overlap_times,
        overlap_mean,
        overlap_ci_lower,
        overlap_ci_upper,
        lambda_var,
    )
end

end # module
