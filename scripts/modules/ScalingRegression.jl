module ScalingRegression

using LinearAlgebra
using Statistics
using Distributions
using Random

export ScalingFitResult,
       ScalingProfileResult,
       ScalingBootstrapResult,
       fit_scaling_ols,
       fit_scaling_wls,
       fit_scaling_profile,
       fit_scaling_bootstrap,
       select_scaling_points_robust,
       select_optimal_window,
       ols_var_covar

"""
    ScalingFitResult

Result from a single OLS or WLS scaling fit.
"""
struct ScalingFitResult
    alpha::Float64
    beta_hat::Float64
    beta_se::Float64
    beta_ci::Tuple{Float64,Float64}
    t_statistic::Float64
    p_value::Float64
    r2::Float64
    n_points::Int
    residuals::Vector{Float64}
    cooks_d::Vector{Float64}
    dw_stat::Float64
    method::Symbol
end

"""
    ScalingProfileResult

Result from profile-likelihood fit with free kappa*.
"""
struct ScalingProfileResult
    kappa_star_eff::Float64
    kappa_star_ci::Tuple{Float64,Float64}
    alpha::Float64
    beta_hat::Float64
    beta_se::Float64
    beta_ci::Tuple{Float64,Float64}
    t_statistic::Float64
    p_value::Float64
    r2::Float64
    C_fit::Float64
    n_points::Int
    rss_profile::Vector{Float64}
    kappa_star_grid::Vector{Float64}
    method::Symbol
    converged::Bool
end

"""
    ScalingBootstrapResult

Result from full-pipeline bootstrap.
"""
struct ScalingBootstrapResult
    beta_samples::Vector{Float64}
    kappa_star_samples::Vector{Float64}
    beta_hat::Float64
    beta_ci::Tuple{Float64,Float64}
    kappa_star_hat::Float64
    kappa_star_ci::Tuple{Float64,Float64}
    n_boot::Int
    n_successful::Int
    method::Symbol
end

"""
    ols_var_covar(X, sigma_sq) -> Matrix{Float64}

QR-based variance-covariance matrix for OLS/WLS.
"""
function ols_var_covar(X::Matrix{Float64}, sigma_sq::Float64)::Matrix{Float64}
    F = qr(X)
    R = F.R
    Ri = R \ Matrix{Float64}(I, size(R)...)
    return sigma_sq * (Ri * Ri')
end

"""
    fit_scaling_ols(x, y; beta_null=0.5, alpha=0.05)

OLS fit of y = a + βx with diagnostics.
"""
function fit_scaling_ols(
    x::Vector{Float64},
    y::Vector{Float64};
    beta_null::Float64 = 0.5,
    alpha::Float64 = 0.05,
)::ScalingFitResult
    n = length(x)
    @assert n >= 3 "Need at least 3 points for OLS (got $n)"

    X = hcat(ones(n), x)
    F = qr(X)
    coeffs = F \ y
    y_pred = X * coeffs
    residuals = y .- y_pred

    ss_res = sum(residuals .^ 2)
    ss_tot = sum((y .- mean(y)) .^ 2)
    r2 = ss_tot > 0 ? 1.0 - ss_res / ss_tot : NaN
    df = n - 2
    sigma_sq = ss_res / max(1, df)

    var_covar = ols_var_covar(X, sigma_sq)
    se_beta = sqrt(var_covar[2, 2])

    t_stat = (coeffs[2] - beta_null) / se_beta
    p_val = 2.0 * ccdf(TDist(df), abs(t_stat))
    t_crit = quantile(TDist(df), 1.0 - alpha / 2.0)
    ci = (coeffs[2] - t_crit * se_beta, coeffs[2] + t_crit * se_beta)

    # Cook's distance
    H = X * (F.R \ (F.R' \ X'))
    h = diag(H)
    p_params = 2
    cooks = (residuals .^ 2 ./ (p_params * sigma_sq)) .* (h ./ (1.0 .- h) .^ 2)

    # Durbin-Watson
    diffs = diff(residuals)
    dw = sum(diffs .^ 2) / ss_res

    return ScalingFitResult(
        coeffs[1], coeffs[2], se_beta, ci,
        t_stat, p_val, r2, n, residuals, cooks, dw, :ols,
    )
end

"""
    fit_scaling_wls(x, y, weights; beta_null=0.5, alpha=0.05)

Weighted least squares fit with diagnostics.
"""
function fit_scaling_wls(
    x::Vector{Float64},
    y::Vector{Float64},
    w::Vector{Float64};
    beta_null::Float64 = 0.5,
    alpha::Float64 = 0.05,
)::ScalingFitResult
    n = length(x)
    @assert n >= 3 "Need at least 3 points for WLS (got $n)"
    @assert length(w) == n "Weight vector length mismatch"

    w_safe = copy(w)
    valid_w = filter(wi -> isfinite(wi) && wi > 0, w_safe)
    med_w = isempty(valid_w) ? 1.0 : median(valid_w)
    for i in eachindex(w_safe)
        if !isfinite(w_safe[i]) || w_safe[i] <= 0
            w_safe[i] = med_w
        end
    end

    sqW = Diagonal(sqrt.(w_safe))
    X = hcat(ones(n), x)
    Xw = sqW * X
    yw = sqW * y

    F = qr(Xw)
    coeffs = F \ yw
    y_pred = X * coeffs
    residuals = y .- y_pred
    weighted_residuals = sqW * residuals
    ss_res_w = sum(weighted_residuals .^ 2)
    df = n - 2
    sigma_sq = ss_res_w / max(1, df)

    var_covar = ols_var_covar(Xw, sigma_sq)
    se_beta = sqrt(var_covar[2, 2])

    t_stat = (coeffs[2] - beta_null) / se_beta
    p_val = 2.0 * ccdf(TDist(df), abs(t_stat))
    t_crit = quantile(TDist(df), 1.0 - alpha / 2.0)
    ci = (coeffs[2] - t_crit * se_beta, coeffs[2] + t_crit * se_beta)

    ss_tot_w = sum(w_safe .* (y .- mean(y)) .^ 2)
    r2 = ss_tot_w > 0 ? 1.0 - ss_res_w / ss_tot_w : NaN

    Hw = Xw * (F.R \ (F.R' \ Xw'))
    hw = diag(Hw)
    cooks = (weighted_residuals .^ 2 ./ (2 * sigma_sq)) .* (hw ./ (1.0 .- hw) .^ 2)

    diffs = diff(weighted_residuals)
    dw = sum(diffs .^ 2) / ss_res_w

    return ScalingFitResult(
        coeffs[1], coeffs[2], se_beta, ci,
        t_stat, p_val, r2, n, residuals, cooks, dw, :wls,
    )
end

"""
    fit_scaling_profile(kappas, m_corr, kappa_star_init; ...)

Profile-likelihood fit with free kappa*.
"""
function fit_scaling_profile(
    kappas::Vector{Float64},
    m_corr::Vector{Float64},
    kappa_star_init::Float64;
    weights::Union{Nothing,Vector{Float64}} = nothing,
    grid_half_width::Float64 = 0.15,
    grid_n::Int = 201,
    beta_null::Float64 = 0.5,
    alpha::Float64 = 0.05,
    min_supercritical::Int = 8,
    amp_floor::Float64 = 1e-6,
)::ScalingProfileResult
    k_lo = kappa_star_init * (1.0 - grid_half_width)
    k_hi = kappa_star_init * (1.0 + grid_half_width)
    k_hi = min(k_hi, minimum(kappas) - 1e-8)
    if k_lo >= k_hi
        return _empty_profile_result(kappa_star_init, beta_null)
    end

    kstar_grid = collect(range(k_lo, k_hi; length=grid_n))
    rss_vec = fill(Inf, grid_n)
    fits = Vector{Union{Nothing,ScalingFitResult}}(nothing, grid_n)

    for (gi, kstar_cand) in enumerate(kstar_grid)
        supercrit = kappas .> kstar_cand
        above_floor = m_corr .> amp_floor
        valid = supercrit .& above_floor .& isfinite.(kappas) .& isfinite.(m_corr)
        if sum(valid) < min_supercritical
            continue
        end
        x = log.(kappas[valid] .- kstar_cand)
        y = log.(m_corr[valid])
        if !all(isfinite, x) || !all(isfinite, y)
            continue
        end
        try
            if weights !== nothing
                w_valid = weights[valid]
                f = fit_scaling_wls(x, y, w_valid; beta_null=beta_null, alpha=alpha)
            else
                f = fit_scaling_ols(x, y; beta_null=beta_null, alpha=alpha)
            end
            rss_vec[gi] = sum(f.residuals .^ 2)
            fits[gi] = f
        catch
            continue
        end
    end

    best_gi = argmin(rss_vec)
    if !isfinite(rss_vec[best_gi]) || fits[best_gi] === nothing
        return _empty_profile_result(kappa_star_init, beta_null)
    end

    best_fit = fits[best_gi]
    kstar_eff = kstar_grid[best_gi]
    rss_min = rss_vec[best_gi]
    n_best = best_fit.n_points
    chi2_thresh = rss_min * (1.0 + quantile(FDist(1, n_best - 2), 1.0 - alpha) / (n_best - 2))
    in_ci = findall(r -> isfinite(r) && r <= chi2_thresh, rss_vec)
    kstar_ci = isempty(in_ci) ? (kstar_eff, kstar_eff) :
               (kstar_grid[first(in_ci)], kstar_grid[last(in_ci)])

    return ScalingProfileResult(
        kstar_eff, kstar_ci,
        best_fit.alpha, best_fit.beta_hat, best_fit.beta_se, best_fit.beta_ci,
        best_fit.t_statistic, best_fit.p_value, best_fit.r2,
        exp(best_fit.alpha),
        best_fit.n_points,
        rss_vec, kstar_grid,
        :profile, true,
    )
end

function _empty_profile_result(kstar_init::Float64, beta_null::Float64)::ScalingProfileResult
    return ScalingProfileResult(
        kstar_init, (NaN, NaN),
        NaN, NaN, NaN, (NaN, NaN),
        NaN, NaN, NaN, NaN, 0,
        Float64[], Float64[],
        :profile, false,
    )
end

"""
    fit_scaling_bootstrap(kappas, m_corr, kappa_star_init; ...)

Bootstrap of the full scaling pipeline.
"""
function fit_scaling_bootstrap(
    kappas::Vector{Float64},
    m_corr::Vector{Float64},
    kappa_star_init::Float64;
    weights::Union{Nothing,Vector{Float64}} = nothing,
    n_boot::Int = 2000,
    alpha::Float64 = 0.05,
    min_supercritical::Int = 8,
    rng::AbstractRNG = Random.default_rng(),
)::ScalingBootstrapResult
    n = length(kappas)
    beta_samples = Float64[]
    kstar_samples = Float64[]
    sizehint!(beta_samples, n_boot)
    sizehint!(kstar_samples, n_boot)

    for _ in 1:n_boot
        idx = rand(rng, 1:n, n)
        k_b = kappas[idx]
        m_b = m_corr[idx]
        w_b = weights !== nothing ? weights[idx] : nothing
        try
            pf = fit_scaling_profile(
                k_b, m_b, kappa_star_init;
                weights=w_b,
                grid_n=101,
                min_supercritical=min_supercritical,
            )
            if pf.converged && isfinite(pf.beta_hat)
                push!(beta_samples, pf.beta_hat)
                push!(kstar_samples, pf.kappa_star_eff)
            end
        catch
            continue
        end
    end

    n_ok = length(beta_samples)
    if n_ok < 100
        return ScalingBootstrapResult(
            beta_samples, kstar_samples,
            NaN, (NaN, NaN), NaN, (NaN, NaN),
            n_boot, n_ok, :bootstrap,
        )
    end

    half_alpha = alpha / 2.0
    return ScalingBootstrapResult(
        beta_samples, kstar_samples,
        median(beta_samples),
        (quantile(beta_samples, half_alpha), quantile(beta_samples, 1.0 - half_alpha)),
        median(kstar_samples),
        (quantile(kstar_samples, half_alpha), quantile(kstar_samples, 1.0 - half_alpha)),
        n_boot, n_ok, :bootstrap,
    )
end

"""
    select_scaling_points_robust(df, kappa_star; ...)

Select points for scaling regression using robust filters.
"""
function select_scaling_points_robust(
    df,
    kappa_star::Float64;
    delta_window::Tuple{Float64,Float64} = (1e-2, 1e-1),
    amp_floor::Float64 = 1e-6,
    convergence_gate::Bool = true,
    decided_gate::Bool = true,
    pdec_min::Float64 = 0.80,
    cooks_threshold_factor::Float64 = 4.0,
)
    mask = isfinite.(df.m_corr) .& isfinite.(df.kappa) .& isfinite.(df.delta_abs)
    mask .&= df.delta_rel .>= delta_window[1]
    mask .&= df.delta_rel .<= delta_window[2]
    mask .&= df.delta_abs .> 0.0
    mask .&= df.m_corr .> amp_floor

    n_window = sum(mask)

    if convergence_gate && hasproperty(df, :converged)
        mask .&= df.converged
    end
    n_converged = sum(mask)

    if decided_gate && hasproperty(df, :decided_share)
        mask .&= df.decided_share .>= pdec_min
    end
    n_decided = sum(mask)

    n_pre_cook = sum(mask)
    n_cook_removed = 0
    if n_pre_cook >= 8
        x = log.(df.kappa[mask] .- kappa_star)
        y = log.(df.m_corr[mask])
        if all(isfinite, x) && all(isfinite, y)
            prelim = fit_scaling_ols(x, y)
            cook_thresh = cooks_threshold_factor / length(x)
            mask_indices = findall(mask)
            for (j, mi) in enumerate(mask_indices)
                if prelim.cooks_d[j] > cook_thresh
                    mask[mi] = false
                    n_cook_removed += 1
                end
            end
        end
    end

    n_final = sum(mask)
    diagnostics = (
        n_window = n_window,
        n_converged = n_converged,
        n_decided = n_decided,
        n_pre_cook = n_pre_cook,
        n_cook_removed = n_cook_removed,
        n_final = n_final,
        cooks_threshold = n_pre_cook >= 8 ? cooks_threshold_factor / n_pre_cook : NaN,
    )

    return mask, diagnostics
end

"""
    select_optimal_window(kappas, m_corr, kappa_star; ...)

BIC-based window selection for scaling regression.
"""
function select_optimal_window(
    kappas::Vector{Float64},
    m_corr::Vector{Float64},
    kappa_star::Float64;
    candidate_windows::Vector{Tuple{Float64,Float64}} = [
        (5e-3, 5e-2), (8e-3, 8e-2), (1e-2, 1e-1),
        (1e-2, 1.5e-1), (2e-2, 1e-1), (5e-3, 2e-1),
    ],
    min_points::Int = 12,
    amp_floor::Float64 = 1e-6,
)
    best_bic = Inf
    best_window = candidate_windows[1]
    results = NamedTuple[]

    delta = (kappas .- kappa_star) ./ kappa_star

    for (dmin, dmax) in candidate_windows
        sel = (delta .>= dmin) .& (delta .<= dmax) .&
              (kappas .> kappa_star) .& (m_corr .> amp_floor) .&
              isfinite.(kappas) .& isfinite.(m_corr)
        n = sum(sel)
        if n < min_points
            push!(results, (window=(dmin, dmax), n=n, bic=Inf, beta=NaN, r2=NaN))
            continue
        end

        x = log.(kappas[sel] .- kappa_star)
        y = log.(m_corr[sel])
        if !all(isfinite, x) || !all(isfinite, y)
            push!(results, (window=(dmin, dmax), n=n, bic=Inf, beta=NaN, r2=NaN))
            continue
        end

        fit = fit_scaling_ols(x, y)
        rss = sum(fit.residuals .^ 2)
        bic = n * log(rss / n + 1e-30) + 2 * log(n)
        push!(results, (window=(dmin, dmax), n=n, bic=bic, beta=fit.beta_hat, r2=fit.r2))

        if bic < best_bic
            best_bic = bic
            best_window = (dmin, dmax)
        end
    end

    return best_window, results
end

end
