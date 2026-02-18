module BifurcationTests

using Statistics
using StatsBase
using HypothesisTests
using Random

using ..OrderParameter
using ..ScalingRegression

export test_scaling_exponent_multi,
       test_scaling_exponent,
       test_hysteresis,
       bootstrap_kappa_star,
       estimate_kappa_star_from_data,
       aggregate_verdict,
       ScalingTestMultiResult

"""
    ScalingTestMultiResult

Composite scaling test result holding OLS/WLS/profile/boot fits.
"""
struct ScalingTestMultiResult
    ols::ScalingFitResult
    wls::Union{ScalingFitResult,Nothing}
    profile::Union{ScalingProfileResult,Nothing}
    bootstrap::Union{ScalingBootstrapResult,Nothing}

    n_methods_pass::Int
    pass::Bool
    primary_beta::Float64
    primary_ci::Tuple{Float64,Float64}
    primary_method::Symbol
    verdict::String

    kappa_star_eff::Float64
    optimal_window::Tuple{Float64,Float64}
    m0::Float64
    m0_se::Float64
end

"""
    test_scaling_exponent_multi(kappas, m_abs, m_abs_se, kappa_star_B; ...)

Multi-method scaling test with consensus.
"""
function test_scaling_exponent_multi(
    kappas::Vector{Float64},
    m_abs::Vector{Float64},
    m_abs_se::Vector{Float64},
    kappa_star_B::Float64;
    m0_delta_range::Tuple{Float64,Float64} = (-0.30, -0.10),
    scaling_alpha::Float64 = 0.05,
    min_points::Int = 12,
    run_bootstrap::Bool = false,
    n_boot::Int = 2000,
)::ScalingTestMultiResult
    m0, m0_se, _ = OrderParameter.estimate_baseline_m0(
        kappas, m_abs, kappa_star_B; delta_range=m0_delta_range)
    m_corr, m_corr_se = OrderParameter.correct_amplitude_vector(m_abs, m_abs_se, m0, m0_se)

    best_window, _ = ScalingRegression.select_optimal_window(kappas, m_corr, kappa_star_B)

    delta_rel = (kappas .- kappa_star_B) ./ kappa_star_B
    sel = (delta_rel .>= best_window[1]) .& (delta_rel .<= best_window[2]) .&
          (kappas .> kappa_star_B) .& (m_corr .> 1e-6) .&
          isfinite.(kappas) .& isfinite.(m_corr)

    if sum(sel) < min_points
        return _empty_multi_result(kappa_star_B, best_window, m0, m0_se)
    end

    x_fixed = log.(kappas[sel] .- kappa_star_B)
    y = log.(m_corr[sel])

    ols_fit = ScalingRegression.fit_scaling_ols(x_fixed, y; beta_null=0.5, alpha=scaling_alpha)

    cook_thresh = 4.0 / length(x_fixed)
    clean = ols_fit.cooks_d .<= cook_thresh
    if sum(clean) >= min_points && sum(.!clean) > 0
        ols_fit = ScalingRegression.fit_scaling_ols(x_fixed[clean], y[clean];
            beta_null=0.5, alpha=scaling_alpha)
    end

    wls_fit = nothing
    se_valid = m_corr_se[sel]
    se_in_log = se_valid ./ (m_corr[sel] .+ 1e-12)
    w = 1.0 ./ (se_in_log .^ 2 .+ 1e-12)
    if all(isfinite, w) && any(w .> 0)
        try
            wls_fit = ScalingRegression.fit_scaling_wls(x_fixed, y, w;
                beta_null=0.5, alpha=scaling_alpha)
        catch
        end
    end

    prof_fit = ScalingRegression.fit_scaling_profile(
        kappas[isfinite.(m_corr) .& (m_corr .> 1e-6)],
        m_corr[isfinite.(m_corr) .& (m_corr .> 1e-6)],
        kappa_star_B;
        weights=nothing,
    )

    boot_fit = nothing
    if run_bootstrap
        boot_fit = ScalingRegression.fit_scaling_bootstrap(
            kappas[isfinite.(m_corr) .& (m_corr .> 1e-6)],
            m_corr[isfinite.(m_corr) .& (m_corr .> 1e-6)],
            kappa_star_B;
            n_boot=n_boot,
        )
    end

    contains_half(ci) = ci[1] <= 0.5 <= ci[2]
    ols_pass = contains_half(ols_fit.beta_ci)
    wls_pass = wls_fit !== nothing && contains_half(wls_fit.beta_ci)
    prof_pass = prof_fit.converged && contains_half(prof_fit.beta_ci)
    _boot_pass = boot_fit !== nothing && boot_fit.n_successful >= 100 &&
                 contains_half(boot_fit.beta_ci)

    methods_tested = 2 + (wls_fit !== nothing ? 1 : 0)
    methods_pass = ols_pass + wls_pass + prof_pass
    pass = methods_pass >= 2

    if prof_fit.converged
        primary_beta = prof_fit.beta_hat
        primary_ci = prof_fit.beta_ci
        primary_method = :profile
    else
        primary_beta = ols_fit.beta_hat
        primary_ci = ols_fit.beta_ci
        primary_method = :ols
    end

    verdict = if pass
        "PASS: $(methods_pass)/$(methods_tested) methods have CI containing 0.5 (primary: $(primary_method))"
    else
        "FAIL: only $(methods_pass)/$(methods_tested) methods have CI containing 0.5"
    end

    kstar_eff = prof_fit.converged ? prof_fit.kappa_star_eff : kappa_star_B

    return ScalingTestMultiResult(
        ols_fit, wls_fit, prof_fit, boot_fit,
        methods_pass, pass, primary_beta, primary_ci, primary_method, verdict,
        kstar_eff, best_window, m0, m0_se,
    )
end

function _empty_multi_result(kstar::Float64, window::Tuple{Float64,Float64}, m0::Float64, m0_se::Float64)
    empty_ols = ScalingFitResult(
        NaN, NaN, NaN, (NaN, NaN), NaN, NaN, NaN, 0,
        Float64[], Float64[], NaN, :ols)
    return ScalingTestMultiResult(
        empty_ols, nothing, nothing, nothing,
        0, false, NaN, (NaN, NaN), :none,
        "FAIL: insufficient data for any scaling method",
        kstar, window, m0, m0_se,
    )
end

"""
    test_scaling_exponent(kappas, m_abs, kappa_star; ...)

Legacy single-method scaling test.
"""
function test_scaling_exponent(
    kappas::Vector{Float64},
    m_abs::Vector{Float64},
    kappa_star::Float64;
    beta_null::Float64 = 0.5,
    alpha::Float64 = 0.05,
)::NamedTuple
    sel = (kappas .> kappa_star) .& (m_abs .> 0) .& isfinite.(m_abs)
    if sum(sel) < 3
        return (beta_hat=NaN, beta_se=NaN, beta_ci=(NaN, NaN),
                t_statistic=NaN, p_value=NaN, n_points=sum(sel),
                pass=false, verdict="INSUFFICIENT DATA")
    end

    x = log.(kappas[sel] .- kappa_star)
    y = log.(m_abs[sel])
    fit = ScalingRegression.fit_scaling_ols(x, y; beta_null=beta_null, alpha=alpha)
    pass = fit.beta_ci[1] <= beta_null <= fit.beta_ci[2]

    return (
        beta_hat = fit.beta_hat,
        beta_se = fit.beta_se,
        beta_ci = fit.beta_ci,
        t_statistic = fit.t_statistic,
        p_value = fit.p_value,
        n_points = fit.n_points,
        pass = pass,
        verdict = pass ? "PASS" : "FAIL",
    )
end

"""
    test_hysteresis(forward_vals, backward_vals)

Two-sample t-test on forward/backward order parameter sequences.
"""
function test_hysteresis(
    forward_vals::Vector{Float64},
    backward_vals::Vector{Float64},
)::NamedTuple
    if isempty(forward_vals) || isempty(backward_vals)
        return (mean_difference=NaN, max_difference=NaN, t_statistic=NaN,
                p_value=NaN, pass=false, verdict="INSUFFICIENT DATA")
    end

    min_len = min(length(forward_vals), length(backward_vals))
    f = forward_vals[1:min_len]
    b = backward_vals[1:min_len]
    diffs = f .- b
    mean_diff = mean(abs.(diffs))
    max_diff = maximum(abs.(diffs))

    ttest = HypothesisTests.OneSampleTTest(diffs)
    p_val = pvalue(ttest)
    pass = p_val > 0.05
    verdict = pass ? "PASS" : "FAIL"

    return (
        mean_difference = mean_diff,
        max_difference = max_diff,
        t_statistic = ttest.t,
        p_value = p_val,
        pass = pass,
        verdict = verdict,
    )
end

"""
    estimate_kappa_star_from_data(kappas, values)

Linear interpolation of the zero-crossing of `values` as a function of `kappas`.
"""
function estimate_kappa_star_from_data(
    kappas::Vector{Float64},
    values::Vector{Float64},
)::Float64
    idx = sortperm(kappas)
    k = kappas[idx]
    v = values[idx]
    for i in 1:(length(k) - 1)
        if v[i] == 0
            return k[i]
        elseif v[i] * v[i + 1] < 0
            t = abs(v[i]) / (abs(v[i]) + abs(v[i + 1]))
            return k[i] * (1 - t) + k[i + 1] * t
        end
    end
    return k[argmin(abs.(v))]
end

"""
    bootstrap_kappa_star(kappas, values; n_boot=200)

Bootstrap CI for kappa* based on zero-crossing.
"""
function bootstrap_kappa_star(
    kappas::Vector{Float64},
    values::Vector{Float64};
    n_boot::Int = 200,
    rng::AbstractRNG = Random.default_rng(),
)::NamedTuple
    n = length(kappas)
    boots = Vector{Float64}(undef, n_boot)
    for b in 1:n_boot
        idx = rand(rng, 1:n, n)
        boots[b] = estimate_kappa_star_from_data(kappas[idx], values[idx])
    end
    lo = quantile(boots, 0.025)
    hi = quantile(boots, 0.975)
    return (kappa_star_ci=(lo, hi), verdict="$(round((lo+hi)/2, digits=3)) +/- $(round((hi-lo)/2, digits=3))")
end

"""
    aggregate_verdict(results...) -> Bool

Aggregate multiple test pass flags.
"""
function aggregate_verdict(results::Bool...)
    return all(results)
end

end
