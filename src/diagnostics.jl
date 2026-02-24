module Diagnostics

using Statistics
using Random
using Distributions
using FFTW

using BeliefSim.Types: Params

export fit_log_growth, estimate_kappa_star_from_scan, growth_scan_kappa_star,
       estimate_V0_from_sweep, stationarity_gate, compute_atilde_stats,
       jump_detector, hysteresis_detector, fit_beta_window, fit_beta_windows,
       fit_alpha, psd_peak_metrics

"""
    fit_log_growth(times, values, window)

Log-linear fit of log(|m(t)|) over a time window. Returns slope, se, r2, n.
"""
function fit_log_growth(times::Vector{Float64}, values::Vector{Float64}, window::Tuple{Float64,Float64})
    t0, t1 = window
    idx = findall(t -> t >= t0 && t <= t1, times)
    if length(idx) < 3
        return (slope=NaN, intercept=NaN, r2=NaN, se=NaN, n=length(idx))
    end
    t_fit = times[idx]
    m_fit = abs.(values[idx])
    valid = findall(m_fit .> 1e-12)
    if length(valid) < 3
        return (slope=NaN, intercept=NaN, r2=NaN, se=NaN, n=length(valid))
    end
    t_fit = t_fit[valid]
    log_m = log.(m_fit[valid])
    X = hcat(ones(length(t_fit)), t_fit)
    coeffs = X \ log_m
    yhat = X * coeffs
    ss_res = sum((log_m .- yhat).^2)
    ss_tot = sum((log_m .- mean(log_m)).^2)
    r2 = ss_tot > 0 ? 1 - ss_res / ss_tot : NaN
    se = sqrt(ss_res / max(1, length(t_fit) - 2) / sum((t_fit .- mean(t_fit)).^2))
    return (slope=coeffs[2], intercept=coeffs[1], r2=r2, se=se, n=length(t_fit))
end

function estimate_kappa_star_from_scan(kappa_grid::Vector{Float64}, slopes::Vector{Float64})
    idx = sortperm(kappa_grid)
    kappa = kappa_grid[idx]
    s = slopes[idx]
    for i in 1:(length(kappa) - 1)
        if s[i] == 0
            return kappa[i]
        elseif s[i] * s[i + 1] < 0
            t = abs(s[i]) / (abs(s[i]) + abs(s[i + 1]))
            return kappa[i] * (1 - t) + kappa[i + 1] * t
        end
    end
    return kappa[argmin(abs.(s))]
end

"""
    growth_scan_kappa_star(p, kappa_grid; ...)

Operational kappa_star_eff via growth scan of |m(t)|. Returns slope grid,
bootstrap slopes, bootstrapped kappa*, and SE.
"""
function growth_scan_kappa_star(
    run_ensemble_fn::Function,
    p::Params,
    kappa_grid::Vector{Float64};
    N::Int,
    T::Float64,
    dt::Float64,
    n_ensemble::Int,
    seed_base::Int,
    impl::Symbol,
    mt_stride::Int,
    epsilon::Float64,
    window::Tuple{Float64,Float64},
    n_boot::Int,
    ordered_value::Float64,
)
    slopes = Vector{Float64}(undef, length(kappa_grid))
    boot_slopes = n_boot > 0 ? Matrix{Float64}(undef, n_boot, length(kappa_grid)) : Matrix{Float64}(undef, 0, 0)

    for (i, kappa) in enumerate(kappa_grid)
        res = run_ensemble_fn(
            p;
            kappa=kappa,
            N=N,
            T=T,
            dt=dt,
            n_ensemble=n_ensemble,
            seed_base=seed_base + 10000 * i,
            impl=impl,
            mt_stride=mt_stride,
            init_mode=:odd,
            epsilon=epsilon,
            ordered_value=ordered_value,
        )
        mean_abs = vec(mean(abs.(res.mean_traj); dims=1))
        fit = fit_log_growth(res.time_grid, mean_abs, window)
        slopes[i] = fit.slope

        if n_boot > 0
            for b in 1:n_boot
                sample_idx = rand(1:n_ensemble, n_ensemble)
                mean_boot = vec(mean(abs.(res.mean_traj[sample_idx, :]); dims=1))
                boot_fit = fit_log_growth(res.time_grid, mean_boot, window)
                boot_slopes[b, i] = boot_fit.slope
            end
        end
    end

    boot_kappa = Float64[]
    if n_boot > 0 && size(boot_slopes, 1) > 0
        sizehint!(boot_kappa, size(boot_slopes, 1))
        for b in 1:size(boot_slopes, 1)
            ks = estimate_kappa_star_from_scan(kappa_grid, boot_slopes[b, :])
            push!(boot_kappa, ks)
        end
    end

    kappa_star = estimate_kappa_star_from_scan(kappa_grid, slopes)
    kappa_se = !isempty(boot_kappa) ? std(boot_kappa) : NaN

    return (kappa_star=kappa_star, kappa_se=kappa_se, slopes=slopes, boot_kappa=boot_kappa)
end

"""
    estimate_V0_from_sweep(variances, kappas, kappa_star; fit_frac=0.3)

Quadratic fit around kappa_star; fallback to minimum if too few points.
Matches fig6_ensemble_enhanced2.jl operational definition.
"""
function estimate_V0_from_sweep(
    variances::Vector{Float64},
    kappas::Vector{Float64},
    kappa_star::Float64;
    fit_frac::Float64 = 0.3,
)
    rel = (kappas .- kappa_star) ./ kappa_star
    mask = abs.(rel) .<= fit_frac
    if sum(mask) < 5
        finite_v = variances[isfinite.(variances)]
        if isempty(finite_v)
            return (V0=NaN, method="none")
        end
        return (V0=minimum(finite_v), method="minimum")
    end
    X = [ones(sum(mask)) (rel[mask].^2)]
    y = variances[mask]
    coeffs = X \ y
    V0 = max(coeffs[1], 0.0)
    return (V0=V0, method="quadratic")
end

"""
    compute_atilde_stats(kappas, per_kappa_samples, V0, theta)

Compute Atilde mean and SE for each kappa.
"""
function compute_atilde_stats(
    kappas::Vector{Float64},
    per_kappa_samples::Vector{Vector{Float64}},
    V0::Float64,
    theta::Float64,
)
    n = length(kappas)
    atilde_mean = fill(NaN, n)
    atilde_se = fill(NaN, n)
    for i in 1:n
        runs = per_kappa_samples[i]
        if isempty(runs)
            continue
        end
        A = [sqrt(max(v - V0, 0.0)) / theta for v in runs]
        atilde_mean[i] = mean(A)
        atilde_se[i] = std(A) / sqrt(max(length(A), 1))
    end
    return (atilde_mean=atilde_mean, atilde_se=atilde_se)
end

"""
    stationarity_gate(block_means; sigma=2.5)

Returns (stationary::Bool, max_dev::Float64, se::Float64).
"""
function stationarity_gate(block_means::Vector{Float64}; sigma::Float64 = 2.5)
    if length(block_means) < 2
        return (stationary=true, max_dev=0.0, se=NaN)
    end
    m = mean(block_means)
    se = std(block_means) / sqrt(length(block_means))
    max_dev = maximum(abs.(block_means .- m))
    if !isfinite(se) || se == 0.0
        return (stationary=true, max_dev=max_dev, se=se)
    end
    return (stationary=max_dev <= sigma * se, max_dev=max_dev, se=se)
end

"""
    jump_detector(kappas, atilde_mean, atilde_se; ci_sigma, min_jump)

Detects discontinuous jump between adjacent points.
"""
function jump_detector(
    kappas::Vector{Float64},
    atilde_mean::Vector{Float64},
    atilde_se::Vector{Float64};
    ci_sigma::Float64 = 3.0,
    min_jump::Float64 = 0.05,
)
    idx = sortperm(kappas)
    k = kappas[idx]
    a = atilde_mean[idx]
    se = atilde_se[idx]
    best = (jump=false, kappa=NaN, jump_size=NaN, idx=0)
    for i in 1:(length(k) - 1)
        if !isfinite(a[i]) || !isfinite(a[i + 1])
            continue
        end
        delta = abs(a[i + 1] - a[i])
        se_comb = sqrt(se[i]^2 + se[i + 1]^2)
        if delta > ci_sigma * se_comb && delta > min_jump
            return (jump=true, kappa=0.5 * (k[i] + k[i + 1]), jump_size=delta, idx=i)
        end
    end
    return best
end

"""
    hysteresis_detector(kappas, atilde_up, atilde_down, se_up, se_down; ...)

Returns gap flag and relative width.
"""
function hysteresis_detector(
    kappas::Vector{Float64},
    atilde_up::Vector{Float64},
    atilde_down::Vector{Float64},
    se_up::Vector{Float64},
    se_down::Vector{Float64};
    ci_sigma::Float64 = 3.0,
    min_gap::Float64 = 0.04,
)
    idx = sortperm(kappas)
    k = kappas[idx]
    a_u = atilde_up[idx]
    a_d = atilde_down[idx]
    se_u = se_up[idx]
    se_d = se_down[idx]
    gap_mask = falses(length(k))
    for i in 1:length(k)
        if !isfinite(a_u[i]) || !isfinite(a_d[i])
            continue
        end
        gap = abs(a_u[i] - a_d[i])
        se_comb = sqrt(se_u[i]^2 + se_d[i]^2)
        if gap > ci_sigma * se_comb && gap > min_gap
            gap_mask[i] = true
        end
    end
    if any(gap_mask)
        k_low = minimum(k[gap_mask])
        k_high = maximum(k[gap_mask])
        return (hysteresis=true, k_low=k_low, k_high=k_high, width=k_high - k_low)
    end
    return (hysteresis=false, k_low=NaN, k_high=NaN, width=NaN)
end

"""
    fit_beta_window(kappas, atilde, kappa_star, window)

Log-log fit of atilde vs delta within window. Returns slope beta, se, r2, n.
"""
function fit_beta_window(
    kappas::Vector{Float64},
    atilde::Vector{Float64},
    kappa_star::Float64,
    window::Tuple{Float64,Float64},
)
    delta = (kappas .- kappa_star) ./ kappa_star
    mask = (delta .>= window[1]) .& (delta .<= window[2]) .& (atilde .> 0) .& isfinite.(atilde)
    if sum(mask) < 3
        return (beta=NaN, se=NaN, r2=NaN, n=sum(mask))
    end
    x = log.(delta[mask])
    y = log.(atilde[mask])
    X = hcat(ones(length(x)), x)
    coeffs = X \ y
    yhat = X * coeffs
    ss_res = sum((y .- yhat).^2)
    ss_tot = sum((y .- mean(y)).^2)
    r2 = ss_tot > 0 ? 1 - ss_res / ss_tot : NaN
    sigma2 = ss_res / max(length(x) - 2, 1)
    se = sqrt(sigma2 / sum((x .- mean(x)).^2))
    return (beta=coeffs[2], se=se, r2=r2, n=length(x))
end

function fit_beta_windows(
    kappas::Vector{Float64},
    atilde::Vector{Float64},
    kappa_star::Float64,
    windows::Dict{String,Tuple{Float64,Float64}},
)
    out = Dict{String,Any}()
    for (name, win) in windows
        out[name] = fit_beta_window(kappas, atilde, kappa_star, win)
    end
    return out
end

"""
    fit_alpha(N_values, atilde_at_kstar)

Fit finite-size scaling Atilde ~ N^{-alpha}. Returns alpha, se, r2, n.
"""
function fit_alpha(N_values::Vector{Int}, atilde_at_kstar::Vector{Float64})
    mask = (atilde_at_kstar .> 0) .& isfinite.(atilde_at_kstar)
    if sum(mask) < 3
        return (alpha=NaN, se=NaN, r2=NaN, n=sum(mask))
    end
    x = log.(Float64.(N_values[mask]))
    y = log.(atilde_at_kstar[mask])
    X = hcat(ones(length(x)), x)
    coeffs = X \ y
    yhat = X * coeffs
    ss_res = sum((y .- yhat).^2)
    ss_tot = sum((y .- mean(y)).^2)
    r2 = ss_tot > 0 ? 1 - ss_res / ss_tot : NaN
    sigma2 = ss_res / max(length(x) - 2, 1)
    se = sqrt(sigma2 / sum((x .- mean(x)).^2))
    alpha = -coeffs[2]
    return (alpha=alpha, se=se, r2=r2, n=length(x))
end

"""
    psd_peak_metrics(x, dt; min_omega)

Compute a simple Welch PSD and extract peak metrics.
"""
function psd_peak_metrics(x::Vector{Float64}, dt::Float64; min_omega::Float64 = 0.0)
    if length(x) < 16
        return (omega_peak=NaN, prominence=NaN, dc_ratio=NaN)
    end
    x0 = x .- mean(x)
    n = length(x0)
    n_seg = max(2, min(8, div(n, 64)))
    seg_len = max(16, div(n, n_seg))
    step = max(1, div(seg_len, 2))
    win = 0.5 .- 0.5 .* cos.(2 * pi .* (0:(seg_len - 1)) ./ (seg_len - 1))
    nfft = seg_len
    psd_acc = zeros(div(nfft, 2) + 1)
    n_used = 0
    for start in 1:step:(n - seg_len + 1)
        seg = x0[start:(start + seg_len - 1)] .* win
        fftv = rfft(seg)
        pxx = abs.(fftv).^2
        psd_acc .+= pxx
        n_used += 1
    end
    if n_used == 0
        return (omega_peak=NaN, prominence=NaN, dc_ratio=NaN)
    end
    psd = psd_acc ./ n_used
    freqs = (0:(length(psd) - 1)) ./ (nfft * dt)
    omega = 2 * pi .* freqs
    mask = omega .>= min_omega
    if sum(mask) < 2
        return (omega_peak=NaN, prominence=NaN, dc_ratio=NaN)
    end
    psd_sub = psd[mask]
    omega_sub = omega[mask]
    peak_idx = argmax(psd_sub)
    omega_peak = omega_sub[peak_idx]
    peak_val = psd_sub[peak_idx]
    median_hf = median(psd_sub)
    prominence = median_hf > 0 ? peak_val / median_hf : NaN
    dc_val = psd[2] > 0 ? psd[2] : psd[1]
    dc_ratio = dc_val > 0 ? peak_val / dc_val : NaN
    return (omega_peak=omega_peak, prominence=prominence, dc_ratio=dc_ratio)
end

end # module
