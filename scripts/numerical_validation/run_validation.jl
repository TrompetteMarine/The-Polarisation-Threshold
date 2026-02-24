#!/usr/bin/env julia
using Pkg
Pkg.activate(".")

using YAML
using CSV
using DataFrames
using Random
using Statistics
using Printf
using Dates
using JSON3
using LibGit2
using SHA
using Distributions
using Plots

using BeliefSim
using BeliefSim.Types: Params, StepHazard, LogisticHazard
using BeliefSim.Model: euler_maruyama_step!, reset_step!
using BeliefSim.Stats: critical_kappa

# -----------------------------
# IO helpers
# -----------------------------
struct TeeIO <: IO
    a::IO
    b::IO
end
Base.write(t::TeeIO, data::Vector{UInt8}) = (write(t.a, data); write(t.b, data))
Base.write(t::TeeIO, data::UInt8) = (write(t.a, data); write(t.b, data))
Base.write(t::TeeIO, data::AbstractString) = (write(t.a, data); write(t.b, data))
Base.flush(t::TeeIO) = (flush(t.a); flush(t.b))
Base.isopen(t::TeeIO) = isopen(t.a) && isopen(t.b)

function to_string_dict(x)
    if x isa Dict
        return Dict(string(k) => to_string_dict(v) for (k, v) in x)
    elseif x isa AbstractVector
        return [to_string_dict(v) for v in x]
    else
        return x
    end
end

function load_config(path::String)
    cfg_raw = YAML.load_file(path)
    return to_string_dict(cfg_raw)
end

function git_commit_hash(path::String)
    try
        repo = LibGit2.GitRepo(path)
        return string(LibGit2.head_oid(repo))
    catch
        return "unknown"
    end
end

function sha_hex(s::AbstractString)
    return bytes2hex(sha1(s))
end

function mkparams(pdict::Dict{String,Any})
    hazard = pdict["hazard"]
    kind = lowercase(String(hazard["kind"]))
    h = if kind == "step"
        StepHazard(Float64(hazard["nu0"]))
    elseif kind == "logistic"
        LogisticHazard(Float64(hazard["numax"]), Float64(hazard["beta"]))
    else
        error("Unknown hazard kind: $(hazard["kind"]) ")
    end
    return Params(
        λ = Float64(pdict["lambda"]),
        σ = Float64(pdict["sigma"]),
        Θ = Float64(pdict["theta"]),
        c0 = Float64(pdict["c0"]),
        hazard = h,
    )
end

# -----------------------------
# Boundary reset implementations
# -----------------------------
@inline function boundary_reset_first!(u::Vector{Float64}, p::Params)
    theta = p.Θ
    c0 = p.c0
    @inbounds for i in eachindex(u)
        if abs(u[i]) >= theta
            s = sign(u[i])
            s = s == 0 ? 1.0 : s
            u[i] = c0 * s * theta
        end
    end
    return nothing
end

@inline function boundary_reset_interp!(u::Vector{Float64}, u_prev::Vector{Float64}, p::Params)
    theta = p.Θ
    c0 = p.c0
    @inbounds for i in eachindex(u)
        if abs(u[i]) >= theta
            if abs(u_prev[i]) < theta
                s = sign(u[i])
                s = s == 0 ? sign(u_prev[i]) : s
                s = s == 0 ? 1.0 : s
                u[i] = c0 * s * theta
            else
                s = sign(u[i])
                s = s == 0 ? 1.0 : s
                u[i] = c0 * s * theta
            end
        end
    end
    return nothing
end

# -----------------------------
# Simulation helpers
# -----------------------------
function simulate_path(
    p::Params;
    kappa::Float64,
    N::Int,
    T::Float64,
    dt::Float64,
    seed::Int,
    impl::Symbol,
    mt_stride::Int,
    init_mode::Symbol,
    epsilon::Float64,
)
    Random.seed!(seed)
    steps = Int(round(T / dt))
    n_time = Int(floor(steps / mt_stride)) + 1
    time_grid = collect(0:mt_stride:steps) .* dt

    sigma = p.σ
    lambda = p.λ
    u = randn(N) .* (sigma / sqrt(2 * lambda))
    if init_mode == :odd
        n1 = div(N, 2)
        @inbounds u[1:n1] .+= epsilon
        @inbounds u[(n1 + 1):end] .-= epsilon
    end

    u_prev = similar(u)
    mean_traj = Vector{Float64}(undef, n_time)
    var_traj = Vector{Float64}(undef, n_time)
    mean_traj[1] = mean(u)
    var_traj[1] = var(u)

    idx = 2
    for step in 1:steps
        gbar = mean(u)
        copyto!(u_prev, u)
        euler_maruyama_step!(u, kappa, gbar, p, dt)
        if impl == :hazard
            reset_step!(u, p, dt)
        elseif impl == :first_crossing
            boundary_reset_first!(u, p)
        elseif impl == :interp
            boundary_reset_interp!(u, u_prev, p)
        else
            error("Unknown impl: $impl")
        end

        if step % mt_stride == 0
            mean_traj[idx] = mean(u)
            var_traj[idx] = var(u)
            idx += 1
        end
    end

    return time_grid, mean_traj, var_traj, u
end

function run_ensemble(
    p::Params;
    kappa::Float64,
    N::Int,
    T::Float64,
    dt::Float64,
    n_ensemble::Int,
    seed_base::Int,
    impl::Symbol,
    mt_stride::Int,
    init_mode::Symbol,
    epsilon::Float64,
)
    steps = Int(round(T / dt))
    n_time = Int(floor(steps / mt_stride)) + 1
    mean_traj = Matrix{Float64}(undef, n_ensemble, n_time)
    var_traj = Matrix{Float64}(undef, n_ensemble, n_time)
    finals = Vector{Vector{Float64}}(undef, n_ensemble)
    time_grid_ref = collect(0:mt_stride:steps) .* dt

    for i in 1:n_ensemble
        seed = seed_base + (i - 1) * 1000
        t, m, v, u_final = simulate_path(
            p;
            kappa=kappa,
            N=N,
            T=T,
            dt=dt,
            seed=seed,
            impl=impl,
            mt_stride=mt_stride,
            init_mode=init_mode,
            epsilon=epsilon,
        )
        mean_traj[i, :] .= m
        var_traj[i, :] .= v
        finals[i] = u_final
    end

    return (time_grid=time_grid_ref, mean_traj=mean_traj, var_traj=var_traj, finals=finals)
end

# -----------------------------
# Diagnostics
# -----------------------------
function fit_log_growth(t::Vector{Float64}, y::Vector{Float64}, window::Tuple{Float64,Float64})
    t0, t1 = window
    mask = (t .>= t0) .& (t .<= t1)
    tt = t[mask]
    yy = y[mask]
    yy = log.(yy .+ 1e-12)
    good = isfinite.(yy)
    tt = tt[good]
    yy = yy[good]
    if length(tt) < 3
        return (slope=NaN, intercept=NaN, r2=NaN, se=NaN, n=length(tt))
    end
    X = hcat(ones(length(tt)), tt)
    coeffs = X \ yy
    yhat = X * coeffs
    ss_res = sum((yy .- yhat).^2)
    ss_tot = sum((yy .- mean(yy)).^2)
    r2 = ss_tot > 0 ? 1 - ss_res / ss_tot : NaN
    se = sqrt(ss_res / max(1, length(tt) - 2) / sum((tt .- mean(tt)).^2))
    return (slope=coeffs[2], intercept=coeffs[1], r2=r2, se=se, n=length(tt))
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

function growth_scan(
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
)
    slopes = Vector{Float64}(undef, length(kappa_grid))
    boot_slopes = n_boot > 0 ? Matrix{Float64}(undef, n_boot, length(kappa_grid)) : Matrix{Float64}(undef, 0, 0)

    for (i, kappa) in enumerate(kappa_grid)
        res = run_ensemble(
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

    return slopes, boot_slopes, boot_kappa
end

function kappa_star_from_growth(
    p::Params;
    kappa_grid::Vector{Float64},
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
    bisection_iters::Int,
    bisection_tol::Float64,
)
    slopes, boot_slopes, boot_kappa = growth_scan(
        p,
        kappa_grid;
        N=N,
        T=T,
        dt=dt,
        n_ensemble=n_ensemble,
        seed_base=seed_base,
        impl=impl,
        mt_stride=mt_stride,
        epsilon=epsilon,
        window=window,
        n_boot=n_boot,
    )

    kappa_star = estimate_kappa_star_from_scan(kappa_grid, slopes)

    # bracket for bisection
    kappa_sorted = sort(kappa_grid)
    s_sorted = slopes[sortperm(kappa_grid)]
    k_low = NaN
    k_high = NaN
    s_low = NaN
    s_high = NaN
    for i in 1:(length(kappa_sorted) - 1)
        if s_sorted[i] == 0
            k_low = kappa_sorted[i]
            k_high = kappa_sorted[i]
            s_low = s_sorted[i]
            s_high = s_sorted[i]
            break
        elseif s_sorted[i] * s_sorted[i + 1] < 0
            k_low = kappa_sorted[i]
            k_high = kappa_sorted[i + 1]
            s_low = s_sorted[i]
            s_high = s_sorted[i + 1]
            break
        end
    end

    status = "ok"
    if !isfinite(k_low) || !isfinite(k_high)
        status = "no_bracket"
    else
        for iter in 1:bisection_iters
            if abs(k_high - k_low) < bisection_tol
                break
            end
            k_mid = 0.5 * (k_low + k_high)
            res_mid = run_ensemble(
                p;
                kappa=k_mid,
                N=N,
                T=T,
                dt=dt,
                n_ensemble=n_ensemble,
                seed_base=seed_base + 777 + iter * 1000,
                impl=impl,
                mt_stride=mt_stride,
                init_mode=:odd,
                epsilon=epsilon,
            )
            mean_abs = vec(mean(abs.(res_mid.mean_traj); dims=1))
            fit_mid = fit_log_growth(res_mid.time_grid, mean_abs, window)
            if !isfinite(fit_mid.slope)
                break
            end
            if fit_mid.slope == 0
                k_low = k_mid
                k_high = k_mid
                break
            elseif fit_mid.slope * s_low > 0
                k_low = k_mid
                s_low = fit_mid.slope
            else
                k_high = k_mid
                s_high = fit_mid.slope
            end
        end
        kappa_star = 0.5 * (k_low + k_high)
    end

    kappa_se = NaN
    if n_boot > 0 && !isempty(boot_kappa)
        kappa_se = std(boot_kappa)
    end

    return (kappa_star=kappa_star, kappa_se=kappa_se, slopes=slopes, boot_kappa=boot_kappa, status=status)
end

function sample_skew_kurt(x::Vector{Float64})
    n = length(x)
    if n == 0
        return 0.0, 3.0
    end
    mu = mean(x)
    m2 = 0.0
    m3 = 0.0
    m4 = 0.0
    @inbounds for xi in x
        d = xi - mu
        d2 = d * d
        m2 += d2
        m3 += d2 * d
        m4 += d2 * d2
    end
    m2 /= n
    if m2 <= 0.0
        return 0.0, 3.0
    end
    m3 /= n
    m4 /= n
    skew = m3 / (m2^(3 / 2))
    kurt = m4 / (m2^2)
    return skew, kurt
end

function bimodality_coeff(skew::Float64, kurt::Float64, n::Int)
    if n < 4
        return NaN
    end
    correction = 3 * (n - 1)^2 / ((n - 2) * (n - 3))
    return (skew^2 + 1) / (kurt + correction)
end

# Adaptive simulation time to mitigate critical slowing down
function adaptive_T(kappa::Float64, kappa_star::Float64, T_base::Float64, T_max::Float64, tau0::Float64)
    delta = abs(kappa / kappa_star - 1.0)
    if delta < 1e-8
        return T_max
    end
    T_adapt = T_base * max(1.0, tau0 / delta)
    return min(T_max, T_adapt)
end

# Check if the ensemble mean of |m| has reached a plateau
function is_ensemble_converged(mean_abs_traj::Vector{Float64}, time_grid::Vector{Float64}, frac::Float64 = 0.2)
    n = length(mean_abs_traj)
    if n < 10
        return false
    end
    late_start = max(1, Int(floor((1 - frac) * n)))
    tail_half = max(1, Int(floor(n * frac / 2)))
    early_late = mean_abs_traj[late_start:max(late_start, n - tail_half)]
    late_late = mean_abs_traj[max(1, n - tail_half + 1):n]
    if length(early_late) < 3 || length(late_late) < 3
        return true
    end
    mean_early = mean(early_late)
    mean_late = mean(late_late)
    rel_diff = abs(mean_late - mean_early) / (abs(mean_early) + 1e-6)
    return rel_diff < 0.05
end

# Estimate m0 from deeply subcritical simulations (e.g., kappa = 0.5 * kappa*_theory)
function estimate_m0(
    p::Params;
    kappa_frac::Float64 = 0.5,
    N::Int,
    T::Float64,
    dt::Float64,
    n_ensemble::Int,
    seed_base::Int,
    impl::Symbol,
    mt_stride::Int,
    epsilon::Float64,
    late_window_frac::Float64,
    convergence_frac::Float64,
    n_boot::Int = 200,
)
    kappa_theory = critical_kappa(p)
    kappa0 = kappa_frac * kappa_theory
    res = run_ensemble(
        p;
        kappa=kappa0,
        N=N,
        T=T,
        dt=dt,
        n_ensemble=n_ensemble,
        seed_base=seed_base + 50000,
        impl=impl,
        mt_stride=mt_stride,
        init_mode=:random,
        epsilon=epsilon,
    )
    n_time = size(res.mean_traj, 2)
    late_start = max(1, Int(floor((1 - late_window_frac) * n_time)))
    mbar = Float64[]
    for j in 1:n_ensemble
        abs_traj = abs.(res.mean_traj[j, :])
        if is_ensemble_converged(abs_traj, res.time_grid, convergence_frac)
            push!(mbar, mean(abs_traj[late_start:end]))
        end
    end
    if isempty(mbar)
        mbar = [mean(abs.(res.mean_traj[j, late_start:end])) for j in 1:n_ensemble]
    end
    m0 = mean(mbar)
    boot_m0 = Float64[]
    sizehint!(boot_m0, n_boot)
    for _ in 1:n_boot
        idx = rand(1:length(mbar), length(mbar))
        push!(boot_m0, mean(mbar[idx]))
    end
    m0_se = std(boot_m0)
    return m0, m0_se, mbar
end

# Select delta window by minimizing BIC over a grid of (delta_lo, delta_hi)
function select_delta_window(
    kappas::Vector{Float64},
    amps::Vector{Float64},
    kappa_star::Float64;
    delta_min::Float64 = 1e-2,
    delta_max::Float64 = 0.5,
    n_steps::Int = 20,
    min_points::Int = 5,
)
    delta = (kappas .- kappa_star) ./ kappa_star
    delta_grid_lo = exp.(range(log(delta_min), log(delta_max); length=n_steps))
    delta_grid_hi = exp.(range(log(delta_min), log(delta_max); length=n_steps))
    best_bic = Inf
    best_window = (delta_min, delta_max)
    best_n = 0
    for delta_lo in delta_grid_lo
        for delta_hi in delta_grid_hi
            delta_hi <= delta_lo && continue
            mask = (delta .>= delta_lo) .& (delta .<= delta_hi) .& (amps .> 0) .& isfinite.(amps)
            if sum(mask) < min_points
                continue
            end
            x = log.(delta[mask])
            y = log.(amps[mask])
            X = hcat(ones(length(x)), x)
            coeffs = X \ y
            rss = sum((y .- X * coeffs).^2)
            n = length(x)
            bic = n * log(rss / n) + 2 * log(n)
            if bic < best_bic
                best_bic = bic
                best_window = (delta_lo, delta_hi)
                best_n = n
            end
        end
    end
    return best_window, best_n, best_bic
end

# Theil-Sen robust slope (median of pairwise slopes)
function theil_sen_slope(x::Vector{Float64}, y::Vector{Float64})
    n = length(x)
    slopes = Float64[]
    sizehint!(slopes, div(n * (n - 1), 2))
    for i in 1:n
        for j in (i + 1):n
            if x[i] != x[j]
                push!(slopes, (y[j] - y[i]) / (x[j] - x[i]))
            end
        end
    end
    return isempty(slopes) ? NaN : median(slopes)
end

# Nested bootstrap for beta that resamples kappa*, per-kappa runs, and re-selects delta window
function bootstrap_beta_nested(
    kappas::Vector{Float64},
    per_kappa_samples::Vector{Vector{Float64}},
    m0::Float64,
    kappa_star_boot::Vector{Float64},
    delta_range::Tuple{Float64,Float64};
    n_outer::Int = 200,
    delta_steps::Int = 20,
    min_points::Int = 5,
)
    n_kappa = length(kappas)
    if n_kappa == 0 || isempty(kappa_star_boot)
        return (beta_hat=NaN, beta_se=NaN, beta_ci=(NaN, NaN),
                theil_sen=NaN, theil_ci=(NaN, NaN), n_boot=0)
    end

    function fit_beta_window(kappa_vals, samples, kappa_star_ref)
        M_corr = Float64[]
        kappa_valid = Float64[]
        for i in eachindex(kappa_vals)
            runs = samples[i]
            isempty(runs) && continue
            mbar = mean(runs)
            m_corr = sqrt(max(mbar^2 - m0^2, 0.0))
            if m_corr > 1e-8
                push!(M_corr, m_corr)
                push!(kappa_valid, kappa_vals[i])
            end
        end
        if length(kappa_valid) < min_points
            return (beta_hat=NaN, window=(NaN, NaN), n_points=0, x=Float64[], y=Float64[])
        end
        win, _, _ = select_delta_window(
            kappa_valid, M_corr, kappa_star_ref;
            delta_min=delta_range[1], delta_max=delta_range[2],
            n_steps=delta_steps, min_points=min_points,
        )
        delta = (kappa_valid .- kappa_star_ref) ./ kappa_star_ref
        mask = (delta .>= win[1]) .& (delta .<= win[2]) .& (M_corr .> 0)
        if sum(mask) < min_points
            return (beta_hat=NaN, window=win, n_points=sum(mask), x=Float64[], y=Float64[])
        end
        x = log.(delta[mask])
        y = log.(M_corr[mask])
        X = hcat(ones(length(x)), x)
        coeffs = X \ y
        return (beta_hat=coeffs[2], window=win, n_points=length(x), x=x, y=y)
    end

    # Baseline estimate for BCa
    kappa_star_ref = median(kappa_star_boot)
    fit0 = fit_beta_window(kappas, per_kappa_samples, kappa_star_ref)
    beta_hat0 = fit0.beta_hat

    # Jackknife for acceleration (leave-one-kappa out)
    jack = Float64[]
    sizehint!(jack, n_kappa)
    for i in 1:n_kappa
        kappa_j = [kappas[j] for j in 1:n_kappa if j != i]
        s_j = [per_kappa_samples[j] for j in 1:n_kappa if j != i]
        fit_j = fit_beta_window(kappa_j, s_j, kappa_star_ref)
        isfinite(fit_j.beta_hat) && push!(jack, fit_j.beta_hat)
    end

    # Nested bootstrap
    boot_beta = Float64[]
    boot_theil = Float64[]
    sizehint!(boot_beta, n_outer)
    sizehint!(boot_theil, n_outer)
    for _ in 1:n_outer
        kappa_s = kappa_star_boot[rand(1:length(kappa_star_boot))]
        M_corr = Float64[]
        kappa_valid = Float64[]
        for i in 1:n_kappa
            runs = per_kappa_samples[i]
            isempty(runs) && continue
            idx = rand(1:length(runs), length(runs))
            mbar_b = mean(runs[idx])
            m_corr = sqrt(max(mbar_b^2 - m0^2, 0.0))
            if m_corr > 1e-8
                push!(M_corr, m_corr)
                push!(kappa_valid, kappas[i])
            end
        end
        if length(kappa_valid) < min_points
            continue
        end
        win, _, _ = select_delta_window(
            kappa_valid, M_corr, kappa_s;
            delta_min=delta_range[1], delta_max=delta_range[2],
            n_steps=delta_steps, min_points=min_points,
        )
        delta = (kappa_valid .- kappa_s) ./ kappa_s
        mask = (delta .>= win[1]) .& (delta .<= win[2]) .& (M_corr .> 0)
        if sum(mask) < min_points
            continue
        end
        x = log.(delta[mask])
        y = log.(M_corr[mask])
        X = hcat(ones(length(x)), x)
        coeffs = X \ y
        push!(boot_beta, coeffs[2])
        push!(boot_theil, theil_sen_slope(x, y))
    end

    beta_ci = (NaN, NaN)
    theil_ci = (NaN, NaN)
    if length(boot_beta) >= 100 && isfinite(beta_hat0)
        z0 = quantile(Normal(), mean(boot_beta .< beta_hat0))
        a = 0.0
        if length(jack) >= 5
            jack_mean = mean(jack)
            num = sum((jack_mean .- jack).^3)
            den = 6.0 * (sum((jack_mean .- jack).^2))^(3 / 2)
            a = den != 0.0 ? num / den : 0.0
        end
        zα1 = quantile(Normal(), 0.025)
        zα2 = quantile(Normal(), 0.975)
        α1 = cdf(Normal(), z0 + (z0 + zα1) / (1 - a * (z0 + zα1)))
        α2 = cdf(Normal(), z0 + (z0 + zα2) / (1 - a * (z0 + zα2)))
        α1 = clamp(α1, 0.0, 1.0)
        α2 = clamp(α2, 0.0, 1.0)
        beta_ci = (quantile(boot_beta, α1), quantile(boot_beta, α2))
    elseif length(boot_beta) >= 100
        beta_ci = (quantile(boot_beta, 0.025), quantile(boot_beta, 0.975))
    end
    if length(boot_theil) >= 100
        theil_ci = (quantile(boot_theil, 0.025), quantile(boot_theil, 0.975))
    end

    return (
        beta_hat = isempty(boot_beta) ? NaN : median(boot_beta),
        beta_se = isempty(boot_beta) ? NaN : std(boot_beta),
        beta_ci = beta_ci,
        theil_sen = isempty(boot_theil) ? NaN : median(boot_theil),
        theil_ci = theil_ci,
        n_boot = length(boot_beta),
    )
end

function kappa_star_from_bimodality(
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
    threshold::Float64,
)
    bimod_means = Vector{Float64}(undef, length(kappa_grid))
    bimod_ses = Vector{Float64}(undef, length(kappa_grid))

    for (i, kappa) in enumerate(kappa_grid)
        res = run_ensemble(
            p;
            kappa=kappa,
            N=N,
            T=T,
            dt=dt,
            n_ensemble=n_ensemble,
            seed_base=seed_base + 20000 * i,
            impl=impl,
            mt_stride=mt_stride,
            init_mode=:random,
            epsilon=epsilon,
        )
        coeffs = Vector{Float64}(undef, n_ensemble)
        for j in 1:n_ensemble
            skew, kurt = sample_skew_kurt(res.finals[j])
            coeffs[j] = bimodality_coeff(skew, kurt, length(res.finals[j]))
        end
        bimod_means[i] = mean(coeffs)
        bimod_ses[i] = std(coeffs) / sqrt(n_ensemble)
    end

    kappa_star = NaN
    for (i, kappa) in enumerate(kappa_grid)
        if bimod_means[i] >= threshold
            kappa_star = kappa
            break
        end
    end

    return (kappa_star=kappa_star, bimod_mean=bimod_means, bimod_se=bimod_ses)
end

function fit_beta(kappas::Vector{Float64}, amps::Vector{Float64}; kappa_star::Float64, min_points::Int)
    mask = (kappas .> kappa_star) .& (amps .> 0) .& isfinite.(amps)
    if sum(mask) < min_points
        return (beta_hat=NaN, beta_se=NaN, beta_ci=(NaN, NaN), r2=NaN, n=0)
    end
    x = log.(kappas[mask] .- kappa_star)
    y = log.(amps[mask])
    X = hcat(ones(length(x)), x)
    coeffs = X \ y
    yhat = X * coeffs
    ss_res = sum((y .- yhat).^2)
    ss_tot = sum((y .- mean(y)).^2)
    r2 = ss_tot > 0 ? 1 - ss_res / ss_tot : NaN
    df = length(x) - 2
    sigma2 = ss_res / max(1, df)
    se_beta = sqrt(sigma2 / sum((x .- mean(x)).^2))
    tcrit = quantile(TDist(max(1, df)), 0.975)
    ci = (coeffs[2] - tcrit * se_beta, coeffs[2] + tcrit * se_beta)
    return (beta_hat=coeffs[2], beta_se=se_beta, beta_ci=ci, r2=r2, n=length(x))
end

function beta_from_sweep(
    p::Params;
    kappa_star::Float64,
    deltas::Vector{Float64},
    N::Int,
    T::Float64,
    dt::Float64,
    n_ensemble::Int,
    seed_base::Int,
    impl::Symbol,
    mt_stride::Int,
    epsilon::Float64,
    late_window_frac::Float64,
    n_boot::Int,
    min_points::Int,
)
    kappas = kappa_star .* (1 .+ deltas)
    amps = Vector{Float64}(undef, length(kappas))
    amps_se = Vector{Float64}(undef, length(kappas))
    per_kappa_samples = Vector{Vector{Float64}}(undef, length(kappas))

    for (i, kappa) in enumerate(kappas)
        res = run_ensemble(
            p;
            kappa=kappa,
            N=N,
            T=T,
            dt=dt,
            n_ensemble=n_ensemble,
            seed_base=seed_base + 30000 * i,
            impl=impl,
            mt_stride=mt_stride,
            init_mode=:random,
            epsilon=epsilon,
        )
        n_time = size(res.mean_traj, 2)
        late_start = max(1, Int(floor((1 - late_window_frac) * n_time)))
        mbar = Vector{Float64}(undef, n_ensemble)
        for j in 1:n_ensemble
            mbar[j] = mean(abs.(res.mean_traj[j, late_start:end]))
        end
        per_kappa_samples[i] = mbar
        amps[i] = mean(mbar)
        amps_se[i] = std(mbar) / sqrt(n_ensemble)
    end

    fit = fit_beta(kappas, amps; kappa_star=kappa_star, min_points=min_points)

    beta_se = fit.beta_se
    if n_boot > 0
        boot = Float64[]
        sizehint!(boot, n_boot)
        for _ in 1:n_boot
            amps_b = Vector{Float64}(undef, length(kappas))
            for i in eachindex(kappas)
                samples = per_kappa_samples[i]
                idx = rand(1:length(samples), length(samples))
                amps_b[i] = mean(samples[idx])
            end
            fit_b = fit_beta(kappas, amps_b; kappa_star=kappa_star, min_points=min_points)
            if isfinite(fit_b.beta_hat)
                push!(boot, fit_b.beta_hat)
            end
        end
        if !isempty(boot)
            beta_se = std(boot)
        end
    end

    return (beta_hat=fit.beta_hat, beta_se=beta_se, beta_ci=fit.beta_ci, r2=fit.r2, n=fit.n, kappas=kappas, amps=amps, amps_se=amps_se)
end

# -----------------------------
# TeX table
# -----------------------------
function write_tex_table(df::DataFrame, path::String)
    impl_map = Dict(
        "first_crossing" => "First-crossing (A)",
        "interp" => "Interp.\\ to boundary (B)",
    )
    df = filter(row -> haskey(impl_map, row.impl), df)
    dt_order = sort(unique(df.dt), rev=true)

    open(path, "w") do io
        println(io, "% Auto-generated by run_validation.jl")
        println(io, "\\begin{table}[!t]")
        println(io, "\\centering")
        println(io, "\\footnotesize")
        println(io, "\\caption{Numerical robustness checks for the simulation-based diagnostics.")
        println(io, "\$\\kappa^\\star_{\\mathrm{eff}}\$ denotes the estimated instability onset in the finite-\$N\$ ABM under the operational criterion described in the replication scripts; \$\\hat\\beta\$ is the fitted scaling exponent from the order-parameter curve near onset (used only as supportive evidence for a supercritical normal form).")
        println(io, "Standard errors (s.e.) are computed by bootstrap, accounting for uncertainty in \\(\\kappa^*\\), window selection, and sampling variability.}")
        println(io, "\\label{tab:robustness}")
        println(io, "\\begin{tabular}{lcccc}")
        println(io, "\\toprule")
        println(io, "Implementation & \$\\Delta t\$ & \$N\$ & \$\\kappa^\\star_{\\mathrm{eff}}\$ & \$\\hat\\beta\$ (s.e.) \\\\")
        println(io, "\\midrule")

        for impl in sort(unique(df.impl))
            impl_display = impl_map[impl]
            df_impl = df[df.impl .== impl, :]
            for dt in dt_order
                df_dt = df_impl[df_impl.dt .== dt, :]
                for row in eachrow(df_dt)
                    dt_latex = if dt == 0.01
                        "\$10^{-2}\$"
                    elseif dt == 0.005
                        "\$5\\times 10^{-3}\$"
                    elseif dt == 0.001
                        "\$10^{-3}\$"
                    else
                        "\$" * string(dt) * "\$"
                    end
                    beta_str = if isfinite(row.beta_hat)
                        @sprintf("%.4f (%.4f)", row.beta_hat, row.beta_se)
                    else
                        "---"
                    end
                    @printf(io, "%s & %s & %d & %.5f & %s \\\\\n",
                            impl_display, dt_latex, row.N, row.kappa_star_eff, beta_str)
                end
                if dt != last(dt_order)
                    println(io, "\\addlinespace")
                end
            end
            if impl != last(sort(unique(df.impl)))
                println(io, "\\midrule")
            end
        end
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
        println(io, "\\end{table}")
    end
end

function plot_summary(df::DataFrame, outpath::String)
    impls = unique(df.impl)
    p1 = plot(title="kappa* (growth scan)", xlabel="dt", ylabel="kappa*", legend=:right)
    p2 = plot(title="beta (scaling)", xlabel="dt", ylabel="beta", legend=:right)

    for impl in impls
        df_impl = df[df.impl .== impl, :]
        for N in unique(df_impl.N)
            sub = df_impl[df_impl.N .== N, :]
            order = sortperm(sub.dt)
            label = "$(impl), N=$(N)"
            plot!(p1, sub.dt[order], sub.kappa_star_eff[order]; marker=:circle, label=label)
            plot!(p2, sub.dt[order], sub.beta_hat[order]; marker=:circle, label=label)
        end
    end
    plt = plot(p1, p2; layout=(1, 2), size=(900, 360))
    savefig(plt, outpath)
end

# -----------------------------
# Main
# -----------------------------
function main()
    if length(ARGS) == 0
        println("Usage: julia --project=. scripts/numerical_validation/run_validation.jl --config configs/validation.yaml [--quick]")
        exit(1)
    end

    config_path = ""
    quick = false
    for (i, arg) in enumerate(ARGS)
        if arg == "--config" && i < length(ARGS)
            config_path = ARGS[i + 1]
        elseif arg == "--quick"
            quick = true
        elseif arg == "--full"
            quick = false
        end
    end
    isempty(config_path) && error("--config is required")

    cfg = load_config(config_path)
    outdir = cfg["output_dir"]
    isdir(outdir) || mkpath(outdir)
    isdir(joinpath(outdir, "figs")) || mkpath(joinpath(outdir, "figs"))
    isdir(joinpath(outdir, "cache")) || mkpath(joinpath(outdir, "cache"))

    log_path = joinpath(outdir, "run.log")
    open(log_path, "w") do log_io
        println("============================================================")
        println("NUMERICAL VALIDATION PIPELINE (RIGOROUS MODE)")
        println("============================================================")
        println("Config: $config_path")
        println("Mode: ", quick ? "quick" : "full")
        println("Timestamp: ", string(now()))
        println(log_io, "============================================================")
        println(log_io, "NUMERICAL VALIDATION PIPELINE (RIGOROUS MODE)")
        println(log_io, "============================================================")
        println(log_io, "Config: $config_path")
        println(log_io, "Mode: ", quick ? "quick" : "full")
        println(log_io, "Timestamp: ", string(now()))

        p = mkparams(cfg["params"])
        grid = cfg["grid"]
        runtime = cfg["runtime"]

        seeds = Int(get(grid, "seeds", 20))
        seed_base = Int(get(grid, "seed_base", 0))
        T_base = Float64(runtime["T"])
        burn_in = Float64(runtime["burn_in"])
        mt_stride = Int(runtime["mt_stride"])
        T_max = Float64(get(runtime, "T_max", 1200.0))
        tau0 = Float64(get(runtime, "tau0", 50.0))

        if quick
            seeds = Int(get(cfg["quick"], "seeds", seeds))
            T_base = Float64(get(cfg["quick"], "T", T_base))
            burn_in = Float64(get(cfg["quick"], "burn_in", burn_in))
        end

        growth_cfg = runtime["growth_scan"]
        epsilon = Float64(growth_cfg["epsilon"])
        window = (Float64(growth_cfg["window"][1]), Float64(growth_cfg["window"][2]))
        kappa_min_factor = Float64(growth_cfg["kappa_min_factor"])
        kappa_max_factor = Float64(growth_cfg["kappa_max_factor"])
        grid_points = Int(growth_cfg["grid_points"])
        bisection_iters = Int(growth_cfg["bisection_iters"])
        bisection_tol = Float64(growth_cfg["bisection_tol"])
        n_boot_growth = Int(growth_cfg["n_boot"])

        if quick
            bisection_iters = Int(get(cfg["quick"], "bisection_iters", bisection_iters))
            bisection_tol = Float64(get(cfg["quick"], "bisection_tol", bisection_tol))
            n_boot_growth = Int(get(cfg["quick"], "growth_scan_n_boot", n_boot_growth))
        end

        beta_cfg = runtime["beta_fit"]
        delta_window = (Float64(beta_cfg["delta_window"][1]), Float64(beta_cfg["delta_window"][2]))
        kappa_points = Int(beta_cfg["kappa_points"])
        late_window_frac = Float64(beta_cfg["late_window_frac"])
        n_boot_beta = Int(beta_cfg["n_boot"])
        min_points = Int(beta_cfg["min_points"])
        convergence_frac = Float64(get(beta_cfg, "convergence_frac", 0.2))

        if quick
            kappa_points = Int(get(cfg["quick"], "kappa_points", kappa_points))
            n_boot_beta = Int(get(cfg["quick"], "beta_n_boot", n_boot_beta))
        end

        bimod_cfg = runtime["bimodality"]
        bimod_threshold = Float64(bimod_cfg["threshold"])

        dt_list = [Float64(x) for x in grid["dt"]]
        N_list = [Int(x) for x in grid["N"]]
        impl_list = [Symbol(x) for x in grid["impl"]]

        kappa_theory = critical_kappa(p)
        @printf("Theoretical kappa* (spectral): %.6f\n", kappa_theory)
        @printf(log_io, "Theoretical kappa* (spectral): %.6f\n", kappa_theory)

        results = DataFrame(
            impl=String[],
            dt=Float64[],
            N=Int[],
            kappa_star_eff=Float64[],
            kappa_se=Float64[],
            beta_hat=Float64[],
            beta_se=Float64[],
            beta_ci_lower=Float64[],
            beta_ci_upper=Float64[],
            theil_sen_beta=Float64[],
            n_scaling_points=Int[],
            delta_window_lo=Float64[],
            delta_window_hi=Float64[],
            fraction_converged=Float64[],
            m0=Float64[],
            seeds=Int[],
            horizon=Float64[],
            burnin=Float64[],
            criterion=String[],
        )

        for impl in impl_list
            for dt in dt_list
                for N in N_list
                    println("------------------------------------------------------------")
                    @printf("impl=%s  dt=%.4g  N=%d  seeds=%d\n", String(impl), dt, N, seeds)
                    println(log_io, "------------------------------------------------------------")
                    @printf(log_io, "impl=%s  dt=%.4g  N=%d  seeds=%d\n", String(impl), dt, N, seeds)

                    # Step 1: baseline m0 from deep subcritical
                    m0, m0_se, _ = estimate_m0(
                        p;
                        kappa_frac=0.5,
                        N=N,
                        T=T_base,
                        dt=dt,
                        n_ensemble=seeds,
                        seed_base=seed_base + 1000000,
                        impl=impl,
                        mt_stride=mt_stride,
                        epsilon=epsilon,
                        late_window_frac=late_window_frac,
                        convergence_frac=convergence_frac,
                        n_boot=200,
                    )
                    @printf("  m0 = %.6f (se=%.6f)\n", m0, m0_se)
                    @printf(log_io, "  m0 = %.6f (se=%.6f)\n", m0, m0_se)

                    # Step 2: growth scan for kappa* (with bootstrap)
                    kappa_min = kappa_theory * kappa_min_factor
                    kappa_max = kappa_theory * kappa_max_factor
                    kappa_grid = collect(range(kappa_min, kappa_max; length=grid_points))

                    growth = kappa_star_from_growth(
                        p;
                        kappa_grid=kappa_grid,
                        N=N,
                        T=T_base,
                        dt=dt,
                        n_ensemble=seeds,
                        seed_base=seed_base,
                        impl=impl,
                        mt_stride=mt_stride,
                        epsilon=epsilon,
                        window=window,
                        n_boot=n_boot_growth,
                        bisection_iters=bisection_iters,
                        bisection_tol=bisection_tol,
                    )

                    @printf("  kappa*_growth = %.6f (se=%.6f, status=%s)\n",
                            growth.kappa_star, growth.kappa_se, growth.status)
                    @printf(log_io, "  kappa*_growth = %.6f (se=%.6f, status=%s)\n",
                            growth.kappa_star, growth.kappa_se, growth.status)

                    kappa_eff = growth.kappa_star
                    if !isfinite(kappa_eff) || kappa_eff <= 0
                        bimod = kappa_star_from_bimodality(
                            p,
                            kappa_grid;
                            N=N,
                            T=T_base,
                            dt=dt,
                            n_ensemble=seeds,
                            seed_base=seed_base,
                            impl=impl,
                            mt_stride=mt_stride,
                            epsilon=epsilon,
                            threshold=bimod_threshold,
                        )
                        @printf("  kappa*_bimodality = %.6f (threshold=%.2f)\n",
                                bimod.kappa_star, bimod_threshold)
                        @printf(log_io, "  kappa*_bimodality = %.6f (threshold=%.2f)\n",
                                bimod.kappa_star, bimod_threshold)
                        kappa_eff = bimod.kappa_star
                    end

                    if !isfinite(kappa_eff) || kappa_eff <= 0
                        @warn "No valid kappa* for impl=$(impl), dt=$(dt), N=$(N); skipping beta fit."
                        push!(results, (
                            impl=String(impl),
                            dt=dt,
                            N=N,
                            kappa_star_eff=NaN,
                            kappa_se=NaN,
                            beta_hat=NaN,
                            beta_se=NaN,
                            beta_ci_lower=NaN,
                            beta_ci_upper=NaN,
                            theil_sen_beta=NaN,
                            n_scaling_points=0,
                            delta_window_lo=NaN,
                            delta_window_hi=NaN,
                            fraction_converged=NaN,
                            m0=m0,
                            seeds=seeds,
                            horizon=T_base,
                            burnin=burn_in,
                            criterion="growth_scan + bimodality",
                        ))
                        continue
                    end

                    # Step 3: supercritical sweep with adaptive T and convergence filter
                    delta_sweep = exp.(range(log(delta_window[1]), log(delta_window[2]); length=kappa_points))
                    kappas_sweep = kappa_eff .* (1 .+ delta_sweep)
                    per_kappa_samples = Vector{Vector{Float64}}(undef, length(kappas_sweep))

                    for (i, kappa_val) in enumerate(kappas_sweep)
                        T_adapt = adaptive_T(kappa_val, kappa_eff, T_base, T_max, tau0)
                        res = run_ensemble(
                            p;
                            kappa=kappa_val,
                            N=N,
                            T=T_adapt,
                            dt=dt,
                            n_ensemble=seeds,
                            seed_base=seed_base + 30000 * i,
                            impl=impl,
                            mt_stride=mt_stride,
                            init_mode=:random,
                            epsilon=epsilon,
                        )
                        n_time = size(res.mean_traj, 2)
                        late_start = max(1, Int(floor((1 - late_window_frac) * n_time)))
                        mbar = Float64[]
                        for j in 1:seeds
                            abs_traj = abs.(res.mean_traj[j, :])
                            if is_ensemble_converged(abs_traj, res.time_grid, convergence_frac)
                                push!(mbar, mean(abs_traj[late_start:end]))
                            end
                        end
                        if isempty(mbar)
                            mbar = [mean(abs.(res.mean_traj[j, late_start:end])) for j in 1:seeds]
                            @warn "No converged runs at kappa=$(round(kappa_val, digits=5)); using all runs."
                        end
                        per_kappa_samples[i] = mbar
                    end

                    valid_idx = findall(i -> !isempty(per_kappa_samples[i]), eachindex(per_kappa_samples))
                    per_kappa_samples = per_kappa_samples[valid_idx]
                    kappas_sweep = kappas_sweep[valid_idx]

                    cache_rows = Vector{NamedTuple}()
                    for i in eachindex(kappas_sweep)
                        runs = per_kappa_samples[i]
                        isempty(runs) && continue
                        push!(cache_rows, (
                            kappa = kappas_sweep[i],
                            m_abs_mean = mean(runs),
                            m_abs_se = std(runs) / sqrt(max(length(runs), 1)),
                            n_runs = length(runs),
                        ))
                    end
                    if !isempty(cache_rows)
                        dt_tag = replace(string(dt), "." => "p")
                        cache_name = "sweep_impl=$(String(impl))_dt=$(dt_tag)_N=$(N).csv"
                        CSV.write(joinpath(outdir, "cache", cache_name), DataFrame(cache_rows))
                    end

                    # Baseline window selection (for reporting)
                    M_corr_full = Float64[]
                    for i in eachindex(kappas_sweep)
                        mbar = mean(per_kappa_samples[i])
                        push!(M_corr_full, sqrt(max(mbar^2 - m0^2, 0.0)))
                    end
                    opt_win, n_win, _ = select_delta_window(
                        kappas_sweep, M_corr_full, kappa_eff;
                        delta_min=delta_window[1], delta_max=delta_window[2],
                        n_steps=20, min_points=min_points,
                    )
                    delta_vals = (kappas_sweep .- kappa_eff) ./ kappa_eff
                    win_mask = (delta_vals .>= opt_win[1]) .& (delta_vals .<= opt_win[2]) .& (M_corr_full .> 0)
                    n_scaling = sum(win_mask)

                    # Step 4: nested bootstrap for beta
                    beta_hat = NaN
                    beta_se = NaN
                    beta_ci = (NaN, NaN)
                    theil_beta = NaN

                    if !isempty(growth.boot_kappa) && length(kappas_sweep) >= min_points
                        boot_result = bootstrap_beta_nested(
                            kappas_sweep,
                            per_kappa_samples,
                            m0,
                            growth.boot_kappa,
                            delta_window;
                            n_outer=n_boot_beta,
                            delta_steps=20,
                            min_points=min_points,
                        )
                        beta_hat = boot_result.beta_hat
                        beta_se = boot_result.beta_se
                        beta_ci = boot_result.beta_ci
                        theil_beta = boot_result.theil_sen
                    else
                        if n_scaling >= min_points
                            x = log.(delta_vals[win_mask])
                            y = log.(M_corr_full[win_mask])
                            X = hcat(ones(length(x)), x)
                            coeffs = X \ y
                            beta_hat = coeffs[2]
                            resid = y - X * coeffs
                            sigma2 = sum(resid.^2) / max(length(x) - 2, 1)
                            se_beta = sqrt(sigma2 / sum((x .- mean(x)).^2))
                            beta_se = se_beta
                            beta_ci = (beta_hat - 1.96 * se_beta, beta_hat + 1.96 * se_beta)
                            theil_beta = theil_sen_slope(x, y)
                        end
                    end

                    @printf("  beta_hat = %.4f (se=%.4f) [95%% CI: %.4f, %.4f]\n",
                            beta_hat, beta_se, beta_ci[1], beta_ci[2])
                    @printf("  Theil-Sen beta = %.4f\n", theil_beta)
                    @printf(log_io, "  beta_hat = %.4f (se=%.4f) [95%% CI: %.4f, %.4f]\n",
                            beta_hat, beta_se, beta_ci[1], beta_ci[2])
                    @printf(log_io, "  Theil-Sen beta = %.4f\n", theil_beta)

                    all_runs = isempty(per_kappa_samples) ? Float64[] : vcat(per_kappa_samples...)
                    denom = seeds * max(length(kappas_sweep), 1)
                    frac_conv = denom > 0 ? length(all_runs) / denom : NaN

                    push!(results, (
                        impl=String(impl),
                        dt=dt,
                        N=N,
                        kappa_star_eff=kappa_eff,
                        kappa_se=growth.kappa_se,
                        beta_hat=beta_hat,
                        beta_se=beta_se,
                        beta_ci_lower=beta_ci[1],
                        beta_ci_upper=beta_ci[2],
                        theil_sen_beta=theil_beta,
                        n_scaling_points=n_scaling,
                        delta_window_lo=opt_win[1],
                        delta_window_hi=opt_win[2],
                        fraction_converged=frac_conv,
                        m0=m0,
                        seeds=seeds,
                        horizon=T_base,
                        burnin=burn_in,
                        criterion="growth_scan + bimodality",
                    ))

                    open(joinpath(outdir, "diagnostics.log"), "a") do diag_io
                        println(diag_io, "Configuration: impl=$(impl), dt=$(dt), N=$(N)")
                        println(diag_io, "  m0 = $(m0) (se=$(m0_se))")
                        println(diag_io, "  kappa* = $(kappa_eff) (se=$(growth.kappa_se))")
                        println(diag_io, "  beta = $(beta_hat) (se=$(beta_se))  CI: [$(beta_ci[1]), $(beta_ci[2])]")
                        println(diag_io, "  Theil-Sen beta = $(theil_beta)")
                        println(diag_io, "  n_scaling_points = $(n_scaling)")
                        println(diag_io, "  delta window = [$(opt_win[1]), $(opt_win[2])]")
                        println(diag_io, "  fraction converged = $(frac_conv)")
                        println(diag_io)
                    end
                end
            end
        end

        CSV.write(joinpath(outdir, "robustness_table.csv"), results)
        write_tex_table(results, joinpath(outdir, "robustness_table.tex"))
        plot_summary(results, joinpath(outdir, "figs", "robustness_summary.pdf"))

        cfg_text = read(config_path, String)
        manifest = Dict(
            "git_commit" => git_commit_hash(pwd()),
            "timestamp" => string(now()),
            "host" => Sys.hostname(),
            "julia_version" => string(VERSION),
            "config_path" => config_path,
            "config_hash" => sha_hex(cfg_text),
            "quick" => quick,
            "pkg_status" => let
                io = IOBuffer()
                Pkg.status(io=io)
                String(take!(io))
            end,
        )
        JSON3.write(joinpath(outdir, "run_manifest.json"), manifest)

        println("\nOutputs written to: $outdir")
        println(log_io, "")
        println(log_io, "Outputs written to: $outdir")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
