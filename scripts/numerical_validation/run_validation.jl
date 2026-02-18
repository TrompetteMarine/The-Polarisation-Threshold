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

    return slopes, boot_slopes
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
    slopes, boot_slopes = growth_scan(
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
        for _ in 1:bisection_iters
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
                seed_base=seed_base + 777,
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
    if n_boot > 0 && size(boot_slopes, 1) > 0
        boot_kappa = Float64[]
        sizehint!(boot_kappa, size(boot_slopes, 1))
        for b in 1:size(boot_slopes, 1)
            k_b = estimate_kappa_star_from_scan(kappa_grid, boot_slopes[b, :])
            push!(boot_kappa, k_b)
        end
        kappa_se = std(boot_kappa)
    end

    return (kappa_star=kappa_star, kappa_se=kappa_se, slopes=slopes, status=status)
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
    open(path, "w") do io
        println(io, "% Auto-generated by run_validation.jl")
        println(io, "\\begin{tabular}{lcccccc}")
        println(io, "\\toprule")
        latex_inline(s) = "\\(" * s * "\\)"
        header = "Impl & " *
                 latex_inline("\\Delta t") * " & " *
                 latex_inline("N") * " & " *
                 latex_inline("\\kappa^*_{\\mathrm{eff}}") * " & s.e. & " *
                 latex_inline("\\beta") * " & s.e. \\\\"
        println(io, header)
        println(io, "\\midrule")
        for impl in unique(df.impl)
            df_impl = df[df.impl .== impl, :]
            for dt in unique(df_impl.dt)
                df_dt = df_impl[df_impl.dt .== dt, :]
                for row in eachrow(df_dt)
                    @printf(io, "%s & %.4g & %d & %.5f & %.5f & %.4f & %.4f \\\\\n",
                            row.impl, row.dt, row.N, row.kappa_star_eff, row.kappa_se, row.beta_hat, row.beta_se)
                end
            end
            println(io, "\\midrule")
        end
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
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

    log_path = joinpath(outdir, "run.log")
    open(log_path, "w") do log_io
        println("============================================================")
        println("NUMERICAL VALIDATION PIPELINE")
        println("============================================================")
        println("Config: $config_path")
        println("Mode: ", quick ? "quick" : "full")
        println("Timestamp: ", string(now()))
        println(log_io, "============================================================")
        println(log_io, "NUMERICAL VALIDATION PIPELINE")
        println(log_io, "============================================================")
        println(log_io, "Config: $config_path")
        println(log_io, "Mode: ", quick ? "quick" : "full")
        println(log_io, "Timestamp: ", string(now()))

        p = mkparams(cfg["params"])
        grid = cfg["grid"]
        runtime = cfg["runtime"]

        seeds = Int(get(grid, "seeds", 20))
        seed_base = Int(get(grid, "seed_base", 0))
        T = Float64(runtime["T"])
        burn_in = Float64(runtime["burn_in"])
        mt_stride = Int(runtime["mt_stride"])

        if quick
            seeds = Int(get(cfg["quick"], "seeds", seeds))
            T = Float64(get(cfg["quick"], "T", T))
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

                    kappa_min = kappa_theory * kappa_min_factor
                    kappa_max = kappa_theory * kappa_max_factor
                    kappa_grid = collect(range(kappa_min, kappa_max; length=grid_points))

                    growth = kappa_star_from_growth(
                        p;
                        kappa_grid=kappa_grid,
                        N=N,
                        T=T,
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

                    bimod = kappa_star_from_bimodality(
                        p,
                        kappa_grid;
                        N=N,
                        T=T,
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

                    kappa_eff = growth.kappa_star
                    if !isfinite(kappa_eff)
                        kappa_eff = bimod.kappa_star
                    end

                    delta_grid = exp.(range(log(delta_window[1]), log(delta_window[2]); length=kappa_points))
                    beta = beta_from_sweep(
                        p;
                        kappa_star=kappa_eff,
                        deltas=delta_grid,
                        N=N,
                        T=T,
                        dt=dt,
                        n_ensemble=seeds,
                        seed_base=seed_base,
                        impl=impl,
                        mt_stride=mt_stride,
                        epsilon=epsilon,
                        late_window_frac=late_window_frac,
                        n_boot=n_boot_beta,
                        min_points=min_points,
                    )

                    @printf("  beta_hat = %.4f (se=%.4f, n=%d, R2=%.3f)\n",
                            beta.beta_hat, beta.beta_se, beta.n, beta.r2)
                    @printf(log_io, "  beta_hat = %.4f (se=%.4f, n=%d, R2=%.3f)\n",
                            beta.beta_hat, beta.beta_se, beta.n, beta.r2)

                    push!(results, (
                        impl=String(impl),
                        dt=dt,
                        N=N,
                        kappa_star_eff=kappa_eff,
                        kappa_se=growth.kappa_se,
                        beta_hat=beta.beta_hat,
                        beta_se=beta.beta_se,
                        seeds=seeds,
                        horizon=T,
                        burnin=burn_in,
                        criterion="growth_scan + bimodality",
                    ))
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
