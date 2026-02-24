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
using Base.Threads

using BeliefSim
using BeliefSim.Types: Params, StepHazard
using BeliefSim.Stats: critical_kappa

include(joinpath(@__DIR__, "..", "..", "src", "sim_oubr.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "diagnostics.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "region_classifier.jl"))

using .SimOUBR
using .Diagnostics
using .RegionClassifier

struct ExecutionOptions
    parallel_mode::Symbol
    parallel_level::Symbol
    max_threads_outer::Int
    threaded_logging::Bool
    device::Symbol
    benchmark_mode::Bool
    benchmark_repeats::Int
end

const LOG_FLUSH_EVERY = 50
const LOG_LINE_COUNT = Threads.Atomic{Int}(0)
const LOG_IO_LOCK = ReentrantLock()

function load_config(path::String)
    cfg_raw = YAML.load_file(path)
    return cfg_raw
end

function is_run_config(cfg::AbstractDict)
    return haskey(cfg, "model_defaults") && haskey(cfg, "parameter_sweep")
end

function load_base_region_config()
    base_path = joinpath(@__DIR__, "..", "..", "configs", "region_id.yaml")
    if !isfile(base_path)
        error("Base config not found: $base_path")
    end
    return YAML.load_file(base_path)
end

function impl_symbol_from_label(label::AbstractString)
    s = lowercase(String(label))
    if s in ("a_first_cross", "first_crossing", "first_cross")
        return :first_crossing
    elseif s in ("b_interpolate", "interp", "interpolate")
        return :interp
    else
        return Symbol(s)
    end
end

function _compute_sigma_eta(pt::AbstractDict, lambda::Float64, theta::Float64)
    if haskey(pt, "sigma")
        sigma = Float64(pt["sigma"])
        eta = sigma^2 / (2.0 * lambda * theta^2)
        return sigma, eta
    elseif haskey(pt, "eta")
        eta = Float64(pt["eta"])
        sigma = sqrt(2.0 * lambda * theta^2 * eta)
        return sigma, eta
    else
        error("Point must define sigma or eta.")
    end
end

function collect_parameter_points(cfg::AbstractDict)
    defaults = cfg["model_defaults"]
    lambda = Float64(defaults["lambda"])
    theta = Float64(defaults["Theta"])
    param_mode = haskey(defaults, "parameterisation") ? String(defaults["parameterisation"]) : "eta"

    points = Dict{String,NamedTuple{(:label, :c0, :sigma, :eta),Tuple{String,Float64,Float64,Float64}}}()

    sweep = cfg["parameter_sweep"]
    mode = String(sweep["mode"])

    function add_point(pt::AbstractDict)
        c0 = Float64(pt["c0"])
        sigma, eta = _compute_sigma_eta(pt, lambda, theta)
        label = haskey(pt, "label") ? String(pt["label"]) : @sprintf("c0=%.2f,eta=%.3f", c0, eta)
        key = @sprintf("c0=%.4f,eta=%.6f", c0, eta)
        points[key] = (
            label = label,
            c0 = c0,
            sigma = sigma,
            eta = eta,
        )
        return nothing
    end

    if mode == "explicit_points" || mode == "hybrid"
        for pt in sweep["explicit_points"]
            add_point(pt)
        end
    end

    if (mode == "cartesian_grid" || mode == "hybrid") && sweep["cartesian_grid"]["enabled"]
        grid = sweep["cartesian_grid"]
        c0_list = [Float64(x) for x in grid["c0_list"]]
        if param_mode == "sigma" && haskey(grid, "sigma_list")
            sigma_list = [Float64(x) for x in grid["sigma_list"]]
            for c0 in c0_list
                for sigma in sigma_list
                    pt = Dict{String,Any}("c0" => c0, "sigma" => sigma)
                    add_point(pt)
                end
            end
        else
            eta_list = [Float64(x) for x in grid["eta_list"]]
            for c0 in c0_list
                for eta in eta_list
                    pt = Dict{String,Any}("c0" => c0, "eta" => eta)
                    add_point(pt)
                end
            end
        end
    end

    return collect(values(points))
end

function parse_time_value(val, lambda::Float64)
    if val isa Number
        return Float64(val)
    elseif val isa AbstractString
        s = String(val)
        if occursin("/lambda", s)
            num = replace(s, "/lambda" => "")
            return parse(Float64, strip(num)) / lambda
        else
            return parse(Float64, strip(s))
        end
    else
        error("Unsupported time value: $val")
    end
end

@inline function _parse_symbol_opt(raw, default::Symbol, allowed::Tuple)
    raw === nothing && return default
    sym = Symbol(lowercase(String(raw)))
    return sym in allowed ? sym : default
end

function parse_execution_options(cfg::AbstractDict)
    execution = haskey(cfg, "execution") ? cfg["execution"] : Dict{String,Any}()

    parallel_mode = _parse_symbol_opt(get(execution, "parallel_mode", "serial"), :serial, (:serial, :threads))
    parallel_level = _parse_symbol_opt(get(execution, "parallel_level", "panels"), :panels, (:points, :panels, :n))
    max_threads_outer = max(1, Int(get(execution, "max_threads_outer", Threads.nthreads())))
    threaded_logging = Bool(get(execution, "threaded_logging", false))
    device = _parse_symbol_opt(get(execution, "device", "cpu"), :cpu, (:cpu, :gpu, :auto))
    benchmark_mode = Bool(get(execution, "benchmark_mode", false))
    benchmark_repeats = max(1, Int(get(execution, "benchmark_repeats", 1)))

    # Backward-compatible bridge for existing configs with execution.threads.
    if !haskey(execution, "parallel_mode") && haskey(execution, "threads")
        th = execution["threads"]
        if th isa Integer
            max_threads_outer = max(1, min(max_threads_outer, Int(th)))
            parallel_mode = max_threads_outer > 1 ? :threads : :serial
        elseif th isa AbstractString && lowercase(String(th)) == "auto"
            parallel_mode = Threads.nthreads() > 1 ? :threads : :serial
        end
    end

    return ExecutionOptions(
        parallel_mode,
        parallel_level,
        max_threads_outer,
        threaded_logging,
        device,
        benchmark_mode,
        benchmark_repeats,
    )
end

function effective_worker_count(opts::ExecutionOptions, n_jobs::Int)
    if opts.parallel_mode != :threads || n_jobs <= 1 || Threads.nthreads() <= 1
        return 1
    end
    return min(n_jobs, Threads.nthreads(), max(1, opts.max_threads_outer))
end

function threaded_collect(f::Function, jobs::Vector, n_workers::Int)
    out = Vector{Any}(undef, length(jobs))
    if isempty(jobs)
        return out
    end
    if n_workers <= 1
        for i in eachindex(jobs)
            out[i] = f(jobs[i], i)
        end
        return out
    end

    next_idx = Threads.Atomic{Int}(0)
    Threads.@sync for _ in 1:n_workers
        Threads.@spawn begin
            while true
                idx = Threads.atomic_add!(next_idx, 1) + 1
                idx > length(jobs) && break
                out[idx] = f(jobs[idx], idx)
            end
        end
    end
    return out
end

function resolve_run_ensemble_fn(device::Symbol, log_io::Union{IO,Nothing}; threaded::Bool = false)
    cpu_fn = SimOUBR.run_ensemble_oubr
    has_gpu_fn = isdefined(SimOUBR, :run_ensemble_oubr_gpu)

    if device == :cpu
        return cpu_fn
    elseif device == :gpu
        if !has_gpu_fn
            log_line(log_io, "[execution] GPU requested but no GPU kernel is defined; falling back to CPU."; threaded=threaded)
            return cpu_fn
        end
        gpu_fn = getfield(SimOUBR, :run_ensemble_oubr_gpu)
        log_line(log_io, "[execution] Using GPU kernel when available; runtime errors fall back to CPU."; threaded=threaded)
        return (p; kwargs...) -> begin
            try
                gpu_fn(p; kwargs...)
            catch
                cpu_fn(p; kwargs...)
            end
        end
    else
        if !has_gpu_fn
            log_line(log_io, "[execution] device=auto with no GPU kernel; using CPU.", threaded=threaded)
            return cpu_fn
        end
        gpu_fn = getfield(SimOUBR, :run_ensemble_oubr_gpu)
        log_line(log_io, "[execution] device=auto detected GPU path; runtime errors fall back to CPU.", threaded=threaded)
        return (p; kwargs...) -> begin
            try
                gpu_fn(p; kwargs...)
            catch
                cpu_fn(p; kwargs...)
            end
        end
    end
end

@inline function sample_col_range(time_grid::Vector{Float64}, t0::Float64, t1::Float64)
    i0 = searchsortedfirst(time_grid, t0)
    i1 = searchsortedlast(time_grid, t1)
    i0 > i1 && return 1:0
    return i0:i1
end

function row_means_over_range!(out::Vector{Float64}, A::Matrix{Float64}, cols::UnitRange{Int})
    n_rows = size(A, 1)
    inv_n = isempty(cols) ? NaN : (1.0 / length(cols))
    @inbounds for r in 1:n_rows
        s = 0.0
        for c in cols
            s += A[r, c]
        end
        out[r] = s * inv_n
    end
    return out
end

function col_means_over_range(A::Matrix{Float64}, cols::UnitRange{Int})
    n_rows = size(A, 1)
    out = Vector{Float64}(undef, length(cols))
    inv_n = n_rows > 0 ? (1.0 / n_rows) : NaN
    j = 1
    @inbounds for c in cols
        s = 0.0
        for r in 1:n_rows
            s += A[r, c]
        end
        out[j] = s * inv_n
        j += 1
    end
    return out
end

@inline function atilde_from_variance(v::Float64, V0::Float64, theta::Float64)
    return sqrt(max(v - V0, 0.0)) / theta
end

function block_means_atilde(series::Vector{Float64}, V0::Float64, theta::Float64, n_blocks::Int)
    n = length(series)
    n == 0 && return Float64[]

    if n_blocks <= 1
        s = 0.0
        @inbounds for i in 1:n
            s += atilde_from_variance(series[i], V0, theta)
        end
        return [s / n]
    end

    block_size = max(1, div(n, n_blocks))
    out = Float64[]
    sizehint!(out, n_blocks)
    start = 1
    for b in 1:n_blocks
        stop = b == n_blocks ? n : min(n, start + block_size - 1)
        s = 0.0
        @inbounds for i in start:stop
            s += atilde_from_variance(series[i], V0, theta)
        end
        push!(out, s / max(stop - start + 1, 1))
        start = stop + 1
        start > n && break
    end
    return out
end

function fill_atilde_from_row!(out::Vector{Float64}, A::Matrix{Float64}, row::Int, V0::Float64, theta::Float64)
    @inbounds for c in 1:size(A, 2)
        out[c] = atilde_from_variance(A[row, c], V0, theta)
    end
    return out
end

function nearest_index(values::Vector{Float64}, target::Float64)
    best_i = 1
    best_d = Inf
    @inbounds for i in eachindex(values)
        d = abs(values[i] - target)
        if d < best_d
            best_d = d
            best_i = i
        end
    end
    return best_i
end

function log_line(io::Union{IO,Nothing}, msg::AbstractString; flush_now::Bool = false, threaded::Bool = false)
    println(msg)
    if io !== nothing
        if threaded
            lock(LOG_IO_LOCK) do
                println(io, msg)
                n = Threads.atomic_add!(LOG_LINE_COUNT, 1) + 1
                if flush_now || (n % LOG_FLUSH_EVERY == 0)
                    flush(io)
                end
            end
        else
            println(io, msg)
            n = Threads.atomic_add!(LOG_LINE_COUNT, 1) + 1
            if flush_now || (n % LOG_FLUSH_EVERY == 0)
                flush(io)
            end
        end
    end
end

function logf(io::Union{IO,Nothing}, fmt::AbstractString, args...; flush_now::Bool = false, threaded::Bool = false)
    msg = isempty(args) ? String(fmt) : Printf.format(Printf.Format(fmt), args...)
    log_line(io, msg; flush_now=flush_now, threaded=threaded)
end

function make_params(pdict::AbstractDict)
    hazard = StepHazard(0.0)
    return Params(
        λ = Float64(pdict["lambda"]),
        σ = Float64(pdict["sigma"]),
        Θ = Float64(pdict["theta"]),
        c0 = Float64(pdict["c0"]),
        hazard = hazard,
    )
end

function build_delta_grid(cfg::AbstractDict)
    coarse = haskey(cfg, "coarse_delta") ? cfg["coarse_delta"] : nothing
    refined = haskey(cfg, "refined_delta") ? cfg["refined_delta"] : nothing
    tc_refined = haskey(cfg, "tc_refined_delta") ? cfg["tc_refined_delta"] : nothing
    deltas = Float64[]
    for c in (coarse, refined, tc_refined)
        c === nothing && continue
        if haskey(c, "enabled") && !Bool(c["enabled"])
            continue
        end
        dmin = Float64(c["min"])
        dmax = Float64(c["max"])
        step = Float64(c["step"])
        n = Int(floor((dmax - dmin) / step)) + 1
        for i in 0:(n - 1)
            push!(deltas, dmin + i * step)
        end
    end
    deltas = unique(deltas)
    sort!(deltas)
    return deltas
end

function block_means(series::Vector{Float64}, idx::Vector{Int}, n_blocks::Int)
    if isempty(idx)
        return Float64[]
    end
    n = length(idx)
    if n_blocks <= 1
        return [mean(series[idx])]
    end
    block_size = max(1, div(n, n_blocks))
    out = Float64[]
    start = 1
    for b in 1:n_blocks
        stop = b == n_blocks ? n : min(n, start + block_size - 1)
        push!(out, mean(series[idx[start:stop]]))
        start = stop + 1
        start > n && break
    end
    return out
end

function run_panel(
    p::Params;
    kappa_theory::Float64,
    N::Int,
    dt::Float64,
    impl::Symbol,
    seeds::Int,
    seed_base::Int,
    mt_stride::Int,
    burn_in::Float64,
    sample::Float64,
    ordered_value::Float64,
    growth_cfg::AbstractDict,
    scan_cfg::AbstractDict,
    station_cfg::AbstractDict,
    beta_cfg::AbstractDict,
    psd_cfg::AbstractDict,
    thresholds::AbstractDict,
    compute_hysteresis::Bool = true,
    compute_psd::Bool = true,
    compute_beta::Bool = true,
    run_ensemble_fn::Function = SimOUBR.run_ensemble_oubr,
    log_io::Union{IO,Nothing} = nothing,
    log_enabled::Bool = true,
    log_threaded::Bool = false,
    progress_every::Int = 10,
)
    function panel_log(fmt::AbstractString, args...)
        log_enabled || return nothing
        logf(log_io, fmt, args...; threaded=log_threaded)
        return nothing
    end

    total_T = burn_in + sample
    growth_kappa_min_factor = Float64(growth_cfg["kappa_min_factor"])
    growth_kappa_max_factor = Float64(growth_cfg["kappa_max_factor"])
    growth_grid_points = Int(growth_cfg["grid_points"])
    growth_epsilon = Float64(growth_cfg["epsilon"])
    growth_window = (Float64(growth_cfg["window"][1]), Float64(growth_cfg["window"][2]))
    growth_n_boot = Int(growth_cfg["n_boot"])

    station_blocks = Int(station_cfg["blocks"])
    station_sigma = Float64(station_cfg["sigma"])

    noise_ref_delta = Float64(thresholds["noise_floor"]["reference_delta"])
    noise_min_effect = Float64(thresholds["noise_floor"]["min_effect_on_top"])
    jump_ci_sigma = Float64(thresholds["jump_detector"]["ci_sigma"])
    jump_min_jump = Float64(thresholds["jump_detector"]["min_jump_size"])
    hyst_ci_sigma = Float64(thresholds["hysteresis"]["ci_sigma"])
    hyst_min_gap = Float64(thresholds["hysteresis"]["min_gap"])

    refined_has = haskey(scan_cfg, "refined_delta")
    refined_enabled = refined_has ? !haskey(scan_cfg["refined_delta"], "enabled") || Bool(scan_cfg["refined_delta"]["enabled"]) : false
    refined_min = refined_has ? Float64(scan_cfg["refined_delta"]["min"]) : -Inf
    refined_max = refined_has ? Float64(scan_cfg["refined_delta"]["max"]) : Inf

    beta_wide_window = (Float64(beta_cfg["windows"]["wide"][1]), Float64(beta_cfg["windows"]["wide"][2]))
    beta_mid_window = (Float64(beta_cfg["windows"]["mid"][1]), Float64(beta_cfg["windows"]["mid"][2]))
    beta_tight_window = (Float64(beta_cfg["windows"]["tight"][1]), Float64(beta_cfg["windows"]["tight"][2]))

    psd_min_omega = Float64(psd_cfg["min_omega_factor"]) * p.λ
    psd_seed_cv = Float64(get(psd_cfg, "seed_coherence_cv", 0.15))
    psd_prominence_threshold = Float64(psd_cfg["prominence_ratio"])
    psd_dc_threshold = Float64(psd_cfg["dc_dominance_ratio"])

    # Step 1: kappa_star_eff from growth scan
    kappa_min = kappa_theory * growth_kappa_min_factor
    kappa_max = kappa_theory * growth_kappa_max_factor
    kappa_grid = collect(range(kappa_min, kappa_max; length=growth_grid_points))

    panel_log("[panel] N=%d dt=%.4g impl=%s seeds=%d", N, dt, String(impl), seeds)
    panel_log("[panel] growth scan kappa grid: %d points", length(kappa_grid))
    growth = Diagnostics.growth_scan_kappa_star(
        run_ensemble_fn,
        p,
        kappa_grid;
        N=N,
        T=total_T,
        dt=dt,
        n_ensemble=seeds,
        seed_base=seed_base,
        impl=impl,
        mt_stride=mt_stride,
        epsilon=growth_epsilon,
        window=growth_window,
        n_boot=growth_n_boot,
        ordered_value=ordered_value,
    )

    kappa_star_eff = growth.kappa_star
    if !isfinite(kappa_star_eff) || kappa_star_eff <= 0
        return Dict{String,Any}(
            "kappa_star_eff" => kappa_star_eff,
            "kappa_star_se" => growth.kappa_se,
            "signal_points" => 0,
            "jump_flag" => false,
            "jump_size" => NaN,
            "hysteresis_flag" => false,
            "delta_kappa_rel" => NaN,
            "beta_wide" => NaN,
            "beta_mid" => NaN,
            "beta_tight" => NaN,
            "beta_mid_se" => NaN,
            "beta_mid_r2" => NaN,
            "beta_mid_n" => 0,
            "psd_flag" => false,
            "omega_peak" => NaN,
            "psd_prominence" => NaN,
            "psd_dc_ratio" => NaN,
            "V0" => NaN,
            "noise_floor" => NaN,
            "kappas" => Float64[],
            "atilde_mean" => Float64[],
            "atilde_se" => Float64[],
        )
    end

    # Step 2: sweep grid
    deltas = build_delta_grid(scan_cfg)
    kappas = kappa_star_eff .* (1 .+ deltas)
    n_kappa = length(kappas)

    per_kappa_samples = Vector{Vector{Float64}}(undef, n_kappa)
    mean_sq_sample_series = Vector{Vector{Float64}}(undef, n_kappa)
    sample_cols = 1:0
    idx_psd = findfirst(d -> d > 0, deltas)
    mean_sq_psd::Union{Nothing,Matrix{Float64}} = nothing
    per_run_tmp = Vector{Float64}(undef, seeds)

    panel_log("[panel] sweep grid: %d kappa points", n_kappa)
    for (i, kappa) in enumerate(kappas)
        res = run_ensemble_fn(
            p;
            kappa=kappa,
            N=N,
            T=total_T,
            dt=dt,
            n_ensemble=seeds,
            seed_base=seed_base + 10000 * i,
            impl=impl,
            mt_stride=mt_stride,
            init_mode=:random,
            epsilon=growth_epsilon,
            ordered_value=ordered_value,
        )
        if i == 1
            sample_cols = sample_col_range(res.time_grid, burn_in, total_T)
            isempty(sample_cols) && error("No sample indices found for burn_in=$burn_in sample=$sample")
        end

        if idx_psd !== nothing && i == idx_psd
            mean_sq_psd = res.mean_sq_traj
        end

        row_means_over_range!(per_run_tmp, res.mean_sq_traj, sample_cols)
        per_kappa_samples[i] = copy(per_run_tmp)
        mean_sq_sample_series[i] = col_means_over_range(res.mean_sq_traj, sample_cols)

        if (i % progress_every == 0) || (i == n_kappa)
            panel_log("[panel] sweep progress %d/%d", i, n_kappa)
        end
    end

    # V0 baseline
    V_mean = Vector{Float64}(undef, n_kappa)
    for i in 1:n_kappa
        V_mean[i] = mean(per_kappa_samples[i])
    end
    V0_res = Diagnostics.estimate_V0_from_sweep(V_mean, kappas, kappa_star_eff)
    V0 = V0_res.V0

    # Atilde stats
    stats = Diagnostics.compute_atilde_stats(kappas, per_kappa_samples, V0, p.Θ)

    # Stationarity gate
    stationary_mask = BitVector(undef, n_kappa)
    for i in 1:n_kappa
        bmeans = block_means_atilde(mean_sq_sample_series[i], V0, p.Θ, station_blocks)
        gate = Diagnostics.stationarity_gate(bmeans; sigma=station_sigma)
        stationary_mask[i] = gate.stationary
    end

    # Noise floor
    ref_idx = argmin(abs.(deltas .- noise_ref_delta))
    ref_runs = per_kappa_samples[ref_idx]
    ref_atilde = Vector{Float64}(undef, length(ref_runs))
    @inbounds for i in eachindex(ref_runs)
        ref_atilde[i] = atilde_from_variance(ref_runs[i], V0, p.Θ)
    end
    noise_med = median(ref_atilde)
    noise_mad = median(abs.(ref_atilde .- noise_med))
    noise_floor = noise_med + 5.0 * noise_mad

    # Signal gate count in refined band
    signal_count = 0
    for i in 1:n_kappa
        if stationary_mask[i] && isfinite(stats.atilde_mean[i]) &&
           stats.atilde_mean[i] >= noise_floor + noise_min_effect
            if deltas[i] >= refined_min && deltas[i] <= refined_max
                signal_count += 1
            end
        end
    end

    # Use only stationary points for fits/diagnostics
    kappas_fit = kappas[stationary_mask]
    atilde_fit = stats.atilde_mean[stationary_mask]
    atilde_se_fit = stats.atilde_se[stationary_mask]

    # Jump detector on refined grid
    jump = Diagnostics.jump_detector(kappas_fit, atilde_fit, atilde_se_fit;
                                     ci_sigma=jump_ci_sigma,
                                     min_jump=jump_min_jump)

    # Hysteresis detector (backward sweep, ordered init) on refined grid
    hyst = (hysteresis=false, k_low=NaN, k_high=NaN, width=NaN)
    if compute_hysteresis && refined_enabled
        refined_mask = (deltas .>= refined_min) .& (deltas .<= refined_max)
        kappas_ref = kappas[refined_mask]
        panel_log("[panel] hysteresis sweep: %d kappa points", length(kappas_ref))
        per_kappa_down = Vector{Vector{Float64}}(undef, length(kappas_ref))
        per_run_down_tmp = Vector{Float64}(undef, seeds)
        for (i, kappa) in enumerate(kappas_ref)
            res = run_ensemble_fn(
                p;
                kappa=kappa,
                N=N,
                T=total_T,
                dt=dt,
                n_ensemble=seeds,
                seed_base=seed_base + 20000 * i,
                impl=impl,
                mt_stride=mt_stride,
                init_mode=:ordered_plus,
                epsilon=growth_epsilon,
                ordered_value=ordered_value,
            )
            row_means_over_range!(per_run_down_tmp, res.mean_sq_traj, sample_cols)
            per_kappa_down[i] = copy(per_run_down_tmp)
            if (i % progress_every == 0) || (i == length(kappas_ref))
                panel_log("[panel] hysteresis progress %d/%d", i, length(kappas_ref))
            end
        end
        stats_down = Diagnostics.compute_atilde_stats(kappas_ref, per_kappa_down, V0, p.Θ)
        stats_up = Diagnostics.compute_atilde_stats(kappas_ref, per_kappa_samples[refined_mask], V0, p.Θ)
        stat_ref_mask = stationary_mask[refined_mask]
        kappas_ref_fit = kappas_ref[stat_ref_mask]
        up_mean_fit = stats_up.atilde_mean[stat_ref_mask]
        down_mean_fit = stats_down.atilde_mean[stat_ref_mask]
        up_se_fit = stats_up.atilde_se[stat_ref_mask]
        down_se_fit = stats_down.atilde_se[stat_ref_mask]
        hyst = Diagnostics.hysteresis_detector(kappas_ref_fit, up_mean_fit, down_mean_fit,
                                               up_se_fit, down_se_fit;
                                               ci_sigma=hyst_ci_sigma,
                                               min_gap=hyst_min_gap)
    end

    delta_kappa_rel = isfinite(hyst.width) ? hyst.width / kappa_star_eff : NaN

    # Beta fits
    beta_wide_fit = (beta=NaN, se=NaN, r2=NaN, n=0)
    beta_mid_fit = (beta=NaN, se=NaN, r2=NaN, n=0)
    beta_tight_fit = (beta=NaN, se=NaN, r2=NaN, n=0)
    if compute_beta
        beta_wide_fit = Diagnostics.fit_beta_window(kappas_fit, atilde_fit, kappa_star_eff, beta_wide_window)
        beta_mid_fit = Diagnostics.fit_beta_window(kappas_fit, atilde_fit, kappa_star_eff, beta_mid_window)
        beta_tight_fit = Diagnostics.fit_beta_window(kappas_fit, atilde_fit, kappa_star_eff, beta_tight_window)
    end

    # PSD metrics on a representative supercritical point
    psd_flag = false
    omega_peak = NaN
    prominence = NaN
    dc_ratio = NaN
    omega_cv = NaN
    if compute_psd && mean_sq_psd !== nothing
        omega_peaks = Float64[]
        prominences = Float64[]
        dc_ratios = Float64[]
        atilde_series = Vector{Float64}(undef, size(mean_sq_psd, 2))
        for j in 1:size(mean_sq_psd, 1)
            fill_atilde_from_row!(atilde_series, mean_sq_psd, j, V0, p.Θ)
            pm = Diagnostics.psd_peak_metrics(atilde_series, dt * mt_stride;
                                              min_omega=psd_min_omega)
            if isfinite(pm.omega_peak) && isfinite(pm.prominence) && isfinite(pm.dc_ratio)
                push!(omega_peaks, pm.omega_peak)
                push!(prominences, pm.prominence)
                push!(dc_ratios, pm.dc_ratio)
            end
        end
        if length(omega_peaks) >= 3
            omega_peak = median(omega_peaks)
            omega_cv = std(omega_peaks) / max(abs(mean(omega_peaks)), 1e-12)
            prominence = median(prominences)
            dc_ratio = median(dc_ratios)
            psd_flag = omega_cv <= psd_seed_cv &&
                       prominence >= psd_prominence_threshold &&
                       dc_ratio >= psd_dc_threshold
        end
    end

    # Re-entrant detection: ordering then return to noise at high kappa
    reentrant_flag = false
    kappa_high = maximum(kappas)
    atilde_high = stats.atilde_mean[argmax(kappas)]
    if isfinite(kappa_high) && kappa_high >= max(3.0 * kappa_star_eff, kappa_star_eff + 2.0 * p.λ)
        if isfinite(atilde_high) && atilde_high <= noise_floor + noise_min_effect
            # count peaks above noise floor
            a_sorted = stats.atilde_mean[sortperm(kappas)]
            peak_count = 0
            for i in 2:(length(a_sorted) - 1)
                if a_sorted[i] > a_sorted[i - 1] && a_sorted[i] > a_sorted[i + 1] &&
                   a_sorted[i] >= noise_floor + noise_min_effect
                    peak_count += 1
                end
            end
            if peak_count >= 2
                reentrant_flag = true
            end
        end
    end

    diag = Dict{String,Any}(
        "kappa_star_eff" => kappa_star_eff,
        "kappa_star_se" => growth.kappa_se,
        "signal_points" => signal_count,
        "jump_flag" => jump.jump,
        "jump_size" => jump.jump_size,
        "hysteresis_flag" => hyst.hysteresis,
        "delta_kappa_rel" => delta_kappa_rel,
        "beta_wide" => beta_wide_fit.beta,
        "beta_mid" => beta_mid_fit.beta,
        "beta_tight" => beta_tight_fit.beta,
        "beta_mid_se" => beta_mid_fit.se,
        "beta_mid_r2" => beta_mid_fit.r2,
        "beta_mid_n" => beta_mid_fit.n,
        "psd_flag" => psd_flag,
        "omega_peak" => omega_peak,
        "psd_prominence" => prominence,
        "psd_dc_ratio" => dc_ratio,
        "reentrant_flag" => reentrant_flag,
        "V0" => V0,
        "noise_floor" => noise_floor,
        "kappas" => kappas,
        "atilde_mean" => stats.atilde_mean,
        "atilde_se" => stats.atilde_se,
    )

    return diag
end

function run_panel_set(
    p::Params,
    run_N_list::Vector{Int};
    kappa_theory::Float64,
    dt::Float64,
    impl::Symbol,
    seeds::Int,
    seed_base::Int,
    mt_stride::Int,
    burn_in::Float64,
    sample::Float64,
    ordered_value::Float64,
    growth_cfg::AbstractDict,
    scan_cfg::AbstractDict,
    station_cfg::AbstractDict,
    beta_cfg::AbstractDict,
    psd_cfg::AbstractDict,
    thresholds::AbstractDict,
    compute_hysteresis::Bool = true,
    compute_psd::Bool = true,
    compute_beta::Bool = true,
    run_ensemble_fn::Function = SimOUBR.run_ensemble_oubr,
    n_workers::Int = 1,
    log_io::Union{IO,Nothing} = nothing,
    log_enabled::Bool = true,
    threaded_logging::Bool = false,
)
    n_panels = length(run_N_list)
    diag_vec = Vector{Dict{String,Any}}(undef, n_panels)
    panel_log_enabled = log_enabled && (n_workers == 1 || threaded_logging)

    if n_workers <= 1 || n_panels <= 1
        for i in eachindex(run_N_list)
            N = run_N_list[i]
            diag_vec[i] = run_panel(
                p;
                kappa_theory=kappa_theory,
                N=N,
                dt=dt,
                impl=impl,
                seeds=seeds,
                seed_base=seed_base,
                mt_stride=mt_stride,
                burn_in=burn_in,
                sample=sample,
                ordered_value=ordered_value,
                growth_cfg=growth_cfg,
                scan_cfg=scan_cfg,
                station_cfg=station_cfg,
                beta_cfg=beta_cfg,
                psd_cfg=psd_cfg,
                thresholds=thresholds,
                compute_hysteresis=compute_hysteresis,
                compute_psd=compute_psd,
                compute_beta=compute_beta,
                run_ensemble_fn=run_ensemble_fn,
                log_io=log_io,
                log_enabled=panel_log_enabled,
                log_threaded=threaded_logging,
            )
        end
    else
        next_idx = Threads.Atomic{Int}(0)
        Threads.@sync for _ in 1:n_workers
            Threads.@spawn begin
                while true
                    i = Threads.atomic_add!(next_idx, 1) + 1
                    i > n_panels && break
                    N = run_N_list[i]
                    diag_vec[i] = run_panel(
                        p;
                        kappa_theory=kappa_theory,
                        N=N,
                        dt=dt,
                        impl=impl,
                        seeds=seeds,
                        seed_base=seed_base,
                        mt_stride=mt_stride,
                        burn_in=burn_in,
                        sample=sample,
                        ordered_value=ordered_value,
                        growth_cfg=growth_cfg,
                        scan_cfg=scan_cfg,
                        station_cfg=station_cfg,
                        beta_cfg=beta_cfg,
                        psd_cfg=psd_cfg,
                        thresholds=thresholds,
                        compute_hysteresis=compute_hysteresis,
                        compute_psd=compute_psd,
                        compute_beta=compute_beta,
                        run_ensemble_fn=run_ensemble_fn,
                        log_io=log_io,
                        log_enabled=panel_log_enabled,
                        log_threaded=threaded_logging,
                    )
                end
            end
        end
    end

    panel_diags = Dict{Int,Dict{String,Any}}()
    for i in eachindex(run_N_list)
        panel_diags[run_N_list[i]] = diag_vec[i]
    end
    return panel_diags
end

function compute_combo_row(
    p::Params,
    pt::NamedTuple;
    kappa_theory::Float64,
    dt::Float64,
    impl::Symbol,
    panel_N_list::Vector{Int},
    alpha_N_list::Vector{Int},
    run_N_list::Vector{Int},
    seeds::Int,
    seed_base_pt::Int,
    mt_stride::Int,
    burn_in::Float64,
    sample::Float64,
    ordered_value::Float64,
    growth_cfg::AbstractDict,
    scan_cfg::AbstractDict,
    station_cfg::AbstractDict,
    beta_cfg::AbstractDict,
    psd_cfg::AbstractDict,
    thresholds::AbstractDict,
    compute_hysteresis::Bool = true,
    compute_psd::Bool = true,
    compute_beta::Bool = true,
    compute_alpha::Bool = true,
    kappa_star_analytic::Float64 = NaN,
    run_ensemble_fn::Function = SimOUBR.run_ensemble_oubr,
    parallel_mode::Symbol = :serial,
    parallel_level::Symbol = :panels,
    max_threads_outer::Int = 1,
    log_io::Union{IO,Nothing} = nothing,
    log_enabled::Bool = true,
    threaded_logging::Bool = false,
)
    n_workers_n = 1
    if parallel_mode == :threads && parallel_level == :n
        n_workers_n = min(length(run_N_list), max_threads_outer, Threads.nthreads())
    end

    panel_diags = run_panel_set(
        p,
        run_N_list;
        kappa_theory=kappa_theory,
        dt=dt,
        impl=impl,
        seeds=seeds,
        seed_base=seed_base_pt,
        mt_stride=mt_stride,
        burn_in=burn_in,
        sample=sample,
        ordered_value=ordered_value,
        growth_cfg=growth_cfg,
        scan_cfg=scan_cfg,
        station_cfg=station_cfg,
        beta_cfg=beta_cfg,
        psd_cfg=psd_cfg,
        thresholds=thresholds,
        compute_hysteresis=compute_hysteresis,
        compute_psd=compute_psd,
        compute_beta=compute_beta,
        run_ensemble_fn=run_ensemble_fn,
        n_workers=n_workers_n,
        log_io=log_io,
        log_enabled=log_enabled,
        threaded_logging=threaded_logging,
    )

    atilde_at_kstar = Float64[]
    Ns_for_alpha = Int[]
    if compute_alpha
        for N in alpha_N_list
            diag = panel_diags[N]
            ks = Float64(diag["kappa_star_eff"])
            if isfinite(ks)
                kappas = diag["kappas"]::Vector{Float64}
                atilde = diag["atilde_mean"]::Vector{Float64}
                idx = nearest_index(kappas, ks)
                push!(atilde_at_kstar, atilde[idx])
                push!(Ns_for_alpha, N)
            end
        end
    end

    alpha_fit = compute_alpha ? Diagnostics.fit_alpha(Ns_for_alpha, atilde_at_kstar) :
               (alpha=NaN, se=NaN, r2=NaN, n=0)

    vote_rows = Vector{Dict{String,Any}}(undef, length(panel_N_list))
    for (i, N) in enumerate(panel_N_list)
        diag = panel_diags[N]
        diag["alpha_hat"] = alpha_fit.alpha
        diag["alpha_se"] = alpha_fit.se
        diag["alpha_r2"] = alpha_fit.r2
        diag["alpha_n"] = alpha_fit.n

        (label, _, anchor) = RegionClassifier.classify_region(diag; thresholds=thresholds)
        vote_rows[i] = Dict(
            "impl" => String(impl),
            "dt" => dt,
            "N" => N,
            "region_label" => label,
            "anchor_pass" => anchor,
        )
        diag["region_label"] = label
    end

    agg = RegionClassifier.aggregate_robustness(vote_rows)

    n_panel = length(panel_N_list)
    kappa_eff_vals = Vector{Float64}(undef, n_panel)
    jump_sizes = Vector{Float64}(undef, n_panel)
    delta_kappas = Vector{Float64}(undef, n_panel)
    beta_mids = Vector{Float64}(undef, n_panel)
    beta_mid_ses = Vector{Float64}(undef, n_panel)
    jump_flag = false
    hyst_flag = false
    for (i, N) in enumerate(panel_N_list)
        diag = panel_diags[N]
        kappa_eff_vals[i] = Float64(diag["kappa_star_eff"])
        jump_sizes[i] = Float64(diag["jump_size"])
        delta_kappas[i] = Float64(diag["delta_kappa_rel"])
        beta_mids[i] = Float64(diag["beta_mid"])
        beta_mid_ses[i] = Float64(diag["beta_mid_se"])
        jump_flag |= Bool(diag["jump_flag"])
        hyst_flag |= Bool(diag["hysteresis_flag"])
    end

    psd_diag = panel_diags[panel_N_list[1]]
    return (
        c0 = pt.c0,
        eta = pt.eta,
        lambda = p.λ,
        sigma = p.σ,
        Theta = p.Θ,
        kappa_star_analytic = kappa_star_analytic,
        kappa_star_eff = median(kappa_eff_vals),
        jump_flag = jump_flag,
        jump_size_Atilde = median(jump_sizes),
        hysteresis_flag = hyst_flag,
        Delta_kappa_rel = median(delta_kappas),
        beta_hat_mid = median(beta_mids),
        se_beta_mid = median(beta_mid_ses),
        alpha_hat = alpha_fit.alpha,
        se_alpha = alpha_fit.se,
        psd_peak_type = Bool(psd_diag["psd_flag"]) ? "omega" : "DC",
        omega_peak = Float64(psd_diag["omega_peak"]),
        region_label = agg.region_label,
        confidence = agg.confidence,
        abstain_reason = agg.abstain_reason,
    )
end

function maybe_run_panel_benchmark(
    outdir::String,
    repeats::Int,
    p::Params;
    kappa_theory::Float64,
    N::Int,
    dt::Float64,
    impl::Symbol,
    seeds::Int,
    seed_base::Int,
    mt_stride::Int,
    burn_in::Float64,
    sample::Float64,
    ordered_value::Float64,
    growth_cfg::AbstractDict,
    scan_cfg::AbstractDict,
    station_cfg::AbstractDict,
    beta_cfg::AbstractDict,
    psd_cfg::AbstractDict,
    thresholds::AbstractDict,
    compute_hysteresis::Bool,
    compute_psd::Bool,
    compute_beta::Bool,
    run_ensemble_fn::Function,
)
    times = Vector{Float64}(undef, repeats)
    bytes = Vector{Int}(undef, repeats)
    for r in 1:repeats
        GC.gc()
        timed = @timed run_panel(
            p;
            kappa_theory=kappa_theory,
            N=N,
            dt=dt,
            impl=impl,
            seeds=seeds,
            seed_base=seed_base,
            mt_stride=mt_stride,
            burn_in=burn_in,
            sample=sample,
            ordered_value=ordered_value,
            growth_cfg=growth_cfg,
            scan_cfg=scan_cfg,
            station_cfg=station_cfg,
            beta_cfg=beta_cfg,
            psd_cfg=psd_cfg,
            thresholds=thresholds,
            compute_hysteresis=compute_hysteresis,
            compute_psd=compute_psd,
            compute_beta=compute_beta,
            run_ensemble_fn=run_ensemble_fn,
            log_io=nothing,
            log_enabled=false,
        )
        times[r] = timed.time
        bytes[r] = timed.bytes
    end

    bench_path = joinpath(outdir, "benchmark_summary.txt")
    open(bench_path, "w") do io
        println(io, "run_panel benchmark (optional)")
        println(io, "repeats: $repeats")
        println(io, "N: $N, dt: $dt, impl: $(String(impl)), seeds: $seeds")
        println(io, @sprintf("median_time_s: %.6f", median(times)))
        println(io, @sprintf("median_alloc_bytes: %.0f", median(Float64.(bytes))))
        println(io, "all_time_s: $(times)")
        println(io, "all_alloc_bytes: $(bytes)")
    end
end

function write_signature_tables(outdir::String; write_csv::Bool = true, write_tex::Bool = true)
    rows = RegionClassifier.region_signature_rows()
    df = DataFrame(rows)
    if write_csv
        CSV.write(joinpath(outdir, "region_signature_table.csv"), df)
    end

    if write_tex
        tex_path = joinpath(outdir, "region_signature_table.tex")
        open(tex_path, "w") do io
            println(io, "% Auto-generated region signature table")
            println(io, "\\begin{tabular}{lllllllllll}")
            println(io, "\\toprule")
            println(io, "Region & Name & Transition & Jump? & Hysteresis? & Delta kappa & Beta & Alpha & PSD & Large-kappa & Runs \\\\")
            println(io, "\\midrule")
            for row in eachrow(df)
                @printf(io, "%s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s \\\\\n",
                        row.Region, row.Name, row.Transition, row.Jump, row.Hysteresis,
                        row.Delta_kappa_rel, row.Beta, row.Alpha, row.PSD, row.Large_kappa, row.Runs)
            end
            println(io, "\\bottomrule")
            println(io, "\\end{tabular}")
        end
    end
end

function write_region_table(rows::Vector{NamedTuple}, outdir::String; write_csv::Bool = true, write_tex::Bool = true)
    df = DataFrame(rows)
    if write_csv
        CSV.write(joinpath(outdir, "region_identification_table.csv"), df)
    end

    if write_tex
        tex_path = joinpath(outdir, "region_identification_table.tex")
        open(tex_path, "w") do io
            println(io, "% Auto-generated region identification table")
            println(io, "\\begin{tabular}{llllllllllllllllllll}")
            println(io, "\\toprule")
            println(io, "c0 & eta & lambda & sigma & theta & kappa_star_analytic & kappa_star_eff & jump & jump_size & hysteresis & delta_kappa_rel & beta_mid & se_beta_mid & alpha & se_alpha & psd & omega & region & confidence & abstain \\\\")
            println(io, "\\midrule")
            for row in eachrow(df)
                @printf(io,
                        "%.3f & %.3f & %.3f & %.3f & %.3f & %.5f & %.5f & %s & %.4f & %s & %.4f & %.4f & %.4f & %.4f & %.4f & %s & %.4f & %s & %.2f & %s \\\\\n",
                        row.c0, row.eta, row.lambda, row.sigma, row.Theta,
                        row.kappa_star_analytic, row.kappa_star_eff,
                        row.jump_flag ? "Y" : "N", row.jump_size_Atilde,
                        row.hysteresis_flag ? "Y" : "N", row.Delta_kappa_rel,
                        row.beta_hat_mid, row.se_beta_mid,
                        row.alpha_hat, row.se_alpha,
                        row.psd_peak_type, row.omega_peak,
                        row.region_label, row.confidence, row.abstain_reason)
            end
            println(io, "\\bottomrule")
            println(io, "\\end{tabular}")
        end
    end
end

function write_provenance(outdir::String, cfg_path::String, cfg::AbstractDict)
    prov = joinpath(outdir, "README_provenance.txt")
    open(prov, "w") do io
        println(io, "Region identification provenance")
        println(io, "Timestamp: $(Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))")
        println(io, "Git hash: $(try string(LibGit2.head_oid(LibGit2.GitRepo(pwd()))) catch; "unknown" end)")
        println(io, "Julia version: $(VERSION)")
        println(io, "Config path: $cfg_path")
        if is_run_config(cfg)
            println(io, "Run mode: $(cfg["run_mode"])")
            println(io, "Model defaults: $(cfg["model_defaults"])")
            println(io, "Parameter sweep: $(cfg["parameter_sweep"])")
            println(io, "Execution: $(get(cfg, "execution", Dict{String,Any}()))")
            println(io, "IO: $(get(cfg, "io", Dict{String,Any}()))")
        else
            println(io, "Grid dt: $(cfg["grid"]["dt"])")
            println(io, "Grid N: $(cfg["grid"]["N"])")
            println(io, "Impl: $(cfg["grid"]["impl"])")
            println(io, "Seeds: $(cfg["grid"]["seeds"])")
            println(io, "Burn-in: $(cfg["runtime"]["burn_in"])")
            println(io, "Sample: $(cfg["runtime"]["sample"])")
            println(io, "Kappa scan: $(cfg["runtime"]["kappa_scan"])")
            println(io, "Stationarity: $(cfg["runtime"]["stationarity"])")
            println(io, "Beta fit: $(cfg["runtime"]["beta_fit"])")
            println(io, "Alpha fit: $(cfg["runtime"]["alpha_fit"])")
            println(io, "PSD: $(cfg["runtime"]["psd_peak"])")
        end
    end
end

function main()
    config_path = ""
    for (i, arg) in enumerate(ARGS)
        if arg == "--config" && i < length(ARGS)
            config_path = ARGS[i + 1]
        end
    end
    if isempty(config_path)
        println("Usage: julia --project=. scripts/region_id/run_region_identification.jl --config configs/region_id.yaml")
        exit(1)
    end

    cfg = load_config(config_path)
    if is_run_config(cfg)
        base_cfg = load_base_region_config()

        defaults = cfg["model_defaults"]
        lambda = Float64(defaults["lambda"])
        theta = Float64(defaults["Theta"])
        ordered_value = Float64(get(defaults, "ordered_value", 1.0))

        mode_name = String(cfg["run_mode"])
        mode_cfg = cfg["modes"][mode_name]
        panel_cfg = mode_cfg["robustness_panel"]
        diag_cfg = mode_cfg["diagnostics"]

        dt_list = [Float64(x) for x in panel_cfg["dt_values"]]
        impl_list = [impl_symbol_from_label(x) for x in panel_cfg["boundary_impl"]]
        panel_N_list = [Int(x) for x in panel_cfg["N_values"]]
        seeds = Int(panel_cfg["seeds"])

        compute_hysteresis = Bool(diag_cfg["compute_hysteresis"])
        compute_psd = Bool(diag_cfg["compute_psd"])
        compute_beta = Bool(diag_cfg["compute_beta"])
        compute_alpha = Bool(diag_cfg["compute_alpha"])

        alpha_N_list = panel_N_list
        if compute_alpha && haskey(diag_cfg, "alpha_fit_N_values")
            alpha_N_list = [Int(x) for x in diag_cfg["alpha_fit_N_values"]]
        end

        run_N_list = unique(vcat(panel_N_list, alpha_N_list))

        runtime = base_cfg["runtime"]
        scan_cfg = deepcopy(runtime["kappa_scan"])
        scan_flags = mode_cfg["kappa_scan"]
        scan_cfg["coarse_delta"]["enabled"] = Bool(scan_flags["use_coarse"])
        scan_cfg["refined_delta"]["enabled"] = Bool(scan_flags["use_refined"])
        scan_cfg["tc_refined_delta"]["enabled"] = Bool(scan_flags["use_tc_refined"])

        # overrides
        if haskey(cfg, "overrides")
            overrides = cfg["overrides"]
            if haskey(overrides, "kappa_scan_overrides")
                for (k, v) in overrides["kappa_scan_overrides"]
                    scan_cfg[k] = v
                end
            end
            if haskey(overrides, "time_window_overrides")
                tw = overrides["time_window_overrides"]
                if haskey(tw, "burn_in")
                    runtime["burn_in"] = tw["burn_in"]
                end
                if haskey(tw, "sample")
                    runtime["sample"] = tw["sample"]
                end
            end
        end

        station_cfg = runtime["stationarity"]
        beta_cfg = runtime["beta_fit"]
        psd_cfg = runtime["psd_peak"]
        growth_cfg = runtime["growth_scan"]
        mt_stride = Int(runtime["mt_stride"])
        burn_in = parse_time_value(runtime["burn_in"], lambda)
        sample = parse_time_value(runtime["sample"], lambda)

        thresholds = haskey(cfg, "thresholds") ? cfg["thresholds"] :
                     (haskey(base_cfg, "thresholds") ? base_cfg["thresholds"] : RegionClassifier.default_thresholds())

        execution_opts = parse_execution_options(cfg)
        execution = haskey(cfg, "execution") ? cfg["execution"] : Dict{String,Any}()
        seed_base = Int(get(execution, "rng_seed_base", 0))

        out_root = String(cfg["meta"]["output_root"])
        output_subdir = String(cfg["io"]["output_subdir"])
        exp_name = String(cfg["meta"]["experiment_name"])
        outdir = joinpath(out_root, output_subdir, exp_name)
        isdir(outdir) || mkpath(outdir)
        isdir(joinpath(outdir, "configs")) || mkpath(joinpath(outdir, "configs"))

        write_csv = Bool(cfg["io"]["write_csv"])
        write_tex = Bool(cfg["io"]["write_tex"])
        write_configs = Bool(cfg["io"]["write_configs"])
        write_provenance_flag = Bool(cfg["io"]["write_provenance"])

        if write_csv || write_tex
            write_signature_tables(outdir; write_csv=write_csv, write_tex=write_tex)
        end

        if write_configs
            try
                cp(config_path, joinpath(outdir, "configs", basename(config_path)); force=true)
            catch
            end
        end

        points = collect_parameter_points(cfg)
        point_models = Vector{NamedTuple}(undef, length(points))
        for (pi, pt) in enumerate(points)
            p = Params(λ=lambda, σ=pt.sigma, Θ=theta, c0=pt.c0, hazard=StepHazard(0.0))
            point_models[pi] = (
                pt = pt,
                p = p,
                kappa_theory = critical_kappa(p),
                seed_base_pt = seed_base + (pi - 1) * 1_000_000,
            )
        end

        kappa_star_analytic_enabled = haskey(cfg, "kappa_star_analytic") && Bool(cfg["kappa_star_analytic"]["enabled"])

        results_rows = Vector{NamedTuple}()
        log_path = joinpath(outdir, "run.log")
        open(log_path, "a") do log_io
            log_line(log_io, "============================================================"; flush_now=true)
            log_line(log_io, "OU-BR Region Identification Run")
            log_line(log_io, "Config: $(config_path)")
            log_line(log_io, "Mode: $(mode_name)")
            ts = Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS")
            log_line(log_io, "Timestamp: $ts")
            log_line(log_io, "Points: $(length(points))")
            log_line(log_io, "dt: $(dt_list)")
            log_line(log_io, "impl: $(impl_list)")
            log_line(log_io, "panel N: $(panel_N_list)")
            log_line(log_io, "alpha N: $(alpha_N_list)")
            log_line(log_io, "seeds: $(seeds)")
            log_line(log_io, "parallel_mode: $(execution_opts.parallel_mode)")
            log_line(log_io, "parallel_level: $(execution_opts.parallel_level)")
            log_line(log_io, "max_threads_outer: $(execution_opts.max_threads_outer)")
            log_line(log_io, "threaded_logging: $(execution_opts.threaded_logging)")
            log_line(log_io, "device: $(execution_opts.device)")
            log_line(log_io, "============================================================"; flush_now=true)

            run_ensemble_fn = resolve_run_ensemble_fn(execution_opts.device, log_io; threaded=execution_opts.threaded_logging)

            if execution_opts.benchmark_mode && !isempty(point_models)
                m0 = point_models[1]
                maybe_run_panel_benchmark(
                    outdir,
                    execution_opts.benchmark_repeats,
                    m0.p;
                    kappa_theory=m0.kappa_theory,
                    N=run_N_list[1],
                    dt=dt_list[1],
                    impl=impl_list[1],
                    seeds=seeds,
                    seed_base=m0.seed_base_pt,
                    mt_stride=mt_stride,
                    burn_in=burn_in,
                    sample=sample,
                    ordered_value=ordered_value,
                    growth_cfg=growth_cfg,
                    scan_cfg=scan_cfg,
                    station_cfg=station_cfg,
                    beta_cfg=beta_cfg,
                    psd_cfg=psd_cfg,
                    thresholds=thresholds,
                    compute_hysteresis=compute_hysteresis,
                    compute_psd=compute_psd,
                    compute_beta=compute_beta,
                    run_ensemble_fn=run_ensemble_fn,
                )
                log_line(log_io, "Benchmark written: $(joinpath(outdir, "benchmark_summary.txt"))")
            end

            if execution_opts.parallel_level == :points
                jobs = collect(1:length(point_models))
                n_workers = effective_worker_count(execution_opts, length(jobs))
                row_blocks = threaded_collect(jobs, n_workers) do pi, _
                    model = point_models[pi]
                    local_rows = NamedTuple[]
                    local_log_enabled = (n_workers == 1) || execution_opts.threaded_logging
                    if local_log_enabled
                        logf(log_io, "[point %d/%d] label=%s c0=%.3f eta=%.4f sigma=%.4f",
                             pi, length(point_models), model.pt.label, model.pt.c0, model.pt.eta, model.pt.sigma;
                             threaded=execution_opts.threaded_logging)
                    end
                    kappa_star_analytic = kappa_star_analytic_enabled ? model.kappa_theory : NaN
                    for dt in dt_list
                        for impl in impl_list
                            row = compute_combo_row(
                                model.p,
                                model.pt;
                                kappa_theory=model.kappa_theory,
                                dt=dt,
                                impl=impl,
                                panel_N_list=panel_N_list,
                                alpha_N_list=alpha_N_list,
                                run_N_list=run_N_list,
                                seeds=seeds,
                                seed_base_pt=model.seed_base_pt,
                                mt_stride=mt_stride,
                                burn_in=burn_in,
                                sample=sample,
                                ordered_value=ordered_value,
                                growth_cfg=growth_cfg,
                                scan_cfg=scan_cfg,
                                station_cfg=station_cfg,
                                beta_cfg=beta_cfg,
                                psd_cfg=psd_cfg,
                                thresholds=thresholds,
                                compute_hysteresis=compute_hysteresis,
                                compute_psd=compute_psd,
                                compute_beta=compute_beta,
                                compute_alpha=compute_alpha,
                                kappa_star_analytic=kappa_star_analytic,
                                run_ensemble_fn=run_ensemble_fn,
                                parallel_mode=execution_opts.parallel_mode,
                                parallel_level=execution_opts.parallel_level,
                                max_threads_outer=execution_opts.max_threads_outer,
                                log_io=log_io,
                                log_enabled=local_log_enabled,
                                threaded_logging=execution_opts.threaded_logging,
                            )
                            push!(local_rows, row)
                        end
                    end
                    return local_rows
                end

                for rows in row_blocks
                    append!(results_rows, rows)
                end
            elseif execution_opts.parallel_level == :panels
                jobs = NamedTuple[]
                for pi in 1:length(point_models)
                    for dt in dt_list
                        for impl in impl_list
                            push!(jobs, (pi=pi, dt=dt, impl=impl))
                        end
                    end
                end

                n_workers = effective_worker_count(execution_opts, length(jobs))
                panel_rows = threaded_collect(jobs, n_workers) do job, _
                    model = point_models[job.pi]
                    kappa_star_analytic = kappa_star_analytic_enabled ? model.kappa_theory : NaN
                    local_log_enabled = (n_workers == 1) || execution_opts.threaded_logging
                    if local_log_enabled
                        logf(log_io, "[point %d/%d] label=%s dt=%.4g impl=%s",
                             job.pi, length(point_models), model.pt.label, job.dt, String(job.impl);
                             threaded=execution_opts.threaded_logging)
                    end
                    return compute_combo_row(
                        model.p,
                        model.pt;
                        kappa_theory=model.kappa_theory,
                        dt=job.dt,
                        impl=job.impl,
                        panel_N_list=panel_N_list,
                        alpha_N_list=alpha_N_list,
                        run_N_list=run_N_list,
                        seeds=seeds,
                        seed_base_pt=model.seed_base_pt,
                        mt_stride=mt_stride,
                        burn_in=burn_in,
                        sample=sample,
                        ordered_value=ordered_value,
                        growth_cfg=growth_cfg,
                        scan_cfg=scan_cfg,
                        station_cfg=station_cfg,
                        beta_cfg=beta_cfg,
                        psd_cfg=psd_cfg,
                        thresholds=thresholds,
                        compute_hysteresis=compute_hysteresis,
                        compute_psd=compute_psd,
                        compute_beta=compute_beta,
                        compute_alpha=compute_alpha,
                        kappa_star_analytic=kappa_star_analytic,
                        run_ensemble_fn=run_ensemble_fn,
                        parallel_mode=execution_opts.parallel_mode,
                        parallel_level=execution_opts.parallel_level,
                        max_threads_outer=execution_opts.max_threads_outer,
                        log_io=log_io,
                        log_enabled=local_log_enabled,
                        threaded_logging=execution_opts.threaded_logging,
                    )
                end
                append!(results_rows, panel_rows)
            else
                for (pi, model) in enumerate(point_models)
                    logf(log_io, "[point %d/%d] label=%s c0=%.3f eta=%.4f sigma=%.4f",
                         pi, length(point_models), model.pt.label, model.pt.c0, model.pt.eta, model.pt.sigma)
                    kappa_star_analytic = kappa_star_analytic_enabled ? model.kappa_theory : NaN
                    for dt in dt_list
                        for impl in impl_list
                            row = compute_combo_row(
                                model.p,
                                model.pt;
                                kappa_theory=model.kappa_theory,
                                dt=dt,
                                impl=impl,
                                panel_N_list=panel_N_list,
                                alpha_N_list=alpha_N_list,
                                run_N_list=run_N_list,
                                seeds=seeds,
                                seed_base_pt=model.seed_base_pt,
                                mt_stride=mt_stride,
                                burn_in=burn_in,
                                sample=sample,
                                ordered_value=ordered_value,
                                growth_cfg=growth_cfg,
                                scan_cfg=scan_cfg,
                                station_cfg=station_cfg,
                                beta_cfg=beta_cfg,
                                psd_cfg=psd_cfg,
                                thresholds=thresholds,
                                compute_hysteresis=compute_hysteresis,
                                compute_psd=compute_psd,
                                compute_beta=compute_beta,
                                compute_alpha=compute_alpha,
                                kappa_star_analytic=kappa_star_analytic,
                                run_ensemble_fn=run_ensemble_fn,
                                parallel_mode=execution_opts.parallel_mode,
                                parallel_level=execution_opts.parallel_level,
                                max_threads_outer=execution_opts.max_threads_outer,
                                log_io=log_io,
                                log_enabled=true,
                                threaded_logging=execution_opts.threaded_logging,
                            )
                            push!(results_rows, row)
                        end
                    end
                end
            end
        end

        if write_csv || write_tex
            write_region_table(results_rows, outdir; write_csv=write_csv, write_tex=write_tex)
        end
        if write_provenance_flag
            write_provenance(outdir, config_path, cfg)
        end
        println("Outputs written to: $outdir")
        return
    end

    # Legacy config path (configs/region_id.yaml)
    outdir = cfg["outputs"]["dir"]
    isdir(outdir) || mkpath(outdir)
    isdir(joinpath(outdir, "configs")) || mkpath(joinpath(outdir, "configs"))

    write_signature_tables(outdir)

    # copy config
    try
        cp(config_path, joinpath(outdir, "configs", basename(config_path)); force=true)
    catch
    end

    p = make_params(cfg["params"])
    ordered_value = Float64(get(cfg["params"], "ordered_value", 1.0))

    grid = cfg["grid"]
    runtime = cfg["runtime"]
    execution_opts = parse_execution_options(cfg)

    seeds = Int(grid["seeds"])
    seed_base = Int(get(grid, "seed_base", 0))

    burn_in = parse_time_value(runtime["burn_in"], p.λ)
    sample = parse_time_value(runtime["sample"], p.λ)
    mt_stride = Int(runtime["mt_stride"])

    growth_cfg = runtime["growth_scan"]
    scan_cfg = runtime["kappa_scan"]
    station_cfg = runtime["stationarity"]
    beta_cfg = runtime["beta_fit"]
    psd_cfg = runtime["psd_peak"]
    thresholds = haskey(cfg, "thresholds") ? cfg["thresholds"] : RegionClassifier.default_thresholds()

    dt_list = [Float64(x) for x in grid["dt"]]
    N_list = [Int(x) for x in grid["N"]]
    impl_list = [Symbol(x) for x in grid["impl"]]

    kappa_theory = critical_kappa(p)
    pt_legacy = (
        label = "legacy",
        c0 = p.c0,
        sigma = p.σ,
        eta = Float64(get(cfg["params"], "eta", NaN)),
    )

    results_rows = Vector{NamedTuple}()
    log_path = joinpath(outdir, "run.log")
    open(log_path, "a") do log_io
        log_line(log_io, "============================================================"; flush_now=true)
        log_line(log_io, "OU-BR Region Identification Run (legacy config)")
        log_line(log_io, "Config: $(config_path)")
        ts = Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS")
        log_line(log_io, "Timestamp: $ts")
        log_line(log_io, "dt: $(dt_list)")
        log_line(log_io, "impl: $(impl_list)")
        log_line(log_io, "N: $(N_list)")
        log_line(log_io, "seeds: $(seeds)")
        log_line(log_io, "parallel_mode: $(execution_opts.parallel_mode)")
        log_line(log_io, "parallel_level: $(execution_opts.parallel_level)")
        log_line(log_io, "max_threads_outer: $(execution_opts.max_threads_outer)")
        log_line(log_io, "threaded_logging: $(execution_opts.threaded_logging)")
        log_line(log_io, "device: $(execution_opts.device)")
        log_line(log_io, "============================================================"; flush_now=true)

        run_ensemble_fn = resolve_run_ensemble_fn(execution_opts.device, log_io; threaded=execution_opts.threaded_logging)

        if execution_opts.benchmark_mode
            maybe_run_panel_benchmark(
                outdir,
                execution_opts.benchmark_repeats,
                p;
                kappa_theory=kappa_theory,
                N=N_list[1],
                dt=dt_list[1],
                impl=impl_list[1],
                seeds=seeds,
                seed_base=seed_base,
                mt_stride=mt_stride,
                burn_in=burn_in,
                sample=sample,
                ordered_value=ordered_value,
                growth_cfg=growth_cfg,
                scan_cfg=scan_cfg,
                station_cfg=station_cfg,
                beta_cfg=beta_cfg,
                psd_cfg=psd_cfg,
                thresholds=thresholds,
                compute_hysteresis=true,
                compute_psd=true,
                compute_beta=true,
                run_ensemble_fn=run_ensemble_fn,
            )
            log_line(log_io, "Benchmark written: $(joinpath(outdir, "benchmark_summary.txt"))")
        end

        jobs = NamedTuple[]
        for dt in dt_list
            for impl in impl_list
                push!(jobs, (dt=dt, impl=impl))
            end
        end

        n_workers = execution_opts.parallel_level == :panels ? effective_worker_count(execution_opts, length(jobs)) : 1
        combo_rows = threaded_collect(jobs, n_workers) do job, _
            local_log_enabled = (n_workers == 1) || execution_opts.threaded_logging
            return compute_combo_row(
                p,
                pt_legacy;
                kappa_theory=kappa_theory,
                dt=job.dt,
                impl=job.impl,
                panel_N_list=N_list,
                alpha_N_list=N_list,
                run_N_list=N_list,
                seeds=seeds,
                seed_base_pt=seed_base,
                mt_stride=mt_stride,
                burn_in=burn_in,
                sample=sample,
                ordered_value=ordered_value,
                growth_cfg=growth_cfg,
                scan_cfg=scan_cfg,
                station_cfg=station_cfg,
                beta_cfg=beta_cfg,
                psd_cfg=psd_cfg,
                thresholds=thresholds,
                compute_hysteresis=true,
                compute_psd=true,
                compute_beta=true,
                compute_alpha=true,
                kappa_star_analytic=kappa_theory,
                run_ensemble_fn=run_ensemble_fn,
                parallel_mode=execution_opts.parallel_mode,
                parallel_level=execution_opts.parallel_level,
                max_threads_outer=execution_opts.max_threads_outer,
                log_io=log_io,
                log_enabled=local_log_enabled,
                threaded_logging=execution_opts.threaded_logging,
            )
        end
        append!(results_rows, combo_rows)
    end

    write_region_table(results_rows, outdir)
    write_provenance(outdir, config_path, cfg)

    println("Outputs written to: $outdir")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
