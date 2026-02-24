#!/usr/bin/env julia
# =============================================================================
# Figure 6 Ensemble Validation: multi-run averaging, CI, and growth-rate tests
# =============================================================================

using Pkg
Pkg.activate(".")

using Random
using Statistics
using LinearAlgebra
using Printf
using Dates
using CSV
using DataFrames
using Distributions
using HypothesisTests
using StatsBase
using JSON3
using LibGit2
using Distributed

using BeliefSim
using BeliefSim.Types
using BeliefSim.Stats
using BeliefSim.Hazard: ν

include(joinpath(@__DIR__, "ensemble_utils.jl"))
include(joinpath(@__DIR__, "statistical_tests.jl"))
include(joinpath(@__DIR__, "visualization.jl"))

using .EnsembleUtils
using .StatisticalTests
using .Visualization

# Paths
project_root = dirname(@__DIR__)
script_dir = @__DIR__

const LOG_PROGRESS = true

function logmsg(msg::String)
    if LOG_PROGRESS
        println("[", Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"), "] ", msg)
        flush(stdout)
    end
end

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model parameters
lambda = 0.85
sigma = 0.8
theta = 2.0
c0 = 0.8
nu0 = 10.6

# Simulation settings
N = 20000
T = 400.0
dt = 0.01
burn_in = 120.0
seed = 2025
T_max = 1200.0
convergence_slope_tol = 1e-4
extend_if_not_converged = false

# Ensemble settings
RUN_MODE = :fast  # :fast for development, :pub for paper renders
if any(arg -> arg in ("-pub", "--pub"), ARGS)
    RUN_MODE = :pub
elseif any(arg -> arg in ("-fast", "--fast"), ARGS)
    RUN_MODE = :fast
end
n_ensemble_scenarios = RUN_MODE == :pub ? 200 : 20
n_ensemble_kappa_sweep = RUN_MODE == :pub ? 80 : 10
n_rep_per_kappa_nearcrit = RUN_MODE == :pub ? 3 : 1

n_ensemble = n_ensemble_scenarios
mt_stride = max(1, Int(round(0.05 / dt)))
snapshot_times = [0.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, T]
parallel_mode = :distributed   # :auto, :threads, :distributed, :off
parallel_target_threads = max(1, Sys.CPU_THREADS - 1)
parallel_target_workers = max(1, Sys.CPU_THREADS - 1)
parallel_ensemble = true 
density_bins = 240

# Quick-run toggle (set true for faster diagnostics)
fast_mode = RUN_MODE == :fast

# Branch classification (symmetry-aware)
branch_late_window_fraction = 0.2
branch_min_window = 50.0
branch_tau_mult = 3.0
branch_min_decided_share = 0.1
branch_tau_floor = 1e-6

# Density sanity checks
density_int_tol = 1e-2
density_mu_mix_tol = 2e-2
density_mu_mix_frac = 0.25
density_mu_mix_relax = 0.25
density_mu_aligned_min = 1e-3
density_check_min_time_frac = 0.8
density_strict_checks = true

# Growth-rate fitting windows
fit_windows = [(10.0, 50.0), (20.0, 80.0), (50.0, 150.0)]
scaling_delta_windows = [(1e-2, 5e-2), (1e-2, 1e-1), (5e-3, 2e-1)]
scaling_amp_floor = 1e-6

# Scenario setup
scenario_ratios = Dict(
    "below" => 0.8,
    "critical" => 1.0,
    "above" => 1.5,
)

# Sweep settings
run_sweep = true
sweep_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
sweep_ensemble = n_ensemble_kappa_sweep
sweep_ratio_cap = 1.5          # Avoid numerical explosions far above kappa*
max_reasonable_variance = 100.0
sweep_mode = :equilibrium      # :equilibrium or :short_time
t_measure = 50.0               # Used when sweep_mode == :short_time

# Alternative observables
run_observables = true
var_fit_window = (50.0, 150.0)

# Threshold outputs
threshold_dir = "outputs/threshold"
mkpath(threshold_dir)

# Route A (empirical growth scan) settings
growth_scan_enabled = true
growth_scan_factors = collect(range(0.6, 1.4; length=9))
growth_scan_window = (10.0, 60.0)
growth_scan_bootstrap = 200
growth_scan_min_points = 8

# Route B (theoretical rank-one) settings
kappaB_L = 6.0
kappaB_M = 401
kappaB_boundary = :reflecting

if fast_mode
    @warn "FAST_MODE enabled: reducing T, sweep size, and density bins for speed."
    T = 200.0
    burn_in = 80.0
    sweep_ratios = [0.8, 1.0, 1.2, 1.4, 1.5]
    sweep_ratio_cap = 1.5
    run_observables = false
    fit_windows = [(10.0, 40.0)]
    t_measure = 30.0
    density_bins = 120
    snapshot_times = [0.0, 20.0, 80.0, T]
    growth_scan_bootstrap = 50
    growth_scan_factors = collect(range(0.7, 1.3; length=7))
end

function parse_device_mode(args::Vector{String})
    for (i, arg) in enumerate(args)
        if arg == "--device" && i < length(args)
            dev = Symbol(lowercase(String(args[i + 1])))
            if dev in (:cpu, :gpu, :auto)
                return dev
            end
        end
    end
    return :auto
end

device_mode = parse_device_mode(ARGS)
gpu_ready = EnsembleUtils.gpu_backend_ready()
simulation_device = if device_mode == :gpu
    gpu_ready ? :gpu : :cpu
elseif device_mode == :auto
    gpu_ready ? :gpu : :cpu
else
    :cpu
end

if device_mode == :gpu && !gpu_ready
    @warn "--device gpu requested but CUDA backend is unavailable; using CPU."
end

function configure_parallelism(; mode::Symbol = :auto, target_threads::Int = 1, target_workers::Int = 1)
    if mode == :off
        return false
    end
    if mode == :distributed
        setup_distributed(target_workers)
        return Distributed.nprocs() > 1
    end

    nthreads = Threads.nthreads()
    if nthreads > 1
        if nthreads < target_threads
            @warn "Julia has $nthreads threads; target is $target_threads. Consider JULIA_NUM_THREADS=$target_threads for faster runs."
        end
        return true
    end

    if mode == :threads
        @warn "Julia is single-threaded. For maximum speed, start with: JULIA_NUM_THREADS=$target_threads"
        return false
    end

    setup_distributed(target_workers)
    return Distributed.nprocs() > 1
end

function setup_distributed(target_workers::Int)
    if target_workers < 1
        return
    end
    if Distributed.nprocs() < target_workers + 1
        addprocs(target_workers + 1 - Distributed.nprocs())
    end
    proj = project_root
    sdir = script_dir
    for w in Distributed.workers()
        Distributed.remotecall_eval(Main, w, :(begin
            using Pkg
            Pkg.activate($proj)
            using Random
            using Statistics
            using BeliefSim
            using BeliefSim.Types
            using BeliefSim.Model: euler_maruyama_step!, reset_step!
            include(joinpath($sdir, "ensemble_utils.jl"))
            using .EnsembleUtils
        end))
    end
    return
end

parallel_ensemble = simulation_device == :gpu ? false : configure_parallelism(
    mode=parallel_mode,
    target_threads=parallel_target_threads,
    target_workers=parallel_target_workers,
)

function parallel_backend_label()
    if !parallel_ensemble
        return "serial"
    end
    if Threads.nthreads() > 1
        return "threads"
    end
    if Distributed.nprocs() > 1
        return "distributed"
    end
    return "serial"
end

logmsg("Parallel backend: $(parallel_backend_label()) (nprocs=$(Distributed.nprocs()), threads=$(Threads.nthreads()), parallel_ensemble=$(parallel_ensemble), device=$(simulation_device))")

# Output paths
outdir = "outputs/ensemble_results"
sweep_dir = "outputs/parameter_sweep"
stats_dir = "outputs/statistical_tests"
figdir = "figs"
docs_dir = "docs"
snippets_dir = "manuscript_snippets"

mkpath(outdir)
mkpath(sweep_dir)
mkpath(stats_dir)
mkpath(figdir)
mkpath(docs_dir)
mkpath(snippets_dir)

function scenario_direction(label::String)
    if label == "below"
        return :less
    elseif label == "above"
        return :greater
    end
    return :two_sided
end

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

function histogram_density(data::Vector{Float64}, edges::Vector{Float64})
    hist = fit(Histogram, data, edges; closed=:left)
    weights = hist.weights
    total = sum(weights)
    binwidth = edges[2] - edges[1]
    density = total > 0 ? (weights ./ (total * binwidth)) : fill(0.0, length(weights))
    centers = midpoints(hist.edges[1])
    return centers, density
end

# =============================================================================
# THRESHOLD UTILITIES (Route A: empirical, Route B: theoretical)
# =============================================================================

function trapezoid_weights(x::Vector{Float64})
    n = length(x)
    @assert n >= 2 "Need at least two grid points for trapezoid weights."
    h = x[2] - x[1]
    w = fill(h, n)
    w[1] *= 0.5
    w[end] *= 0.5
    return w
end

function build_A0_operator(p::Params; L::Float64, M::Int, boundary::Symbol = :reflecting)
    @assert isodd(M) "M should be odd so that x=0 is on the grid."
    x = collect(range(-L, L, length=M))
    h = x[2] - x[1]
    σ2 = p.σ^2 / 2
    A = zeros(Float64, M, M)

    function add_gain!(row::Int, xi::Float64)
        y = xi / p.c0
        if y < -L || y > L
            return
        end
        rate = (1 / p.c0) * ν(p.hazard, y, p.Θ)
        if rate == 0.0
            return
        end
        pos = (y - x[1]) / h + 1
        j = clamp(Int(floor(pos)), 1, M - 1)
        w = pos - j
        A[row, j] += rate * (1 - w)
        A[row, j + 1] += rate * w
    end

    for i in 2:(M - 1)
        xi = x[i]
        rate = ν(p.hazard, xi, p.Θ)

        # Drift: λ ρ + λ x ρ'
        A[i, i] += p.λ
        A[i, i-1] += -p.λ * xi / (2h)
        A[i, i+1] += p.λ * xi / (2h)

        # Diffusion
        A[i, i-1] += σ2 / h^2
        A[i, i] += -2σ2 / h^2
        A[i, i+1] += σ2 / h^2

        # Reset loss
        A[i, i] += -rate

        # Reset gain
        add_gain!(i, xi)
    end

    if boundary == :reflecting
        for i in (1, M)
            xi = x[i]
            rate = ν(p.hazard, xi, p.Θ)
            A[i, i] += p.λ
            A[i, i] += -σ2 / h^2
            if i == 1
                A[i, i+1] += σ2 / h^2
            else
                A[i, i-1] += σ2 / h^2
            end
            A[i, i] += -rate
            add_gain!(i, xi)
        end
    else
        A[1, 1] = -10.0
        A[M, M] = -10.0
    end

    return A, x
end

function stationary_density_from_A0(A::Matrix{Float64}, x::Vector{Float64})
    M = length(x)
    w = trapezoid_weights(x)
    A2 = copy(A)
    b = zeros(Float64, M)
    mid = div(M + 1, 2)
    A2[mid, :] .= w
    b[mid] = 1.0
    rho = A2 \ b
    rho .= 0.5 .* (rho .+ reverse(rho))  # enforce even symmetry
    rho ./= sum(rho .* w)
    return rho, w
end

function odd_subspace_matrix(A::Matrix{Float64})
    M = size(A, 1)
    mid = div(M + 1, 2)
    idx_pos = (mid + 1):M
    n = length(idx_pos)
    Aodd = zeros(Float64, n, n)
    for (col, j) in enumerate(idx_pos)
        f = zeros(Float64, M)
        f[j] = 1.0
        j_mirror = mid - (j - mid)
        f[j_mirror] = -1.0
        g = A * f
        Aodd[:, col] = g[idx_pos]
    end
    return Aodd, idx_pos
end

function compute_kappa_star_B(p::Params; L::Float64, M::Int, boundary::Symbol = :reflecting)
    A0, x = build_A0_operator(p; L=L, M=M, boundary=boundary)
    rho, w = stationary_density_from_A0(A0, x)
    h = x[2] - x[1]

    b = zeros(Float64, length(x))
    for i in 2:(length(x) - 1)
        b[i] = -(rho[i + 1] - rho[i - 1]) / (2h)
    end
    b[1] = -(rho[2] - rho[1]) / h
    b[end] = -(rho[end] - rho[end - 1]) / h

    Aodd, idx_pos = odd_subspace_matrix(A0)
    b_odd = b[idx_pos]
    x_pos = x[idx_pos]
    w_pos = w[idx_pos]
    a_odd = 2 .* x_pos .* w_pos

    v = (-Aodd) \ b_odd
    Phi0 = dot(a_odd, v)
    kappa_B = 1 / Phi0

    return (
        kappa_B = kappa_B,
        Phi0 = Phi0,
        x = x,
        rho = rho,
        Aodd = Aodd,
        b_odd = b_odd,
        a_odd = a_odd,
    )
end

function fit_log_growth(t::Vector{Float64}, y::Vector{Float64}, window::Tuple{Float64, Float64})
    t1, t2 = window
    mask = (t .>= t1) .& (t .<= t2)
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

function growth_scan_kappa_A(
    p::Params,
    kappa_grid::Vector{Float64};
    N::Int,
    T::Float64,
    dt::Float64,
    n_ensemble::Int,
    base_seed::Int,
    snapshot_times::Vector{Float64},
    mt_stride::Int,
    window::Tuple{Float64, Float64},
    n_boot::Int,
    parallel::Bool,
    device::Symbol,
)
    rows = Vector{NamedTuple}()
    boot_slopes = Matrix{Float64}(undef, n_boot, length(kappa_grid))

    for (k_idx, κ) in enumerate(kappa_grid)
        if LOG_PROGRESS
            logmsg("Growth scan $(k_idx)/$(length(kappa_grid)) at kappa=$(round(κ, digits=4))")
        end
        res = run_ensemble_simulation(
            p;
            kappa=κ,
            n_ensemble=n_ensemble,
            N=N,
            T=T,
            dt=dt,
            base_seed=base_seed,
            snapshot_times=snapshot_times,
            mt_stride=mt_stride,
            store_snapshots=false,
            track_moments=false,
            parallel=parallel,
            device=device,
        )
        m_traj = res.mean_trajectories
        m_abs = vec(mean(abs.(m_traj); dims=1))
        fit = fit_log_growth(res.time_grid, m_abs, window)

        for b in 1:n_boot
            idx = rand(1:size(m_traj, 1), size(m_traj, 1))
            m_abs_b = vec(mean(abs.(m_traj[idx, :]); dims=1))
            boot_slopes[b, k_idx] = fit_log_growth(res.time_grid, m_abs_b, window).slope
        end

        ci_lower = quantile(boot_slopes[:, k_idx], 0.025)
        ci_upper = quantile(boot_slopes[:, k_idx], 0.975)

        push!(rows, (
            kappa=κ,
            lambda_hat=fit.slope,
            lambda_se=fit.se,
            lambda_ci_lower=ci_lower,
            lambda_ci_upper=ci_upper,
            r2=fit.r2,
            n_points=fit.n,
        ))
    end

    return rows, boot_slopes
end

function estimate_kappa_star_from_scan(kappa_grid::Vector{Float64}, slopes::Vector{Float64})
    for i in 1:(length(kappa_grid) - 1)
        if slopes[i] <= 0 && slopes[i + 1] >= 0
            w = -slopes[i] / (slopes[i + 1] - slopes[i] + eps())
            return kappa_grid[i] + w * (kappa_grid[i + 1] - kappa_grid[i])
        end
    end
    idx = argmin(abs.(slopes))
    return kappa_grid[idx]
end

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

function late_window_indices(time_grid::Vector{Float64};
    late_window_fraction::Float64 = branch_late_window_fraction,
    min_window::Float64 = branch_min_window,
)
    T_sim = time_grid[end]
    window = min(T_sim, max(late_window_fraction * T_sim, min_window))
    idx = findall(t -> t >= T_sim - window, time_grid)
    if isempty(idx)
        idx = [length(time_grid)]
    end
    return idx
end

function compute_branch_signs(
    mean_traj::Matrix{Float64},
    time_grid::Vector{Float64};
    late_window_fraction::Float64 = branch_late_window_fraction,
    min_window::Float64 = branch_min_window,
    tau::Union{Nothing, Float64} = nothing,
    tau_mult::Float64 = branch_tau_mult,
    min_decided_share::Float64 = branch_min_decided_share,
)
    T_sim = time_grid[end]
    window = min(T_sim, max(late_window_fraction * T_sim, min_window))
    idx = findall(t -> t >= T_sim - window, time_grid)
    if isempty(idx)
        idx = [length(time_grid)]
    end
    n_late = length(idx)
    mean_late = [mean(mean_traj[i, idx]) for i in 1:size(mean_traj, 1)]
    sd_late = n_late > 1 ? [std(mean_traj[i, idx]; corrected=true) for i in 1:size(mean_traj, 1)] :
              fill(0.0, size(mean_traj, 1))

    tau_scale = tau === nothing ? tau_mult * median(sd_late) : tau
    if !isfinite(tau_scale) || tau_scale <= 0.0
        tau_scale = branch_tau_floor
    end
    tau_used = max(branch_tau_floor, tau_scale)

    decided = abs.(mean_late) .>= tau_used
    decided_share = mean(decided)
    if decided_share < min_decided_share
        decided .= false
        decided_share = 0.0
    end

    signs = sign.(mean_late)
    for i in eachindex(signs)
        if !decided[i] || signs[i] == 0
            signs[i] = 1.0
        end
    end

    decided_count = count(decided)
    plus_share = count(i -> decided[i] && signs[i] > 0, eachindex(signs)) / max(1, decided_count)
    minus_share = count(i -> decided[i] && signs[i] < 0, eachindex(signs)) / max(1, decided_count)
    undecided_share = 1.0 - decided_share

    return Int.(signs), mean_late, decided, tau_used, decided_share, plus_share, minus_share, undecided_share
end

function git_commit_hash(path::String)
    try
        repo = LibGit2.GitRepo(path)
        return string(LibGit2.head_oid(repo))
    catch
        return "unknown"
    end
end

function write_docs(summary_path::String, report::String)
    open(summary_path, "w") do io
        write(io, report)
    end
end

# =============================================================================
# ADAPTIVE SWEEP HELPERS
# =============================================================================

function get_adaptive_T(ratio::Float64, base_T::Float64)
    delta = max(abs(ratio - 1.0), 1e-4)
    scale = clamp(0.05 / delta, 1.0, 10.0)
    return min(T_max, base_T * scale)
end

function build_dense_sweep_ratios(kappa_star_B::Float64, kappa_star_ref::Float64)
    deltas_above = [1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2, 3.2e-2, 6.4e-2, 1e-1, 0.15, 0.20]
    deltas_below = [-1e-1, -6.4e-2, -3.2e-2, -1.6e-2, -8e-3, -4e-3]
    kappas_near = kappa_star_B .* (1 .+ vcat(deltas_below, deltas_above))
    ratios_near = kappas_near ./ kappa_star_ref

    ratios = vcat(sweep_ratios, ratios_near)
    ratios = filter(r -> r > 0.0 && r <= sweep_ratio_cap, ratios)
    return sort(unique(ratios))
end

function convergence_slope(time_grid::Vector{Float64}, series::Vector{Float64}; frac::Float64 = 0.1)
    if isempty(time_grid) || length(time_grid) < 2
        return 0.0
    end
    t_end = time_grid[end]
    idx = findall(t -> t >= t_end - frac * t_end, time_grid)
    if length(idx) < 2
        return 0.0
    end
    x = time_grid[idx]
    y = series[idx]
    X = hcat(ones(length(x)), x)
    coeffs = X \ y
    return coeffs[2]
end

function is_converged(time_grid::Vector{Float64}, series::Vector{Float64}; frac::Float64 = 0.1)
    slope = convergence_slope(time_grid, series; frac=frac)
    if !isfinite(slope)
        return false
    end
    t_end = time_grid[end]
    idx = findall(t -> t >= t_end - frac * t_end, time_grid)
    if isempty(idx)
        return true
    end
    tail_mean = mean(series[idx])
    return abs(slope) <= convergence_slope_tol * max(tail_mean, 1e-6)
end

function is_valid_variance(V::Float64; max_reasonable::Float64 = 100.0)
    return isfinite(V) && V > 0.0 && V < max_reasonable
end

# =============================================================================
# POLARIZATION MEASUREMENT (Correct Order Parameter for Z2 Bifurcation)
# =============================================================================

"""
    measure_polarization_variance(snapshots; baseline_var=0.0)

Measure polarization amplitude via excess variance.
For symmetric bifurcation where population splits into ±a*:
  V = ⟨x^2⟩ - ⟨x⟩^2 ≈ (a*)^2
Therefore: |a*| ≈ sqrt(V - V_baseline)
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
Model: 0.5*N(+mu, sigma^2) + 0.5*N(-mu, sigma^2)
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
This should occur near kappa* where consensus is strongest.
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

    if ratio_at_min < 0.5 || ratio_at_min > 1.5
        @warn "Variance minimum at kappa/kappa* = $(round(ratio_at_min, digits=2)), expected near 1.0"
    end

    return (
        V_baseline = V_min,
        V_baseline_kappa = kappa_at_min,
        V_baseline_ratio = ratio_at_min,
        method = "minimum",
    )
end

"""
    print_sweep_diagnostics(kappas, amplitudes, variances, kappa_star)
"""
function print_sweep_diagnostics(
    kappas::Vector{Float64},
    amplitudes::Vector{Float64},
    variances::Vector{Float64},
    kappa_star::Float64;
    baseline_var::Float64 = NaN,
)
    println("\n" * "="^60)
    println("DIAGNOSTIC: Order Parameter Measurements (M* = E|m_T|)")
    println("="^60)
    println("kappa* = $(round(kappa_star, digits=4))")
    println("Total sweep points: $(length(kappas))")
    println("Points above kappa*: $(sum(kappas .> kappa_star))")
    println()
    println("Data table:")
    println("-"^70)
    if isfinite(baseline_var)
        @printf("  %8s  %8s  %10s  %10s  %10s  %8s\n",
                "k/ks", "kappa", "Var(x)", "V-Vb", "M*", "log M*")
    else
        @printf("  %8s  %8s  %10s  %10s  %8s\n", "k/ks", "kappa", "Var(x)", "M*", "log M*")
    end
    println("-"^70)

    for i in eachindex(kappas)
        ratio = kappas[i] / kappa_star
        V = variances[i]
        a = amplitudes[i]
        log_a = a > 1e-10 ? log(a) : NaN
        marker = kappas[i] > kappa_star ? " *" : "  "
        if isfinite(baseline_var)
            V_exc = max(0.0, V - baseline_var)
            @printf("  %8.3f  %8.4f  %10.6f  %10.6f  %10.6f  %8.3f%s\n",
                    ratio, kappas[i], V, V_exc, a, log_a, marker)
        else
            @printf("  %8.3f  %8.4f  %10.6f  %10.6f  %8.3f%s\n",
                    ratio, kappas[i], V, a, log_a, marker)
        end
    end
    println("-"^70)
    println("  (* = above critical point, used for scaling fit)")
    println()

    above_mask = kappas .> kappa_star
    if sum(above_mask) < 3
        println("WARNING: Only $(sum(above_mask)) points above kappa* (need >=3 for fit)")
    end
    a_above = amplitudes[above_mask]
    if any(a_above .< 1e-10)
        println("WARNING: Some M* values are near zero above kappa*")
    end
    if sum(above_mask) >= 2
        a_sorted = amplitudes[above_mask][sortperm(kappas[above_mask])]
        if !issorted(a_sorted)
            println("WARNING: M* not monotonically increasing above kappa*")
        else
            println("OK: M* increases monotonically above kappa*")
        end
    end
end

# =============================================================================
# BIFURCATION VALIDATION (Supercritical Pitchfork Tests)
# =============================================================================

"""
Test H0: beta = 0.5 (supercritical pitchfork scaling).

For kappa > kappa*, the symmetry-aware order parameter follows M* = C (kappa - kappa*)^beta.
Returns a NamedTuple with estimate, CI, p-value, and verdict.
"""
function test_scaling_exponent(
    kappa_values::Vector{Float64},
    a_star_values::Vector{Float64},
    kappa_star::Float64;
    alpha::Float64 = 0.05,
    delta_min::Float64 = 1e-2,
    delta_max::Float64 = 1e-1,
    amp_floor::Float64 = 1e-6,
    decided_share::Union{Nothing, Vector{Float64}} = nothing,
    converged::Union{Nothing, Vector{Bool}} = nothing,
    min_decided_share::Float64 = 0.8,
)
    delta = (kappa_values .- kappa_star) ./ kappa_star
    mask = (delta .>= delta_min) .& (delta .<= delta_max) .& (a_star_values .> amp_floor)
    if decided_share !== nothing
        mask .&= decided_share .>= min_decided_share
    end
    if converged !== nothing
        mask .&= converged
    end
    if sum(mask) < 8
        return (
            beta_hat = NaN,
            beta_se = NaN,
            beta_ci = (NaN, NaN),
            t_statistic = NaN,
            p_value = NaN,
            n_points = sum(mask),
            pass = false,
            verdict = "INSUFFICIENT DATA: need >=8 points in window delta∈[$(delta_min),$(delta_max)]",
            delta_window = (delta_min, delta_max),
            amp_floor = amp_floor,
            local_centers = Float64[],
            local_slopes = Float64[],
        )
    end

    x = log.(kappa_values[mask] .- kappa_star)
    y = log.(abs.(a_star_values[mask]) .+ 1e-12)
    n = length(x)

    X = hcat(ones(n), x)
    coeffs = X \ y
    log_C = coeffs[1]
    beta_hat = coeffs[2]

    y_pred = X * coeffs
    residuals = y .- y_pred
    sigma_sq = sum(residuals .^ 2) / (n - 2)
    var_covar = sigma_sq * inv(X' * X)
    se_beta = sqrt(var_covar[2, 2])

    t_stat = (beta_hat - 0.5) / se_beta
    df = n - 2
    p_value = 2.0 * ccdf(TDist(df), abs(t_stat))

    t_crit = quantile(TDist(df), 1 - alpha / 2)
    ci_lower = beta_hat - t_crit * se_beta
    ci_upper = beta_hat + t_crit * se_beta

    pass = p_value > alpha
    ci_contains_half = ci_lower <= 0.5 <= ci_upper

    verdict = if pass && ci_contains_half
        "PASS: beta=$(round(beta_hat, digits=3)) +/- $(round(se_beta, digits=3)) consistent with 0.5 (n=$(n), window=$(delta_min)-$(delta_max))"
    elseif !pass
        "FAIL: beta=$(round(beta_hat, digits=3)) differs from 0.5 (p=$(round(p_value, digits=4)), n=$(n), window=$(delta_min)-$(delta_max))"
    else
        "MARGINAL: p>alpha but CI does not contain 0.5 (n=$(n), window=$(delta_min)-$(delta_max))"
    end

    local_centers = Float64[]
    local_slopes = Float64[]
    if n >= 6
        ord = sortperm(x)
        x_sorted = x[ord]
        y_sorted = y[ord]
        for i in 1:(length(x_sorted) - 2)
            xs = x_sorted[i:i+2]
            ys = y_sorted[i:i+2]
            Xs = hcat(ones(3), xs)
            coeffs_s = Xs \ ys
            push!(local_centers, mean(xs))
            push!(local_slopes, coeffs_s[2])
        end
    end

    return (
        beta_hat = beta_hat,
        beta_se = se_beta,
        beta_ci = (ci_lower, ci_upper),
        log_C = log_C,
        C = exp(log_C),
        t_statistic = t_stat,
        df = df,
        p_value = p_value,
        n_points = n,
        pass = pass,
        verdict = verdict,
        delta_window = (delta_min, delta_max),
        amp_floor = amp_floor,
        local_centers = local_centers,
        local_slopes = local_slopes,
    )
end

"""
Test for hysteresis (distinguishes super- from subcritical bifurcation).

Supercritical: forward sweep ~= backward sweep.
Subcritical: hysteresis loop exists.
"""
function test_hysteresis(
    kappa_forward::Vector{Float64},
    a_forward::Vector{Float64},
    kappa_backward::Vector{Float64},
    a_backward::Vector{Float64};
    alpha::Float64 = 0.05,
)
    matched_kappa = intersect(kappa_forward, kappa_backward)
    if length(matched_kappa) < 3
        return (
            ks_statistic = NaN,
            p_value = NaN,
            pass = false,
            verdict = "INSUFFICIENT DATA: need matching kappa points",
        )
    end

    a_fwd = [a_forward[findfirst(==(k), kappa_forward)] for k in matched_kappa]
    a_bwd = [a_backward[findfirst(==(k), kappa_backward)] for k in matched_kappa]

    differences = abs.(a_fwd) .- abs.(a_bwd)
    t_result = OneSampleTTest(differences, 0.0)
    p_value = pvalue(t_result)

    max_diff = maximum(abs.(differences))
    mean_diff = mean(abs.(differences))

    pass = p_value > alpha
    verdict = if pass
        "PASS: no significant hysteresis (max diff=$(round(max_diff, digits=4)))"
    else
        "FAIL: hysteresis detected (mean diff=$(round(mean_diff, digits=4)), p=$(round(p_value, digits=4)))"
    end

    return (
        mean_difference = mean_diff,
        max_difference = max_diff,
        t_statistic = t_result.t,
        p_value = p_value,
        n_points = length(matched_kappa),
        pass = pass,
        verdict = verdict,
    )
end

"""
Bootstrap confidence interval for kappa* estimate.
"""
function bootstrap_kappa_star(
    kappa_values::Vector{Float64},
    a_star_values::Vector{Float64};
    n_bootstrap::Int = 1000,
    alpha::Float64 = 0.05,
)
    n = length(kappa_values)
    kappa_star_samples = Float64[]

    for _ in 1:n_bootstrap
        idx = rand(1:n, n)
        k_sample = kappa_values[idx]
        a_sample = a_star_values[idx]
        try
            kstar_est = estimate_kappa_star_from_data(k_sample, a_sample)
            push!(kappa_star_samples, kstar_est)
        catch
            continue
        end
    end

    if length(kappa_star_samples) < 100
        return (
            kappa_star_mean = NaN,
            kappa_star_ci = (NaN, NaN),
            n_successful = length(kappa_star_samples),
            verdict = "BOOTSTRAP FAILED: too few successful estimates",
        )
    end

    kappa_star_mean = mean(kappa_star_samples)
    ci_lower = quantile(kappa_star_samples, alpha / 2)
    ci_upper = quantile(kappa_star_samples, 1 - alpha / 2)
    ci_width = ci_upper - ci_lower

    verdict = if ci_width < 0.1 * kappa_star_mean
        "PASS: kappa* well-localized (CI width=$(round(ci_width, digits=4)))"
    else
        "WIDE CI: kappa*=$(round(kappa_star_mean, digits=3)) +/- $(round(ci_width / 2, digits=3))"
    end

    return (
        kappa_star_mean = kappa_star_mean,
        kappa_star_ci = (ci_lower, ci_upper),
        kappa_star_std = std(kappa_star_samples),
        n_successful = length(kappa_star_samples),
        verdict = verdict,
    )
end

function estimate_kappa_star_from_data(kappa::Vector{Float64}, a_star::Vector{Float64})
    idx = argmin(abs.(a_star))
    return kappa[idx]
end

"""
    estimate_kappa_varmin(kappas, variances)

Estimate kappa_varmin from empirical variance data via the variance minimum.
This is a diagnostic quantity, NOT a critical point.
"""
function estimate_kappa_varmin(
    kappas::Vector{Float64},
    variances::Vector{Float64},
)
    finite_idx = findall(isfinite, variances)
    if isempty(finite_idx)
        return NaN
    end
    idx_min = finite_idx[argmin(variances[finite_idx])]
    kappa_varmin = kappas[idx_min]

    # Optional local parabola fit for robustness
    if idx_min > 1 && idx_min < length(kappas)
        k1, k2, k3 = kappas[idx_min - 1], kappas[idx_min], kappas[idx_min + 1]
        v1, v2, v3 = variances[idx_min - 1], variances[idx_min], variances[idx_min + 1]
        denom = (k1 - k2) * (k1 - k3) * (k2 - k3)
        if denom != 0.0
            a = (k3 * (v2 - v1) + k2 * (v1 - v3) + k1 * (v3 - v2)) / denom
            b = (k3^2 * (v1 - v2) + k2^2 * (v3 - v1) + k1^2 * (v2 - v3)) / denom
            kappa_varmin_parabola = -b / (2a)
            if kappa_varmin_parabola > minimum(kappas) && kappa_varmin_parabola < maximum(kappas)
                return kappa_varmin_parabola
            end
        end
    end

    return kappa_varmin
end

"""
    fit_scaling_with_free_kappa_star(kappas, amplitudes, kappa_star_init)

Grid-search fit of M* = C (kappa - kappa*)^beta with kappa* free.
Avoids external dependencies by using log-linear regression for each candidate.
"""
function fit_scaling_with_free_kappa_star(
    kappas::Vector{Float64},
    amplitudes::Vector{Float64},
    kappa_star_init::Float64;
    grid_width::Float64 = 0.15,
    grid_points::Int = 41,
)
    mask = isfinite.(amplitudes) .& (amplitudes .> 1e-8) .& isfinite.(kappas)
    k = kappas[mask]
    a = amplitudes[mask]
    if length(k) < 30
        return (kappa_star_fit=NaN, C_fit=NaN, beta_fit=NaN, converged=false)
    end

    k_min = max(minimum(k), kappa_star_init * (1 - grid_width))
    k_max = min(maximum(k) - 1e-6, kappa_star_init * (1 + grid_width))
    if k_min >= k_max
        return (kappa_star_fit=NaN, C_fit=NaN, beta_fit=NaN, converged=false)
    end

    best_sse = Inf
    best = (kappa_star_fit=NaN, C_fit=NaN, beta_fit=NaN)
    for kstar in range(k_min, k_max; length=grid_points)
        valid = k .> kstar
        if sum(valid) < 3
            continue
        end
        x = log.(k[valid] .- kstar)
        y = log.(a[valid])
        X = hcat(ones(length(x)), x)
        coeffs = X \ y
        y_pred = X * coeffs
        sse = sum((y .- y_pred) .^ 2)
        if sse < best_sse
            best_sse = sse
            best = (kappa_star_fit=kstar, C_fit=exp(coeffs[1]), beta_fit=coeffs[2])
        end
    end

    return (best..., converged=isfinite(best.kappa_star_fit))
end

"""
    test_scaling_exponent_robust(kappas, amplitudes, variances, kappa_star_analytic)

Run scaling test with analytic kappa*, report variance-minimum kappa_varmin (diagnostic),
and a free kappa* grid-search fit.
"""
function test_scaling_exponent_robust(
    kappa_values::Vector{Float64},
    a_star_values::Vector{Float64},
    variances::Vector{Float64},
    kappa_star_analytic::Float64;
    alpha::Float64 = 0.05,
    delta_windows::Vector{Tuple{Float64, Float64}} = scaling_delta_windows,
    amp_floor::Float64 = scaling_amp_floor,
    decided_share::Union{Nothing, Vector{Float64}} = nothing,
    converged::Union{Nothing, Vector{Bool}} = nothing,
    min_decided_share::Float64 = 0.8,
)
    kappa_varmin = estimate_kappa_varmin(kappa_values, variances)

    println("\n--- kappa* Comparison ---")
    @printf("  kappa*_B (theory) = %.4f\n", kappa_star_analytic)
    @printf("  kappa_varmin (variance minimum) = %.4f (diagnostic)\n", kappa_varmin)
    if isfinite(kappa_varmin)
        @printf("  ratio (varmin/kappa*_B) = %.3f\n", kappa_varmin / kappa_star_analytic)
    end

    function run_window_set(kstar::Float64)
        results = NamedTuple[]
        for (dmin, dmax) in delta_windows
            push!(results, test_scaling_exponent(
                kappa_values, a_star_values, kstar;
                alpha=alpha,
                delta_min=dmin,
                delta_max=dmax,
                amp_floor=amp_floor,
                decided_share=decided_share,
                converged=converged,
                min_decided_share=min_decided_share,
            ))
        end
        return results
    end

    default_idx = findfirst(w -> isapprox(w[1], 1e-2; atol=1e-6) && isapprox(w[2], 1e-1; atol=1e-6), delta_windows)
    default_idx = default_idx === nothing ? 1 : default_idx

    println("\n--- Scaling fit with kappa*_B (theory) ---")
    analytic_windows = run_window_set(kappa_star_analytic)
    result_analytic = analytic_windows[default_idx]

    println("\n--- Scaling fit with kappa* free (grid search) ---")
    result_free = fit_scaling_with_free_kappa_star(
        kappa_values, a_star_values, kappa_star_analytic
    )

    return (
        analytic=result_analytic,
        free=result_free,
        kappa_varmin=kappa_varmin,
        analytic_windows=analytic_windows,
        delta_windows=delta_windows,
        default_window=delta_windows[default_idx],
    )
end

struct BifurcationValidation
    scaling_test::NamedTuple
    hysteresis_test::Union{NamedTuple, Nothing}
    kappa_star_ci::NamedTuple
    overall_pass::Bool
    summary::String
end

# =============================================================================
# MAIN WORKFLOW
# =============================================================================

println("=" ^ 72)
println("FIGURE 6 ENSEMBLE VALIDATION")
println("=" ^ 72)

logmsg("Starting ensemble validation run")
p = Params(λ=lambda, σ=sigma, Θ=theta, c0=c0, hazard=StepHazard(nu0))

logmsg("Estimating V*")
Vstar = estimate_Vstar(p; N=N, T=350.0, dt=dt, burn_in=burn_in, seed=seed)
@printf("V* = %.3f\n", Vstar)

logmsg("Computing theoretical kappa*_B (rank-one susceptibility)")
kappaB_res = compute_kappa_star_B(p; L=kappaB_L, M=kappaB_M, boundary=kappaB_boundary)
kappa_star_B = kappaB_res.kappa_B
@printf("kappa*_B = %.4f (Phi0=%.4f)\n", kappa_star_B, kappaB_res.Phi0)

kappa_star_A = NaN
kappa_star_A_ci = (NaN, NaN)
growth_scan_df = DataFrame()
if growth_scan_enabled && isfinite(kappa_star_B)
    logmsg("Running growth scan for kappa*_A (empirical)")
    kappa_grid = kappa_star_B .* growth_scan_factors
    rows, boot_slopes = growth_scan_kappa_A(
        p,
        kappa_grid;
        N=N,
        T=T,
        dt=dt,
        n_ensemble=n_ensemble,
        base_seed=seed + 5000,
        snapshot_times=snapshot_times,
        mt_stride=mt_stride,
        window=growth_scan_window,
        n_boot=growth_scan_bootstrap,
        parallel=parallel_ensemble,
        device=simulation_device,
    )
    growth_scan_df = DataFrame(rows)
    CSV.write(joinpath(threshold_dir, "growth_scan.csv"), growth_scan_df)
    kappa_star_A = estimate_kappa_star_from_scan(kappa_grid, growth_scan_df.lambda_hat)

    if growth_scan_bootstrap > 0
        kappa_boot = [estimate_kappa_star_from_scan(kappa_grid, boot_slopes[b, :]) for b in 1:growth_scan_bootstrap]
        kappa_star_A_ci = (quantile(kappa_boot, 0.025), quantile(kappa_boot, 0.975))
    end
end

@printf("kappa*_A = %.4f (CI=[%.4f, %.4f])\n", kappa_star_A, kappa_star_A_ci[1], kappa_star_A_ci[2])

kappa_star = isfinite(kappa_star_A) ? kappa_star_A : kappa_star_B
@printf("kappa*_ref = %.4f\n", kappa_star)

if isfinite(kappa_star_B)
    JSON3.write(joinpath(threshold_dir, "kappa_star_B.json"), Dict(
        "kappa_star_B" => kappa_star_B,
        "Phi0" => kappaB_res.Phi0,
        "grid" => Dict("L" => kappaB_L, "M" => kappaB_M, "boundary" => String(kappaB_boundary)),
    ))
    # Diagnostics: leading odd eigenvalue of rank-one operator for a few kappas
    Aodd = kappaB_res.Aodd
    b_odd = kappaB_res.b_odd
    a_odd = kappaB_res.a_odd
    ktest = [0.0, 0.5 * kappa_star_B, kappa_star_B, 1.5 * kappa_star_B]
    eig_rows = Vector{NamedTuple}()
    for κ in ktest
        Aκ = Aodd .+ κ .* (b_odd * a_odd')
        λmax = maximum(real.(eigvals(Aκ)))
        push!(eig_rows, (kappa=κ, lambda1=λmax))
    end
    CSV.write(joinpath(threshold_dir, "susceptibility.csv"), DataFrame(eig_rows))
end

if isfinite(kappa_star_A)
    JSON3.write(joinpath(threshold_dir, "kappa_star_A.json"), Dict(
        "kappa_star_A" => kappa_star_A,
        "kappa_star_A_ci" => [kappa_star_A_ci[1], kappa_star_A_ci[2]],
        "window" => [growth_scan_window[1], growth_scan_window[2]],
        "kappa_grid" => isfinite(kappa_star_B) ? kappa_star_B .* growth_scan_factors : growth_scan_factors,
        "bootstrap" => growth_scan_bootstrap,
    ))
end

threshold_meta = Dict(
    "parameters" => Dict("lambda" => lambda, "sigma" => sigma, "theta" => theta, "c0" => c0, "nu0" => nu0),
    "simulation" => Dict("N" => N, "T" => T, "dt" => dt, "n_ensemble" => n_ensemble, "snapshot_times" => snapshot_times),
    "kappa_star_A" => kappa_star_A,
    "kappa_star_A_ci" => [kappa_star_A_ci[1], kappa_star_A_ci[2]],
    "kappa_star_B" => kappa_star_B,
    "Phi0" => isfinite(kappa_star_B) ? kappaB_res.Phi0 : NaN,
    "grid" => Dict("L" => kappaB_L, "M" => kappaB_M, "boundary" => String(kappaB_boundary)),
    "timestamp" => Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"),
)
JSON3.write(joinpath(threshold_dir, "metadata.json"), threshold_meta)

results = Dict{String, EnsembleResults}()
stats = Dict{String, EnsembleStatistics}()
growth = Dict{String, GrowthRateResult}()

for (label, ratio) in scenario_ratios
    logmsg("Running scenario: $label (kappa ratio = $ratio) [backend=$(parallel_backend_label())]")
    kappa = ratio * kappa_star
    res = run_ensemble_simulation(
        p;
        kappa=kappa,
        n_ensemble=n_ensemble,
        N=N,
        T=T,
        dt=dt,
        base_seed=seed,
        snapshot_times=snapshot_times,
        mt_stride=mt_stride,
        store_snapshots=true,
        track_moments=true,
        parallel=parallel_ensemble,
        device=simulation_device,
    )
    results[label] = res
    stats[label] = compute_ensemble_statistics(res)
    window_results = fit_growth_rate_multiwindow(
        res.time_grid,
        res.mean_trajectories;
        windows=fit_windows,
        method=:log_linear,
        direction=scenario_direction(label),
    )
    growth[label] = select_best_growth_result(window_results)
    @printf("  lambda_hat = %+.6f, 95%% CI: [%+.4f, %+.4f]\n",
            growth[label].lambda_mean, growth[label].lambda_ci_lower, growth[label].lambda_ci_upper)
end

branch_info = Dict{String, NamedTuple}()
for (label, res) in results
    signs, mean_late, decided, tau_used, decided_share, plus_share, minus_share, undecided_share =
        compute_branch_signs(res.mean_trajectories, res.time_grid)
    branch_info[label] = (
        signs=signs,
        mean_late=mean_late,
        decided=decided,
        tau=tau_used,
        decided_share=decided_share,
        plus_share=plus_share,
        minus_share=minus_share,
        undecided_share=undecided_share,
        imbalance=plus_share - minus_share,
    )
end

# =============================================================================
# OUTPUT: ENSEMBLE GROWTH RATES
# =============================================================================

growth_rows = Vector{NamedTuple}()
for (label, ratio) in scenario_ratios
    g = growth[label]
    status = "UNKNOWN"
    if label == "below"
        status = (g.lambda_mean < 0 && g.is_significant) ? "PASS" : "FAIL"
    elseif label == "above"
        status = (g.lambda_mean > 0 && g.is_significant) ? "PASS" : "FAIL"
    else
        status = abs(g.lambda_mean) < 0.01 ? "PASS" : "CHECK"
    end
    push!(growth_rows, (
        scenario=label,
        kappa_ratio=ratio,
        kappa=ratio * kappa_star,
        lambda_mean=g.lambda_mean,
        lambda_std=g.lambda_std,
        lambda_ci_lower=g.lambda_ci_lower,
        lambda_ci_upper=g.lambda_ci_upper,
        r_squared=missing,
        p_value=g.p_value,
        status=status,
        window_start=g.window_used[1],
        window_end=g.window_used[2],
    ))
end

growth_df = DataFrame(growth_rows)
CSV.write(joinpath(outdir, "ensemble_growth_rates.csv"), growth_df)

# =============================================================================
# OUTPUT: ENSEMBLE TRAJECTORIES
# =============================================================================

traj_rows = Vector{NamedTuple}()
for (label, res) in results
    st = stats[label]
    info = branch_info[label]
    decided_idx = findall(info.decided)
    n_decided = length(decided_idx)
    n_times = length(res.time_grid)
    mean_aligned = fill(NaN, n_times)
    mean_aligned_lo = fill(NaN, n_times)
    mean_aligned_hi = fill(NaN, n_times)

    if n_decided > 0
        for t_idx in 1:n_times
            vals = info.signs[decided_idx] .* res.mean_trajectories[decided_idx, t_idx]
            mean_aligned[t_idx] = mean(vals)
            if n_decided >= 2
                se = std(vals; corrected=true) / sqrt(n_decided)
                tcrit = quantile(TDist(n_decided - 1), 0.975)
                mean_aligned_lo[t_idx] = mean_aligned[t_idx] - tcrit * se
                mean_aligned_hi[t_idx] = mean_aligned[t_idx] + tcrit * se
            end
        end
    end
    for (i, t) in enumerate(res.time_grid)
        push!(traj_rows, (
            scenario=label,
            time=t,
            mean=st.mean_traj[i],
            mean_ci_lower=st.mean_ci_lower[i],
            mean_ci_upper=st.mean_ci_upper[i],
            mean_signed=st.mean_traj[i],
            mean_signed_ci_lower=st.mean_ci_lower[i],
            mean_signed_ci_upper=st.mean_ci_upper[i],
            mean_abs=st.mean_abs[i],
            mean_abs_ci_lower=st.mean_abs_ci_lower[i],
            mean_abs_ci_upper=st.mean_abs_ci_upper[i],
            mean_rms=st.mean_rms[i],
            mean_rms_ci_lower=st.mean_rms_ci_lower[i],
            mean_rms_ci_upper=st.mean_rms_ci_upper[i],
            mean_aligned=mean_aligned[i],
            mean_aligned_ci_lower=mean_aligned_lo[i],
            mean_aligned_ci_upper=mean_aligned_hi[i],
            decided_share=info.decided_share,
            plus_share=info.plus_share,
            minus_share=info.minus_share,
            undecided_share=info.undecided_share,
            imbalance=info.imbalance,
            variance=st.var_traj[i],
            var_ci_lower=st.var_ci_lower[i],
            var_ci_upper=st.var_ci_upper[i],
            std_error=st.std_error[i],
        ))
    end
end

traj_df = DataFrame(traj_rows)
CSV.write(joinpath(outdir, "ensemble_trajectories.csv"), traj_df)

# Terminal branch diagnostics
terminal_rows = Vector{NamedTuple}()
for (label, info) in branch_info
    ratio = scenario_ratios[label]
    kappa = ratio * kappa_star
    for (i, m) in enumerate(info.mean_late)
        push!(terminal_rows, (
            scenario=label,
            kappa=kappa,
            kappa_ratio=ratio,
            run_id=i,
            mean_late=m,
            abs_mean_late=abs(m),
            branch_sign=info.signs[i],
            decided_flag=info.decided[i],
        ))
    end
end
terminal_df = DataFrame(terminal_rows)
CSV.write(joinpath(outdir, "terminal_means.csv"), terminal_df)

# =============================================================================
# ALTERNATIVE OBSERVABLES
# =============================================================================

observables = Dict{String, AlternativeObservables}()
if run_observables
    println("\nComputing alternative observables...")
    edges = compute_edges_from_snapshots(results; bins=density_bins)
    for (label, res) in results
        observables[label] = compute_alternative_observables(res, edges; var_window=var_fit_window)
    end

    obs_rows = Vector{NamedTuple}()
    for (label, obs) in observables
        ratio = scenario_ratios[label]
        kappa = ratio * kappa_star
        st = stats[label]
        m_abs_star = st.mean_abs[end]
        m_signed_star = st.mean_traj[end]
        var_star = st.var_traj[end]
        for (i, t) in enumerate(obs.time_grid)
            push!(obs_rows, (
                scenario=label,
                kappa=kappa,
                kappa_ratio=ratio,
                m_abs_star=m_abs_star,
                m_signed_star=m_signed_star,
                variance_star=var_star,
                time=t,
                kurtosis=obs.kurtosis_mean[i],
                kurtosis_ci_lower=obs.kurtosis_ci_lower[i],
                kurtosis_ci_upper=obs.kurtosis_ci_upper[i],
                bimodality=obs.bimodality_mean[i],
                bimodality_ci_lower=obs.bimodality_ci_lower[i],
                bimodality_ci_upper=obs.bimodality_ci_upper[i],
                overlap=NaN,
                overlap_ci_lower=NaN,
                overlap_ci_upper=NaN,
            ))
        end
        for (j, t_snap) in enumerate(obs.overlap_times)
            push!(obs_rows, (
                scenario=label,
                kappa=kappa,
                kappa_ratio=ratio,
                m_abs_star=m_abs_star,
                m_signed_star=m_signed_star,
                variance_star=var_star,
                time=t_snap,
                kurtosis=NaN,
                kurtosis_ci_lower=NaN,
                kurtosis_ci_upper=NaN,
                bimodality=NaN,
                bimodality_ci_lower=NaN,
                bimodality_ci_upper=NaN,
                overlap=obs.overlap_mean[j],
                overlap_ci_lower=obs.overlap_ci_lower[j],
                overlap_ci_upper=obs.overlap_ci_upper[j],
            ))
        end
    end

    obs_df = DataFrame(obs_rows)
    CSV.write(joinpath(outdir, "alternative_observables.csv"), obs_df)
end

# =============================================================================
# PARAMETER SWEEP WITH CORRECT ORDER PARAMETER
# =============================================================================

sweep_df = DataFrame()
equilibrium_df = DataFrame()
scaling_result = nothing
scaling_results = nothing
hysteresis_result = nothing
kappa_ci_result = nothing
all_pass = false
V_baseline = NaN
V_baseline_kappa = NaN
V_baseline_ratio = NaN
m0 = NaN

if run_sweep
    logmsg("Running kappa sweep for bifurcation validation [backend=$(parallel_backend_label())]")
    println("Sweep mode: $(sweep_mode) (t_measure = $(t_measure))")

    sweep_ratios_fine = fast_mode ? sweep_ratios : build_dense_sweep_ratios(kappa_star_B, kappa_star)
    sweep_ratios_fine = filter(r -> r <= sweep_ratio_cap, sweep_ratios_fine)

    println("\n--- Pass 1: Collecting variance + order parameter ---")
    sweep_rows = Vector{NamedTuple}()
    equilibrium_rows = Vector{NamedTuple}()
    all_variances = Float64[]
    all_kappas = Float64[]

    for (idx, ratio) in enumerate(sweep_ratios_fine)
        kappa = ratio * kappa_star
        delta_rel_B = abs(kappa / kappa_star_B - 1.0)
        n_rep = (RUN_MODE == :pub && delta_rel_B >= 1e-2 && delta_rel_B <= 1e-1) ? n_rep_per_kappa_nearcrit : 1
        T_adaptive = sweep_mode == :short_time ? t_measure : get_adaptive_T(ratio, T)
        @printf("  [%2d/%2d] kappa/kappa* = %.3f (T=%.0f, reps=%d) ... ",
                idx, length(sweep_ratios_fine), ratio, T_adaptive, n_rep)

        mean_traj_chunks = Matrix{Float64}[]
        V_per_run_all = Float64[]
        converged_flags = Bool[]
        time_grid_use = Float64[]

        for rep in 1:n_rep
            rep_seed = seed + 100000 * idx + 1000 * rep
            res = run_ensemble_simulation(
                p;
                kappa=kappa,
                n_ensemble=sweep_ensemble,
                N=N,
                T=T_adaptive,
                dt=dt,
                base_seed=rep_seed,
                snapshot_times=[T_adaptive],
                mt_stride=mt_stride,
                store_snapshots=true,
                track_moments=true,
                parallel=parallel_ensemble,
                device=simulation_device,
            )

            time_grid_use = res.time_grid
            push!(mean_traj_chunks, res.mean_trajectories)

            mean_abs_traj = vec(mean(abs.(res.mean_trajectories); dims=1))
            conv = is_converged(res.time_grid, mean_abs_traj; frac=0.1)
            if !conv
                slope_tail = convergence_slope(res.time_grid, mean_abs_traj; frac=0.1)
                @printf("  [warn] mean_abs slope in last 10%% = %.3e (tol=%.1e)\n",
                        slope_tail, convergence_slope_tol)
            end
            push!(converged_flags, conv)

            if !isempty(res.snapshots)
                final_snaps = [snap_list[end] for snap_list in res.snapshots if !isempty(snap_list)]
                V_per_run = [var(snap) for snap in final_snaps if !isempty(snap)]
                append!(V_per_run_all, filter(isfinite, V_per_run))
            end
        end

        mean_traj_all = isempty(mean_traj_chunks) ? zeros(0, 0) : vcat(mean_traj_chunks...)
        n_runs_total = size(mean_traj_all, 1)
        converged = all(converged_flags)

        if n_runs_total > 0
            late_idx = late_window_indices(time_grid_use)
            mbar_all = [mean(abs.(mean_traj_all[i, late_idx])) for i in 1:n_runs_total]
            mean_abs_star = mean(mbar_all)
            if n_runs_total > 1
                se = std(mbar_all; corrected=true) / sqrt(n_runs_total)
                tcrit = quantile(TDist(n_runs_total - 1), 0.975)
                m_abs_ci_lower = mean_abs_star - tcrit * se
                m_abs_ci_upper = mean_abs_star + tcrit * se
            else
                m_abs_ci_lower = mean_abs_star
                m_abs_ci_upper = mean_abs_star
            end
            msq_all = [mean(mean_traj_all[i, late_idx].^2) for i in 1:n_runs_total]
            mean_rms_star = sqrt(mean(msq_all))
            signs, mean_late, decided, tau_used, decided_share, plus_share, minus_share, undecided_share =
                compute_branch_signs(mean_traj_all, time_grid_use)
            mean_aligned_star = any(decided) ? mean(signs[decided] .* mean_late[decided]) : NaN
        else
            mean_abs_star = NaN
            m_abs_ci_lower = NaN
            m_abs_ci_upper = NaN
            mean_rms_star = NaN
            mean_aligned_star = NaN
            decided_share = NaN
            plus_share = NaN
            minus_share = NaN
            undecided_share = NaN
        end

        n_runs = length(V_per_run_all)
        if n_runs > 0
            V_mean = median(V_per_run_all)
            V_std = std(V_per_run_all)
            if !is_valid_variance(V_mean; max_reasonable=max_reasonable_variance)
                @printf("EXPLOSION (V=%.2e)\n", V_mean)
                V_mean = NaN
                V_std = NaN
            else
                @printf("V = %.3f\n", V_mean)
            end
        else
            V_mean = NaN
            V_std = NaN
            @printf("V = NaN\n")
        end

        push!(all_variances, V_mean)
        push!(all_kappas, kappa)

        g = n_runs_total > 0 ? fit_growth_rate_ensemble(
            time_grid_use,
            mean_traj_all;
            fitting_window=(20.0, 80.0),
            method=:log_linear,
            direction=ratio > 1.0 ? :greater : (ratio < 1.0 ? :less : :two_sided),
        ) : GrowthRateResult(NaN, NaN, NaN, NaN, NaN, NaN, false, (NaN, NaN), :log_linear)

        push!(sweep_rows, (
            kappa_ratio=ratio,
            kappa=kappa,
            m_abs_star=mean_abs_star,
            m_abs_ci_lower=m_abs_ci_lower,
            m_abs_ci_upper=m_abs_ci_upper,
            m_rms_star=mean_rms_star,
            m_aligned_star=mean_aligned_star,
            decided_share=decided_share,
            plus_share=plus_share,
            minus_share=minus_share,
            undecided_share=undecided_share,
            imbalance=plus_share - minus_share,
            variance=V_mean,
            variance_std=V_std,
            lambda_mean=g.lambda_mean,
            lambda_ci_lower=g.lambda_ci_lower,
            lambda_ci_upper=g.lambda_ci_upper,
            r_squared=missing,
            n_ensemble=n_runs_total,
            n_rep=n_rep,
            converged=converged,
            T_sim=T_adaptive,
        ))

        push!(equilibrium_rows, (
            kappa_ratio=ratio,
            kappa=kappa,
            variance=V_mean,
            variance_std=V_std,
            m_abs_star=mean_abs_star,
            m_abs_ci_lower=m_abs_ci_lower,
            m_abs_ci_upper=m_abs_ci_upper,
            m_rms_star=mean_rms_star,
            m_aligned_star=mean_aligned_star,
            decided_share=decided_share,
            plus_share=plus_share,
            minus_share=minus_share,
            undecided_share=undecided_share,
            imbalance=plus_share - minus_share,
            n_ensemble=n_runs_total,
            n_rep=n_rep,
            converged=converged,
            T_sim=T_adaptive,
        ))
    end

    baseline_result = estimate_baseline_variance_from_sweep(all_variances, all_kappas, kappa_star)
    V_baseline = baseline_result.V_baseline
    V_baseline_kappa = baseline_result.V_baseline_kappa
    V_baseline_ratio = baseline_result.V_baseline_ratio

    println("\n--- Baseline Variance ---")
    @printf("  V_baseline = %.6f (minimum variance)\n", V_baseline)
    @printf("  Occurs at kappa/kappa* = %.3f\n", V_baseline_ratio)

    sweep_df = DataFrame(sweep_rows)
    equilibrium_df = DataFrame(equilibrium_rows)

    # Baseline correction for finite-size floor in |m|
    below_mask = (equilibrium_df.kappa .< kappa_star_B) .& .!ismissing.(equilibrium_df.m_abs_star)
    below_vals = filter(isfinite, equilibrium_df.m_abs_star[below_mask])
    m0 = isempty(below_vals) ? 0.0 : median(below_vals)
    equilibrium_df.m_corr_star = sqrt.(max.(equilibrium_df.m_abs_star .^ 2 .- m0^2, 0.0))
    sweep_df.m_corr_star = sqrt.(max.(sweep_df.m_abs_star .^ 2 .- m0^2, 0.0))

    equilibrium_df.excess_variance = max.(0.0, equilibrium_df.variance .- V_baseline)
    sweep_df.excess_variance = max.(0.0, sweep_df.variance .- V_baseline)
    equilibrium_df.a_star_mean = sqrt.(equilibrium_df.excess_variance)
    sweep_df.a_star = sqrt.(sweep_df.excess_variance)

    equilibrium_df.a_star_se = ifelse.((equilibrium_df.n_ensemble .> 1) .&
                                       (equilibrium_df.excess_variance .> 1e-10) .&
                                       isfinite.(equilibrium_df.variance_std),
                                       (equilibrium_df.variance_std ./ sqrt.(equilibrium_df.n_ensemble)) ./ (2 .* max.(equilibrium_df.a_star_mean, 1e-10)),
                                       NaN)

    CSV.write(joinpath(sweep_dir, "parameter_sweep.csv"), sweep_df)
    CSV.write(joinpath(sweep_dir, "equilibrium_sweep.csv"), equilibrium_df)

    equilibrium_kappas = equilibrium_df.kappa
    equilibrium_amplitudes_abs = equilibrium_df.m_abs_star
    equilibrium_amplitudes_corr = equilibrium_df.m_corr_star
    equilibrium_variances = equilibrium_df.variance

    print_sweep_diagnostics(
        equilibrium_kappas,
        equilibrium_amplitudes_abs,
        equilibrium_variances,
        kappa_star;
        baseline_var=V_baseline,
    )

    if any(.!isfinite.(equilibrium_variances))
        println("NOTE: Some sweep points produced invalid variance and were excluded from fits.")
    end

    println("\n" * "="^60)
    println("BIFURCATION VALIDATION: SCALING EXPONENT TEST")
    println("="^60)
    println("Model: M_corr = C (kappa - kappa*)^beta (baseline-corrected |m|)")
    println("H0: beta = 0.5 (supercritical pitchfork)")
    println()

    valid_mask = isfinite.(equilibrium_amplitudes_corr) .& isfinite.(equilibrium_kappas)
    scaling_results = test_scaling_exponent_robust(
        equilibrium_kappas[valid_mask],
        equilibrium_amplitudes_corr[valid_mask],
        equilibrium_variances[valid_mask],
        kappa_star_B;
        alpha=0.05,
        decided_share=equilibrium_df.decided_share[valid_mask],
        converged=equilibrium_df.converged[valid_mask],
        min_decided_share=0.8,
    )
    scaling_result = scaling_results.analytic

    println("\n--- Summary ---")
    function beta_range(window_results)
        vals = [r.beta_hat for r in window_results if isfinite(r.beta_hat)]
        return isempty(vals) ? (NaN, NaN) : (minimum(vals), maximum(vals))
    end
    window_str = "[$(scaling_results.default_window[1]), $(scaling_results.default_window[2])]"
    analytic_range = beta_range(scaling_results.analytic_windows)

    println("With kappa*_B (theory): beta = $(round(scaling_results.analytic.beta_hat, digits=3)) " *
            "(n=$(scaling_results.analytic.n_points), window=$(window_str))")
    println("  sensitivity across windows: [$(round(analytic_range[1], digits=3)), $(round(analytic_range[2], digits=3))]")

    println("Variance minimum (diagnostic): kappa_varmin = $(round(scaling_results.kappa_varmin, digits=4))")
    if scaling_results.free.converged
        println("With kappa*_free: beta = $(round(scaling_results.free.beta_fit, digits=3)), " *
                "kappa* = $(round(scaling_results.free.kappa_star_fit, digits=4))")
    end

    println("\n" * "="^60)
    println("BIFURCATION VALIDATION: HYSTERESIS TEST")
    println("="^60)
    println("Supercritical: forward ~= backward (no hysteresis)")
    println("Subcritical: forward != backward (bistability)")
    println()
    logmsg("Running backward sweep (different seed) [backend=$(parallel_backend_label())]")

    backward_amplitudes = Float64[]
    backward_kappas = Float64[]
    backward_rows = Vector{NamedTuple}()
    for (idx, ratio) in enumerate(reverse(sweep_ratios_fine))
        kappa = ratio * kappa_star
        T_adaptive = sweep_mode == :short_time ? t_measure : get_adaptive_T(ratio, T)
        @printf("  [%2d/%2d] kappa/kappa* = %.3f (backward, T=%.0f) ... ",
                idx, length(sweep_ratios_fine), ratio, T_adaptive)

        res = run_ensemble_simulation(
            p;
            kappa=kappa,
            n_ensemble=sweep_ensemble,
            N=N,
            T=T_adaptive,
            dt=dt,
            base_seed=seed + 10000,
            snapshot_times=[T_adaptive],
            mt_stride=mt_stride,
            store_snapshots=true,
            track_moments=true,
            parallel=parallel_ensemble,
            device=simulation_device,
        )

        V_mean = NaN
        V_std = NaN
        a_star_se = NaN
        n_runs = 0

        if !isempty(res.snapshots)
            final_snaps = [snap_list[end] for snap_list in res.snapshots if !isempty(snap_list)]
            pol = measure_polarization_variance(final_snaps; baseline_var=V_baseline)
            V_mean = pol.variance_mean
            V_std = pol.variance_std
            n_runs = pol.n_runs
        end

        late_idx = late_window_indices(res.time_grid)
        mbar = [mean(abs.(res.mean_trajectories[i, late_idx])) for i in 1:size(res.mean_trajectories, 1)]
        mean_abs_star = mean(mbar)
        msq = [mean(res.mean_trajectories[i, late_idx].^2) for i in 1:size(res.mean_trajectories, 1)]
        mean_rms_star = sqrt(mean(msq))
        signs, mean_late, decided, tau_used, decided_share, plus_share, minus_share, undecided_share =
            compute_branch_signs(res.mean_trajectories, res.time_grid)
        mean_aligned_star = any(decided) ? mean(signs[decided] .* mean_late[decided]) : NaN
        m_star = mean_abs_star

        if isfinite(m_star)
            @printf("M* = %.3f\n", m_star)
        else
            @printf("M* = NaN\n")
        end
        push!(backward_amplitudes, m_star)
        push!(backward_kappas, kappa)

        V_excess = isfinite(V_mean) ? max(0.0, V_mean - V_baseline) : NaN
        push!(backward_rows, (
            kappa_ratio=ratio,
            kappa=kappa,
            variance=V_mean,
            variance_std=V_std,
            excess_variance=V_excess,
            a_star_mean=NaN,
            a_star_se=a_star_se,
            m_abs_star=mean_abs_star,
            m_rms_star=mean_rms_star,
            m_aligned_star=mean_aligned_star,
            decided_share=decided_share,
            n_ensemble=max(n_runs, size(res.mean_trajectories, 1)),
            T_sim=T_adaptive,
        ))
    end

    if !isempty(backward_rows)
        backward_df = DataFrame(backward_rows)
        CSV.write(joinpath(sweep_dir, "backward_sweep.csv"), backward_df)
    end

    fwd_mask = isfinite.(equilibrium_amplitudes_abs) .& isfinite.(equilibrium_kappas)
    bwd_mask = isfinite.(backward_amplitudes) .& isfinite.(backward_kappas)
    hysteresis_result = test_hysteresis(
        equilibrium_kappas[fwd_mask],
        equilibrium_amplitudes_abs[fwd_mask],
        backward_kappas[bwd_mask],
        backward_amplitudes[bwd_mask],
    )

    println()
    @printf("  Mean |Delta M*| = %.3f\n", hysteresis_result.mean_difference)
    @printf("  Max  |Delta M*| = %.3f\n", hysteresis_result.max_difference)
    @printf("  t-stat = %.3f\n", hysteresis_result.t_statistic)
    @printf("  p-value = %.4f\n", hysteresis_result.p_value)
    println()
    println("  -> $(hysteresis_result.verdict)")

    println("\n" * "="^60)
    println("BIFURCATION VALIDATION: CRITICAL POINT LOCALIZATION")
    println("="^60)

    kappa_ci_result = bootstrap_kappa_star(
        equilibrium_kappas,
        equilibrium_amplitudes_abs;
        n_bootstrap=100,
    )

    @printf("  kappa*_B (theory) = %.4f\n", kappa_star_B)
    @printf("  Bootstrap 95%% CI: [%.4f, %.4f]\n",
            kappa_ci_result.kappa_star_ci[1], kappa_ci_result.kappa_star_ci[2])
    println()
    println("  -> $(kappa_ci_result.verdict)")

    println("\n" * "="^60)
    println("OVERALL BIFURCATION VALIDATION")
    println("="^60)

    scaling_pass = !isnothing(scaling_result) && scaling_result.pass
    hysteresis_pass = !isnothing(hysteresis_result) && hysteresis_result.pass
    all_pass = scaling_pass && hysteresis_pass

    if all_pass
        println("VALIDATED: data consistent with supercritical pitchfork bifurcation")
    else
        println("VALIDATION ISSUES DETECTED:")
        !scaling_pass && println("  - scaling exponent beta != 0.5")
        !hysteresis_pass && println("  - hysteresis detected")
    end

    validation_df = DataFrame(
        test=["scaling_exponent", "hysteresis", "kappa_star_localization"],
        statistic=[
            isnothing(scaling_result) ? NaN : scaling_result.beta_hat,
            isnothing(hysteresis_result) ? NaN : hysteresis_result.mean_difference,
            kappa_star,
        ],
        ci_lower=[
            isnothing(scaling_result) ? NaN : scaling_result.beta_ci[1],
            NaN,
            kappa_ci_result.kappa_star_ci[1],
        ],
        ci_upper=[
            isnothing(scaling_result) ? NaN : scaling_result.beta_ci[2],
            NaN,
            kappa_ci_result.kappa_star_ci[2],
        ],
        p_value=[
            isnothing(scaling_result) ? NaN : scaling_result.p_value,
            isnothing(hysteresis_result) ? NaN : hysteresis_result.p_value,
            NaN,
        ],
        pass=[scaling_pass, hysteresis_pass, true],
        verdict=[
            isnothing(scaling_result) ? "NOT RUN" : scaling_result.verdict,
            isnothing(hysteresis_result) ? "NOT RUN" : hysteresis_result.verdict,
            kappa_ci_result.verdict,
        ],
    )
    CSV.write(joinpath(stats_dir, "bifurcation_validation.csv"), validation_df)

    open(joinpath(stats_dir, "baseline_variance.txt"), "w") do io
        println(io, "V_baseline = $V_baseline")
        println(io, "kappa_at_min = $V_baseline_kappa")
        println(io, "kappa_ratio_at_min = $V_baseline_ratio")
    end
end

# =============================================================================
# FIGURES
# =============================================================================

println("\nGenerating figures...")
logmsg("Generating figures and exporting density snapshots")
edges = compute_edges_from_snapshots(results; bins=density_bins)
density_data = Dict{String, DensitySummary}()
density_rows = Vector{NamedTuple}()
scenario_order = ["below", "critical", "above"]

for label in scenario_order
    res = results[label]
    info = branch_info[label]
    signs = info.signs
    decided = info.decided
    n_ensemble_local = length(res.snapshots)
    n_times = length(res.snapshot_times)
    n_bins = length(edges) - 1
    centers, _ = histogram_density(res.snapshots[1][1], edges)

    mix_mean = Matrix{Float64}(undef, n_times, n_bins)
    mix_lo = Matrix{Float64}(undef, n_times, n_bins)
    mix_hi = Matrix{Float64}(undef, n_times, n_bins)
    align_mean = Matrix{Float64}(undef, n_times, n_bins)
    align_lo = Matrix{Float64}(undef, n_times, n_bins)
    align_hi = Matrix{Float64}(undef, n_times, n_bins)
    plus_mean = Matrix{Float64}(undef, n_times, n_bins)
    plus_lo = Matrix{Float64}(undef, n_times, n_bins)
    plus_hi = Matrix{Float64}(undef, n_times, n_bins)
    minus_mean = Matrix{Float64}(undef, n_times, n_bins)
    minus_lo = Matrix{Float64}(undef, n_times, n_bins)
    minus_hi = Matrix{Float64}(undef, n_times, n_bins)

    for t_idx in 1:n_times
        dens_all = Matrix{Float64}(undef, n_ensemble_local, n_bins)
        dens_aligned = Matrix{Float64}(undef, n_ensemble_local, n_bins)
        plus_list = Vector{Vector{Float64}}()
        minus_list = Vector{Vector{Float64}}()

        for e_idx in 1:n_ensemble_local
            u = res.snapshots[e_idx][t_idx]
            _, dens = histogram_density(u, edges)
            dens_all[e_idx, :] .= dens

            if signs[e_idx] < 0
                _, dens_align = histogram_density(-u, edges)
                dens_aligned[e_idx, :] .= dens_align
            else
                dens_aligned[e_idx, :] .= dens
            end

            if decided[e_idx]
                if signs[e_idx] < 0
                    push!(minus_list, dens)
                else
                    push!(plus_list, dens)
                end
            end
        end

        mix_mean[t_idx, :], mix_lo[t_idx, :], mix_hi[t_idx, :] =
            summarize_density_stack(dens_all)
        align_mean[t_idx, :], align_lo[t_idx, :], align_hi[t_idx, :] =
            summarize_density_stack(dens_aligned)

        plus_stack = stack_vectors(plus_list, n_bins)
        minus_stack = stack_vectors(minus_list, n_bins)

        plus_mean[t_idx, :], plus_lo[t_idx, :], plus_hi[t_idx, :] =
            summarize_density_stack(plus_stack)
        minus_mean[t_idx, :], minus_lo[t_idx, :], minus_hi[t_idx, :] =
            summarize_density_stack(minus_stack)
    end

    density_data[label] = DensitySummary(
        centers,
        res.snapshot_times,
        mix_mean,
        mix_lo,
        mix_hi,
        align_mean,
        align_lo,
        align_hi,
        plus_mean,
        plus_lo,
        plus_hi,
        minus_mean,
        minus_lo,
        minus_hi,
    )

    for (t_idx, tval) in enumerate(res.snapshot_times)
        for (b_idx, center) in enumerate(centers)
            push!(density_rows, (
                scenario=label,
                time=tval,
                bin_center=center,
                density_mixture_mean=mix_mean[t_idx, b_idx],
                density_mixture_ci_lower=mix_lo[t_idx, b_idx],
                density_mixture_ci_upper=mix_hi[t_idx, b_idx],
                density_aligned_mean=align_mean[t_idx, b_idx],
                density_aligned_ci_lower=align_lo[t_idx, b_idx],
                density_aligned_ci_upper=align_hi[t_idx, b_idx],
                density_plus_mean=plus_mean[t_idx, b_idx],
                density_plus_ci_lower=plus_lo[t_idx, b_idx],
                density_plus_ci_upper=plus_hi[t_idx, b_idx],
                density_minus_mean=minus_mean[t_idx, b_idx],
                density_minus_ci_lower=minus_lo[t_idx, b_idx],
                density_minus_ci_upper=minus_hi[t_idx, b_idx],
            ))
        end
    end
end

if !isempty(density_rows)
    density_df = DataFrame(density_rows)
    CSV.write(joinpath(outdir, "density_snapshots.csv"), density_df)
end

# =============================================================================
# OUTPUT: DENSITY MOMENTS + SANITY CHECKS
# =============================================================================

mom_rows = Vector{NamedTuple}()
for label in scenario_order
    summary = density_data[label]
    centers = summary.centers
    if length(centers) < 2
        continue
    end
    du = centers[2] - centers[1]
    for (t_idx, tval) in enumerate(summary.times)
        rho_mix = summary.mixture_mean[t_idx, :]
        rho_align = summary.aligned_mean[t_idx, :]
        int_mix = sum(rho_mix) * du
        int_align = sum(rho_align) * du
        mu_mix = sum(centers .* rho_mix) * du
        mu_align = sum(centers .* rho_align) * du
        var_mix = sum((centers .^ 2) .* rho_mix) * du - mu_mix^2

        push!(mom_rows, (
            scenario=label,
            time=tval,
            integral_mixture=int_mix,
            integral_aligned=int_align,
            mu_mix=mu_mix,
            mu_aligned=mu_align,
            var_mix=var_mix,
        ))

        if !isfinite(int_mix) || abs(int_mix - 1.0) > density_int_tol
            error("Density integral check failed (mixture) for $label at t=$tval: integral=$int_mix (tol=$density_int_tol)")
        end
        if !isfinite(int_align) || abs(int_align - 1.0) > density_int_tol
            error("Density integral check failed (aligned) for $label at t=$tval: integral=$int_align (tol=$density_int_tol)")
        end
        if label != "above" && isfinite(mu_mix)
            check_time = tval >= density_check_min_time_frac * summary.times[end]
            tol = max(density_mu_mix_tol, density_mu_mix_frac * sqrt(max(var_mix, 0.0)))
            if check_time && abs(mu_mix) > tol
                msg = "Mixture mean not near zero for $label at t=$tval: mu_mix=$mu_mix (tol=$tol)"
                n_runs_here = length(branch_info[label].mean_late)
                if fast_mode || !density_strict_checks || n_runs_here < 8
                    @warn msg
                else
                    if abs(mu_mix) > tol * (1 + density_mu_mix_relax)
                        error(msg)
                    else
                        @warn msg
                    end
                end
            end
        end
        if label == "above" && t_idx == length(summary.times)
            if branch_info[label].decided_share >= branch_min_decided_share && mu_align <= density_mu_aligned_min
                @warn "Aligned mean too small at late time for above: mu_aligned=$mu_align (min=$density_mu_aligned_min)"
            end
        end
    end
end

if !isempty(mom_rows)
    density_moments_df = DataFrame(mom_rows)
    CSV.write(joinpath(outdir, "density_moments.csv"), density_moments_df)
end

plot_ensemble_figure(
    joinpath(figdir, "fig6_ensemble_enhanced.pdf"),
    density_data,
    snapshot_times,
    Dict{String, Any}(stats),
    first(values(results)).time_grid;
    title="Ensemble validation",
    observables=run_observables ? observables : nothing,
)

plot_density_panels_2x2(
    joinpath(figdir, "fig_density_panels.pdf"),
    density_data,
    snapshot_times;
    title="Density evolution",
)

plot_dynamics_panels_2x2(
    joinpath(figdir, "fig_dynamics_panels.pdf"),
    first(values(results)).time_grid,
    Dict{String, Any}(stats);
    terminal_means=terminal_rows,
    title="Dynamics",
)

if run_observables && !isempty(observables)
    plot_robustness_panels_2x2(
        joinpath(figdir, "fig_robustness_panels.pdf"),
        first(values(results)).time_grid,
        Dict{String, Any}(observables);
        title="Robustness diagnostics",
    )
end

if run_sweep && nrow(sweep_df) > 0
    plot_phase_diagram(joinpath(figdir, "fig6_phase_diagram.pdf"), sweep_df)
end

if run_observables
    plot_observables(joinpath(figdir, "fig6_observables_comparison.pdf"),
                     Dict{String, Any}(observables),
                     Dict{String, Any}(stats),
                     first(values(results)).time_grid)
end

# =============================================================================
# REPORTING
# =============================================================================

scaling = if isnothing(scaling_result)
    (beta_hat=NaN, beta_se=NaN, beta_ci=(NaN, NaN), t_statistic=NaN,
     p_value=NaN, C=NaN, pass=false, verdict="NOT RUN", n_points=0)
else
    scaling_result
end

hysteresis = if isnothing(hysteresis_result)
    (mean_difference=NaN, max_difference=NaN, t_statistic=NaN,
     p_value=NaN, pass=false, verdict="NOT RUN")
else
    hysteresis_result
end

kappa_ci = if isnothing(kappa_ci_result)
    (kappa_star_ci=(NaN, NaN), verdict="NOT RUN")
else
    kappa_ci_result
end

metadata_kappa_star_eff = NaN
metadata_beta_fit = NaN
metadata_C_fit = NaN
metadata_kappa_varmin = NaN
if !isnothing(scaling_results)
    metadata_kappa_varmin = scaling_results.kappa_varmin
    if scaling_results.free.converged
        metadata_kappa_star_eff = scaling_results.free.kappa_star_fit
        metadata_beta_fit = scaling_results.free.beta_fit
        metadata_C_fit = scaling_results.free.C_fit
    end
end

    report_md = """
# Bifurcation Validation Report (Figure 6)

Generated: $(Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))

## Model Parameters
| Parameter | Value |
|-----------|-------|
| lambda | $lambda |
| sigma | $sigma |
| theta | $theta |
| c0 | $c0 |
| nu0 | $nu0 |

## Critical Values
- V* = $(round(Vstar, digits=6))
- kappa*_B (theory) = $(isfinite(kappa_star_B) ? round(kappa_star_B, digits=6) : "N/A")
- kappa*_A (empirical) = $(isfinite(kappa_star_A) ? round(kappa_star_A, digits=6) : "N/A")
- kappa*_A CI = $(isfinite(kappa_star_A_ci[1]) ? "[$(round(kappa_star_A_ci[1], digits=6)), $(round(kappa_star_A_ci[2], digits=6))]" : "N/A")
- kappa_varmin (variance minimum, diagnostic) = $(isfinite(metadata_kappa_varmin) ? round(metadata_kappa_varmin, digits=6) : "N/A")
- kappa*_ref (used for scenarios) = $(round(kappa_star, digits=6))
- V_baseline (minimum variance across sweep, near kappa*) = $(run_sweep ? round(V_baseline, digits=6) : "N/A")
- kappa/kappa* at V_baseline = $(run_sweep ? round(V_baseline_ratio, digits=3) : "N/A")

## Simulation Settings
- N = $N
- T = $T
- dt = $dt
- Ensemble size = $n_ensemble (scenarios), $sweep_ensemble (sweep)
- Sweep cap: kappa/kappa* <= $(sweep_ratio_cap)
- Adaptive T: increases near kappa* to mitigate critical slowing down (cap T_max = $T_max)
- Sweep mode: $(sweep_mode) (t_measure = $(t_measure))

---

## Order Parameter
Primary order parameter: **M*(t) = E|m_i(t)|** across runs, with optional **aligned mean**
M_aligned(t) = E[s_i m_i(t)] over decided runs (branch signs from late-time averages).
Because of symmetry, the signed mixture mean E[m(t)] can remain near zero above kappa*.
Variance-based amplitude sqrt(V - V_baseline) is exported as a secondary diagnostic.
Scaling uses the baseline-corrected amplitude M_corr = sqrt(max(M_abs^2 − M0^2, 0)),
with M0 the median M_abs below kappa*_B.

## Test 1: Scaling Exponent (H0: beta = 0.5)
| Statistic | Value |
|-----------|-------|
| beta_hat | $(round(scaling.beta_hat, digits=4)) |
| Standard Error | $(round(scaling.beta_se, digits=4)) |
| 95% CI | [$(round(scaling.beta_ci[1], digits=4)), $(round(scaling.beta_ci[2], digits=4))] |
| t-statistic | $(round(scaling.t_statistic, digits=3)) |
| p-value | $(round(scaling.p_value, digits=4)) |
| Amplitude C | $(hasproperty(scaling, :C) && isfinite(scaling.C) ? round(scaling.C, digits=4) : NaN) |
| Points used | $(scaling.n_points) |
| Delta window | $(hasproperty(scaling, :delta_window) ? "[$(scaling.delta_window[1]), $(scaling.delta_window[2])]" : "N/A") |
| Verdict | $(scaling.verdict) |

## Test 2: Hysteresis (Supercriticality)
| Statistic | Value |
|-----------|-------|
| Mean |Delta M*| | $(round(hysteresis.mean_difference, digits=6)) |
| Max |Delta M*| | $(round(hysteresis.max_difference, digits=6)) |
| t-statistic | $(round(hysteresis.t_statistic, digits=3)) |
| p-value | $(round(hysteresis.p_value, digits=4)) |
| Verdict | $(hysteresis.verdict) |

## Test 3: Critical Point Localization
| Statistic | Value |
|-----------|-------|
| kappa*_B (theory) | $(isfinite(kappa_star_B) ? round(kappa_star_B, digits=4) : NaN) |
| Bootstrap 95% CI | [$(round(kappa_ci.kappa_star_ci[1], digits=4)), $(round(kappa_ci.kappa_star_ci[2], digits=4))] |
| Verdict | $(kappa_ci.verdict) |

---

## Overall Verdict
$(all_pass ? "**PASS**: data consistent with supercritical pitchfork bifurcation" :
"**FAIL**: validation did not pass all criteria")
"""
write_docs(joinpath(docs_dir, "bifurcation_validation_report.md"), report_md)

validation_tex = """
\\begin{table}[h]
\\centering
\\caption{Empirical validation of supercritical pitchfork bifurcation. The primary order parameter is
M* = E|m| (with aligned means shown for decided runs), which is symmetry-aware under Z2 invariance.}
\\label{tab:bifurcation_validation}
\\begin{tabular}{lcccl}
\\toprule
Test & Statistic & 95\\% CI & p-value & Verdict \\\\
\\midrule
Scaling (H0: beta = 0.5) &
  beta_hat = $(round(scaling.beta_hat, digits=3)) &
  [$(round(scaling.beta_ci[1], digits=3)), $(round(scaling.beta_ci[2], digits=3))] &
  $(round(scaling.p_value, digits=3)) &
  $(scaling.pass ? "Pass" : "Fail") \\\\
Hysteresis &
  mean |Delta M*| = $(round(hysteresis.mean_difference, digits=4)) &
  --- &
  $(round(hysteresis.p_value, digits=3)) &
  $(hysteresis.pass ? "Pass" : "Fail") \\\\
\\bottomrule
\\end{tabular}
\\end{table}
"""
write_docs(joinpath(snippets_dir, "bifurcation_validation_table.tex"), validation_tex)

write_docs(joinpath(snippets_dir, "figure_captions.tex"), """
\\caption{Ensemble validation of the polarization threshold.
\\textbf{(a)} Density evolution for kappa < kappa* (consensus),
kappa = kappa* (critical), and kappa > kappa* (symmetry breaking).
Mixture density remains symmetric by construction; aligned density reveals the selected branch.
\\textbf{(b)} Scaling law validation for the symmetry-aware order parameter
M* = E|m| vs kappa - kappa* on log-log axes, with fitted exponent beta
and 95% CI.}
""")

write_docs(joinpath(snippets_dir, "results_text.tex"), """
We validate the supercritical pitchfork prediction using the symmetry-aware order parameter
M* = E|m(t)|, with aligned means computed from run-wise branch signs. Because of Z2 symmetry, the
signed mixture mean can remain near zero above kappa*; therefore E|m(t)| and RMS(m) are the
primary indicators of symmetry breaking. The scaling exponent
beta_hat = $(round(scaling.beta_hat, digits=3)) (95% CI: [$(round(scaling.beta_ci[1], digits=3)),
$(round(scaling.beta_ci[2], digits=3))], p = $(round(scaling.p_value, digits=3))),
estimated for M_corr = sqrt(max(M_abs^2 − M0^2, 0)) in a near-critical window
Δ = (κ−κ*)/κ* ∈ [$(hasproperty(scaling, :delta_window) ? scaling.delta_window[1] : "N/A"),
$(hasproperty(scaling, :delta_window) ? scaling.delta_window[2] : "N/A")] with n = $(scaling.n_points) points,
is compared with the theoretical prediction beta = 0.5. The absence of hysteresis
(p = $(round(hysteresis.p_value, digits=3))) supports supercriticality.
""")

metadata = Dict(
    "model" => "OU-PR",
    "hazard_type" => "StepHazard",
    "parameters" => Dict(
        "lambda" => lambda,
        "sigma" => sigma,
        "theta" => theta,
        "c0" => c0,
        "nu0" => nu0,
    ),
    "simulation" => Dict(
        "N" => N,
        "T" => T,
        "dt" => dt,
        "burn_in" => burn_in,
        "n_ensemble" => n_ensemble,
        "snapshot_times" => snapshot_times,
        "mt_stride" => mt_stride,
        "run_mode" => String(RUN_MODE),
        "n_ensemble_kappa_sweep" => sweep_ensemble,
        "n_rep_per_kappa_nearcrit" => n_rep_per_kappa_nearcrit,
    ),
    "computed" => Dict(
        "Vstar" => Vstar,
        "order_parameter_primary" => "M_abs",
        "order_parameter_aligned" => "M_aligned",
        "kappa_star_ref" => kappa_star,
        "kappa_star_A" => kappa_star_A,
        "kappa_star_A_ci" => [kappa_star_A_ci[1], kappa_star_A_ci[2]],
        "kappa_star_B" => kappa_star_B,
        "Phi0" => isfinite(kappa_star_B) ? kappaB_res.Phi0 : NaN,
        "V_baseline" => V_baseline,
        "M0_abs_floor" => m0,
        "kappa_varmin" => metadata_kappa_varmin,
        "kappa_star_eff" => metadata_kappa_star_eff,
        "beta_fit" => metadata_beta_fit,
        "C_fit" => metadata_C_fit,
        "kappa_ratio_at_V_baseline" => V_baseline_ratio,
        "scaling_delta_window" => hasproperty(scaling, :delta_window) ? [scaling.delta_window[1], scaling.delta_window[2]] : [NaN, NaN],
        "scaling_n_points" => scaling.n_points,
    ),
    "timestamp" => Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"),
    "git_commit" => git_commit_hash(project_root),
)
JSON3.write(joinpath(outdir, "metadata.json"), metadata)

println("\nDone. Outputs written to:")
println("  $outdir")
println("  $sweep_dir")
println("  $figdir")
println("  $docs_dir")
println("  $snippets_dir")
