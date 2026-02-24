module Config

using Dates

export PipelineConfig,
       parse_config,
       setup_output_dirs

const SCALING_DELTA_WINDOWS = [
    (1e-2, 5e-2),
    (1e-2, 1e-1),
    (5e-3, 2e-1),
]
const SCALING_N_MIN = 15               # minimum points for a stable log-log scaling fit
const SCALING_WINDOW_MAX = 0.12        # max delta in valid pitchfork power-law window

"""
    PipelineConfig

Strongly-typed configuration for the bifurcation scaling pipeline.
"""
struct PipelineConfig
    run_mode::Symbol
    fast_mode::Bool
    run_growth_scan::Bool
    run_scaling_regression::Bool
    reuse_scaling_data::Bool

    # Model parameters
    lambda::Float64
    sigma::Float64
    theta::Float64
    c0::Float64
    nu0::Float64

    # Simulation settings
    N::Int
    T::Float64
    dt::Float64
    burn_in::Float64
    seed::Int
    T_max::Float64
    convergence_slope_tol::Float64
    extend_if_not_converged::Bool

    # Ensemble settings
    n_ensemble_scenarios::Int
    n_ensemble_kappa_sweep::Int
    n_rep_per_kappa_nearcrit::Int
    mt_stride::Int
    snapshot_times::Vector{Float64}
    parallel_mode::Symbol
    parallel_target_threads::Int
    parallel_target_workers::Int
    parallel_ensemble::Bool
    density_bins::Int

    # Branch classification
    branch_late_window_fraction::Float64
    branch_min_window::Float64
    branch_tau_mult::Float64
    branch_min_decided_share::Float64
    branch_tau_floor::Float64

    # Density checks
    density_int_tol::Float64
    density_mu_mix_tol::Float64
    density_mu_mix_frac::Float64
    density_mu_mix_relax::Float64
    density_mu_aligned_min::Float64
    density_check_min_time_frac::Float64
    density_strict_checks::Bool

    # Growth scan
    growth_scan_enabled::Bool
    growth_scan_factors::Vector{Float64}
    growth_scan_window::Tuple{Float64,Float64}
    growth_scan_bootstrap::Int
    growth_scan_min_points::Int

    # Scaling regression
    fit_windows::Vector{Tuple{Float64,Float64}}
    scaling_delta_windows::Vector{Tuple{Float64,Float64}}
    scaling_amp_floor::Float64
    scaling_n_min::Int
    scaling_n_target::Int
    scaling_delta_window_default::Tuple{Float64,Float64}
    scaling_allow_window_expand::Bool
    scaling_max_window_expand::Tuple{Float64,Float64}
    scaling_saturation_filter::Bool
    scaling_saturation_cutoff_quantile::Float64
    scaling_convergence_gate::Bool
    scaling_decided_gate::Bool
    scaling_pdec_min::Float64

    # Scenario ratios
    scenario_ratios::Dict{String,Float64}

    # Sweep settings
    run_sweep::Bool
    sweep_ratios::Vector{Float64}
    sweep_ratio_cap::Float64
    max_reasonable_variance::Float64
    sweep_mode::Symbol
    t_measure::Float64

    # Optional analyses
    run_observables::Bool
    run_kappa_scan::Bool

    # Threshold B settings
    kappaB_L::Float64
    kappaB_M::Int
    kappaB_boundary::Symbol

    # Output paths
    outdir::String
    sweep_dir::String
    threshold_dir::String
    figs_dir::String
    docs_dir::String
    snippets_dir::String
    stats_dir::String
end

"""
    parse_config(args::Vector{String}) -> PipelineConfig

Parse CLI flags and return a fully-populated PipelineConfig.
"""
function parse_config(args::Vector{String})::PipelineConfig
    run_mode = :fast
    if any(arg -> arg in ("-pub", "--pub"), args)
        run_mode = :pub
    elseif any(arg -> arg in ("-fast", "--fast"), args)
        run_mode = :fast
    end

    run_growth_scan = !any(arg -> arg in ("-skip-growth", "--skip-growth-scan"), args)
    run_scaling_regression = !any(arg -> arg in ("-skip-scaling", "--skip-scaling"), args)
    reuse_scaling_data = any(arg -> arg in ("-reuse-scaling", "--reuse-scaling"), args)

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
    n_ensemble_scenarios = run_mode == :pub ? 200 : 20
    n_ensemble_kappa_sweep = run_mode == :pub ? 80 : 30
    n_rep_per_kappa_nearcrit = run_mode == :pub ? 3 : 1
    mt_stride = max(1, Int(round(0.05 / dt)))
    snapshot_times = [0.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, T]
    parallel_mode = :distributed
    parallel_target_threads = max(1, Sys.CPU_THREADS - 1)
    parallel_target_workers = max(1, Sys.CPU_THREADS - 1)
    parallel_ensemble = true
    density_bins = 240
    fast_mode = run_mode == :fast

    # Branch classification
    branch_late_window_fraction = 0.2
    branch_min_window = 50.0
    branch_tau_mult = 3.0
    branch_min_decided_share = 0.1
    branch_tau_floor = 1e-6

    # Density checks
    density_int_tol = 1e-2
    density_mu_mix_tol = 2e-2
    density_mu_mix_frac = 0.25
    density_mu_mix_relax = 0.25
    density_mu_aligned_min = 1e-3
    density_check_min_time_frac = 0.8
    density_strict_checks = true

    # Growth scan
    growth_scan_enabled = true
    growth_scan_factors = collect(range(0.70, 1.30; length=15))
    growth_scan_window = (10.0, 60.0)
    growth_scan_bootstrap = run_mode == :pub ? 200 : 100
    growth_scan_min_points = 8

    # Scaling regression
    fit_windows = [(10.0, 50.0), (20.0, 80.0), (50.0, 150.0)]
    scaling_delta_windows = [(1e-2, 5e-2), (1e-2, 1e-1), (5e-3, 2e-1)]
    scaling_amp_floor = 1e-6
    scaling_n_min = SCALING_N_MIN
    scaling_n_target = 30
    scaling_delta_window_default = (1.0e-2, SCALING_WINDOW_MAX)
    scaling_allow_window_expand = true
    scaling_max_window_expand = (5.0e-3, 1.5e-1)
    scaling_saturation_filter = true
    scaling_saturation_cutoff_quantile = 0.85
    scaling_convergence_gate = true
    scaling_decided_gate = true
    scaling_pdec_min = 0.80

    scenario_ratios = Dict(
        "below" => 0.8,
        "critical" => 1.0,
        "above" => 1.5,
    )

    # Sweep settings
    run_sweep = true
    sweep_ratios = fast_mode ? [0.8, 1.0, 1.2, 1.4, 1.5] :
                   [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    sweep_ratio_cap = 1.5
    max_reasonable_variance = 100.0
    sweep_mode = :equilibrium
    t_measure = 50.0

    # Optional analyses
    run_observables = true
    run_kappa_scan = false

    # Threshold B settings
    kappaB_L = 6.0
    kappaB_M = 801
    kappaB_boundary = :reflecting

    project_root = dirname(@__DIR__)
    outdir = joinpath(project_root, "outputs", "ensemble_results")
    sweep_dir = joinpath(project_root, "outputs", "parameter_sweep")
    threshold_dir = joinpath(project_root, "outputs", "threshold")
    figs_dir = joinpath(project_root, "figs")
    docs_dir = joinpath(project_root, "docs")
    snippets_dir = joinpath(project_root, "manuscript_snippets")
    stats_dir = joinpath(project_root, "outputs", "statistical_tests")

    return PipelineConfig(
        run_mode, fast_mode, run_growth_scan, run_scaling_regression, reuse_scaling_data,
        lambda, sigma, theta, c0, nu0,
        N, T, dt, burn_in, seed, T_max, convergence_slope_tol, extend_if_not_converged,
        n_ensemble_scenarios, n_ensemble_kappa_sweep, n_rep_per_kappa_nearcrit,
        mt_stride, snapshot_times, parallel_mode, parallel_target_threads,
        parallel_target_workers, parallel_ensemble, density_bins,
        branch_late_window_fraction, branch_min_window, branch_tau_mult,
        branch_min_decided_share, branch_tau_floor,
        density_int_tol, density_mu_mix_tol, density_mu_mix_frac, density_mu_mix_relax,
        density_mu_aligned_min, density_check_min_time_frac, density_strict_checks,
        growth_scan_enabled, growth_scan_factors, growth_scan_window,
        growth_scan_bootstrap, growth_scan_min_points,
        fit_windows, scaling_delta_windows, scaling_amp_floor, scaling_n_min, scaling_n_target,
        scaling_delta_window_default, scaling_allow_window_expand, scaling_max_window_expand,
        scaling_saturation_filter, scaling_saturation_cutoff_quantile,
        scaling_convergence_gate, scaling_decided_gate, scaling_pdec_min,
        scenario_ratios, run_sweep, sweep_ratios, sweep_ratio_cap, max_reasonable_variance,
        sweep_mode, t_measure, run_observables, run_kappa_scan,
        kappaB_L, kappaB_M, kappaB_boundary,
        outdir, sweep_dir, threshold_dir, figs_dir, docs_dir, snippets_dir, stats_dir
    )
end

"""
    setup_output_dirs(cfg::PipelineConfig)

Create output directories referenced by the pipeline.
"""
function setup_output_dirs(cfg::PipelineConfig)
    for path in (cfg.outdir, cfg.sweep_dir, cfg.threshold_dir, cfg.figs_dir, cfg.docs_dir, cfg.snippets_dir, cfg.stats_dir)
        isdir(path) || mkpath(path)
    end
    return nothing
end

end # module
