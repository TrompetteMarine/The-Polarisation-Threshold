#!/usr/bin/env julia
# =============================================================================
# Plot figures from existing outputs (no simulations)
# =============================================================================

using Pkg
Pkg.activate(".")

using CSV
using DataFrames
using Statistics
using Tables

include(joinpath(@__DIR__, "visualization.jl"))
using .Visualization

const PROJECT_ROOT = dirname(@__DIR__)
const OUTDIR = joinpath(PROJECT_ROOT, "outputs", "ensemble_results")
const FIGDIR = joinpath(PROJECT_ROOT, "figs")

mkpath(FIGDIR)

function normalize_names!(df::DataFrame)
    old = names(df)
    new = Symbol.(strip.(string.(old)))
    if old != new
        rename!(df, Pair.(old, new))
    end
    return df
end

function load_density_summaries(path::String)
    df = CSV.read(path, DataFrame)
    normalize_names!(df)
    scenarios = sort(unique(df.scenario))
    density_data = Dict{String, DensitySummary}()
    snapshot_times = sort(unique(df.time))

    for scen in scenarios
        sub = df[df.scenario .== scen, :]
        times = sort(unique(sub.time))
        centers = sort(unique(sub.bin_center))
        n_times = length(times)
        n_bins = length(centers)

        function mat_from(col::Symbol)
            mat = Matrix{Float64}(undef, n_times, n_bins)
            for (ti, tval) in enumerate(times)
                s2 = sub[sub.time .== tval, :]
                ord = sortperm(s2.bin_center)
                mat[ti, :] = s2[ord, col]
            end
            return mat
        end

        density_data[scen] = DensitySummary(
            centers,
            times,
            mat_from(:density_mixture_mean),
            mat_from(:density_mixture_ci_lower),
            mat_from(:density_mixture_ci_upper),
            mat_from(:density_aligned_mean),
            mat_from(:density_aligned_ci_lower),
            mat_from(:density_aligned_ci_upper),
            mat_from(:density_plus_mean),
            mat_from(:density_plus_ci_lower),
            mat_from(:density_plus_ci_upper),
            mat_from(:density_minus_mean),
            mat_from(:density_minus_ci_lower),
            mat_from(:density_minus_ci_upper),
        )
    end

    return density_data, snapshot_times
end

function load_stats_dict(path::String)
    df = CSV.read(path, DataFrame)
    normalize_names!(df)
    scenarios = sort(unique(df.scenario))
    stats = Dict{String, NamedTuple}()
    time_grid = nothing

    for scen in scenarios
        sub = df[df.scenario .== scen, :]
        sub = sort(sub, :time)
        time_grid = time_grid === nothing ? sub.time : time_grid
        stats[scen] = (
            mean_abs = sub.mean_abs,
            mean_abs_ci_lower = sub.mean_abs_ci_lower,
            mean_abs_ci_upper = sub.mean_abs_ci_upper,
            mean_signed = hasproperty(sub, :mean_signed) ? sub.mean_signed : sub.mean,
            mean_signed_ci_lower = hasproperty(sub, :mean_signed_ci_lower) ? sub.mean_signed_ci_lower : sub.mean_ci_lower,
            mean_signed_ci_upper = hasproperty(sub, :mean_signed_ci_upper) ? sub.mean_signed_ci_upper : sub.mean_ci_upper,
            var_traj = sub.variance,
            var_ci_lower = sub.var_ci_lower,
            var_ci_upper = sub.var_ci_upper,
            decided_share = sub.decided_share,
            plus_share = sub.plus_share,
            minus_share = sub.minus_share,
            imbalance = sub.imbalance,
        )
    end

    return stats, time_grid
end

function load_terminal_rows(path::String)
    df = CSV.read(path, DataFrame)
    normalize_names!(df)
    return Tables.rowtable(df)
end

function load_obs_dict(path::String)
    df = CSV.read(path, DataFrame)
    normalize_names!(df)
    scenarios = sort(unique(df.scenario))
    obs = Dict{String, NamedTuple}()
    for scen in scenarios
        sub = df[df.scenario .== scen, :]
        sub = sort(sub, :time)
        obs[scen] = (
            bimodality_mean = sub.bimodality,
            overlap_mean = sub.overlap,
            overlap_times = sub.time,
            variance_mean = sub.variance_star,
            kurtosis_mean = sub.kurtosis,
        )
    end
    return obs
end

function main()
    args = Set(ARGS)
    plot_density = isempty(args) || ("--all" in args) || ("--density" in args)
    plot_dynamics = isempty(args) || ("--all" in args) || ("--dynamics" in args)
    plot_robustness = ("--robustness" in args) || ("--all" in args)

    density_path = joinpath(OUTDIR, "density_snapshots.csv")
    traj_path = joinpath(OUTDIR, "ensemble_trajectories.csv")
    terminal_path = joinpath(OUTDIR, "terminal_means.csv")
    obs_path = joinpath(OUTDIR, "alternative_observables.csv")

    if plot_density
        if isfile(density_path)
            density_data, snapshot_times = load_density_summaries(density_path)
            plot_density_panels_2x2(joinpath(FIGDIR, "fig_density_panels.pdf"), density_data, snapshot_times)
            plot_density_panels_2x2(joinpath(FIGDIR, "fig_density_evolution.pdf"), density_data, snapshot_times)
            println("Saved density panels to figs/fig_density_panels.pdf and figs/fig_density_evolution.pdf")
        else
            @warn "Missing density_snapshots.csv; run fig6_ensemble_enhanced.jl first."
        end
    end

    if plot_dynamics
        if isfile(traj_path)
            stats, time_grid = load_stats_dict(traj_path)
            terminal_rows = isfile(terminal_path) ? load_terminal_rows(terminal_path) : nothing
            plot_dynamics_panels_2x2(joinpath(FIGDIR, "fig_dynamics_panels.pdf"), time_grid, stats;
                                    terminal_means=terminal_rows)
            println("Saved dynamics panels to figs/fig_dynamics_panels.pdf")
        else
            @warn "Missing ensemble_trajectories.csv; run fig6_ensemble_enhanced.jl first."
        end
    end

    if plot_robustness
        if isfile(obs_path) && isfile(traj_path)
            stats, time_grid = load_stats_dict(traj_path)
            obs = load_obs_dict(obs_path)
            plot_robustness_panels_2x2(joinpath(FIGDIR, "fig_robustness_panels.pdf"), time_grid, obs)
            println("Saved robustness panels to figs/fig_robustness_panels.pdf")
        else
            @warn "Missing alternative_observables.csv or ensemble_trajectories.csv; skipping robustness figure."
        end
    end
end

main()

