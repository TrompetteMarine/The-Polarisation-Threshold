module Visualization

using Statistics
using Distributions
using StatsBase

include(joinpath(@__DIR__, "plot_grammar.jl"))
using .PlotGrammar: GRAMMAR, apply_plot_grammar!, style_axis!, add_time_colorbar!, add_style_legend!, time_color

export DensityCI,
       DensitySummary,
       compute_density_ci,
       plot_ensemble_dashboard,
       plot_density_panels_2x2,
       plot_dynamics_panels_2x2,
       plot_robustness_panels_2x2,
       plot_ensemble_figure,
       plot_phase_diagram,
       plot_observables

struct DensityCI
    centers::Vector{Float64}
    mean_density::Matrix{Float64}   # n_times x n_bins
    ci_lower::Matrix{Float64}
    ci_upper::Matrix{Float64}
    snapshot_times::Vector{Float64}
end

struct DensitySummary
    centers::Vector{Float64}
    times::Vector{Float64}
    mixture_mean::Matrix{Float64}
    mixture_ci_lower::Matrix{Float64}
    mixture_ci_upper::Matrix{Float64}
    aligned_mean::Matrix{Float64}
    aligned_ci_lower::Matrix{Float64}
    aligned_ci_upper::Matrix{Float64}
    plus_mean::Matrix{Float64}
    plus_ci_lower::Matrix{Float64}
    plus_ci_upper::Matrix{Float64}
    minus_mean::Matrix{Float64}
    minus_ci_lower::Matrix{Float64}
    minus_ci_upper::Matrix{Float64}
end

function density_summary_from_ci(ci::DensityCI)
    n_times = size(ci.mean_density, 1)
    n_bins = size(ci.mean_density, 2)
    nanmat = fill(NaN, n_times, n_bins)
    return DensitySummary(
        ci.centers,
        ci.snapshot_times,
        ci.mean_density,
        ci.ci_lower,
        ci.ci_upper,
        ci.mean_density,
        ci.ci_lower,
        ci.ci_upper,
        nanmat,
        nanmat,
        nanmat,
        nanmat,
        nanmat,
        nanmat,
    )
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

function compute_density_ci(
    snapshots::Vector{Vector{Vector{Float64}}},
    edges::Vector{Float64};
    level::Float64 = 0.95,
    snapshot_times::Vector{Float64} = Float64[],
)
    if isempty(snapshots)
        return DensityCI(Float64[], zeros(0, 0), zeros(0, 0), zeros(0, 0), Float64[])
    end
    n_ensemble = length(snapshots)
    n_times = length(snapshots[1])
    n_bins = length(edges) - 1
    centers, _ = histogram_density(snapshots[1][1], edges)

    mean_density = Matrix{Float64}(undef, n_times, n_bins)
    ci_lower = Matrix{Float64}(undef, n_times, n_bins)
    ci_upper = Matrix{Float64}(undef, n_times, n_bins)

    tcrit = n_ensemble < 2 ? 0.0 : quantile(TDist(n_ensemble - 1), 0.5 + level / 2)

    for t_idx in 1:n_times
        dens_stack = Matrix{Float64}(undef, n_ensemble, n_bins)
        for e_idx in 1:n_ensemble
            _, dens = histogram_density(snapshots[e_idx][t_idx], edges)
            dens_stack[e_idx, :] .= dens
        end
        mean_vec = vec(mean(dens_stack; dims=1))
        std_vec = vec(std(dens_stack; dims=1, corrected=true))
        if n_ensemble < 2
            std_vec .= 0.0
        end
        se = n_ensemble < 2 ? zeros(n_bins) : std_vec ./ sqrt(n_ensemble)
        mean_density[t_idx, :] .= mean_vec
        ci_lower[t_idx, :] .= mean_vec .- tcrit .* se
        ci_upper[t_idx, :] .= mean_vec .+ tcrit .* se
    end

    times = isempty(snapshot_times) ? collect(1:n_times) : snapshot_times
    return DensityCI(centers, mean_density, ci_lower, ci_upper, times)
end

function _load_cairo()
    try
        @eval using CairoMakie
        return true
    catch
        return false
    end
end

function _load_plots()
    try
        @eval using Plots
        return true
    catch
        return false
    end
end

function plot_ensemble_dashboard(
    output_path::AbstractString,
    density_data::AbstractDict{String, DensitySummary},
    snapshot_times::Vector{Float64},
    trajectory_stats::AbstractDict{String, T},
    time_grid::Vector{Float64};
    observables = nothing,
    show_ci::Bool = true,
    show_plus_minus::Bool = true,
    title::String = "Ensemble diagnostics",
) where {T}
    _load_cairo() || error("CairoMakie is required for plot_ensemble_dashboard.")
    apply_plot_grammar!(CairoMakie)

    fig = CairoMakie.Figure(size=(1200, 900), backgroundcolor=:white)

    add_style_legend!(CairoMakie, fig[0, 1:2]; include_plus_minus=false)

    tmin, tmax = isempty(snapshot_times) ? (0.0, 1.0) : extrema(snapshot_times)
    t_ticks = if length(snapshot_times) <= 5
        snapshot_times
    else
        collect(range(tmin, tmax; length=5))
    end
    add_time_colorbar!(CairoMakie, fig[1:2, 3]; tmin=tmin, tmax=tmax, ticks=t_ticks, label="time t")

    ax_below = CairoMakie.Axis(fig[1, 1], xlabel="u", ylabel="rho(u)", title="κ<κ*")
    ax_critical = CairoMakie.Axis(fig[1, 2], xlabel="u", ylabel="rho(u)", title="κ≈κ*")
    ax_above_aligned = CairoMakie.Axis(fig[2, 1], xlabel="u", ylabel="rho(u)", title="κ>κ*")
    ax_above_mix = CairoMakie.Axis(fig[2, 2], xlabel="u", ylabel="rho(u)", title="κ>κ*")

    for (ax, lbl) in zip((ax_below, ax_critical, ax_above_aligned, ax_above_mix), ("(a)", "(b)", "(c)", "(d)"))
        CairoMakie.text!(ax, 0.02, 0.95, text=lbl, space=:relative, align=(:left, :top), fontsize=12)
    end

    CairoMakie.text!(ax_above_aligned, 0.02, 0.95, text="aligned", space=:relative,
                     align=(:left, :top), fontsize=10)
    CairoMakie.text!(ax_above_mix, 0.02, 0.95, text="mixture", space=:relative,
                     align=(:left, :top), fontsize=10)

    function plot_density_series!(
        ax,
        centers,
        series,
        tvals;
        linestyle=:solid,
        linewidth=GRAMMAR.linewidth_main,
        alpha=1.0,
        ci_lower=nothing,
        ci_upper=nothing,
        ci_last=false,
    )
        n_times = length(tvals)
        for (i, tval) in enumerate(tvals)
            col = time_color(CairoMakie, tval, tmin, tmax; colormap=GRAMMAR.colormap_time)
            col_use = alpha < 1.0 ? (col, alpha) : col
            CairoMakie.lines!(ax, centers, series[i, :];
                              color=col_use, linestyle=linestyle, linewidth=linewidth)
            if show_ci && ci_last && i == n_times && ci_lower !== nothing && ci_upper !== nothing
                CairoMakie.band!(ax, centers, ci_lower[i, :], ci_upper[i, :],
                                 color=(col, GRAMMAR.ci_alpha))
            end
        end
    end

    if haskey(density_data, "below")
        data = density_data["below"]
        plot_density_series!(ax_below, data.centers, data.mixture_mean, snapshot_times;
                             linestyle=:dash, linewidth=GRAMMAR.linewidth_secondary,
                             alpha=GRAMMAR.alpha_secondary)
    end
    if haskey(density_data, "critical")
        data = density_data["critical"]
        plot_density_series!(ax_critical, data.centers, data.mixture_mean, snapshot_times;
                             linestyle=:dash, linewidth=GRAMMAR.linewidth_secondary,
                             alpha=GRAMMAR.alpha_secondary)
    end
    if haskey(density_data, "above")
        data = density_data["above"]
        plot_density_series!(ax_above_aligned, data.centers, data.aligned_mean, snapshot_times;
                             linestyle=:solid, linewidth=GRAMMAR.linewidth_main,
                             alpha=1.0, ci_lower=data.aligned_ci_lower,
                             ci_upper=data.aligned_ci_upper, ci_last=true)
        plot_density_series!(ax_above_mix, data.centers, data.mixture_mean, snapshot_times;
                             linestyle=:dash, linewidth=GRAMMAR.linewidth_secondary,
                             alpha=GRAMMAR.alpha_secondary)
        if show_plus_minus && size(data.plus_mean, 1) >= length(snapshot_times)
            idx = length(snapshot_times)
            CairoMakie.lines!(ax_above_aligned, data.centers, data.plus_mean[idx, :],
                              color=(:gray, 0.65), linewidth=GRAMMAR.linewidth_secondary,
                              linestyle=:dot)
            CairoMakie.lines!(ax_above_aligned, data.centers, data.minus_mean[idx, :],
                              color=(:gray, 0.65), linewidth=GRAMMAR.linewidth_secondary,
                              linestyle=:dashdot)
        end
    end

    ax_m = CairoMakie.Axis(fig[3, 1], xlabel="t", ylabel="E|m(t)|", title="Order parameter")
    ax_var = CairoMakie.Axis(fig[3, 2], xlabel="t", ylabel="Var(u)", title="Variance")
    ax_extra = CairoMakie.Axis(fig[3, 3], xlabel="t", ylabel="overlap", title="Overlap")

    for (ax, lbl) in zip((ax_m, ax_var, ax_extra), ("(e)", "(f)", "(g)"))
        CairoMakie.text!(ax, 0.02, 0.95, text=lbl, space=:relative, align=(:left, :top), fontsize=11)
    end

    colors = Dict("below" => :blue, "critical" => :green, "above" => :red)
    labels = Dict("below" => "κ<κ*", "critical" => "κ≈κ*", "above" => "κ>κ*")

    for label in keys(trajectory_stats)
        stats = trajectory_stats[label]
        color = get(colors, label, :black)
        mean_abs = hasproperty(stats, :mean_abs) ? stats.mean_abs : stats.mean
        mean_abs_lo = hasproperty(stats, :mean_abs_ci_lower) ? stats.mean_abs_ci_lower : stats.mean_ci_lower
        mean_abs_hi = hasproperty(stats, :mean_abs_ci_upper) ? stats.mean_abs_ci_upper : stats.mean_ci_upper
        mean_signed = hasproperty(stats, :mean_signed) ? stats.mean_signed :
                      (hasproperty(stats, :mean_traj) ? stats.mean_traj : stats.mean)
        mean_aligned = hasproperty(stats, :mean_aligned) ? stats.mean_aligned : nothing
        mean_aligned_lo = hasproperty(stats, :mean_aligned_ci_lower) ? stats.mean_aligned_ci_lower : nothing
        mean_aligned_hi = hasproperty(stats, :mean_aligned_ci_upper) ? stats.mean_aligned_ci_upper : nothing

        CairoMakie.lines!(ax_m, time_grid, mean_abs, color=color,
                          linewidth=GRAMMAR.linewidth_main)
        CairoMakie.band!(ax_m, time_grid, mean_abs_lo, mean_abs_hi, color=(color, GRAMMAR.ci_alpha))
        if mean_aligned !== nothing && any(isfinite, mean_aligned)
            CairoMakie.lines!(ax_m, time_grid, mean_aligned, color=(color, 0.7),
                              linestyle=:dot, linewidth=GRAMMAR.linewidth_secondary)
            if mean_aligned_lo !== nothing && mean_aligned_hi !== nothing
                CairoMakie.band!(ax_m, time_grid, mean_aligned_lo, mean_aligned_hi,
                                 color=(color, GRAMMAR.ci_alpha * 0.7))
            end
        end
        CairoMakie.lines!(ax_m, time_grid, mean_signed, color=(:gray, 0.6),
                          linestyle=:dash, linewidth=GRAMMAR.linewidth_secondary)

        if hasproperty(stats, :var_traj)
            CairoMakie.lines!(ax_var, time_grid, stats.var_traj, color=color,
                              linewidth=GRAMMAR.linewidth_main)
            if hasproperty(stats, :var_ci_lower) && hasproperty(stats, :var_ci_upper)
                CairoMakie.band!(ax_var, time_grid, stats.var_ci_lower, stats.var_ci_upper,
                                 color=(color, GRAMMAR.ci_alpha))
            end
        end

        if observables !== nothing && haskey(observables, label)
            obs = observables[label]
            if hasproperty(obs, :decided_share_mean)
                ax_extra.ylabel = "decided share"
                ax_extra.title = "Decided share"
                CairoMakie.lines!(ax_extra, time_grid, obs.decided_share_mean, color=color,
                                  linewidth=GRAMMAR.linewidth_main)
                if hasproperty(obs, :decided_share_ci_lower) && hasproperty(obs, :decided_share_ci_upper)
                    CairoMakie.band!(ax_extra, time_grid, obs.decided_share_ci_lower, obs.decided_share_ci_upper,
                                     color=(color, GRAMMAR.ci_alpha))
                end
            elseif hasproperty(obs, :overlap_mean) && !isempty(obs.overlap_times)
                ax_extra.ylabel = "overlap"
                ax_extra.title = "Overlap"
                CairoMakie.lines!(ax_extra, obs.overlap_times, obs.overlap_mean, color=color,
                                  linewidth=GRAMMAR.linewidth_main)
                if hasproperty(obs, :overlap_ci_lower) && hasproperty(obs, :overlap_ci_upper)
                    CairoMakie.band!(ax_extra, obs.overlap_times, obs.overlap_ci_lower, obs.overlap_ci_upper,
                                     color=(color, GRAMMAR.ci_alpha))
                end
            elseif hasproperty(obs, :bimodality_mean)
                ax_extra.ylabel = "bimodality"
                ax_extra.title = "Bimodality"
                CairoMakie.lines!(ax_extra, time_grid, obs.bimodality_mean, color=color,
                                  linewidth=GRAMMAR.linewidth_main)
                if hasproperty(obs, :bimodality_ci_lower) && hasproperty(obs, :bimodality_ci_upper)
                    CairoMakie.band!(ax_extra, time_grid, obs.bimodality_ci_lower, obs.bimodality_ci_upper,
                                     color=(color, GRAMMAR.ci_alpha))
                end
            end
        end
    end

    max_abs = -Inf
    for stats in values(trajectory_stats)
        if hasproperty(stats, :mean_abs) && !isempty(stats.mean_abs)
            max_abs = max(max_abs, maximum(stats.mean_abs))
        end
    end
    if isfinite(max_abs) && max_abs > 0
        CairoMakie.ylims!(ax_m, 0.0, max_abs * 1.05)
    end

    for (i, key) in enumerate(["below", "critical", "above"])
        if haskey(colors, key)
            CairoMakie.text!(ax_m, 0.02, 0.95 - 0.07 * (i - 1);
                             text=labels[key], color=colors[key], space=:relative,
                             align=(:left, :top), fontsize=10)
        end
    end

    CairoMakie.save(output_path, fig)
    return fig
end

function plot_ensemble_figure(
    output_path::String,
    density_data::AbstractDict{String, DensitySummary},
    snapshot_times::Vector{Float64},
    trajectory_stats::AbstractDict{String, T},
    time_grid::Vector{Float64};
    title::String = "Ensemble density snapshots",
    scenario_order::Vector{String} = ["below", "critical", "above"],
    show_ci::Bool = true,
    show_plus_minus::Bool = true,
    observables = nothing,
) where {T}
    if _load_cairo()
        return plot_ensemble_dashboard(
            output_path,
            density_data,
            snapshot_times,
            trajectory_stats,
            time_grid;
            observables=observables,
            show_ci=show_ci,
            show_plus_minus=show_plus_minus,
            title=title,
        )
    end

    if _load_plots()
        Plots.default(fontfamily="Computer Modern")
        colors = Plots.cgrad(:viridis, length(snapshot_times), categorical=false)
        plt1 = Plots.plot(; xlabel="u", ylabel="rho(u)", title="kappa<kappa*", legend=false)
        plt2 = Plots.plot(; xlabel="u", ylabel="rho(u)", title="kappa~kappa*", legend=false)
        plt3 = Plots.plot(; xlabel="u", ylabel="rho(u)", title="kappa>kappa*", legend=false)
        plt4 = Plots.plot(; xlabel="t", ylabel="E|m(t)|", title="Mean trajectory (symmetry-aware)")

        for (i, tval) in enumerate(snapshot_times)
            Plots.plot!(plt1, density_data["below"].centers, density_data["below"].aligned_mean[i, :];
                        color=colors[i], label="")
            Plots.plot!(plt1, density_data["below"].centers, density_data["below"].mixture_mean[i, :];
                        color=colors[i], linestyle=:dash, label="")
            Plots.plot!(plt2, density_data["critical"].centers, density_data["critical"].aligned_mean[i, :];
                        color=colors[i], label="")
            Plots.plot!(plt2, density_data["critical"].centers, density_data["critical"].mixture_mean[i, :];
                        color=colors[i], linestyle=:dash, label="")
            Plots.plot!(plt3, density_data["above"].centers, density_data["above"].aligned_mean[i, :];
                        color=colors[i], label="")
            Plots.plot!(plt3, density_data["above"].centers, density_data["above"].mixture_mean[i, :];
                        color=colors[i], linestyle=:dash, label="")
        end

        for (label, stats) in trajectory_stats
            if label == "below"
                Plots.plot!(plt4, time_grid, stats.mean_abs; color=:blue, label="below |m|")
                Plots.plot!(plt4, time_grid, stats.mean_traj; color=:gray, linestyle=:dash, label="E[m(t)] (signed)")
            elseif label == "critical"
                Plots.plot!(plt4, time_grid, stats.mean_abs; color=:green, label="critical |m|")
                Plots.plot!(plt4, time_grid, stats.mean_traj; color=:gray, linestyle=:dash, label="")
            elseif label == "above"
                Plots.plot!(plt4, time_grid, stats.mean_abs; color=:red, label="above |m|")
                Plots.plot!(plt4, time_grid, stats.mean_traj; color=:gray, linestyle=:dash, label="")
            end
        end
        max_abs = -Inf
        for stats in values(trajectory_stats)
            if hasproperty(stats, :mean_abs) && !isempty(stats.mean_abs)
                max_abs = max(max_abs, maximum(stats.mean_abs))
            end
        end
        if isfinite(max_abs) && max_abs > 0
            Plots.ylims!(plt4, (0.0, max_abs * 1.05))
        end

        fig = Plots.plot(plt1, plt2, plt3, plt4; layout=(4, 1), size=(900, 1200), title=title)
        Plots.savefig(fig, output_path)
        return
    end

    error("No plotting backend available (CairoMakie or Plots).")
end

function plot_ensemble_figure(
    output_path::String,
    density_data::AbstractDict{String, DensityCI},
    snapshot_times::Vector{Float64},
    trajectory_stats::AbstractDict{String, T},
    time_grid::Vector{Float64};
    title::String = "Ensemble density snapshots",
    scenario_order::Vector{String} = ["below", "critical", "above"],
    show_ci::Bool = true,
    show_plus_minus::Bool = true,
    observables = nothing,
) where {T}
    summary = Dict{String, DensitySummary}()
    for (k, v) in density_data
        summary[k] = density_summary_from_ci(v)
    end
    return plot_ensemble_figure(
        output_path,
        summary,
        snapshot_times,
        trajectory_stats,
        time_grid;
        title=title,
        scenario_order=scenario_order,
        show_ci=show_ci,
        show_plus_minus=show_plus_minus,
        observables=observables,
    )
end

function plot_phase_diagram(output_path::String, sweep_df)
    if _load_cairo()
        apply_plot_grammar!(CairoMakie)
        fig = CairoMakie.Figure(size=(900, 600), backgroundcolor=:white)
        ax = CairoMakie.Axis(fig[1, 1], xlabel="kappa / kappa*", ylabel="lambda1",
                             title="Leading eigenvalue vs coupling")
        x = sweep_df.kappa_ratio
        y = sweep_df.lambda_mean
        ylow = sweep_df.lambda_ci_lower
        yhigh = sweep_df.lambda_ci_upper
        CairoMakie.errorbars!(ax, x, y, y .- ylow, yhigh .- y; whiskerwidth=6)
        CairoMakie.scatter!(ax, x, y, color=:black)
        CairoMakie.hlines!(ax, [0.0], color=:gray, linestyle=:dash, linewidth=GRAMMAR.linewidth_secondary)
        CairoMakie.vlines!(ax, [1.0], color=:gray, linestyle=:dash, linewidth=GRAMMAR.linewidth_secondary)
        CairoMakie.save(output_path, fig)
        return
    end

    if _load_plots()
        Plots.default(fontfamily="Computer Modern")
        x = sweep_df.kappa_ratio
        y = sweep_df.lambda_mean
        yerr = y .- sweep_df.lambda_ci_lower
        plt = Plots.scatter(x, y; yerr=yerr, xlabel="kappa / kappa*", ylabel="lambda1",
                            title="Leading eigenvalue vs coupling")
        Plots.hline!(plt, [0.0], linestyle=:dash, color=:gray)
        Plots.vline!(plt, [1.0], linestyle=:dash, color=:gray)
        Plots.savefig(plt, output_path)
        return
    end

    error("No plotting backend available (CairoMakie or Plots).")
end

function plot_observables(output_path::String,
                          observables::AbstractDict{String, T},
                          variance_stats::AbstractDict{String, S},
                          time_grid::Vector{Float64}) where {T, S}
    if _load_cairo()
        apply_plot_grammar!(CairoMakie)
        fig = CairoMakie.Figure(size=(1100, 800), backgroundcolor=:white)
        ax1 = CairoMakie.Axis(fig[1, 1], xlabel="t", ylabel="variance", title="Variance")
        ax2 = CairoMakie.Axis(fig[1, 2], xlabel="t", ylabel="bimodality", title="Bimodality coefficient")
        ax3 = CairoMakie.Axis(fig[2, 1], xlabel="t", ylabel="overlap", title="Overlap integral")
        ax4 = CairoMakie.Axis(fig[2, 2], xlabel="t", ylabel="decided share", title="Decided share")

        colors = Dict("below" => :blue, "critical" => :green, "above" => :red)
        labels = Dict("below" => "κ<κ*", "critical" => "κ≈κ*", "above" => "κ>κ*")
        for (label, obs) in observables
            color = colors[label]
            if haskey(variance_stats, label)
                stats = variance_stats[label]
                CairoMakie.lines!(ax1, time_grid, stats.var_traj, color=color, linewidth=GRAMMAR.linewidth_main,
                                  label=labels[label])
                CairoMakie.band!(ax1, time_grid, stats.var_ci_lower, stats.var_ci_upper, color=(color, GRAMMAR.ci_alpha))
            end
            CairoMakie.lines!(ax2, time_grid, obs.bimodality_mean, color=color, linewidth=GRAMMAR.linewidth_main, label=nothing)
            CairoMakie.band!(ax2, time_grid, obs.bimodality_ci_lower, obs.bimodality_ci_upper, color=(color, GRAMMAR.ci_alpha))

            if !isempty(obs.overlap_times)
                CairoMakie.lines!(ax3, obs.overlap_times, obs.overlap_mean, color=color, linewidth=GRAMMAR.linewidth_main, label=nothing)
                if hasproperty(obs, :overlap_ci_lower) && hasproperty(obs, :overlap_ci_upper)
                    CairoMakie.band!(ax3, obs.overlap_times, obs.overlap_ci_lower, obs.overlap_ci_upper, color=(color, GRAMMAR.ci_alpha))
                end
            end

            if hasproperty(obs, :decided_share_mean)
                CairoMakie.lines!(ax4, time_grid, obs.decided_share_mean, color=color, linewidth=GRAMMAR.linewidth_main, label=nothing)
                if hasproperty(obs, :decided_share_ci_lower) && hasproperty(obs, :decided_share_ci_upper)
                    CairoMakie.band!(ax4, time_grid, obs.decided_share_ci_lower, obs.decided_share_ci_upper, color=(color, GRAMMAR.ci_alpha))
                end
            elseif hasproperty(obs, :overlap_mean)
                ax4.ylabel = "overlap"
                ax4.title = "Overlap integral"
                CairoMakie.lines!(ax4, obs.overlap_times, obs.overlap_mean, color=color, linewidth=GRAMMAR.linewidth_main, label=nothing)
            end
        end

        CairoMakie.Legend(fig[0, 1:2], ax1; orientation=:horizontal, framevisible=false)
        CairoMakie.save(output_path, fig)
        return
    end

    if _load_plots()
        Plots.default(fontfamily="Computer Modern")
        plt1 = Plots.plot(; xlabel="t", ylabel="variance", title="Variance")
        plt2 = Plots.plot(; xlabel="t", ylabel="bimodality", title="Bimodality coefficient", legend=false)
        plt3 = Plots.plot(; xlabel="t", ylabel="overlap", title="Overlap integral", legend=false)
        plt4 = Plots.plot(; xlabel="t", ylabel="decided share", title="Decided share", legend=false)

        colors = Dict("below" => :blue, "critical" => :green, "above" => :red)
        labels = Dict("below" => "κ<κ*", "critical" => "κ≈κ*", "above" => "κ>κ*")
        for (label, obs) in observables
            color = colors[label]
            if haskey(variance_stats, label)
                stats = variance_stats[label]
                Plots.plot!(plt1, time_grid, stats.var_traj; color=color, label=labels[label])
            end
            Plots.plot!(plt2, time_grid, obs.bimodality_mean; color=color, label="")
            if !isempty(obs.overlap_times)
                Plots.plot!(plt3, obs.overlap_times, obs.overlap_mean; color=color, label="")
            end
            if hasproperty(obs, :decided_share_mean)
                Plots.plot!(plt4, time_grid, obs.decided_share_mean; color=color, label="")
            elseif hasproperty(obs, :overlap_mean)
                Plots.plot!(plt4, obs.overlap_times, obs.overlap_mean; color=color, label="")
            end
        end
        fig = Plots.plot(plt1, plt2, plt3, plt4; layout=(2, 2), size=(1000, 700))
        Plots.savefig(fig, output_path)
        return
    end

    error("No plotting backend available (CairoMakie or Plots).")
end

function plot_density_panels_2x2(
    outpath::AbstractString,
    density_data::AbstractDict{String, DensitySummary},
    snapshot_times::Vector{Float64};
    title::String = "Density evolution",
    show_mixture_below::Bool = true,
    show_mixture_critical::Bool = true,
    show_aligned_above::Bool = true,
    show_mixture_above::Bool = true,
)
    _load_cairo() || error("CairoMakie is required for plot_density_panels_2x2.")
    apply_plot_grammar!(CairoMakie)

    tmin, tmax = isempty(snapshot_times) ? (0.0, 1.0) : extrema(snapshot_times)
    t_ticks = length(snapshot_times) <= 5 ? snapshot_times : collect(range(tmin, tmax; length=5))

    fig = CairoMakie.Figure(size=(900, 700), backgroundcolor=:white)
    ax1 = CairoMakie.Axis(fig[1, 1], xlabel="u", ylabel="rho(u)", title="κ<κ*")
    ax2 = CairoMakie.Axis(fig[1, 2], xlabel="u", ylabel="rho(u)", title="κ≈κ*")
    ax3 = CairoMakie.Axis(fig[2, 1], xlabel="u", ylabel="rho(u)", title="κ>κ* (aligned)")
    ax4 = CairoMakie.Axis(fig[2, 2], xlabel="u", ylabel="rho(u)", title="κ>κ* (mixture)")

    for (ax, lbl) in zip((ax1, ax2, ax3, ax4), ("(a)", "(b)", "(c)", "(d)"))
        CairoMakie.text!(ax, 0.02, 0.95, text=lbl, space=:relative, align=(:left, :top), fontsize=12)
    end

    add_style_legend!(CairoMakie, fig[0, 1:2]; include_plus_minus=false)
    add_time_colorbar!(CairoMakie, fig[1:2, 3]; tmin=tmin, tmax=tmax, ticks=t_ticks, label="time t")

    function plot_series!(ax, centers, series, linestyle, linewidth, alpha; ci=nothing)
        n_times = length(snapshot_times)
        for (i, tval) in enumerate(snapshot_times)
            col = time_color(CairoMakie, tval, tmin, tmax; colormap=GRAMMAR.colormap_time)
            col_use = alpha < 1.0 ? (col, alpha) : col
            CairoMakie.lines!(ax, centers, series[i, :]; color=col_use, linestyle=linestyle, linewidth=linewidth)
            if ci !== nothing && i == n_times
                CairoMakie.band!(ax, centers, ci[1][i, :], ci[2][i, :], color=(col, GRAMMAR.ci_alpha))
            end
        end
    end

    if haskey(density_data, "below")
        d = density_data["below"]
        if show_mixture_below
            plot_series!(ax1, d.centers, d.mixture_mean, :dash, GRAMMAR.linewidth_secondary, GRAMMAR.alpha_secondary)
        end
        plot_series!(ax1, d.centers, d.aligned_mean, :solid, GRAMMAR.linewidth_main, 1.0)
    end
    if haskey(density_data, "critical")
        d = density_data["critical"]
        if show_mixture_critical
            plot_series!(ax2, d.centers, d.mixture_mean, :dash, GRAMMAR.linewidth_secondary, GRAMMAR.alpha_secondary)
        end
        plot_series!(ax2, d.centers, d.aligned_mean, :solid, GRAMMAR.linewidth_main, 1.0)
    end
    if haskey(density_data, "above")
        d = density_data["above"]
        if show_aligned_above
            plot_series!(ax3, d.centers, d.aligned_mean, :solid, GRAMMAR.linewidth_main, 1.0;
                         ci=(d.aligned_ci_lower, d.aligned_ci_upper))
        end
        if show_mixture_above
            plot_series!(ax4, d.centers, d.mixture_mean, :dash, GRAMMAR.linewidth_secondary, GRAMMAR.alpha_secondary)
        end
    end

    CairoMakie.save(outpath, fig)
    return fig
end

function plot_dynamics_panels_2x2(
    outpath::AbstractString,
    time_grid::Vector{Float64},
    stats_dict::AbstractDict{String, T};
    terminal_means = nothing,
    title::String = "Dynamics",
    include_decided_share::Bool = true,
    include_branch_imbalance::Bool = true,
) where {T}
    _load_cairo() || error("CairoMakie is required for plot_dynamics_panels_2x2.")
    apply_plot_grammar!(CairoMakie)

    fig = CairoMakie.Figure(size=(900, 700), backgroundcolor=:white)
    ax1 = CairoMakie.Axis(fig[1, 1], xlabel="t", ylabel="E|m(t)|", title="Order parameter")
    ax2 = CairoMakie.Axis(fig[1, 2], xlabel="t", ylabel="E[m(t)]", title="Mixture mean")
    ax3 = CairoMakie.Axis(fig[2, 1], xlabel="t", ylabel="Var(u_t)", title="Variance")
    ax4 = CairoMakie.Axis(fig[2, 2], xlabel="t", ylabel="share / imbalance", title="Decision diagnostics")

    for (ax, lbl) in zip((ax1, ax2, ax3, ax4), ("(a)", "(b)", "(c)", "(d)"))
        CairoMakie.text!(ax, 0.02, 0.95, text=lbl, space=:relative, align=(:left, :top), fontsize=12)
    end

    colors = Dict("below" => :blue, "critical" => :green, "above" => :red)
    labels = Dict("below" => "κ<κ*", "critical" => "κ≈κ*", "above" => "κ>κ*")

    for (label, stats) in stats_dict
        color = get(colors, label, :black)
        mean_abs = hasproperty(stats, :mean_abs) ? stats.mean_abs : stats.mean
        mean_abs_lo = hasproperty(stats, :mean_abs_ci_lower) ? stats.mean_abs_ci_lower : stats.mean_ci_lower
        mean_abs_hi = hasproperty(stats, :mean_abs_ci_upper) ? stats.mean_abs_ci_upper : stats.mean_ci_upper
        mean_signed = hasproperty(stats, :mean_signed) ? stats.mean_signed :
                      (hasproperty(stats, :mean_traj) ? stats.mean_traj : stats.mean)
        mean_signed_lo = hasproperty(stats, :mean_signed_ci_lower) ? stats.mean_signed_ci_lower : mean_abs_lo
        mean_signed_hi = hasproperty(stats, :mean_signed_ci_upper) ? stats.mean_signed_ci_upper : mean_abs_hi

        CairoMakie.lines!(ax1, time_grid, mean_abs, color=color, linewidth=GRAMMAR.linewidth_main)
        CairoMakie.band!(ax1, time_grid, mean_abs_lo, mean_abs_hi, color=(color, GRAMMAR.ci_alpha))

    CairoMakie.lines!(ax2, time_grid, mean_signed, color=color, linewidth=GRAMMAR.linewidth_main)
    CairoMakie.band!(ax2, time_grid, mean_signed_lo, mean_signed_hi, color=(color, GRAMMAR.ci_alpha))
    CairoMakie.hlines!(ax2, [0.0], color=:gray, linestyle=:dash, linewidth=GRAMMAR.linewidth_secondary)

        var_traj = hasproperty(stats, :var_traj) ? stats.var_traj : (hasproperty(stats, :variance) ? stats.variance : mean_abs .* 0)
        var_lo = hasproperty(stats, :var_ci_lower) ? stats.var_ci_lower : var_traj
        var_hi = hasproperty(stats, :var_ci_upper) ? stats.var_ci_upper : var_traj
        CairoMakie.lines!(ax3, time_grid, var_traj, color=color, linewidth=GRAMMAR.linewidth_main)
        CairoMakie.band!(ax3, time_grid, var_lo, var_hi, color=(color, GRAMMAR.ci_alpha))
    end

    CairoMakie.ylims!(ax1, 0.0, nothing)

    if terminal_means !== nothing
        by_scen = Dict{String, Vector{Any}}()
        for row in terminal_means
            push!(get!(by_scen, row.scenario, Any[]), row)
        end
        for (label, rows) in by_scen
            color = get(colors, label, :black)
            decided = [r for r in rows if getfield(r, :decided_flag)]
            if isempty(decided)
                continue
            end
            plus = count(r -> getfield(r, :branch_sign) > 0, decided)
            minus = count(r -> getfield(r, :branch_sign) < 0, decided)
            total = length(decided)
            decided_share = total / max(length(rows), 1)
            imbalance = (plus - minus) / max(total, 1)
            if include_decided_share
                CairoMakie.lines!(ax4, time_grid, fill(decided_share, length(time_grid)),
                                  color=color, linewidth=GRAMMAR.linewidth_main)
            end
            if include_branch_imbalance
                CairoMakie.lines!(ax4, time_grid, fill(imbalance, length(time_grid)),
                                  color=color, linewidth=GRAMMAR.linewidth_secondary, linestyle=:dash)
            end
        end
    else
        CairoMakie.text!(ax4, 0.5, 0.5, text="terminal_means missing", space=:relative, align=(:center, :center))
    end

    CairoMakie.text!(ax4, 0.02, 0.92, text="solid: decided\n dashed: imbalance",
                     space=:relative, align=(:left, :top), fontsize=9)
    CairoMakie.ylims!(ax4, -1.0, 1.0)
    CairoMakie.text!(ax2, 0.02, 0.92, text="finite-ensemble imbalance possible",
                     space=:relative, align=(:left, :top), fontsize=9)

    scenario_elements = [
        CairoMakie.LineElement(color=colors["below"], linestyle=:solid, linewidth=GRAMMAR.linewidth_main),
        CairoMakie.LineElement(color=colors["critical"], linestyle=:solid, linewidth=GRAMMAR.linewidth_main),
        CairoMakie.LineElement(color=colors["above"], linestyle=:solid, linewidth=GRAMMAR.linewidth_main),
    ]
    CairoMakie.Legend(fig[0, 1:2], scenario_elements, [labels["below"], labels["critical"], labels["above"]];
                      orientation=:horizontal, framevisible=false, tellwidth=false)

    CairoMakie.Legend(fig[0, 1:2],
                      [CairoMakie.LineElement(color=:blue, linewidth=GRAMMAR.linewidth_main),
                       CairoMakie.LineElement(color=:green, linewidth=GRAMMAR.linewidth_main),
                       CairoMakie.LineElement(color=:red, linewidth=GRAMMAR.linewidth_main)],
                      ["κ<κ*", "κ≈κ*", "κ>κ*"];
                      orientation=:horizontal, framevisible=false, tellwidth=false)

    CairoMakie.save(outpath, fig)
    return fig
end

function plot_robustness_panels_2x2(
    outpath::AbstractString,
    time_grid::Vector{Float64},
    obs_dict::AbstractDict{String, T};
    title::String = "Robustness diagnostics",
) where {T}
    _load_cairo() || error("CairoMakie is required for plot_robustness_panels_2x2.")
    apply_plot_grammar!(CairoMakie)

    fig = CairoMakie.Figure(size=(900, 700), backgroundcolor=:white)
    ax1 = CairoMakie.Axis(fig[1, 1], xlabel="t", ylabel="bimodality", title="Bimodality")
    ax2 = CairoMakie.Axis(fig[1, 2], xlabel="t", ylabel="overlap", title="Overlap")
    ax3 = CairoMakie.Axis(fig[2, 1], xlabel="t", ylabel="variance", title="Variance")
    ax4 = CairoMakie.Axis(fig[2, 2], xlabel="t", ylabel="kurtosis", title="Kurtosis")

    colors = Dict("below" => :blue, "critical" => :green, "above" => :red)
    for (label, obs) in obs_dict
        color = get(colors, label, :black)
        if hasproperty(obs, :bimodality_mean)
            CairoMakie.lines!(ax1, time_grid, obs.bimodality_mean, color=color, linewidth=GRAMMAR.linewidth_main)
        end
        if hasproperty(obs, :overlap_mean) && !isempty(obs.overlap_times)
            CairoMakie.lines!(ax2, obs.overlap_times, obs.overlap_mean, color=color, linewidth=GRAMMAR.linewidth_main)
        end
        if hasproperty(obs, :variance_mean)
            CairoMakie.lines!(ax3, time_grid, obs.variance_mean, color=color, linewidth=GRAMMAR.linewidth_main)
        end
        if hasproperty(obs, :kurtosis_mean)
            CairoMakie.lines!(ax4, time_grid, obs.kurtosis_mean, color=color, linewidth=GRAMMAR.linewidth_main)
        end
    end

    CairoMakie.Legend(fig[0, 1:2],
                      [CairoMakie.LineElement(color=:blue, linewidth=GRAMMAR.linewidth_main),
                       CairoMakie.LineElement(color=:green, linewidth=GRAMMAR.linewidth_main),
                       CairoMakie.LineElement(color=:red, linewidth=GRAMMAR.linewidth_main)],
                      ["κ<κ*", "κ≈κ*", "κ>κ*"];
                      orientation=:horizontal, framevisible=false, tellwidth=false)

    CairoMakie.save(outpath, fig)
    return fig
end

end # module
