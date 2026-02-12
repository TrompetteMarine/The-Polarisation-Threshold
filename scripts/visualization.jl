module Visualization

using Statistics
using Distributions
using StatsBase

export DensityCI,
       DensitySummary,
       compute_density_ci,
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
) where {T}
    if _load_cairo()
        fig = CairoMakie.Figure(size=(1200, 1400), fontsize=14, backgroundcolor=:white)
        ax1 = CairoMakie.Axis(fig[1, 1], xlabel="u", ylabel="rho(u)", title="Below kappa*")
        ax2 = CairoMakie.Axis(fig[2, 1], xlabel="u", ylabel="rho(u)", title="At kappa*")
        ax3 = CairoMakie.Axis(fig[3, 1], xlabel="u", ylabel="rho(u)", title="Above kappa*")
        ax4 = CairoMakie.Axis(fig[4, 1], xlabel="t", ylabel="E|m(t)|", title="Mean trajectory (symmetry-aware)")

        colors = CairoMakie.cgrad(:viridis, length(snapshot_times), categorical=true)
        axes = Dict("below" => ax1, "critical" => ax2, "above" => ax3)
        for label in scenario_order
            data = density_data[label]
            ax = axes[label]
            for (i, tval) in enumerate(snapshot_times)
                mix = data.mixture_mean[i, :]
                aligned = data.aligned_mean[i, :]
                CairoMakie.lines!(ax, data.centers, mix, color=colors[i], linestyle=:dash,
                                  linewidth=1.2, label="mix t=$(round(tval, digits=0))")
                CairoMakie.lines!(ax, data.centers, aligned, color=colors[i],
                                  linewidth=2.0, label="aligned t=$(round(tval, digits=0))")
                if show_ci
                    CairoMakie.band!(ax, data.centers, data.aligned_ci_lower[i, :], data.aligned_ci_upper[i, :],
                                     color=(colors[i], 0.15))
                end
                if show_plus_minus && label == "above"
                    CairoMakie.lines!(ax, data.centers, data.plus_mean[i, :],
                                      color=(colors[i], 0.35), linewidth=1.0)
                    CairoMakie.lines!(ax, data.centers, data.minus_mean[i, :],
                                      color=(colors[i], 0.35), linewidth=1.0)
                end
            end
            CairoMakie.axislegend(ax; position=:rt, framevisible=false)
        end

        for (label, stats) in trajectory_stats
            if label == "below"
                CairoMakie.lines!(ax4, time_grid, stats.mean_abs, color=:blue, linewidth=2.0, label="below |m|")
                CairoMakie.band!(ax4, time_grid, stats.mean_abs_ci_lower, stats.mean_abs_ci_upper, color=(:blue, 0.15))
                CairoMakie.lines!(ax4, time_grid, stats.mean_traj, color=(:gray, 0.6), linewidth=1.0)
            elseif label == "critical"
                CairoMakie.lines!(ax4, time_grid, stats.mean_abs, color=:green, linewidth=2.0, label="critical |m|")
                CairoMakie.band!(ax4, time_grid, stats.mean_abs_ci_lower, stats.mean_abs_ci_upper, color=(:green, 0.15))
                CairoMakie.lines!(ax4, time_grid, stats.mean_traj, color=(:gray, 0.6), linewidth=1.0)
            elseif label == "above"
                CairoMakie.lines!(ax4, time_grid, stats.mean_abs, color=:red, linewidth=2.0, label="above |m|")
                CairoMakie.band!(ax4, time_grid, stats.mean_abs_ci_lower, stats.mean_abs_ci_upper, color=(:red, 0.15))
                CairoMakie.lines!(ax4, time_grid, stats.mean_traj, color=(:gray, 0.6), linewidth=1.0)
            end
        end
        CairoMakie.axislegend(ax4; position=:rb, framevisible=false)

        CairoMakie.save(output_path, fig)
        return
    end

    if _load_plots()
        Plots.default(fontfamily="Computer Modern")
        colors = Plots.cgrad(:viridis, length(snapshot_times), categorical=true)
        plt1 = Plots.plot(; xlabel="u", ylabel="rho(u)", title="Below kappa*")
        plt2 = Plots.plot(; xlabel="u", ylabel="rho(u)", title="At kappa*")
        plt3 = Plots.plot(; xlabel="u", ylabel="rho(u)", title="Above kappa*")
        plt4 = Plots.plot(; xlabel="t", ylabel="E|m(t)|", title="Mean trajectory (symmetry-aware)")

        for (i, tval) in enumerate(snapshot_times)
            Plots.plot!(plt1, density_data["below"].centers, density_data["below"].aligned_mean[i, :];
                        color=colors[i], label="aligned t=$(round(tval, digits=0))")
            Plots.plot!(plt1, density_data["below"].centers, density_data["below"].mixture_mean[i, :];
                        color=colors[i], linestyle=:dash, label="mix t=$(round(tval, digits=0))")
            Plots.plot!(plt2, density_data["critical"].centers, density_data["critical"].aligned_mean[i, :];
                        color=colors[i], label="aligned t=$(round(tval, digits=0))")
            Plots.plot!(plt2, density_data["critical"].centers, density_data["critical"].mixture_mean[i, :];
                        color=colors[i], linestyle=:dash, label="mix t=$(round(tval, digits=0))")
            Plots.plot!(plt3, density_data["above"].centers, density_data["above"].aligned_mean[i, :];
                        color=colors[i], label="aligned t=$(round(tval, digits=0))")
            Plots.plot!(plt3, density_data["above"].centers, density_data["above"].mixture_mean[i, :];
                        color=colors[i], linestyle=:dash, label="mix t=$(round(tval, digits=0))")
        end

        for (label, stats) in trajectory_stats
            if label == "below"
                Plots.plot!(plt4, time_grid, stats.mean_abs; color=:blue, label="below |m|")
                Plots.plot!(plt4, time_grid, stats.mean_traj; color=:gray, label="")
            elseif label == "critical"
                Plots.plot!(plt4, time_grid, stats.mean_abs; color=:green, label="critical |m|")
                Plots.plot!(plt4, time_grid, stats.mean_traj; color=:gray, label="")
            elseif label == "above"
                Plots.plot!(plt4, time_grid, stats.mean_abs; color=:red, label="above |m|")
                Plots.plot!(plt4, time_grid, stats.mean_traj; color=:gray, label="")
            end
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
    )
end

function plot_phase_diagram(output_path::String, sweep_df)
    if _load_cairo()
        fig = CairoMakie.Figure(size=(900, 600), fontsize=14, backgroundcolor=:white)
        ax = CairoMakie.Axis(fig[1, 1], xlabel="kappa / kappa*", ylabel="lambda1",
                             title="Leading eigenvalue vs coupling")
        x = sweep_df.kappa_ratio
        y = sweep_df.lambda_mean
        ylow = sweep_df.lambda_ci_lower
        yhigh = sweep_df.lambda_ci_upper
        CairoMakie.errorbars!(ax, x, y, y .- ylow, yhigh .- y; whiskerwidth=6)
        CairoMakie.scatter!(ax, x, y, color=:black)
        CairoMakie.hlines!(ax, [0.0], color=:gray, linestyle=:dash)
        CairoMakie.vlines!(ax, [1.0], color=:gray, linestyle=:dash)
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
        fig = CairoMakie.Figure(size=(1100, 800), fontsize=14, backgroundcolor=:white)
        ax1 = CairoMakie.Axis(fig[1, 1], xlabel="t", ylabel="variance", title="Variance")
        ax2 = CairoMakie.Axis(fig[1, 2], xlabel="t", ylabel="kurtosis", title="Kurtosis")
        ax3 = CairoMakie.Axis(fig[2, 1], xlabel="t", ylabel="bimodality", title="Bimodality coefficient")
        ax4 = CairoMakie.Axis(fig[2, 2], xlabel="t", ylabel="overlap", title="Overlap integral")

        colors = Dict("below" => :blue, "critical" => :green, "above" => :red)
        for (label, obs) in observables
            color = colors[label]
            if haskey(variance_stats, label)
                stats = variance_stats[label]
                CairoMakie.lines!(ax1, time_grid, stats.var_traj, color=color, label=label)
                CairoMakie.band!(ax1, time_grid, stats.var_ci_lower, stats.var_ci_upper, color=(color, 0.15))
            end
            CairoMakie.lines!(ax2, time_grid, obs.kurtosis_mean, color=color, label=label)
            CairoMakie.band!(ax2, time_grid, obs.kurtosis_ci_lower, obs.kurtosis_ci_upper, color=(color, 0.15))

            CairoMakie.lines!(ax3, time_grid, obs.bimodality_mean, color=color, label=label)
            CairoMakie.band!(ax3, time_grid, obs.bimodality_ci_lower, obs.bimodality_ci_upper, color=(color, 0.15))

            if !isempty(obs.overlap_times)
                CairoMakie.lines!(ax4, obs.overlap_times, obs.overlap_mean, color=color, label=label)
            end
        end

        CairoMakie.axislegend(ax1; position=:rb, framevisible=false)
        CairoMakie.axislegend(ax2; position=:rb, framevisible=false)
        CairoMakie.axislegend(ax3; position=:rb, framevisible=false)
        CairoMakie.axislegend(ax4; position=:rb, framevisible=false)
        CairoMakie.save(output_path, fig)
        return
    end

    if _load_plots()
        Plots.default(fontfamily="Computer Modern")
        plt1 = Plots.plot(; xlabel="t", ylabel="variance", title="Variance")
        plt2 = Plots.plot(; xlabel="t", ylabel="kurtosis", title="Kurtosis")
        plt3 = Plots.plot(; xlabel="t", ylabel="bimodality", title="Bimodality coefficient")
        plt4 = Plots.plot(; xlabel="t", ylabel="overlap", title="Overlap integral")

        colors = Dict("below" => :blue, "critical" => :green, "above" => :red)
        for (label, obs) in observables
            color = colors[label]
            if haskey(variance_stats, label)
                stats = variance_stats[label]
                Plots.plot!(plt1, time_grid, stats.var_traj; color=color, label=label)
            end
            Plots.plot!(plt2, time_grid, obs.kurtosis_mean; color=color, label=label)
            Plots.plot!(plt3, time_grid, obs.bimodality_mean; color=color, label=label)
            if !isempty(obs.overlap_times)
                Plots.plot!(plt4, obs.overlap_times, obs.overlap_mean; color=color, label=label)
            end
        end
        fig = Plots.plot(plt1, plt2, plt3, plt4; layout=(2, 2), size=(1000, 700))
        Plots.savefig(fig, output_path)
        return
    end

    error("No plotting backend available (CairoMakie or Plots).")
end

end # module
