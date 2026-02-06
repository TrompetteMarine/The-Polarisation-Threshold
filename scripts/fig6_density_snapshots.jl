#!/usr/bin/env julia
# =============================================================================
# Figure 6: Density snapshots below vs above kappa* with mean time series
# =============================================================================

using Pkg; Pkg.activate(".")

using Random
using Statistics
using Printf
using Dates
using CSV
using DataFrames
using StatsBase

using BeliefSim
using BeliefSim.Types
using BeliefSim.Stats
using BeliefSim.Model: euler_maruyama_step!, reset_step!

# Optional plotting backends
CAIRO_AVAILABLE = false
try
    using CairoMakie
    CAIRO_AVAILABLE = true
catch
    @warn "CairoMakie not available, falling back to Plots.jl"
    using Plots
end

function simulate_snapshots(p::Params; κ::Float64, N::Int, T::Float64, dt::Float64,
                            seed::Int, snapshot_times::Vector{Float64}, mt_stride::Int)
    Random.seed!(seed)

    steps = Int(round(T / dt))
    time_grid = collect(0.0:dt:T)

    # Initial condition: OU stationary variance (no bias)
    u = randn(N) .* (p.σ / sqrt(2 * p.λ))

    snapshot_indices = [clamp(Int(round(t / dt)) + 1, 1, length(time_grid)) for t in snapshot_times]
    snapshots = Vector{Vector{Float64}}(undef, length(snapshot_indices))

    # Store snapshot at t=0 if requested
    next_snap = 1
    if snapshot_indices[next_snap] == 1
        snapshots[next_snap] = copy(u)
        next_snap += 1
    end

    mt_times = Float64[0.0]
    mt_values = Float64[mean(u)]

    for step in 1:steps
        gbar = mean(u)
        euler_maruyama_step!(u, κ, gbar, p, dt)
        reset_step!(u, p, dt)

        idx = step + 1
        if next_snap <= length(snapshot_indices) && idx == snapshot_indices[next_snap]
            snapshots[next_snap] = copy(u)
            next_snap += 1
        end

        if step % mt_stride == 0
            push!(mt_times, step * dt)
            push!(mt_values, mean(u))
        end
    end

    return snapshots, mt_times, mt_values
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

function main()
    # Baseline parameters matching fig3_bifurcation
    λ = 0.85
    σ = 0.8
    Θ = 2.0
    c0 = 0.8
    nu0 = 10.6

    N = 20000
    T = 200.0
    dt = 0.01
    burn_in = 120.0
    seed = 2025

    snapshot_times = [0.0, 10.0, 20.0, 40.0, 80.0, 160.0, T]
    bins = 240
    mt_stride = max(1, Int(round(0.05 / dt)))

    outdir = "outputs/fig6_density_snapshots"
    figdir = "figs"
    mkpath(outdir)
    mkpath(figdir)

    p = Params(λ=λ, σ=σ, Θ=Θ, c0=c0, hazard=StepHazard(nu0))

    println("=" ^ 72)
    println("FIGURE 6: DENSITY SNAPSHOTS BELOW/ABOVE κ*")
    println("=" ^ 72)
    @printf("λ=%.3f, σ=%.3f, Θ=%.3f, c0=%.3f, ν0=%.3f\n", λ, σ, Θ, c0, nu0)
    @printf("N=%d, T=%.1f, dt=%.3f, burn_in=%.1f\n", N, T, dt, burn_in)
    println("Snapshot times: $(snapshot_times)")
    println("Bins: $bins")
    println("Mean time series stride: $mt_stride steps (Δt=$(mt_stride * dt))")

    # Estimate V* and κ*
    println("Estimating V* and κ*...")
    Vstar = estimate_Vstar(p; N=N, T=350.0, dt=dt, burn_in=burn_in, seed=seed)
    κstar = critical_kappa(p; Vstar=Vstar)

    κ_minus = 0.8 * κstar
    κ_plus = 1.2 * κstar
    @printf("V* = %.6f\n", Vstar)
    @printf("κ* = %.6f, κ- = %.6f, κ+ = %.6f\n", κstar, κ_minus, κ_plus)

    println("Simulating below threshold...")
    snaps_below, mt_t_below, mt_below = simulate_snapshots(p; κ=κ_minus, N=N, T=T, dt=dt,
                                                           seed=seed + 1, snapshot_times=snapshot_times,
                                                           mt_stride=mt_stride)

    println("Simulating above threshold...")
    snaps_above, mt_t_above, mt_above = simulate_snapshots(p; κ=κ_plus, N=N, T=T, dt=dt,
                                                           seed=seed + 2, snapshot_times=snapshot_times,
                                                           mt_stride=mt_stride)

    # Shared bin edges across all snapshots for comparability
    all_vals = vcat(reduce(vcat, snaps_below), reduce(vcat, snaps_above))
    umin = minimum(all_vals)
    umax = maximum(all_vals)
    pad = 0.05 * max(1e-6, umax - umin)
    edges = collect(range(umin - pad, umax + pad; length=bins + 1))

    # Save mean time series
    df_below = DataFrame(t=mt_t_below, m=mt_below)
    df_above = DataFrame(t=mt_t_above, m=mt_above)
    CSV.write(joinpath(outdir, "mt_below.csv"), df_below)
    CSV.write(joinpath(outdir, "mt_above.csv"), df_above)

    # Save histograms
    hist_paths_below = String[]
    hist_paths_above = String[]
    for (idx, t) in enumerate(snapshot_times)
        centers, dens = histogram_density(snaps_below[idx], edges)
        tstr = replace(@sprintf("%.2f", t), "." => "p")
        path = joinpath(outdir, "hist_below_t$(tstr).csv")
        CSV.write(path, DataFrame(bin_center=centers, density=dens))
        push!(hist_paths_below, path)

        centers2, dens2 = histogram_density(snaps_above[idx], edges)
        tstr2 = replace(@sprintf("%.2f", t), "." => "p")
        path2 = joinpath(outdir, "hist_above_t$(tstr2).csv")
        CSV.write(path2, DataFrame(bin_center=centers2, density=dens2))
        push!(hist_paths_above, path2)
    end

    # Write summary
    summary_path = joinpath(outdir, "summary.txt")
    open(summary_path, "w") do io
        println(io, "Figure 6: Density snapshots below/above κ*")
        println(io, "Timestamp: $(Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))")
        @printf(io, "λ=%.4f, σ=%.4f, Θ=%.4f, c0=%.4f, ν0=%.4f\n", λ, σ, Θ, c0, nu0)
        @printf(io, "N=%d, T=%.1f, dt=%.3f, burn_in=%.1f\n", N, T, dt, burn_in)
        @printf(io, "V* = %.6f\n", Vstar)
        @printf(io, "κ* = %.6f, κ- = %.6f, κ+ = %.6f\n", κstar, κ_minus, κ_plus)
        println(io, "Snapshot times: $(snapshot_times)")
        println(io, "Bins: $bins")
        println(io, "Mean time series stride: $mt_stride steps (Δt=$(mt_stride * dt))")
        println(io, "Histogram edges: [$(edges[1]), $(edges[end])] (uniform)")
        println(io, "Mean time series: mt_below.csv, mt_above.csv")
        println(io, "Histogram files below:")
        for pth in hist_paths_below
            println(io, "  - $pth")
        end
        println(io, "Histogram files above:")
        for pth in hist_paths_above
            println(io, "  - $pth")
        end
    end

    # Plot
    if CAIRO_AVAILABLE
        fig = Figure(size=(1000, 900), fontsize=14, backgroundcolor=:white)
        ax1 = Axis(fig[1, 1], xlabel="u", ylabel="ρ(u)", title="Below κ*")
        ax2 = Axis(fig[2, 1], xlabel="u", ylabel="ρ(u)", title="Above κ*")
        ax3 = Axis(fig[3, 1], xlabel="t", ylabel="m(t)", title="Mean trajectory")

        colors = collect(cgrad(:viridis, length(snapshot_times), categorical=true))
        for (i, t) in enumerate(snapshot_times)
            centers, dens = histogram_density(snaps_below[i], edges)
            lines!(ax1, centers, dens, color=colors[i], label=@sprintf("t=%.0f", t))
            centers2, dens2 = histogram_density(snaps_above[i], edges)
            lines!(ax2, centers2, dens2, color=colors[i], label=@sprintf("t=%.0f", t))
        end

        lines!(ax3, mt_t_below, mt_below, color=:blue, linewidth=2.0, label="κ-")
        lines!(ax3, mt_t_above, mt_above, color=:red, linewidth=2.0, label="κ+")

        axislegend(ax1; position=:rb, framevisible=false)
        axislegend(ax2; position=:rb, framevisible=false)
        axislegend(ax3; position=:rb, framevisible=false)

        fig_path = joinpath(figdir, "fig6_density_snapshots.pdf")
        save(fig_path, fig)
    else
        default(fontfamily="Computer Modern")
        colors = cgrad(:viridis, length(snapshot_times), categorical=true)

        plt1 = plot(; xlabel="u", ylabel="ρ(u)", title="Below κ*", legend=:right)
        for (i, t) in enumerate(snapshot_times)
            centers, dens = histogram_density(snaps_below[i], edges)
            plot!(plt1, centers, dens; color=colors[i], label=@sprintf("t=%.0f", t))
        end

        plt2 = plot(; xlabel="u", ylabel="ρ(u)", title="Above κ*", legend=:right)
        for (i, t) in enumerate(snapshot_times)
            centers, dens = histogram_density(snaps_above[i], edges)
            plot!(plt2, centers, dens; color=colors[i], label=@sprintf("t=%.0f", t))
        end

        plt3 = plot(mt_t_below, mt_below; xlabel="t", ylabel="m(t)", title="Mean trajectory",
                    color=:blue, linewidth=2.0, label="κ-")
        plot!(plt3, mt_t_above, mt_above; color=:red, linewidth=2.0, label="κ+")

        fig_path = joinpath(figdir, "fig6_density_snapshots.pdf")
        fig = plot(plt1, plt2, plt3; layout=(3,1), size=(900, 900))
        savefig(fig, fig_path)
    end

    println("Saved: $(joinpath(figdir, "fig6_density_snapshots.pdf"))")
    println("Saved: $summary_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
