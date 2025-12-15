#!/usr/bin/env julia
"""
Welfare Surface Explorer - Volcano-style visualization

Inspired by CairoMakie's volcano contour example, this script creates
clean filled contour plots with colorbars for exploring the welfare
landscape across (κ, V*) parameter space.

Generates simple, publication-ready surface visualizations similar to
topographic maps, emphasizing the overall structure of the welfare function.
"""

using BeliefSim
using BeliefSim.Types
using BeliefSim.OUResets
using BeliefSim.Stats
using Plots
using Statistics
using Random
using Printf

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

const DEBUG = false

# Color schemes for different views
const COLORSCHEMES = (
    terrain = :terrain,      # Topographic-like for welfare surfaces
    thermal = :thermal,      # Heat map style
    deep = :deep,           # Ocean-inspired
    dense = :dense,         # High-contrast
    balance = :balance      # Diverging for differences
)

# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

"""
    median_filter(x, window_size=3)

Smooth ridge lines using median filter.
"""
function median_filter(x, window_size=3)
    n = length(x)
    if n < window_size
        return x
    end
    filtered = similar(x)
    half = div(window_size, 2)
    for i in 1:n
        i_start = max(1, i - half)
        i_end = min(n, i + half)
        filtered[i] = median(x[i_start:i_end])
    end
    return filtered
end

"""
    save_both(plt, stem)

Save plot as both PDF and PNG.
"""
function save_both(plt, stem)
    savefig(plt, "figs/volcano/$(stem).pdf")
    savefig(plt, "figs/volcano/$(stem).png")
end

"""
    compute_welfare_surface(p; α=0.25, β=0.8)

Compute welfare surfaces over (κ, V*) grid by varying σ.
Returns data ready for surface visualization.
"""
function compute_welfare_surface(p; α=0.25, β=0.8)
    DEBUG && println("   → Computing welfare surfaces...")

    # Grid definition
    κgrid = collect(range(0.0, 2.2, length=50))  # Balanced resolution for good contours
    σgrid = collect(range(0.6, 1.3, length=15))  # Balanced sampling of dispersion

    nκ = length(κgrid)
    nσ = length(σgrid)

    # Preallocate surfaces
    Vstar_vals = similar(σgrid)
    W_dec_surface = Matrix{Float64}(undef, nσ, nκ)
    W_soc_surface = similar(W_dec_surface)
    κ_dec_ridge = similar(σgrid)
    κ_soc_ridge = similar(σgrid)

    # Compute surfaces
    for (j, σval) in enumerate(σgrid)
        pσ = Params(λ=p.λ, σ=σval, Θ=p.Θ, c0=p.c0, hazard=p.hazard)

        # Estimate stationary dispersion
        Vstar_vals[j] = estimate_Vstar(pσ; N=6_000, T=220.0, dt=0.01,
                                        burn_in=80.0, seed=500 + j)

        # Compute welfare across κ values
        surf = compute_welfare_curves(pσ, κgrid; α=α, β=β, N=5_000,
                                      T=180.0, dt=0.01, seed=800 + j)

        W_dec_surface[j, :] .= surf.W_dec
        W_soc_surface[j, :] .= surf.W_soc
        κ_dec_ridge[j] = κgrid[argmax(surf.W_dec)]
        κ_soc_ridge[j] = κgrid[argmax(surf.W_soc)]
    end

    # Sort by V* for monotone axis
    perm = sortperm(Vstar_vals)
    Vstar_sorted = Vstar_vals[perm]
    W_dec_surface = W_dec_surface[perm, :]
    W_soc_surface = W_soc_surface[perm, :]
    κ_dec_ridge = κ_dec_ridge[perm]
    κ_soc_ridge = κ_soc_ridge[perm]

    # Smooth ridges
    κ_dec_ridge = median_filter(κ_dec_ridge, 3)
    κ_soc_ridge = median_filter(κ_soc_ridge, 3)

    # Filter non-finite rows
    finite_mask = isfinite.(Vstar_sorted)
    if !all(finite_mask)
        Vstar_sorted = Vstar_sorted[finite_mask]
        W_dec_surface = W_dec_surface[finite_mask, :]
        W_soc_surface = W_soc_surface[finite_mask, :]
        κ_dec_ridge = κ_dec_ridge[finite_mask]
        κ_soc_ridge = κ_soc_ridge[finite_mask]
    end

    return (
        κgrid = κgrid,
        Vstar_sorted = Vstar_sorted,
        W_dec_surface = W_dec_surface,
        W_soc_surface = W_soc_surface,
        κ_dec_ridge = κ_dec_ridge,
        κ_soc_ridge = κ_soc_ridge
    )
end

# ----------------------------------------------------------------------
# Volcano-style visualization functions
# ----------------------------------------------------------------------

"""
    plot_welfare_volcano(κgrid, V, W, ridge_line, title_str;
                        colorscheme=:terrain, nlevels=15)

Create a clean filled contour plot with colorbar, volcano-style.
Similar to the CairoMakie volcano example but using Plots.jl.
"""
function plot_welfare_volcano(κgrid, V, W, ridge_line, title_str;
                             colorscheme=:terrain, nlevels=15)
    # Compute levels from finite values
    finite_vals = filter(isfinite, vec(W))
    if isempty(finite_vals)
        clims = (-1.0, 1.0)
        levels = range(clims[1], clims[2]; length=nlevels)
    else
        wmin, wmax = quantile(finite_vals, [0.02, 0.98])
        clims = (wmin, wmax)
        levels = range(wmin, wmax; length=nlevels)
    end

    # Create figure with colorbar
    plt = plot(size=(900, 600), dpi=300,
               layout=@layout([a{0.85w} b{0.15w}]),
               margin=8Plots.mm)

    # Main contour plot - filled contours
    contourf!(plt[1], κgrid, V, W;
              levels=levels,
              c=colorscheme,
              clims=clims,
              xlabel="Social coupling κ",
              ylabel="Stationary dispersion V*",
              title=title_str,
              fillalpha=1.0,
              linewidth=0,
              nan_color=:white,
              legend=false,
              colorbar=false,
              titlefontsize=14,
              guidefontsize=12)

    # Add contour isolines (for topographic effect)
    contour!(plt[1], κgrid, V, W;
             levels=levels,
             color=:black,
             linewidth=0.8,
             linealpha=0.4,
             label=false,
             colorbar=false)

    # Add ridge line
    plot!(plt[1], ridge_line, V;
          color=:red, linewidth=3.5, linestyle=:solid,
          label=false, alpha=0.9)

    # Colorbar (as a separate heatmap)
    colorbar_vals = reshape(range(clims[1], clims[2]; length=100), :, 1)
    heatmap!(plt[2], [1], range(clims[1], clims[2]; length=100), colorbar_vals;
             c=colorscheme,
             clims=clims,
             colorbar=false,
             axis=false,
             ticks=false,
             legend=false,
             framestyle=:none)

    # Add colorbar labels
    yticks!(plt[2],
           range(clims[1], clims[2]; length=5),
           [@sprintf("%.2f", x) for x in range(clims[1], clims[2]; length=5)])
    plot!(plt[2];
          ylabel="Welfare",
          guidefontsize=11,
          framestyle=:box,
          grid=false,
          showaxis=:y,
          ytickfontsize=9)

    return plt
end

"""
    plot_difference_volcano(κgrid, V, ΔW, ridge_dec, ridge_soc;
                           nlevels=15)

Create volcano-style plot for welfare difference with symmetric colorbar.
"""
function plot_difference_volcano(κgrid, V, ΔW, ridge_dec, ridge_soc;
                                nlevels=15)
    # Symmetric levels around zero
    finite_vals = filter(isfinite, vec(ΔW))
    if isempty(finite_vals)
        clims = (-1.0, 1.0)
    else
        vmax = quantile(abs.(finite_vals), 0.95)
        clims = (-vmax, vmax)
    end
    levels = range(clims[1], clims[2]; length=nlevels)

    # Create figure with colorbar
    plt = plot(size=(900, 600), dpi=300,
               layout=@layout([a{0.85w} b{0.15w}]),
               margin=8Plots.mm)

    # Main contour plot - filled contours
    contourf!(plt[1], κgrid, V, ΔW;
              levels=levels,
              c=:balance,
              clims=clims,
              xlabel="Social coupling κ",
              ylabel="Stationary dispersion V*",
              title="Welfare Difference (Planner − Decentralised)",
              fillalpha=1.0,
              linewidth=0,
              nan_color=:white,
              legend=false,
              colorbar=false,
              titlefontsize=14,
              guidefontsize=12)

    # Add contour isolines (for topographic effect)
    contour!(plt[1], κgrid, V, ΔW;
             levels=levels,
             color=:black,
             linewidth=0.6,
             linealpha=0.35,
             label=false,
             colorbar=false)

    # Zero contour (emphasized)
    contour!(plt[1], κgrid, V, ΔW;
             levels=[0.0],
             color=:black,
             linewidth=2.5,
             linestyle=:solid,
             label=false)

    # Ridge lines
    plot!(plt[1], ridge_dec, V;
          color=:red, linewidth=2.8, linestyle=:dash,
          label=false, alpha=0.8)
    plot!(plt[1], ridge_soc, V;
          color=:cyan, linewidth=2.8, linestyle=:dash,
          label=false, alpha=0.8)

    # Colorbar
    colorbar_vals = reshape(range(clims[1], clims[2]; length=100), :, 1)
    heatmap!(plt[2], [1], range(clims[1], clims[2]; length=100), colorbar_vals;
             c=:balance,
             clims=clims,
             colorbar=false,
             axis=false,
             ticks=false,
             legend=false,
             framestyle=:none)

    # Colorbar labels
    yticks!(plt[2],
           range(clims[1], clims[2]; length=5),
           [@sprintf("%.2f", x) for x in range(clims[1], clims[2]; length=5)])
    plot!(plt[2];
          ylabel="ΔW",
          guidefontsize=11,
          framestyle=:box,
          grid=false,
          showaxis=:y,
          ytickfontsize=9)

    return plt
end

"""
    plot_combined_volcano(data)

Create a 2×2 grid showing different colorscheme views of the welfare surfaces.
"""
function plot_combined_volcano(data)
    κgrid = data.κgrid
    V = data.Vstar_sorted
    W_dec = data.W_dec_surface
    W_soc = data.W_soc_surface
    κ_dec = data.κ_dec_ridge
    κ_soc = data.κ_soc_ridge
    ΔW = W_soc .- W_dec

    # Compute consistent color limits
    finite_dec = filter(isfinite, vec(W_dec))
    finite_soc = filter(isfinite, vec(W_soc))
    wmin = minimum([quantile(finite_dec, 0.02), quantile(finite_soc, 0.02)])
    wmax = maximum([quantile(finite_dec, 0.98), quantile(finite_soc, 0.98)])
    levels = range(wmin, wmax; length=18)

    plt = plot(layout=(2, 2), size=(1400, 1200), dpi=300, margin=8Plots.mm)

    # Panel 1: Decentralised (terrain colorscheme)
    contourf!(plt[1], κgrid, V, W_dec;
              levels=levels, c=:terrain, clims=(wmin, wmax),
              xlabel="κ", ylabel="V*",
              title="Decentralised (terrain)",
              fillalpha=1.0, linewidth=0, nan_color=:white,
              colorbar=true, colorbar_title="Welfare")
    plot!(plt[1], κ_dec, V; color=:black, linewidth=2.5, label=false)

    # Panel 2: Planner (terrain colorscheme)
    contourf!(plt[2], κgrid, V, W_soc;
              levels=levels, c=:terrain, clims=(wmin, wmax),
              xlabel="κ", ylabel="V*",
              title="Planner (terrain)",
              fillalpha=1.0, linewidth=0, nan_color=:white,
              colorbar=true, colorbar_title="Welfare")
    plot!(plt[2], κ_soc, V; color=:black, linewidth=2.5, label=false)

    # Panel 3: Decentralised (thermal colorscheme)
    contourf!(plt[3], κgrid, V, W_dec;
              levels=levels, c=:thermal, clims=(wmin, wmax),
              xlabel="κ", ylabel="V*",
              title="Decentralised (thermal)",
              fillalpha=1.0, linewidth=0, nan_color=:white,
              colorbar=true, colorbar_title="Welfare")
    plot!(plt[3], κ_dec, V; color=:white, linewidth=2.5, label=false, alpha=0.9)

    # Panel 4: Difference (balance colorscheme)
    finite_diff = filter(isfinite, vec(ΔW))
    diff_max = quantile(abs.(finite_diff), 0.95)
    diff_levels = range(-diff_max, diff_max; length=18)

    contourf!(plt[4], κgrid, V, ΔW;
              levels=diff_levels, c=:balance, clims=(-diff_max, diff_max),
              xlabel="κ", ylabel="V*",
              title="Difference (Planner − Decentralised)",
              fillalpha=1.0, linewidth=0, nan_color=:white,
              colorbar=true, colorbar_title="ΔW")
    contour!(plt[4], κgrid, V, ΔW;
             levels=[0.0], color=:black, linewidth=2.5, label=false)
    plot!(plt[4], κ_dec, V; color=:red, linewidth=2.0, linestyle=:dash, label=false, alpha=0.7)
    plot!(plt[4], κ_soc, V; color=:cyan, linewidth=2.0, linestyle=:dash, label=false, alpha=0.7)

    return plt
end

# ----------------------------------------------------------------------
# Main execution
# ----------------------------------------------------------------------

"""
Generate volcano-style welfare surface visualizations.

Creates clean filled contour plots with colorbars inspired by the
CairoMakie volcano example, emphasizing the topographic structure
of the welfare landscape.
"""
function main()
    Random.seed!(2025)
    mkpath("figs/volcano")

    println("Generating Volcano-Style Welfare Surfaces")
    println("="^60)

    # Baseline parameters
    p = Params(λ=0.65, σ=1.85, Θ=2.0, c0=1.0, hazard=StepHazard(0.6))
    α = 0.25
    β = 0.8

    # Compute welfare surfaces
    data = compute_welfare_surface(p; α=α, β=β)

    # Generate individual volcano-style plots
    println("Creating volcano-style visualizations...")

    # Decentralised welfare (terrain colorscheme)
    plt_dec_terrain = plot_welfare_volcano(
        data.κgrid, data.Vstar_sorted, data.W_dec_surface,
        data.κ_dec_ridge, "Decentralised Welfare";
        colorscheme=:terrain, nlevels=18
    )
    save_both(plt_dec_terrain, "decentralised_terrain")
    println("✓ Saved: decentralised_terrain.pdf (volcano-style)")

    # Decentralised welfare (thermal colorscheme)
    plt_dec_thermal = plot_welfare_volcano(
        data.κgrid, data.Vstar_sorted, data.W_dec_surface,
        data.κ_dec_ridge, "Decentralised Welfare";
        colorscheme=:thermal, nlevels=18
    )
    save_both(plt_dec_thermal, "decentralised_thermal")
    println("✓ Saved: decentralised_thermal.pdf (volcano-style)")

    # Planner welfare (terrain colorscheme)
    plt_soc_terrain = plot_welfare_volcano(
        data.κgrid, data.Vstar_sorted, data.W_soc_surface,
        data.κ_soc_ridge, "Planner Welfare";
        colorscheme=:terrain, nlevels=18
    )
    save_both(plt_soc_terrain, "planner_terrain")
    println("✓ Saved: planner_terrain.pdf (volcano-style)")

    # Planner welfare (deep colorscheme)
    plt_soc_deep = plot_welfare_volcano(
        data.κgrid, data.Vstar_sorted, data.W_soc_surface,
        data.κ_soc_ridge, "Planner Welfare";
        colorscheme=:deep, nlevels=18
    )
    save_both(plt_soc_deep, "planner_deep")
    println("✓ Saved: planner_deep.pdf (volcano-style)")

    # Welfare difference
    ΔW = data.W_soc_surface .- data.W_dec_surface
    plt_diff = plot_difference_volcano(
        data.κgrid, data.Vstar_sorted, ΔW,
        data.κ_dec_ridge, data.κ_soc_ridge;
        nlevels=18
    )
    save_both(plt_diff, "difference_balance")
    println("✓ Saved: difference_balance.pdf (volcano-style)")

    # Combined 2×2 grid with different colorschemes
    plt_combined = plot_combined_volcano(data)
    save_both(plt_combined, "combined_colorschemes")
    println("✓ Saved: combined_colorschemes.pdf (2×2 comparison)")

    println("="^60)
    println("Volcano-style surface visualization complete!")
    println("All figures saved to: figs/volcano/")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
