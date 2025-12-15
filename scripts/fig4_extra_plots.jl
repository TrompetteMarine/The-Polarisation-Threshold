#!/usr/bin/env julia
using BeliefSim
using BeliefSim.Types
using BeliefSim.OUResets
using BeliefSim.Stats
using Plots
using Statistics
using Random

# Debug mode: set to true to enable detailed diagnostics
const DEBUG = false

# ----------------------------------------------------------------------
# Shared utilities (consistent with fig4_welfare.jl)
# ----------------------------------------------------------------------

"""
    finite_clims(A...; q=(0.05, 0.95))

Compute robust color limits from finite values across multiple arrays.
Uses quantiles to avoid outlier contamination.
"""
function finite_clims(A...; q=(0.05, 0.95))
    vals = Float64[]
    for X in A
        append!(vals, filter(isfinite, vec(X)))
    end
    if isempty(vals)
        DEBUG && @info "finite_clims: no finite values; defaulting to (-1, 1)"
        return (-1.0, 1.0)
    end
    qv = quantile(vals, collect(q))
    if qv[1] ≈ qv[2]
        lo = minimum(vals)
        hi = maximum(vals)
        if lo == hi
            δ = max(abs(lo), 1.0)
            return (lo - δ, hi + δ)
        else
            padding = 0.05 * (hi - lo)
            return (lo - padding, hi + padding)
        end
    end
    return (qv[1], qv[2])
end

"""
    median_filter(x, window_size=3)

Apply a median filter to smooth ridge lines extracted via argmax.
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

Save plot as both PDF and PNG with stem `figs/<stem>`.
"""
function save_both(plt, stem)
    savefig(plt, "figs/$(stem).pdf")
    savefig(plt, "figs/$(stem).png")
end

"""
    augmented_xlims(xgrid, xdata...; pad_frac=0.08)

Compute augmented x-axis limits with generous padding.
"""
function augmented_xlims(xgrid, xdata...; pad_frac=0.08)
    xmin = minimum(xgrid)
    xmax_data = maximum([maximum(filter(isfinite, x)) for x in xdata])
    xmax = min(maximum(xgrid), xmax_data + 0.3)
    range_width = xmax - xmin
    return (xmin - pad_frac * range_width, xmax + pad_frac * range_width)
end

"""
    augmented_ylims(ygrid; pad_frac=0.06)

Compute augmented y-axis limits with generous padding.
"""
function augmented_ylims(ygrid; pad_frac=0.06)
    ymin, ymax = extrema(filter(isfinite, ygrid))
    range_height = ymax - ymin
    return (ymin - pad_frac * range_height, ymax + pad_frac * range_height)
end

"""
    build_grids()

Construct κ and σ grids for welfare surface computation.
Increased resolution for smoother contours.
"""
function build_grids()
    κgrid = collect(range(0.0, 2.2, length=60))
    σgrid = collect(range(0.6, 1.3, length=15))
    return κgrid, σgrid
end

"""
    compute_welfare_data(p; α=0.25, β=0.8)

Compute welfare surfaces, ridges, and auxiliary data for plotting.
Returns all necessary arrays for generating supplementary figures.
"""
function compute_welfare_data(p; α=0.25, β=0.8)
    DEBUG && println("   → Computing welfare data...")

    κgrid, σgrid = build_grids()
    nκ = length(κgrid)
    nσ = length(σgrid)

    # 1D baseline welfare for reference
    welfare_1d = compute_welfare_curves(p, κgrid; α=α, β=β, N=8_000,
                                        T=260.0, dt=0.01, seed=99)

    # Build 2D surfaces by varying σ
    Vstar_vals = similar(σgrid)
    W_dec_surface = Matrix{Float64}(undef, nσ, nκ)
    W_soc_surface = similar(W_dec_surface)
    κ_dec_ridge = similar(σgrid)
    κ_soc_ridge = similar(σgrid)

    for (j, σval) in enumerate(σgrid)
        pσ = Params(λ=p.λ, σ=σval, Θ=p.Θ, c0=p.c0, hazard=p.hazard)
        Vstar_vals[j] = estimate_Vstar(pσ; N=6_000, T=220.0, dt=0.01,
                                        burn_in=80.0, seed=500 + j)
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

    # Smooth ridges and filter non-finite rows
    κ_dec_ridge = median_filter(κ_dec_ridge, 3)
    κ_soc_ridge = median_filter(κ_soc_ridge, 3)

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
        σgrid = σgrid,
        Vstar_sorted = Vstar_sorted,
        welfare_1d = welfare_1d,
        W_dec_surface = W_dec_surface,
        W_soc_surface = W_soc_surface,
        κ_dec_ridge = κ_dec_ridge,
        κ_soc_ridge = κ_soc_ridge
    )
end

# ----------------------------------------------------------------------
# Supplementary figures (online appendix / presentations)
# ----------------------------------------------------------------------

"""
Figure: Externality wedge Δκ(V*) = κ^soc(V*) - κ^dec(V*).
Shows how the optimal policy gap varies with dispersion.
"""
function plot_externality_ridge(data)
    DEBUG && println("   → Creating externality ridge plot...")

    κ_dec = data.κ_dec_ridge
    κ_soc = data.κ_soc_ridge
    V = data.Vstar_sorted
    wedge = κ_soc .- κ_dec
    max_idx = argmax(abs.(wedge))

    xlim = augmented_ylims(V; pad_frac=0.08)
    ylim = augmented_ylims(wedge; pad_frac=0.12)

    plt = plot(V, wedge;
               xlabel="Stationary dispersion V*",
               ylabel="Optimal κ gap: κ^{soc}(V*) − κ^{dec}(V*)",
               title="Externality Wedge in Coupling Strength",
               color=:black, linewidth=2.8, legend=false,
               size=(900, 450), dpi=300,
               grid=true, gridstyle=:dash, gridalpha=0.25)

    hline!(plt, [0.0]; color=:gray, linestyle=:dash, linewidth=1.8)

    # Shade positive and negative regions
    pos_mask = wedge .> 0
    neg_mask = wedge .< 0
    zero_line = zeros(length(wedge))

    if any(pos_mask)
        plot!(plt, V[pos_mask], wedge[pos_mask];
              fillrange=zero_line[pos_mask], fillalpha=0.2,
              fillcolor=:green, linewidth=0, label=false)
    end
    if any(neg_mask)
        plot!(plt, V[neg_mask], wedge[neg_mask];
              fillrange=zero_line[neg_mask], fillalpha=0.2,
              fillcolor=:red, linewidth=0, label=false)
    end

    # Annotate maximum wedge
    if isfinite(wedge[max_idx])
        annotate!(plt, V[max_idx], wedge[max_idx],
                 text("  Max gap = $(round(abs(wedge[max_idx]); digits=3))",
                      9, :left, :black))
    end

    xlims!(plt, xlim)
    ylims!(plt, ylim)

    return plt
end

"""
Figure: Welfare difference surface ΔW = W^soc - W^dec.
Two-panel: (1) ΔW alone with zero contour, (2) ΔW with ridge overlays.
Also saves individual panels.
"""
function plot_welfare_difference(data)
    DEBUG && println("   → Creating welfare difference surface...")

    κgrid = data.κgrid
    V = data.Vstar_sorted
    ΔW = data.W_soc_surface .- data.W_dec_surface

    # Symmetric color limits
    finite_vals = filter(isfinite, vec(ΔW))
    cmax = isempty(finite_vals) ? 1.0 : quantile(abs.(finite_vals), 0.95)
    clims = (-cmax, cmax)
    levels = range(clims[1], clims[2]; length=15)

    ΔW_plot = copy(ΔW)

    # Augmented axis limits
    xlim = augmented_xlims(κgrid, data.κ_dec_ridge, data.κ_soc_ridge)
    ylim = augmented_ylims(V)

    # Combined 2-panel figure
    plt = plot(layout=(1, 2), size=(1200, 450), dpi=300, margin=5Plots.mm)

    # Panel 1: ΔW with zero contour only
    contourf!(plt[1], κgrid, V, ΔW_plot;
              c=:balance, clims=clims, levels=levels,
              xlabel="Social coupling κ", ylabel="Stationary dispersion V*",
              title="Welfare Difference: ΔW = W^{soc} − W^{dec}",
              fillalpha=0.95, linewidth=0, nan_color=:white)
    contour!(plt[1], κgrid, V, ΔW_plot;
             levels=levels, color=:black, linewidth=0.5,
             linealpha=0.3, legend=false)
    contour!(plt[1], κgrid, V, ΔW_plot;
             levels=[0.0], color=:black, linewidth=2.5,
             linestyle=:solid, label=false)
    xlims!(plt[1], xlim)
    ylims!(plt[1], ylim)

    # Panel 2: ΔW with optimal ridges
    contourf!(plt[2], κgrid, V, ΔW_plot;
              c=:balance, clims=clims, levels=levels,
              xlabel="Social coupling κ", ylabel="Stationary dispersion V*",
              title="ΔW with Optimal Ridges",
              fillalpha=0.95, linewidth=0, nan_color=:white,
              legend=:topright, legendfontsize=9)
    contour!(plt[2], κgrid, V, ΔW_plot;
             levels=levels, color=:black, linewidth=0.5,
             linealpha=0.3, legend=false)
    plot!(plt[2], data.κ_dec_ridge, V;
          color=:red, linewidth=2.5, linestyle=:dash,
          label="κ^{dec}(V*)")
    plot!(plt[2], data.κ_soc_ridge, V;
          color=:cyan, linewidth=2.5, linestyle=:dash,
          label="κ^{soc}(V*)")
    contour!(plt[2], κgrid, V, ΔW_plot;
             levels=[0.0], color=:black, linewidth=2.5,
             linestyle=:solid, label="ΔW = 0")
    xlims!(plt[2], xlim)
    ylims!(plt[2], ylim)

    # Save individual panels
    plt1 = plot(size=(650, 550), dpi=300, margin=5Plots.mm)
    contourf!(plt1, κgrid, V, ΔW_plot;
              c=:balance, clims=clims, levels=levels,
              xlabel="Social coupling κ", ylabel="Stationary dispersion V*",
              title="Welfare Difference: ΔW = W^{soc} − W^{dec}",
              fillalpha=0.95, linewidth=0, nan_color=:white)
    contour!(plt1, κgrid, V, ΔW_plot;
             levels=levels, color=:black, linewidth=0.5,
             linealpha=0.3, legend=false)
    contour!(plt1, κgrid, V, ΔW_plot;
             levels=[0.0], color=:black, linewidth=2.5,
             linestyle=:solid, label=false)
    xlims!(plt1, xlim)
    ylims!(plt1, ylim)
    save_both(plt1, "fig4_panel_difference_simple")

    plt2 = plot(size=(650, 550), dpi=300, margin=5Plots.mm)
    contourf!(plt2, κgrid, V, ΔW_plot;
              c=:balance, clims=clims, levels=levels,
              xlabel="Social coupling κ", ylabel="Stationary dispersion V*",
              title="ΔW with Optimal Ridges",
              fillalpha=0.95, linewidth=0, nan_color=:white,
              legend=:topright, legendfontsize=10)
    contour!(plt2, κgrid, V, ΔW_plot;
             levels=levels, color=:black, linewidth=0.5,
             linealpha=0.3, legend=false)
    plot!(plt2, data.κ_dec_ridge, V;
          color=:red, linewidth=2.5, linestyle=:dash,
          label="κ^{dec}(V*)")
    plot!(plt2, data.κ_soc_ridge, V;
          color=:cyan, linewidth=2.5, linestyle=:dash,
          label="κ^{soc}(V*)")
    contour!(plt2, κgrid, V, ΔW_plot;
             levels=[0.0], color=:black, linewidth=2.5,
             linestyle=:solid, label="ΔW = 0")
    xlims!(plt2, xlim)
    ylims!(plt2, ylim)
    save_both(plt2, "fig4_panel_difference_with_ridges")

    return plt
end

"""
Figure: Cross-sections W(V* | κ) at fixed κ values.
Complementary to the main script's W(κ | V*) slices.
Also saves individual panels.
"""
function plot_slices_kappa(data)
    DEBUG && println("   → Creating W(V* | κ) slices...")

    κgrid = data.κgrid
    V = data.Vstar_sorted
    W_dec = data.W_dec_surface
    W_soc = data.W_soc_surface

    # Select 3 representative κ values (low, mid, high)
    κ_vals = [
        κgrid[round(Int, 0.2 * length(κgrid))],
        κgrid[round(Int, 0.5 * length(κgrid))],
        κgrid[round(Int, 0.8 * length(κgrid))]
    ]
    idxs = [argmin(abs.(κgrid .- κ)) for κ in κ_vals]
    labels = ["Low coupling", "Medium coupling", "High coupling"]

    plt = plot(layout=(1, 3), size=(1800, 450), dpi=300, margin=6Plots.mm)
    individual_plots = []

    for (panel, (κval, idx, label)) in enumerate(zip(κ_vals, idxs, labels))
        W_dec_slice = W_dec[:, idx]
        W_soc_slice = W_soc[:, idx]

        # Mask non-finite values
        valid = isfinite.(W_dec_slice) .& isfinite.(W_soc_slice)
        V_valid = V[valid]

        xlim = augmented_ylims(V_valid; pad_frac=0.08)

        plot!(plt[panel], V_valid, W_dec_slice[valid];
              linewidth=2.8, color=:black, label="Decentralised",
              xlabel="Stationary dispersion V*", ylabel="Welfare",
              title="$label (κ = $(round(κval; digits=3)))",
              legend=:topright, legendfontsize=9,
              grid=true, gridstyle=:dash, gridalpha=0.25)
        plot!(plt[panel], V_valid, W_soc_slice[valid];
              linewidth=2.8, color=:blue, label="Planner")
        xlims!(plt[panel], xlim)

        # Create individual plot
        plt_ind = plot(V_valid, W_dec_slice[valid];
                      linewidth=3.0, color=:black, label="Decentralised",
                      xlabel="Stationary dispersion V*", ylabel="Welfare",
                      title="$label (κ = $(round(κval; digits=3)))",
                      legend=:topright, legendfontsize=10,
                      grid=true, gridstyle=:dash, gridalpha=0.25,
                      size=(650, 500), dpi=300, margin=5Plots.mm)
        plot!(plt_ind, V_valid, W_soc_slice[valid];
              linewidth=3.0, color=:blue, label="Planner")
        xlims!(plt_ind, xlim)
        push!(individual_plots, plt_ind)
    end

    # Save individual slices
    save_both(individual_plots[1], "fig4_slice_low_kappa")
    save_both(individual_plots[2], "fig4_slice_mid_kappa")
    save_both(individual_plots[3], "fig4_slice_high_kappa")

    return plt
end

"""
Figure: Pedagogical contour with region annotations.
Includes textual labels for presentation/teaching purposes.
NOT intended for main paper figures.
Also saves individual panels.
"""
function plot_pedagogical_contour(data)
    DEBUG && println("   → Creating pedagogical contour...")

    κgrid = data.κgrid
    V = data.Vstar_sorted
    W_dec = data.W_dec_surface
    W_soc = data.W_soc_surface

    # Shared color limits and levels
    clims = finite_clims(W_dec, W_soc; q=(0.05, 0.95))
    levels = range(clims[1], clims[2]; length=12)

    W_dec_plot = copy(W_dec)
    W_soc_plot = copy(W_soc)

    xlim = augmented_xlims(κgrid, data.κ_dec_ridge, data.κ_soc_ridge)
    ylim = augmented_ylims(V)

    # Combined 2-panel figure
    plt = plot(layout=(1, 2), size=(1250, 460), dpi=300, margin=5Plots.mm)

    # Panel 1: Decentralised with annotations
    contourf!(plt[1], κgrid, V, W_dec_plot;
              c=:viridis, clims=clims, levels=levels,
              xlabel="Social coupling κ", ylabel="Stationary dispersion V*",
              title="Decentralised Welfare (Pedagogical)",
              fillalpha=0.95, linewidth=0, nan_color=:white)
    contour!(plt[1], κgrid, V, W_dec_plot;
             levels=levels, color=:black, linewidth=0.6,
             linealpha=0.35, legend=false)
    plot!(plt[1], data.κ_dec_ridge, V;
          color=:red, linewidth=3.0, label=false)
    V_mid = median(V)
    annotate!(plt[1], (0.15, V_mid * 0.7), text("Safe\nplateau", 10, :left, :white))
    annotate!(plt[1], (0.5, V_mid * 0.7), text("Welfare\ncrater", 10, :left, :red))
    annotate!(plt[1], (1.2, V_mid * 1.25), text("Danger\nzone", 10, :left, :orange))
    xlims!(plt[1], xlim)
    ylims!(plt[1], ylim)

    # Panel 2: Planner with annotations
    contourf!(plt[2], κgrid, V, W_soc_plot;
              c=:plasma, clims=clims, levels=levels,
              xlabel="Social coupling κ", ylabel="Stationary dispersion V*",
              title="Planner Welfare (Pedagogical)",
              fillalpha=0.95, linewidth=0, nan_color=:white)
    contour!(plt[2], κgrid, V, W_soc_plot;
             levels=levels, color=:black, linewidth=0.6,
             linealpha=0.35, legend=false)
    plot!(plt[2], data.κ_soc_ridge, V;
          color=:cyan, linewidth=3.0, label=false)
    annotate!(plt[2], (0.15, V_mid * 0.7), text("Safe\nplateau", 10, :left, :white))
    annotate!(plt[2], (0.5, V_mid * 0.7), text("Welfare\ncrater", 10, :left, :red))
    annotate!(plt[2], (1.2, V_mid * 1.25), text("Danger\nzone", 10, :left, :orange))
    xlims!(plt[2], xlim)
    ylims!(plt[2], ylim)

    # Save individual pedagogical panels
    plt_dec_ped = plot(size=(650, 550), dpi=300, margin=5Plots.mm)
    contourf!(plt_dec_ped, κgrid, V, W_dec_plot;
              c=:viridis, clims=clims, levels=levels,
              xlabel="Social coupling κ", ylabel="Stationary dispersion V*",
              title="Decentralised Welfare (Pedagogical)",
              fillalpha=0.95, linewidth=0, nan_color=:white)
    contour!(plt_dec_ped, κgrid, V, W_dec_plot;
             levels=levels, color=:black, linewidth=0.6,
             linealpha=0.35, legend=false)
    plot!(plt_dec_ped, data.κ_dec_ridge, V;
          color=:red, linewidth=3.0, label=false)
    annotate!(plt_dec_ped, (0.15, V_mid * 0.7), text("Safe\nplateau", 10, :left, :white))
    annotate!(plt_dec_ped, (0.5, V_mid * 0.7), text("Welfare\ncrater", 10, :left, :red))
    annotate!(plt_dec_ped, (1.2, V_mid * 1.25), text("Danger\nzone", 10, :left, :orange))
    xlims!(plt_dec_ped, xlim)
    ylims!(plt_dec_ped, ylim)
    save_both(plt_dec_ped, "fig4_panel_pedagogical_decentralised")

    plt_soc_ped = plot(size=(650, 550), dpi=300, margin=5Plots.mm)
    contourf!(plt_soc_ped, κgrid, V, W_soc_plot;
              c=:plasma, clims=clims, levels=levels,
              xlabel="Social coupling κ", ylabel="Stationary dispersion V*",
              title="Planner Welfare (Pedagogical)",
              fillalpha=0.95, linewidth=0, nan_color=:white)
    contour!(plt_soc_ped, κgrid, V, W_soc_plot;
             levels=levels, color=:black, linewidth=0.6,
             linealpha=0.35, legend=false)
    plot!(plt_soc_ped, data.κ_soc_ridge, V;
          color=:cyan, linewidth=3.0, label=false)
    annotate!(plt_soc_ped, (0.15, V_mid * 0.7), text("Safe\nplateau", 10, :left, :white))
    annotate!(plt_soc_ped, (0.5, V_mid * 0.7), text("Welfare\ncrater", 10, :left, :red))
    annotate!(plt_soc_ped, (1.2, V_mid * 1.25), text("Danger\nzone", 10, :left, :orange))
    xlims!(plt_soc_ped, xlim)
    ylims!(plt_soc_ped, ylim)
    save_both(plt_soc_ped, "fig4_panel_pedagogical_planner")

    return plt
end

"""
Figure: Ridge comparison with detailed wedge visualization.
Alternative to the optimal comparison in fig4_welfare.jl.
"""
function plot_ridge_comparison(data)
    DEBUG && println("   → Creating ridge comparison...")

    V = data.Vstar_sorted
    κ_dec = data.κ_dec_ridge
    κ_soc = data.κ_soc_ridge

    xlim = augmented_ylims(V; pad_frac=0.08)
    ylim = augmented_xlims(collect(extrema(filter(isfinite, [κ_dec; κ_soc]))),
                           κ_dec, κ_soc; pad_frac=0.1)

    plt = plot(size=(900, 500), dpi=300,
               xlabel="Stationary dispersion V*",
               ylabel="Optimal coupling κ",
               title="Optimal Coupling: Decentralised vs Planner",
               legend=:bottomright, legendfontsize=10,
               grid=true, gridstyle=:dash, gridalpha=0.25,
               margin=5Plots.mm)

    # Ridge lines
    plot!(plt, V, κ_dec;
          linewidth=3.2, color=:red, linestyle=:solid,
          label="κ^{dec}(V*)")
    plot!(plt, V, κ_soc;
          linewidth=3.2, color=:cyan, linestyle=:solid,
          label="κ^{soc}(V*)")

    # Shaded wedge
    plot!(plt, V, κ_soc;
          fillrange=κ_dec, fillalpha=0.25, fillcolor=:purple,
          linewidth=0, label="Externality gap")

    # Mark a few key points
    key_idxs = [
        argmin(abs.(V .- quantile(V, 0.25))),
        argmin(abs.(V .- median(V))),
        argmin(abs.(V .- quantile(V, 0.75)))
    ]
    for idx in key_idxs
        scatter!(plt, [V[idx]], [κ_dec[idx]];
                color=:red, markersize=5, markerstrokewidth=1.5,
                markerstrokecolor=:white, label=false)
        scatter!(plt, [V[idx]], [κ_soc[idx]];
                color=:cyan, markersize=5, markerstrokewidth=1.5,
                markerstrokecolor=:white, label=false)
    end

    xlims!(plt, xlim)
    ylims!(plt, ylim)

    return plt
end

"""
Figure: Labeled contour plots with nonlinear level spacing.
Inspired by CairoMakie's labeled contour example - shows exact welfare values
at different (κ, V*) points with enhanced visualization of structure.
Creates both individual panels and a combined 3-panel figure.
"""
function plot_labeled_contours(data; nlevels=12)
    DEBUG && println("   → Creating labeled contour plots...")

    κgrid = data.κgrid
    V = data.Vstar_sorted
    W_dec = data.W_dec_surface
    W_soc = data.W_soc_surface
    κ_dec = data.κ_dec_ridge
    κ_soc = data.κ_soc_ridge

    # Compute welfare difference
    ΔW = W_soc .- W_dec

    # Augmented axis limits
    xlim = augmented_xlims(κgrid, κ_dec, κ_soc)
    ylim = augmented_ylims(V)

    # For decentralised and planner: use power-law spacing to emphasize structure
    # Map finite values to [0, 1], apply power transform, then back to original range
    function compute_powerlaw_levels(W, nlevels, power=0.5)
        finite_vals = filter(isfinite, vec(W))
        if isempty(finite_vals)
            return range(-1, 1; length=nlevels)
        end
        wmin, wmax = extrema(finite_vals)
        if wmin ≈ wmax
            return fill(wmin, nlevels)
        end
        # Normalize to [0, 1], apply power transform, scale back
        t = range(0, 1; length=nlevels)
        t_transformed = t.^power
        levels = wmin .+ t_transformed .* (wmax - wmin)
        return levels
    end

    # Decentralised: power-law levels to emphasize low-welfare regions
    levels_dec = compute_powerlaw_levels(W_dec, nlevels, 0.6)

    # Planner: similar power-law levels
    levels_soc = compute_powerlaw_levels(W_soc, nlevels, 0.6)

    # Difference: symmetric logarithmic spacing around zero
    finite_diff = filter(isfinite, vec(ΔW))
    if !isempty(finite_diff)
        diff_max = quantile(abs.(finite_diff), 0.95)
        # Symmetric log-spaced levels
        pos_levels = exp.(range(log(0.01*diff_max), log(diff_max); length=div(nlevels, 2)))
        levels_diff = vcat(-reverse(pos_levels), 0.0, pos_levels)
    else
        levels_diff = range(-1, 1; length=nlevels)
    end

    # Individual panel: Decentralised with labeled contours
    plt_dec = plot(size=(700, 580), dpi=300, margin=6Plots.mm)
    contourf!(plt_dec, κgrid, V, W_dec;
              c=:viridis, levels=levels_dec,
              xlabel="Social coupling κ", ylabel="Stationary dispersion V*",
              title="Decentralised Welfare (Labeled Contours)",
              fillalpha=0.85, linewidth=0, nan_color=:white, colorbar=true)
    contour!(plt_dec, κgrid, V, W_dec;
             levels=levels_dec[1:2:end], color=:white, linewidth=1.2,
             linealpha=0.8, legend=false,
             clabels=true, cgrad=:viridis)
    plot!(plt_dec, κ_dec, V; color=:red, linewidth=3.2,
          linestyle=:solid, label="κ^{dec}(V*)", legend=:topright,
          legendfontsize=11)
    xlims!(plt_dec, xlim)
    ylims!(plt_dec, ylim)
    save_both(plt_dec, "fig4_panel_labeled_decentralised")

    # Individual panel: Planner with labeled contours
    plt_soc = plot(size=(700, 580), dpi=300, margin=6Plots.mm)
    contourf!(plt_soc, κgrid, V, W_soc;
              c=:plasma, levels=levels_soc,
              xlabel="Social coupling κ", ylabel="Stationary dispersion V*",
              title="Planner Welfare (Labeled Contours)",
              fillalpha=0.85, linewidth=0, nan_color=:white, colorbar=true)
    contour!(plt_soc, κgrid, V, W_soc;
             levels=levels_soc[1:2:end], color=:white, linewidth=1.2,
             linealpha=0.8, legend=false,
             clabels=true, cgrad=:plasma)
    plot!(plt_soc, κ_soc, V; color=:cyan, linewidth=3.2,
          linestyle=:solid, label="κ^{soc}(V*)", legend=:topright,
          legendfontsize=11)
    xlims!(plt_soc, xlim)
    ylims!(plt_soc, ylim)
    save_both(plt_soc, "fig4_panel_labeled_planner")

    # Individual panel: Welfare difference with labeled contours
    plt_diff = plot(size=(700, 580), dpi=300, margin=6Plots.mm)
    contourf!(plt_diff, κgrid, V, ΔW;
              c=:balance, levels=levels_diff,
              xlabel="Social coupling κ", ylabel="Stationary dispersion V*",
              title="Welfare Difference (Labeled Contours)",
              fillalpha=0.85, linewidth=0, nan_color=:white, colorbar=true)
    contour!(plt_diff, κgrid, V, ΔW;
             levels=levels_diff[2:2:end-1], color=:black, linewidth=1.0,
             linealpha=0.6, legend=false,
             clabels=true, cgrad=:balance)
    # Bold zero contour
    contour!(plt_diff, κgrid, V, ΔW;
             levels=[0.0], color=:black, linewidth=3.0,
             linestyle=:solid, label="ΔW = 0")
    plot!(plt_diff, κ_dec, V; color=:red, linewidth=2.5,
          linestyle=:dash, label="κ^{dec}(V*)")
    plot!(plt_diff, κ_soc, V; color=:cyan, linewidth=2.5,
          linestyle=:dash, label="κ^{soc}(V*)")
    plot!(plt_diff; legend=:topright, legendfontsize=11)
    xlims!(plt_diff, xlim)
    ylims!(plt_diff, ylim)
    save_both(plt_diff, "fig4_panel_labeled_difference")

    # Combined 3-panel figure
    plt_combined = plot(layout=(1, 3), size=(2000, 550), dpi=300, margin=6Plots.mm)

    # Panel 1: Decentralised
    contourf!(plt_combined[1], κgrid, V, W_dec;
              c=:viridis, levels=levels_dec,
              xlabel="Social coupling κ", ylabel="Stationary dispersion V*",
              title="Decentralised Welfare",
              fillalpha=0.85, linewidth=0, nan_color=:white)
    contour!(plt_combined[1], κgrid, V, W_dec;
             levels=levels_dec[1:2:end], color=:white, linewidth=1.0,
             linealpha=0.7, legend=false)
    plot!(plt_combined[1], κ_dec, V; color=:red, linewidth=3.0,
          linestyle=:solid, label="κ^{dec}(V*)", legend=:topright,
          legendfontsize=9)
    xlims!(plt_combined[1], xlim)
    ylims!(plt_combined[1], ylim)

    # Panel 2: Planner
    contourf!(plt_combined[2], κgrid, V, W_soc;
              c=:plasma, levels=levels_soc,
              xlabel="Social coupling κ", ylabel="Stationary dispersion V*",
              title="Planner Welfare",
              fillalpha=0.85, linewidth=0, nan_color=:white)
    contour!(plt_combined[2], κgrid, V, W_soc;
             levels=levels_soc[1:2:end], color=:white, linewidth=1.0,
             linealpha=0.7, legend=false)
    plot!(plt_combined[2], κ_soc, V; color=:cyan, linewidth=3.0,
          linestyle=:solid, label="κ^{soc}(V*)", legend=:topright,
          legendfontsize=9)
    xlims!(plt_combined[2], xlim)
    ylims!(plt_combined[2], ylim)

    # Panel 3: Welfare difference
    contourf!(plt_combined[3], κgrid, V, ΔW;
              c=:balance, levels=levels_diff,
              xlabel="Social coupling κ", ylabel="Stationary dispersion V*",
              title="Planner − Decentralised",
              fillalpha=0.85, linewidth=0, nan_color=:white)
    contour!(plt_combined[3], κgrid, V, ΔW;
             levels=levels_diff[2:2:end-1], color=:black, linewidth=0.8,
             linealpha=0.5, legend=false)
    contour!(plt_combined[3], κgrid, V, ΔW;
             levels=[0.0], color=:black, linewidth=2.8,
             linestyle=:solid, label="ΔW = 0")
    plot!(plt_combined[3], κ_dec, V; color=:red, linewidth=2.2,
          linestyle=:dash, label="κ^{dec}(V*)")
    plot!(plt_combined[3], κ_soc, V; color=:cyan, linewidth=2.2,
          linestyle=:dash, label="κ^{soc}(V*)")
    plot!(plt_combined[3]; legend=:topright, legendfontsize=9)
    xlims!(plt_combined[3], xlim)
    ylims!(plt_combined[3], ylim)

    return plt_combined
end

# ----------------------------------------------------------------------
# Main execution
# ----------------------------------------------------------------------

"""
Extra welfare plots for Figure 4: supplementary and pedagogical figures.

Generates:
- fig4_externality_ridge.pdf: Δκ(V*) wedge
- fig4_welfare_difference_surface.pdf: 2-panel ΔW with ridges
- fig4_panel_difference_simple.pdf: ΔW alone (individual)
- fig4_panel_difference_with_ridges.pdf: ΔW with ridges (individual)
- fig4_welfare_slices_kappa.pdf: W(V* | κ) cross-sections (combined)
- fig4_slice_low/mid/high_kappa.pdf: Individual κ slices
- fig4_pedagogical_contour.pdf: annotated contours (combined)
- fig4_panel_pedagogical_decentralised/planner.pdf: Individual pedagogical panels
- fig4_ridge_comparison.pdf: alternative optimal κ visualization
- fig4_labeled_contours.pdf: 3-panel with nonlinear level spacing (combined)
- fig4_panel_labeled_decentralised/planner/difference.pdf: Individual labeled panels
"""
function main()
    Random.seed!(2025)
    mkpath("figs")

    println("Generating Figure 4: Extra Plots")
    println("="^60)

    # Baseline parameters (match fig4_welfare.jl)
    p = Params(λ=0.65, σ=1.85, Θ=2.0, c0=1.0, hazard=StepHazard(0.6))
    α = 0.25
    β = 0.8

    # Compute welfare data
    data = compute_welfare_data(p; α=α, β=β)

    # Generate supplementary figures
    plt_ridge = plot_externality_ridge(data)
    save_both(plt_ridge, "fig4_externality_ridge")
    println("✓ Saved: fig4_externality_ridge.pdf (Δκ wedge)")

    plt_diff = plot_welfare_difference(data)
    save_both(plt_diff, "fig4_welfare_difference_surface")
    println("✓ Saved: fig4_welfare_difference_surface.pdf (ΔW combined)")
    println("✓ Saved: fig4_panel_difference_simple.pdf (ΔW individual)")
    println("✓ Saved: fig4_panel_difference_with_ridges.pdf (ΔW with ridges individual)")

    plt_slices_k = plot_slices_kappa(data)
    save_both(plt_slices_k, "fig4_welfare_slices_kappa")
    println("✓ Saved: fig4_welfare_slices_kappa.pdf (W vs V* slices combined)")
    println("✓ Saved: fig4_slice_low/mid/high_kappa.pdf (individual κ slices)")

    plt_pedagogical = plot_pedagogical_contour(data)
    save_both(plt_pedagogical, "fig4_pedagogical_contour")
    println("✓ Saved: fig4_pedagogical_contour.pdf (pedagogical combined)")
    println("✓ Saved: fig4_panel_pedagogical_decentralised/planner.pdf (individual pedagogical)")

    plt_ridge_comp = plot_ridge_comparison(data)
    save_both(plt_ridge_comp, "fig4_ridge_comparison")
    println("✓ Saved: fig4_ridge_comparison.pdf (ridge comparison)")

    plt_labeled = plot_labeled_contours(data; nlevels=12)
    save_both(plt_labeled, "fig4_labeled_contours")
    println("✓ Saved: fig4_labeled_contours.pdf (labeled contours combined)")
    println("✓ Saved: fig4_panel_labeled_decentralised/planner/difference.pdf (individual labeled)")

    println("="^60)
    println("Extra plots generation complete!")
    DEBUG && println("(Debug mode: $DEBUG)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
