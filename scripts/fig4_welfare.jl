#!/usr/bin/env julia
using BeliefSim
using BeliefSim.Types
using BeliefSim.OUResets
using BeliefSim.Stats
using Plots
using Statistics
using Random
using Printf

# Debug mode: set to true to enable detailed diagnostics
const DEBUG = false

# ----------------------------------------------------------------------
# Shared utilities
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
    print_histogram(vals; bins=10)

Print a simple ASCII histogram of finite values for diagnostics.
"""
function print_histogram(vals; bins=10)
    isempty(vals) && return DEBUG && @info "Histogram: no finite values"
    lo, hi = minimum(vals), maximum(vals)
    if lo == hi
        lo -= 1.0
        hi += 1.0
    end
    edges = collect(range(lo, hi; length=bins+1))
    counts = zeros(Int, bins)
    for v in vals
        idx = clamp(searchsortedlast(edges, v), 1, bins)
        counts[idx] += 1
    end
    DEBUG && @info "Histogram edges=$(round.(edges; digits=3)) counts=$counts"
end

"""
    diagnose_surface(A, xgrid, ygrid, name; max_lines=5, save_plots=false)

Diagnose a 2D surface: count NaNs/Infs, show min/max, print per-row/col statistics.
Optionally save debug plots if `save_plots=true` (only when DEBUG=true).
"""
function diagnose_surface(A, xgrid, ygrid, name; max_lines=5, save_plots=false)
    DEBUG || return

    nans = count(isnan, A)
    infs = count(x -> isinf(x), A)
    finite_vals = filter(isfinite, vec(A))
    amin = isempty(finite_vals) ? NaN : minimum(finite_vals)
    amax = isempty(finite_vals) ? NaN : maximum(finite_vals)
    amean = isempty(finite_vals) ? NaN : mean(finite_vals)

    @info "Diagnostics for $name" size=size(A) min=amin max=amax mean=amean nans=nans infs=infs
    print_histogram(finite_vals)

    rows_to_show = collect(1:min(size(A, 1), max_lines))
    for r in rows_to_show
        row = A[r, :]
        finite_row = filter(isfinite, row)
        rmin = isempty(finite_row) ? NaN : minimum(finite_row)
        rmax = isempty(finite_row) ? NaN : maximum(finite_row)
        is_const = (!isempty(finite_row)) && all(x -> x ≈ rmin, finite_row)
        @info "Row $r (V*=$(ygrid[r]))" min=rmin max=rmax constant=is_const nans=count(isnan, row)
    end

    cols_to_show = collect(1:min(size(A, 2), max_lines))
    for c in cols_to_show
        col = A[:, c]
        finite_col = filter(isfinite, col)
        cmin = isempty(finite_col) ? NaN : minimum(finite_col)
        cmax = isempty(finite_col) ? NaN : maximum(finite_col)
        is_const = (!isempty(finite_col)) && all(x -> x ≈ cmin, finite_col)
        @info "Col $c (κ=$(xgrid[c]))" min=cmin max=cmax constant=is_const nans=count(isnan, col)
    end

    if save_plots
        mkpath("figs/debug")
        sample_rows = unique(clamp.(round.(Int, range(1, size(A, 1); length=3)), 1, size(A, 1)))
        sample_cols = unique(clamp.(round.(Int, range(1, size(A, 2); length=3)), 1, size(A, 2)))

        plt_rows = plot(title="$name slices: V* rows", xlabel="κ", legend=false)
        for r in sample_rows
            plot!(plt_rows, xgrid, A[r, :]; label="row $r (V*=$(round(ygrid[r]; digits=3)))")
        end
        savefig(plt_rows, "figs/debug/$(name)_rows.pdf")

        plt_cols = plot(title="$name slices: κ cols", xlabel="V*", legend=false)
        for c in sample_cols
            plot!(plt_cols, ygrid, A[:, c]; label="col $c (κ=$(round(xgrid[c]; digits=3)))")
        end
        savefig(plt_cols, "figs/debug/$(name)_cols.pdf")
    end
end

"""
    median_filter(x, window_size=3)

Apply a median filter to smooth ridge lines extracted via argmax.
Removes discontinuous jumps while preserving trends.
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

# ----------------------------------------------------------------------
# Figure 4 (main paper): 1D welfare comparison
# ----------------------------------------------------------------------

"""
Generate the baseline 1D welfare comparison across κ.
This is the main figure for the paper body.
"""
function figure_1d_welfare(p, κgrid; α=0.25, β=0.8)
    DEBUG && println("   → Computing 1D welfare curves...")

    welfare = compute_welfare_curves(p, κgrid; α=α, β=β, N=8_000,
                                     T=260.0, dt=0.01, seed=99)

    κ_dec_idx = argmax(welfare.W_dec)
    κ_soc_idx = argmax(welfare.W_soc)

    plt = plot(; xlabel="Social coupling κ", ylabel="Welfare W(κ)",
               title="Welfare Comparison",
               legend=:bottomright, size=(750, 420), dpi=300,
               grid=true, gridstyle=:dash, gridalpha=0.25)

    plot!(plt, κgrid, welfare.W_dec; color=:black, linewidth=2.5,
          label="Decentralised")
    plot!(plt, κgrid, welfare.W_soc; color=:blue, linewidth=2.5,
          label="Planner")

    scatter!(plt, [κgrid[κ_dec_idx]], [welfare.W_dec[κ_dec_idx]];
             color=:black, markersize=6, markerstrokewidth=2,
             markerstrokecolor=:white, label="κ^dec")
    scatter!(plt, [κgrid[κ_soc_idx]], [welfare.W_soc[κ_soc_idx]];
             color=:blue, markersize=6, markerstrokewidth=2,
             markerstrokecolor=:white, label="κ^soc")

    # Shade externality wedge between optimal points
    κ_lo = min(κgrid[κ_dec_idx], κgrid[κ_soc_idx])
    κ_hi = max(κgrid[κ_dec_idx], κgrid[κ_soc_idx])
    band = range(κ_lo, κ_hi; length=50)
    fill_upper = fill(max(welfare.W_dec[κ_dec_idx], welfare.W_soc[κ_soc_idx]), length(band))
    fill_lower = fill(min(welfare.W_dec[κ_dec_idx], welfare.W_soc[κ_soc_idx]), length(band))
    plot!(plt, band, fill_upper; fillrange=fill_lower, fillalpha=0.15,
          color=:purple, label="Externality wedge")

    # Augmented axis limits
    xlims!(plt, augmented_xlims(κgrid, [κgrid[κ_dec_idx]], [κgrid[κ_soc_idx]]))

    return plt
end

# ----------------------------------------------------------------------
# Enhanced 3-panel figure: Decentralised, Planner, Externality
# ----------------------------------------------------------------------

"""
Build welfare surfaces over (κ, V*) by varying σ to span different dispersions.
Returns sorted arrays ready for contour plotting.
"""
function compute_welfare_surfaces(p, κgrid_contour, σgrid; α=0.25, β=0.8)
    DEBUG && println("   → Computing welfare surfaces over (κ, V*) grid...")

    nκ = length(κgrid_contour)
    nσ = length(σgrid)

    Vstar_vals = similar(σgrid)
    W_dec_surface = Matrix{Float64}(undef, nσ, nκ)
    W_soc_surface = similar(W_dec_surface)
    κ_dec_ridge = similar(σgrid)
    κ_soc_ridge = similar(σgrid)

    for (j, σval) in enumerate(σgrid)
        pσ = Params(λ=p.λ, σ=σval, Θ=p.Θ, c0=p.c0, hazard=p.hazard)
        Vstar_vals[j] = estimate_Vstar(pσ; N=6_000, T=220.0, dt=0.01,
                                        burn_in=80.0, seed=500 + j)
        surf = compute_welfare_curves(pσ, κgrid_contour; α=α, β=β, N=5_000,
                                      T=180.0, dt=0.01, seed=800 + j)

        W_dec_surface[j, :] .= surf.W_dec
        W_soc_surface[j, :] .= surf.W_soc
        κ_dec_ridge[j] = κgrid_contour[argmax(surf.W_dec)]
        κ_soc_ridge[j] = κgrid_contour[argmax(surf.W_soc)]
    end

    # Diagnose raw surfaces
    diagnose_surface(W_dec_surface, κgrid_contour, Vstar_vals,
                     "W_dec_surface_raw"; save_plots=DEBUG)
    diagnose_surface(W_soc_surface, κgrid_contour, Vstar_vals,
                     "W_soc_surface_raw"; save_plots=DEBUG)

    # Sort by V* for monotone axis
    perm = sortperm(Vstar_vals)
    Vstar_sorted = Vstar_vals[perm]
    W_dec_surface = W_dec_surface[perm, :]
    W_soc_surface = W_soc_surface[perm, :]
    κ_dec_ridge = κ_dec_ridge[perm]
    κ_soc_ridge = κ_soc_ridge[perm]

    # Smooth ridge lines
    κ_dec_ridge = median_filter(κ_dec_ridge, 3)
    κ_soc_ridge = median_filter(κ_soc_ridge, 3)

    # Drop non-finite V* rows
    finite_mask = isfinite.(Vstar_sorted)
    if !all(finite_mask)
        Vstar_sorted = Vstar_sorted[finite_mask]
        W_dec_surface = W_dec_surface[finite_mask, :]
        W_soc_surface = W_soc_surface[finite_mask, :]
        κ_dec_ridge = κ_dec_ridge[finite_mask]
        κ_soc_ridge = κ_soc_ridge[finite_mask]
    end

    return (
        Vstar_sorted = Vstar_sorted,
        W_dec_surface = W_dec_surface,
        W_soc_surface = W_soc_surface,
        κ_dec_ridge = κ_dec_ridge,
        κ_soc_ridge = κ_soc_ridge
    )
end

"""
Create individual contour panel for decentralised welfare.
"""
function panel_decentralised(κgrid, V, W_dec, κ_dec, clims, levels)
    xlim = augmented_xlims(κgrid, κ_dec)
    ylim = augmented_ylims(V)

    plt = plot(size=(650, 550), dpi=300, margin=5Plots.mm)
    contourf!(plt, κgrid, V, W_dec;
              c=:viridis, levels=levels, clims=clims,
              xlabel="Social coupling κ", ylabel="Stationary dispersion V*",
              title="Decentralised Welfare",
              fillalpha=0.95, linewidth=0, nan_color=:white)
    contour!(plt, κgrid, V, W_dec;
             levels=levels, color=:black, linewidth=0.6,
             linealpha=0.35, legend=false)
    plot!(plt, κ_dec, V; color=:red, linewidth=3.0,
          linestyle=:solid, label="κ^{dec}(V*)", legend=:topright,
          legendfontsize=10)
    xlims!(plt, xlim)
    ylims!(plt, ylim)
    return plt
end

"""
Create individual contour panel for planner welfare.
"""
function panel_planner(κgrid, V, W_soc, κ_soc, clims, levels)
    xlim = augmented_xlims(κgrid, κ_soc)
    ylim = augmented_ylims(V)

    plt = plot(size=(650, 550), dpi=300, margin=5Plots.mm)
    contourf!(plt, κgrid, V, W_soc;
              c=:plasma, levels=levels, clims=clims,
              xlabel="Social coupling κ", ylabel="Stationary dispersion V*",
              title="Planner Welfare",
              fillalpha=0.95, linewidth=0, nan_color=:white)
    contour!(plt, κgrid, V, W_soc;
             levels=levels, color=:black, linewidth=0.6,
             linealpha=0.35, legend=false)
    plot!(plt, κ_soc, V; color=:cyan, linewidth=3.0,
          linestyle=:solid, label="κ^{soc}(V*)", legend=:topright,
          legendfontsize=10)
    xlims!(plt, xlim)
    ylims!(plt, ylim)
    return plt
end

"""
Create individual contour panel for welfare difference.
"""
function panel_difference(κgrid, V, W_diff, κ_dec, κ_soc, clims_diff, levels_diff)
    xlim = augmented_xlims(κgrid, κ_dec, κ_soc)
    ylim = augmented_ylims(V)

    plt = plot(size=(650, 550), dpi=300, margin=5Plots.mm)
    contourf!(plt, κgrid, V, W_diff;
              c=:balance, levels=levels_diff, clims=clims_diff,
              xlabel="Social coupling κ", ylabel="Stationary dispersion V*",
              title="Planner − Decentralised",
              fillalpha=0.95, linewidth=0, nan_color=:white)
    contour!(plt, κgrid, V, W_diff;
             levels=levels_diff, color=:black, linewidth=0.6,
             linealpha=0.35, legend=false)
    # Bold zero contour
    contour!(plt, κgrid, V, W_diff;
             levels=[0.0], color=:black, linewidth=2.5,
             linestyle=:solid, label="ΔW = 0")
    # Ridge lines (dashed to distinguish from zero contour)
    plot!(plt, κ_dec, V; color=:red, linewidth=2.2,
          linestyle=:dash, label="κ^{dec}(V*)")
    plot!(plt, κ_soc, V; color=:cyan, linewidth=2.2,
          linestyle=:dash, label="κ^{soc}(V*)")
    plot!(plt; legend=:topright, legendfontsize=10)
    xlims!(plt, xlim)
    ylims!(plt, ylim)
    return plt
end

"""
Create publication-grade 3-panel contour figure and save individual panels.
"""
function figure_3panel_enhanced(κgrid, surfaces)
    DEBUG && println("   → Creating 3-panel enhanced contour figure...")

    V = surfaces.Vstar_sorted
    W_dec = surfaces.W_dec_surface
    W_soc = surfaces.W_soc_surface
    κ_dec = surfaces.κ_dec_ridge
    κ_soc = surfaces.κ_soc_ridge

    # Compute shared color limits for welfare levels
    clims_common = finite_clims(W_dec, W_soc; q=(0.02, 0.98))
    levels_common = range(clims_common[1], clims_common[2]; length=20)

    DEBUG && @info "Shared clims for welfare" clims=clims_common
    DEBUG && print_histogram(filter(isfinite, vec(W_dec)); bins=12)
    DEBUG && print_histogram(filter(isfinite, vec(W_soc)); bins=12)

    # Compute welfare difference with symmetric limits
    W_diff = W_soc .- W_dec
    finite_diff = filter(isfinite, vec(W_diff))
    diff_max = isempty(finite_diff) ? 1.0 : quantile(abs.(finite_diff), 0.95)
    clims_diff = (-diff_max, diff_max)
    levels_diff = range(-diff_max, diff_max; length=17)

    # Prepare plotting arrays (keep NaNs for proper masking)
    W_dec_plot = copy(W_dec)
    W_soc_plot = copy(W_soc)
    W_diff_plot = copy(W_diff)

    # Augmented axis limits (shared across all panels)
    xlim = augmented_xlims(κgrid, κ_dec, κ_soc)
    ylim = augmented_ylims(V)

    # Create combined 3-panel figure
    plt = plot(layout=(1, 3), size=(1800, 500), dpi=300, margin=6Plots.mm)

    # Panel 1: Decentralised welfare
    contourf!(plt[1], κgrid, V, W_dec_plot;
              c=:viridis, levels=levels_common, clims=clims_common,
              xlabel="Social coupling κ", ylabel="Stationary dispersion V*",
              title="Decentralised Welfare",
              fillalpha=0.95, linewidth=0, nan_color=:white)
    contour!(plt[1], κgrid, V, W_dec_plot;
             levels=levels_common, color=:black, linewidth=0.6,
             linealpha=0.35, legend=false)
    plot!(plt[1], κ_dec, V; color=:red, linewidth=3.0,
          linestyle=:solid, label="κ^{dec}(V*)", legend=:topright,
          legendfontsize=9)
    xlims!(plt[1], xlim)
    ylims!(plt[1], ylim)

    # Panel 2: Planner welfare
    contourf!(plt[2], κgrid, V, W_soc_plot;
              c=:plasma, levels=levels_common, clims=clims_common,
              xlabel="Social coupling κ", ylabel="Stationary dispersion V*",
              title="Planner Welfare",
              fillalpha=0.95, linewidth=0, nan_color=:white)
    contour!(plt[2], κgrid, V, W_soc_plot;
             levels=levels_common, color=:black, linewidth=0.6,
             linealpha=0.35, legend=false)
    plot!(plt[2], κ_soc, V; color=:cyan, linewidth=3.0,
          linestyle=:solid, label="κ^{soc}(V*)", legend=:topright,
          legendfontsize=9)
    xlims!(plt[2], xlim)
    ylims!(plt[2], ylim)

    # Panel 3: Welfare difference (externality)
    contourf!(plt[3], κgrid, V, W_diff_plot;
              c=:balance, levels=levels_diff, clims=clims_diff,
              xlabel="Social coupling κ", ylabel="Stationary dispersion V*",
              title="Planner − Decentralised",
              fillalpha=0.95, linewidth=0, nan_color=:white)
    contour!(plt[3], κgrid, V, W_diff_plot;
             levels=levels_diff, color=:black, linewidth=0.6,
             linealpha=0.35, legend=false)
    contour!(plt[3], κgrid, V, W_diff_plot;
             levels=[0.0], color=:black, linewidth=2.5,
             linestyle=:solid, label="ΔW = 0")
    plot!(plt[3], κ_dec, V; color=:red, linewidth=2.2,
          linestyle=:dash, label="κ^{dec}(V*)")
    plot!(plt[3], κ_soc, V; color=:cyan, linewidth=2.2,
          linestyle=:dash, label="κ^{soc}(V*)")
    plot!(plt[3]; legend=:topright, legendfontsize=9)
    xlims!(plt[3], xlim)
    ylims!(plt[3], ylim)

    # Save individual panels
    plt_dec = panel_decentralised(κgrid, V, W_dec_plot, κ_dec, clims_common, levels_common)
    save_both(plt_dec, "fig4_panel_decentralised")

    plt_soc = panel_planner(κgrid, V, W_soc_plot, κ_soc, clims_common, levels_common)
    save_both(plt_soc, "fig4_panel_planner")

    plt_diff = panel_difference(κgrid, V, W_diff_plot, κ_dec, κ_soc, clims_diff, levels_diff)
    save_both(plt_diff, "fig4_panel_difference")

    return plt
end

# ----------------------------------------------------------------------
# Supporting figures: Cross-sections and optimal comparison
# ----------------------------------------------------------------------

"""
Plot welfare cross-sections W(κ | V*) at representative dispersion levels.
"""
function figure_welfare_slices(κgrid, surfaces)
    DEBUG && println("   → Creating welfare cross-section slices...")

    V = surfaces.Vstar_sorted
    W_dec = surfaces.W_dec_surface
    W_soc = surfaces.W_soc_surface
    κ_dec = surfaces.κ_dec_ridge
    κ_soc = surfaces.κ_soc_ridge

    # Select 3 representative V* values
    V_targets = [quantile(V, 0.25), median(V), quantile(V, 0.75)]
    labels = ["Low dispersion", "Medium dispersion", "High dispersion"]

    plt = plot(layout=(1, 3), size=(1800, 450), dpi=300, margin=6Plots.mm)
    individual_plots = []

    for (panel, (V_target, label)) in enumerate(zip(V_targets, labels))
        idx = argmin(abs.(V .- V_target))
        V_actual = V[idx]

        W_dec_slice = W_dec[idx, :]
        W_soc_slice = W_soc[idx, :]

        # Mask non-finite values
        valid = isfinite.(W_dec_slice) .& isfinite.(W_soc_slice)
        κ_valid = κgrid[valid]

        # Augmented x-limits
        xlim = augmented_xlims(κ_valid, [κ_dec[idx]], [κ_soc[idx]])

        plot!(plt[panel], κ_valid, W_dec_slice[valid];
              linewidth=2.8, color=:black, label="Decentralised",
              xlabel="Social coupling κ", ylabel="Welfare",
              title="$label (V* = $(round(V_actual; digits=3)))",
              legend=:bottomleft, legendfontsize=9,
              grid=true, gridstyle=:dash, gridalpha=0.25)
        plot!(plt[panel], κ_valid, W_soc_slice[valid];
              linewidth=2.8, color=:blue, label="Planner")

        # Mark optimal κ
        κ_dec_opt = κ_dec[idx]
        κ_soc_opt = κ_soc[idx]
        vline!(plt[panel], [κ_dec_opt];
               color=:black, linestyle=:dash, linewidth=1.8, alpha=0.6,
               label=@sprintf("κ^{dec} = %.2f", κ_dec_opt))
        vline!(plt[panel], [κ_soc_opt];
               color=:blue, linestyle=:dash, linewidth=1.8, alpha=0.6,
               label=@sprintf("κ^{soc} = %.2f", κ_soc_opt))
        xlims!(plt[panel], xlim)

        # Create individual plot
        plt_ind = plot(κ_valid, W_dec_slice[valid];
                      linewidth=3.0, color=:black, label="Decentralised",
                      xlabel="Social coupling κ", ylabel="Welfare",
                      title="$label (V* = $(round(V_actual; digits=3)))",
                      legend=:bottomleft, legendfontsize=10,
                      grid=true, gridstyle=:dash, gridalpha=0.25,
                      size=(650, 500), dpi=300, margin=5Plots.mm)
        plot!(plt_ind, κ_valid, W_soc_slice[valid];
              linewidth=3.0, color=:blue, label="Planner")
        vline!(plt_ind, [κ_dec_opt];
               color=:black, linestyle=:dash, linewidth=2.0, alpha=0.6,
               label=@sprintf("κ^{dec} = %.2f", κ_dec_opt))
        vline!(plt_ind, [κ_soc_opt];
               color=:blue, linestyle=:dash, linewidth=2.0, alpha=0.6,
               label=@sprintf("κ^{soc} = %.2f", κ_soc_opt))
        xlims!(plt_ind, xlim)
        push!(individual_plots, plt_ind)
    end

    # Save individual slices
    save_both(individual_plots[1], "fig4_slice_low_dispersion")
    save_both(individual_plots[2], "fig4_slice_mid_dispersion")
    save_both(individual_plots[3], "fig4_slice_high_dispersion")

    return plt
end

"""
Plot optimal κ comparison: κ^dec(V*) vs κ^soc(V*) with shaded wedge.
"""
function figure_optimal_comparison(surfaces)
    DEBUG && println("   → Creating optimal κ comparison...")

    V = surfaces.Vstar_sorted
    κ_dec = surfaces.κ_dec_ridge
    κ_soc = surfaces.κ_soc_ridge

    # Augmented y-limits
    ylim = augmented_xlims(collect(extrema(filter(isfinite, [κ_dec; κ_soc]))), κ_dec, κ_soc; pad_frac=0.1)
    xlim = augmented_ylims(V; pad_frac=0.08)

    plt = plot(size=(800, 500), dpi=300,
               xlabel="Stationary dispersion V*",
               ylabel="Optimal social coupling κ",
               title="Welfare-Maximizing Coupling Strength",
               legend=:bottomright, legendfontsize=10,
               grid=true, gridstyle=:dash, gridalpha=0.25,
               margin=5Plots.mm)

    plot!(plt, V, κ_dec;
          linewidth=3.0, color=:black, label="κ^{dec}(V*) - Decentralised",
          linestyle=:solid)
    plot!(plt, V, κ_soc;
          linewidth=3.0, color=:blue, label="κ^{soc}(V*) - Planner",
          linestyle=:solid)

    # Shaded externality wedge
    plot!(plt, V, κ_dec;
          fillrange=κ_soc, fillalpha=0.2, fillcolor=:purple,
          linewidth=0, label="Externality gap")

    # Annotation at midpoint
    mid_idx = div(length(V), 2)
    gap = abs(κ_soc[mid_idx] - κ_dec[mid_idx])
    if gap > 0.05
        annotate!(plt, V[mid_idx],
                 (κ_dec[mid_idx] + κ_soc[mid_idx])/2,
                 text("Policy\nwedge", 10, :center, :purple))
    end

    xlims!(plt, xlim)
    ylims!(plt, ylim)

    return plt
end

# ----------------------------------------------------------------------
# Main execution
# ----------------------------------------------------------------------

"""
Figure 4: Welfare analysis for decentralised vs planner equilibria.

Generates:
- fig4_welfare.pdf: 1D welfare comparison (main paper figure)
- fig4_welfare_enhanced.pdf: 3-panel contour with externality
- fig4_panel_decentralised.pdf: Individual panel for decentralised
- fig4_panel_planner.pdf: Individual panel for planner
- fig4_panel_difference.pdf: Individual panel for difference
- fig4_welfare_crosssections.pdf: welfare slices at different V*
- fig4_slice_low/mid/high_dispersion.pdf: Individual slice plots
- fig4_welfare_optimal_comparison.pdf: κ^dec vs κ^soc ridge comparison
"""
function main()
    Random.seed!(2025)
    mkpath("figs")
    DEBUG && mkpath("figs/debug")

    println("Generating Figure 4: Welfare Analysis")
    println("="^60)

    # Baseline parameters
    p = Params(λ=0.65, σ=1.85, Θ=2.0, c0=1.0, hazard=StepHazard(0.6))
    α = 0.25  # Weight on polarisation cost
    β = 0.8   # Discount factor

    # Grid for 1D welfare plot
    κgrid = collect(range(0.0, 2.2, length=28))

    # Generate 1D welfare figure (main paper)
    plt_1d = figure_1d_welfare(p, κgrid; α=α, β=β)
    save_both(plt_1d, "fig4_welfare")
    println("✓ Saved: fig4_welfare.pdf (1D welfare comparison)")

    # Grid for contour surfaces (higher resolution)
    κgrid_contour = collect(range(0.0, 2.2, length=60))
    σgrid = collect(range(0.6, 1.3, length=15))

    # Compute welfare surfaces
    surfaces = compute_welfare_surfaces(p, κgrid_contour, σgrid; α=α, β=β)

    # Generate 3-panel enhanced figure (also saves individual panels)
    plt_3panel = figure_3panel_enhanced(κgrid_contour, surfaces)
    save_both(plt_3panel, "fig4_welfare_enhanced")
    println("✓ Saved: fig4_welfare_enhanced.pdf (3-panel contour)")
    println("✓ Saved: fig4_panel_decentralised.pdf (individual panel)")
    println("✓ Saved: fig4_panel_planner.pdf (individual panel)")
    println("✓ Saved: fig4_panel_difference.pdf (individual panel)")

    # Generate cross-section slices (also saves individual slices)
    plt_slices = figure_welfare_slices(κgrid_contour, surfaces)
    save_both(plt_slices, "fig4_welfare_crosssections")
    println("✓ Saved: fig4_welfare_crosssections.pdf (welfare slices)")
    println("✓ Saved: fig4_slice_low/mid/high_dispersion.pdf (individual slices)")

    # Generate optimal κ comparison
    plt_optimal = figure_optimal_comparison(surfaces)
    save_both(plt_optimal, "fig4_welfare_optimal_comparison")
    println("✓ Saved: fig4_welfare_optimal_comparison.pdf (optimal κ comparison)")

    println("="^60)
    println("Figure 4 generation complete!")
    DEBUG && println("(Debug mode enabled - see figs/debug/ for diagnostics)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
