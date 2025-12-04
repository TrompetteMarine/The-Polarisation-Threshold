#!/usr/bin/env julia
using BeliefSim
using BeliefSim.Types
using BeliefSim.OUResets
using BeliefSim.Stats
using Plots
using Statistics
using Random
using Printf

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

# Compute robust color limits ignoring NaNs; widen if degenerate
function finite_clims(A, B; q=(0.05, 0.95))
    vals = [vec(A); vec(B)]
    vals = filter(isfinite, vals)
    if isempty(vals)
        @info "finite_clims: no finite values; defaulting to (-1, 1)"
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

# Simple histogram for diagnostics (avoids extra dependencies)
function print_histogram(vals; bins=10)
    isempty(vals) && return @info "Histogram: no finite values"
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
    @info "Histogram edges=$(round.(edges; digits=3)) counts=$counts"
end

# Diagnose a 2D surface before cleaning
function diagnose_surface(A, xgrid, ygrid, name; save_plots=true, max_lines=5)
    nans = count(isnan, A)
    infs = count(x -> isinf(x), A)
    finite_vals = filter(isfinite, vec(A))
    amin = isempty(finite_vals) ? NaN : minimum(finite_vals)
    amax = isempty(finite_vals) ? NaN : maximum(finite_vals)
    amean = isempty(finite_vals) ? NaN : mean(finite_vals)

    @info "Diagnostics for $name" size=size(A) min=amin max=amax mean=amean nans=nans infs=infs
    print_histogram(finite_vals)

    # Per-row stats (rows correspond to V* grid)
    rows_to_show = collect(1:min(size(A, 1), max_lines))
    for r in rows_to_show
        row = A[r, :]
        finite_row = filter(isfinite, row)
        rmin = isempty(finite_row) ? NaN : minimum(finite_row)
        rmax = isempty(finite_row) ? NaN : maximum(finite_row)
        is_const = (!isempty(finite_row)) && all(x -> x ≈ rmin, finite_row)
        @info "Row $r (V*=$(ygrid[r]))" min=rmin max=rmax constant=is_const nans=count(isnan, row)
    end

    # Per-column stats (cols correspond to κ grid)
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
        # pick a few rows/cols across the span
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
Figure 4: decentralised vs social welfare across κ.
"""
function main()
    Random.seed!(2025)

    p = Params(λ=0.65, σ=1.85, Θ=2.0, c0=1.0, hazard=StepHazard(0.6))
    κgrid = collect(range(0.0, 2.2, length=28))
    α = 0.25
    β = 0.8

    welfare = compute_welfare_curves(p, κgrid; α=α, β=β, N=8_000,
                                     T=260.0, dt=0.01, seed=99)

    κ_dec_idx = argmax(welfare.W_dec)
    κ_soc_idx = argmax(welfare.W_soc)

    mkpath("figs")
    plt = plot(; xlabel="κ", ylabel="W(κ)", title="Welfare comparison",
               legend=:bottomright, size=(750, 420), dpi=300)
    plot!(plt, κgrid, welfare.W_dec; color=:black, linewidth=2.5, label="Decentralised")
    plot!(plt, κgrid, welfare.W_soc; color=:blue, linewidth=2.5, label="Planner")

    scatter!(plt, [κgrid[κ_dec_idx]], [welfare.W_dec[κ_dec_idx]];
             color=:black, markersize=6, label="κ^dec")
    scatter!(plt, [κgrid[κ_soc_idx]], [welfare.W_soc[κ_soc_idx]];
             color=:blue, markersize=6, label="κ^soc")

    # Shade externality wedge
    κ_lo = min(κgrid[κ_dec_idx], κgrid[κ_soc_idx])
    κ_hi = max(κgrid[κ_dec_idx], κgrid[κ_soc_idx])
    band = range(κ_lo, κ_hi; length=50)
    fill_upper = similar(band)
    fill_lower = similar(band)
    for (i, κ) in enumerate(band)
        fill_upper[i] = max(welfare.W_dec[κ_dec_idx], welfare.W_soc[κ_soc_idx])
        fill_lower[i] = min(welfare.W_dec[κ_dec_idx], welfare.W_soc[κ_soc_idx])
    end
    plot!(plt, band, fill_upper; fillrange=fill_lower, fillalpha=0.15,
          color=:purple, label="Externality wedge")

    savefig(plt, "figs/fig4_welfare.pdf")

    # Welfare contours across (κ, V*) obtained by varying σ to span different dispersions
    # Increased resolution for smoother contours: 9→15 V* levels, 36→60 κ points
    σgrid = collect(range(0.6, 1.3, length=15))
    κgrid_contour = collect(range(first(κgrid), last(κgrid), length=60))
    nκ = length(κgrid_contour)
    nσ = length(σgrid)

    Vstar_vals = similar(σgrid)
    # Surfaces are indexed as (row → V* / σ index, col → κ index)
    W_dec_surface = Matrix{Float64}(undef, nσ, nκ)
    W_soc_surface = similar(W_dec_surface)
    κ_dec_ridge = similar(σgrid)
    κ_soc_ridge = similar(σgrid)

    for (j, σval) in enumerate(σgrid)
        pσ = Params(λ=p.λ, σ=σval, Θ=p.Θ, c0=p.c0, hazard=p.hazard)
        Vstar_vals[j] = estimate_Vstar(pσ; N=6_000, T=220.0, dt=0.01, burn_in=80.0, seed=500 + j)
        surf = compute_welfare_curves(pσ, κgrid_contour; α=α, β=β, N=5_000,
                                      T=180.0, dt=0.01, seed=800 + j)

        W_dec_surface[j, :] .= surf.W_dec
        W_soc_surface[j, :] .= surf.W_soc
        κ_dec_ridge[j] = κgrid_contour[argmax(surf.W_dec)]
        κ_soc_ridge[j] = κgrid_contour[argmax(surf.W_soc)]
    end

    # Diagnose raw surfaces before any cleaning or sorting
    diagnose_surface(W_dec_surface, κgrid_contour, Vstar_vals, "W_dec_surface_raw")
    diagnose_surface(W_soc_surface, κgrid_contour, Vstar_vals, "W_soc_surface_raw")

    # Sort by V* so the vertical axis is monotone
    perm = sortperm(Vstar_vals)
    Vstar_sorted = Vstar_vals[perm]
    W_dec_surface = W_dec_surface[perm, :]
    W_soc_surface = W_soc_surface[perm, :]
    κ_dec_ridge = κ_dec_ridge[perm]
    κ_soc_ridge = κ_soc_ridge[perm]

    # Apply median filter to smooth ridge lines (removes discontinuous jumps from argmax)
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
    κ_dec_ridge = median_filter(κ_dec_ridge, 3)
    κ_soc_ridge = median_filter(κ_soc_ridge, 3)

    # Drop any rows where V* is non-finite to avoid degenerate contours
    finite_mask_rows = isfinite.(Vstar_sorted)
    if !all(finite_mask_rows)
        Vstar_sorted = Vstar_sorted[finite_mask_rows]
        W_dec_surface = W_dec_surface[finite_mask_rows, :]
        W_soc_surface = W_soc_surface[finite_mask_rows, :]
        κ_dec_ridge = κ_dec_ridge[finite_mask_rows]
        κ_soc_ridge = κ_soc_ridge[finite_mask_rows]
    end

    @assert size(W_dec_surface) == (length(Vstar_sorted), length(κgrid_contour))
    @assert size(W_soc_surface) == (length(Vstar_sorted), length(κgrid_contour))

    nan_mask_dec = .!isfinite.(W_dec_surface)
    nan_mask_soc = .!isfinite.(W_soc_surface)

    # Clip color limits to informative quantiles while ignoring NaNs
    clims_common = finite_clims(W_dec_surface, W_soc_surface)
    # Increased contour levels for smoother appearance (14→24)
    levels = range(clims_common[1], clims_common[2]; length=24)
    @info "Shared clims for contours" clims=clims_common levels=levels
    print_histogram(filter(isfinite, vec(W_dec_surface)); bins=12)
    print_histogram(filter(isfinite, vec(W_soc_surface)); bins=12)

    # Prepare plotting arrays: keep NaNs for transparency, but also provide a fallback fill value
    nan_fill = clims_common[1] - 0.1 * max(abs(clims_common[1]), 1.0)
    W_dec_plot = copy(W_dec_surface)
    W_soc_plot = copy(W_soc_surface)
    map!(x -> isfinite(x) ? x : nan_fill, W_dec_plot, W_dec_plot)
    map!(x -> isfinite(x) ? x : nan_fill, W_soc_plot, W_soc_plot)

    κ_max_view = min(maximum(κgrid_contour),
                     max(maximum(κ_dec_ridge), maximum(κ_soc_ridge)) + 0.25)

    plt_contour = plot(layout=(1, 2), size=(1200, 430), dpi=300,
                       margin=5Plots.mm)

    # Surfaces use rows = V* and columns = κ; ensure arrays match the grids above
    contourf!(plt_contour[1], κgrid_contour, Vstar_sorted, W_dec_plot;
              title="Decentralised welfare", xlabel="κ", ylabel="V*",
              c=:viridis, levels=levels, clims=clims_common, legend=:topright,
              fillalpha=0.98, linealpha=0.3, linewidth=0.5, nan_color=:white)
    plot!(plt_contour[1], κ_dec_ridge, Vstar_sorted; color=:black, linewidth=3.0,
          linestyle=:solid, label="κ^dec(V*)")
    xlims!(plt_contour[1], (minimum(κgrid_contour), κ_max_view))

    contourf!(plt_contour[2], κgrid_contour, Vstar_sorted, W_soc_plot;
              title="Planner welfare", xlabel="κ", ylabel="V*",
              c=:plasma, levels=levels, clims=clims_common, legend=:topright,
              fillalpha=0.98, linealpha=0.3, linewidth=0.5, nan_color=:white)
    plot!(plt_contour[2], κ_soc_ridge, Vstar_sorted; color=:black, linewidth=3.0,
          linestyle=:solid, label="κ^soc(V*)")
    xlims!(plt_contour[2], (minimum(κgrid_contour), κ_max_view))

    savefig(plt_contour, "figs/fig4_welfare_contour.pdf")

    # ========================================================================
    # Enhanced three-panel figure: Decentralised, Planner, Externality Wedge
    # ========================================================================

    println("   Creating enhanced three-panel welfare analysis...")

    # Compute welfare difference (externality)
    W_diff = W_soc_surface - W_dec_surface
    W_diff_plot = copy(W_diff)
    # Use symmetric color limits for difference plot
    diff_max = maximum(abs.(filter(isfinite, vec(W_diff))))
    clims_diff = (-diff_max, diff_max)
    # Fill NaNs with zero for difference plot
    map!(x -> isfinite(x) ? x : 0.0, W_diff_plot, W_diff_plot)

    plt_enhanced = plot(layout=(1, 3), size=(1800, 500), dpi=300,
                       margin=6Plots.mm,
                       plot_title="Welfare Analysis: Decentralised vs Planner",
                       plot_titlefontsize=16)

    # Panel 1: Decentralised welfare
    contourf!(plt_enhanced[1], κgrid_contour, Vstar_sorted, W_dec_plot;
              title="Decentralised", xlabel="κ", ylabel="V*",
              c=:viridis, levels=24, clims=clims_common,
              fillalpha=0.95, linewidth=0.8, linecolor=:black, linealpha=0.25)
    plot!(plt_enhanced[1], κ_dec_ridge, Vstar_sorted;
          color=:red, linewidth=3.5, linestyle=:solid,
          label="κ^{dec}(V*)", legend=:topright, legendfontsize=10)
    # Add markers at key V* values
    V_markers = [minimum(Vstar_sorted), median(Vstar_sorted), maximum(Vstar_sorted)]
    for Vm in V_markers
        idx = argmin(abs.(Vstar_sorted .- Vm))
        scatter!(plt_enhanced[1], [κ_dec_ridge[idx]], [Vstar_sorted[idx]];
                markersize=6, color=:red, markerstrokewidth=2,
                markerstrokecolor=:white, label=nothing)
    end
    xlims!(plt_enhanced[1], (minimum(κgrid_contour), κ_max_view))

    # Panel 2: Planner welfare
    contourf!(plt_enhanced[2], κgrid_contour, Vstar_sorted, W_soc_plot;
              title="Planner", xlabel="κ", ylabel="V*",
              c=:plasma, levels=24, clims=clims_common,
              fillalpha=0.95, linewidth=0.8, linecolor=:black, linealpha=0.25)
    plot!(plt_enhanced[2], κ_soc_ridge, Vstar_sorted;
          color=:cyan, linewidth=3.5, linestyle=:solid,
          label="κ^{soc}(V*)", legend=:topright, legendfontsize=10)
    for Vm in V_markers
        idx = argmin(abs.(Vstar_sorted .- Vm))
        scatter!(plt_enhanced[2], [κ_soc_ridge[idx]], [Vstar_sorted[idx]];
                markersize=6, color=:cyan, markerstrokewidth=2,
                markerstrokecolor=:white, label=nothing)
    end
    xlims!(plt_enhanced[2], (minimum(κgrid_contour), κ_max_view))

    # Panel 3: Welfare difference (externality wedge)
    contourf!(plt_enhanced[3], κgrid_contour, Vstar_sorted, W_diff_plot;
              title="Planner - Decentralised (Externality)", xlabel="κ", ylabel="V*",
              c=:RdBu, levels=20, clims=clims_diff,
              fillalpha=0.95, linewidth=0.8, linecolor=:black, linealpha=0.25)
    # Overlay both optimal ridges
    plot!(plt_enhanced[3], κ_dec_ridge, Vstar_sorted;
          color=:red, linewidth=2.5, linestyle=:dash,
          label="κ^{dec}", legend=:topright, legendfontsize=10)
    plot!(plt_enhanced[3], κ_soc_ridge, Vstar_sorted;
          color=:cyan, linewidth=2.5, linestyle=:dash,
          label="κ^{soc}")
    # Add zero contour (where welfare is equal)
    contour!(plt_enhanced[3], κgrid_contour, Vstar_sorted, W_diff_plot;
             levels=[0.0], linewidth=3, linecolor=:black,
             linestyle=:solid, label="W^{soc} = W^{dec}")
    xlims!(plt_enhanced[3], (minimum(κgrid_contour), κ_max_view))

    savefig(plt_enhanced, "figs/fig4_welfare_enhanced.pdf")

    # ========================================================================
    # Cross-section plots: Welfare vs κ at selected V* values
    # ========================================================================

    println("   Creating welfare cross-sections...")

    # Select 3 representative V* values
    V_low = quantile(Vstar_sorted, 0.25)
    V_mid = median(Vstar_sorted)
    V_high = quantile(Vstar_sorted, 0.75)
    V_selected = [V_low, V_mid, V_high]

    plt_crosssec = plot(layout=(1, 3), size=(1800, 450), dpi=300,
                        margin=6Plots.mm,
                        plot_title="Welfare Cross-Sections at Different Dispersion Levels",
                        plot_titlefontsize=16)

    for (panel_idx, V_target) in enumerate(V_selected)
        # Find closest V* in grid
        V_idx = argmin(abs.(Vstar_sorted .- V_target))
        V_actual = Vstar_sorted[V_idx]

        # Extract welfare cross-sections
        W_dec_slice = W_dec_surface[V_idx, :]
        W_soc_slice = W_soc_surface[V_idx, :]
        W_diff_slice = W_diff[V_idx, :]

        # Plot with proper masking
        valid_mask = isfinite.(W_dec_slice) .& isfinite.(W_soc_slice)
        κ_valid = κgrid_contour[valid_mask]

        plot!(plt_crosssec[panel_idx], κ_valid, W_dec_slice[valid_mask];
              linewidth=3, color=:blue, label="Decentralised",
              xlabel="Social coupling κ", ylabel="Welfare",
              title=@sprintf("V* = %.3f", V_actual),
              legend=:bottomright, legendfontsize=9,
              grid=true, gridstyle=:dash, gridalpha=0.3)
        plot!(plt_crosssec[panel_idx], κ_valid, W_soc_slice[valid_mask];
              linewidth=3, color=:red, label="Planner")

        # Mark optimal κ values
        κ_dec_opt = κ_dec_ridge[V_idx]
        κ_soc_opt = κ_soc_ridge[V_idx]

        if κ_dec_opt in κ_valid
            vline!(plt_crosssec[panel_idx], [κ_dec_opt];
                   color=:blue, linestyle=:dash, linewidth=2,
                   alpha=0.6, label=@sprintf("κ^{dec} = %.2f", κ_dec_opt))
        end
        if κ_soc_opt in κ_valid
            vline!(plt_crosssec[panel_idx], [κ_soc_opt];
                   color=:red, linestyle=:dash, linewidth=2,
                   alpha=0.6, label=@sprintf("κ^{soc} = %.2f", κ_soc_opt))
        end
    end

    savefig(plt_crosssec, "figs/fig4_welfare_crosssections.pdf")

    # ========================================================================
    # Optimal κ comparison plot
    # ========================================================================

    println("   Creating optimal κ comparison...")

    plt_optimal = plot(size=(800, 500), dpi=300,
                      xlabel="Stationary dispersion V*",
                      ylabel="Optimal social coupling κ",
                      title="Welfare-Maximizing Coupling Strength",
                      legend=:bottomright,
                      grid=true, gridstyle=:dash, gridalpha=0.3,
                      margin=5Plots.mm)

    plot!(plt_optimal, Vstar_sorted, κ_dec_ridge;
          linewidth=3.5, color=:blue, label="κ^{dec}(V*) - Decentralised",
          linestyle=:solid)
    plot!(plt_optimal, Vstar_sorted, κ_soc_ridge;
          linewidth=3.5, color=:red, label="κ^{soc}(V*) - Planner",
          linestyle=:solid)

    # Fill between to show wedge
    plot!(plt_optimal, Vstar_sorted, κ_dec_ridge;
          fillrange=κ_soc_ridge, fillalpha=0.2, fillcolor=:purple,
          linewidth=0, label="Policy wedge")

    # Add annotations
    mid_idx = div(length(Vstar_sorted), 2)
    annotate!(plt_optimal, Vstar_sorted[mid_idx],
             (κ_dec_ridge[mid_idx] + κ_soc_ridge[mid_idx])/2,
             text("Externality\ngap", 11, :center, :purple))

    savefig(plt_optimal, "figs/fig4_welfare_optimal_comparison.pdf")

    # ========================================================================
    # Diagnostic: NaN mask visualization
    # ========================================================================

    mask_grad = cgrad([:transparent, :black])
    plt_mask = plot(layout=(1, 2), size=(900, 430), dpi=300, margin=5Plots.mm)
    heatmap!(plt_mask[1], κgrid_contour, Vstar_sorted, float(nan_mask_dec);
             xlabel="κ", ylabel="V*", title="NaN mask (Decentralised)",
             c=mask_grad, clims=(0, 1), legend=false)
    heatmap!(plt_mask[2], κgrid_contour, Vstar_sorted, float(nan_mask_soc);
             xlabel="κ", ylabel="V*", title="NaN mask (Planner)",
             c=mask_grad, clims=(0, 1), legend=false)
    savefig(plt_mask, "figs/fig4_welfare_nanmask.pdf")

    println("   ✓ Enhanced welfare figures saved:")
    println("      • fig4_welfare_enhanced.pdf (3-panel with externality)")
    println("      • fig4_welfare_crosssections.pdf (welfare vs κ slices)")
    println("      • fig4_welfare_optimal_comparison.pdf (κ^dec vs κ^soc)")
    println("      • fig4_welfare_nanmask.pdf (diagnostic)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
