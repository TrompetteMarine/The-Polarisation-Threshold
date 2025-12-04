#!/usr/bin/env julia
using BeliefSim
using BeliefSim.Types
using BeliefSim.OUResets
using BeliefSim.Stats
using Plots
using Statistics
using Random

# ----------------------------------------------------------------------
# Helpers (reused from fig4_welfare.jl with minor refactors)
# ----------------------------------------------------------------------

# Robust color limits using finite-only quantiles with safe fallbacks
function finite_clims(A...; q=(0.05, 0.95))
    vals = Float64[]
    for X in A
        append!(vals, filter(isfinite, vec(X)))
    end
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

# Simple histogram for debugging distribution of finite values
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

# Diagnose a 2D surface
function diagnose_surface(A, xgrid, ygrid, name; max_lines=5, save_plots=true)
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

# Build κ and σ grids (kept consistent with fig4_welfare.jl)
# Increased resolution for smoother contours: 9→15 V* levels, 28→60 κ points
function build_grids()
    κgrid = collect(range(0.0, 2.2, length=60))
    σgrid = collect(range(0.6, 1.3, length=15))
    return κgrid, σgrid
end

# Compute welfare curves, surfaces, ridges, and masks
function compute_welfare_data(p; α=0.25, β=0.8)
    κgrid, σgrid = build_grids()
    nκ = length(κgrid)
    nσ = length(σgrid)

    welfare_1d = compute_welfare_curves(p, κgrid; α=α, β=β, N=8_000,
                                        T=260.0, dt=0.01, seed=99)

    Vstar_vals = similar(σgrid)
    W_dec_surface = Matrix{Float64}(undef, nσ, nκ)  # rows = V* (σ), cols = κ
    W_soc_surface = similar(W_dec_surface)
    κ_dec_ridge = similar(σgrid)
    κ_soc_ridge = similar(σgrid)

    for (j, σval) in enumerate(σgrid)
        pσ = Params(λ=p.λ, σ=σval, Θ=p.Θ, c0=p.c0, hazard=p.hazard)
        Vstar_vals[j] = estimate_Vstar(pσ; N=6_000, T=220.0, dt=0.01, burn_in=80.0, seed=500 + j)
        surf = compute_welfare_curves(pσ, κgrid; α=α, β=β, N=5_000,
                                      T=180.0, dt=0.01, seed=800 + j)
        W_dec_surface[j, :] .= surf.W_dec
        W_soc_surface[j, :] .= surf.W_soc
        κ_dec_ridge[j] = κgrid[argmax(surf.W_dec)]
        κ_soc_ridge[j] = κgrid[argmax(surf.W_soc)]
    end

    diagnose_surface(W_dec_surface, κgrid, Vstar_vals, "W_dec_surface_raw")
    diagnose_surface(W_soc_surface, κgrid, Vstar_vals, "W_soc_surface_raw")

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

    finite_mask_rows = isfinite.(Vstar_sorted)
    if !all(finite_mask_rows)
        Vstar_sorted = Vstar_sorted[finite_mask_rows]
        W_dec_surface = W_dec_surface[finite_mask_rows, :]
        W_soc_surface = W_soc_surface[finite_mask_rows, :]
        κ_dec_ridge = κ_dec_ridge[finite_mask_rows]
        κ_soc_ridge = κ_soc_ridge[finite_mask_rows]
    end

    @assert size(W_dec_surface) == (length(Vstar_sorted), length(κgrid))
    @assert size(W_soc_surface) == (length(Vstar_sorted), length(κgrid))

    nan_mask_dec = .!isfinite.(W_dec_surface)
    nan_mask_soc = .!isfinite.(W_soc_surface)

    return (
        κgrid = κgrid,
        σgrid = σgrid,
        Vstar_sorted = Vstar_sorted,
        welfare_1d = welfare_1d,
        W_dec_surface = W_dec_surface,
        W_soc_surface = W_soc_surface,
        κ_dec_ridge = κ_dec_ridge,
        κ_soc_ridge = κ_soc_ridge,
        nan_mask_dec = nan_mask_dec,
        nan_mask_soc = nan_mask_soc
    )
end

# Shared clims/levels based on finite values
function shared_clims_levels(W_dec_surface, W_soc_surface; nlevels=14)
    clims = finite_clims(W_dec_surface, W_soc_surface)
    levels = collect(range(clims[1], clims[2]; length=nlevels))
    @info "Shared clims" clims=clims levels=levels
    print_histogram(filter(isfinite, vec(W_dec_surface)); bins=12)
    print_histogram(filter(isfinite, vec(W_soc_surface)); bins=12)
    return clims, levels
end

# ----------------------------------------------------------------------
# Figure helpers
# ----------------------------------------------------------------------

function save_both(plt, stem)
    savefig(plt, "figs/$(stem).pdf")
    savefig(plt, "figs/$(stem).png")
end

# Figure A: Δκ(V*) ridge difference
function plot_externality_ridge(data)
    κ_dec = data.κ_dec_ridge
    κ_soc = data.κ_soc_ridge
    V = data.Vstar_sorted
    wedge = κ_soc .- κ_dec
    max_idx = argmax(abs.(wedge))

    plt = plot(V, wedge; xlabel="V*", ylabel="Δκ(V*)",
               title="Externality wedge in κ: planner vs decentralised",
               color=:black, linewidth=2.4, legend=false, size=(900, 450), dpi=300)
    hline!(plt, [0.0]; color=:gray, linestyle=:dash, linewidth=1.5)
    pos_mask = wedge .> 0
    neg_mask = wedge .< 0
    band = 0.0 .* wedge
    plot!(plt, V[pos_mask], wedge[pos_mask]; fillrange=band[pos_mask], fillalpha=0.15, color=:green, label=false)
    plot!(plt, V[neg_mask], wedge[neg_mask]; fillrange=band[neg_mask], fillalpha=0.15, color=:red, label=false)
    annotate!(plt, (V[max_idx], wedge[max_idx]),
              text("largest wedge (|Δκ|=$(round(abs(wedge[max_idx]); digits=3)))", 10, :left))
    return plt
end

# Figure B: ΔW surface with ridges
function plot_welfare_difference(data; nlevels=13)
    ΔW = data.W_soc_surface .- data.W_dec_surface
    V = data.Vstar_sorted
    κgrid = data.κgrid

    finite_vals = filter(isfinite, vec(ΔW))
    cmax = isempty(finite_vals) ? 1.0 : quantile(abs.(finite_vals), 0.95)
    clims = (-cmax, cmax)
    levels = collect(range(clims[1], clims[2]; length=nlevels))

    # Keep NaNs as missing for plotting; mask with nan_color
    ΔW_plot = copy(ΔW)
    map!(x -> isfinite(x) ? x : NaN, ΔW_plot, ΔW_plot)

    # Restrict x-limits to columns with finite values to avoid vast blank regions
    finite_cols = findall(j -> any(isfinite, ΔW[:, j]), 1:size(ΔW, 2))
    κ_minmax = isempty(finite_cols) ? (minimum(κgrid), maximum(κgrid)) :
                (minimum(κgrid[finite_cols]), maximum(κgrid[finite_cols]))
    pad = 0.05 * max(κ_minmax[2] - κ_minmax[1], 1e-6)
    κ_lims = (κ_minmax[1] - pad, κ_minmax[2] + pad)

    plt = plot(layout=(1, 2), size=(1200, 450), dpi=300, margin=5Plots.mm)

    contourf!(plt[1], κgrid, V, ΔW_plot; c=:balance, clims=clims, levels=levels,
              xlabel="κ", ylabel="V*", title="ΔW = W_soc − W_dec",
              fillalpha=0.95, linealpha=0.6, colorbar_title="ΔW", nan_color=:white)
    contour!(plt[1], κgrid, V, ΔW_plot; levels=[0.0], color=:black, linewidth=2.5, legend=false)
    xlims!(plt[1], κ_lims)

    contourf!(plt[2], κgrid, V, ΔW_plot; c=:balance, clims=clims, levels=levels,
              xlabel="κ", ylabel="V*", title="ΔW with ridges",
              fillalpha=0.95, linealpha=0.6, colorbar_title="ΔW", nan_color=:white, legend=:topright)
    plot!(plt[2], data.κ_dec_ridge, V; color=:black, linewidth=2.8, label="κ^dec(V*)")
    plot!(plt[2], data.κ_soc_ridge, V; color=:gray, linewidth=2.8, label="κ^soc(V*)")
    xlims!(plt[2], κ_lims)

    # Mark κ of max ΔW at median V* (if finite)
    mid_idx = round(Int, length(V) / 2)
    if any(isfinite, ΔW[mid_idx, :])
        κ_max_idx = argmax(ΔW[mid_idx, :])
        κ_star = κgrid[κ_max_idx]
        v_star = V[mid_idx]
        v_max = ΔW[mid_idx, κ_max_idx]
        vline!(plt[2], [κ_star]; color=:red, linestyle=:dash, linewidth=2, label="max ΔW at V*≈$(round(v_star; digits=3))")
        annotate!(plt[2], (κ_star, v_star), text("κ≈$(round(κ_star; digits=3))\nΔW≈$(round(v_max; digits=3))", 9, :left))
    end
    return plt
end

# Figure C: slices at representative V*
function plot_slices_Vstar(data)
    κgrid = data.κgrid
    V = data.Vstar_sorted
    dec = data.W_dec_surface
    soc = data.W_soc_surface
    κ_dec = data.κ_dec_ridge
    κ_soc = data.κ_soc_ridge

    finite_V = filter(isfinite, V)
    idxs = map(q -> begin
        target = quantile(finite_V, q)
        argmin(abs.(V .- target))
    end, (0.1, 0.5, 0.9))

    plt = plot(layout=(3, 1), size=(950, 900), dpi=300, margin=5Plots.mm)
    subtitles = ["Low dispersion", "Mid dispersion", "High dispersion"]
    colors = (:black, :blue)
    for (panel, idx) in enumerate(idxs)
        y_dec = dec[idx, :]
        y_soc = soc[idx, :]
        plot!(plt[panel], κgrid, y_dec; color=colors[1], linewidth=2.3, label="W_dec(κ | V*)")
        plot!(plt[panel], κgrid, y_soc; color=colors[2], linewidth=2.3, label="W_soc(κ | V*)")
        vline!(plt[panel], [κ_dec[idx]]; color=colors[1], linestyle=:dash, linewidth=1.8, label="κ^dec")
        vline!(plt[panel], [κ_soc[idx]]; color=colors[2], linestyle=:dash, linewidth=1.8, label="κ^soc")
        scatter!(plt[panel], [κ_dec[idx]], [y_dec[argmin(abs.(κgrid .- κ_dec[idx]))]]; color=colors[1], markerstrokecolor=:white, label=false)
        scatter!(plt[panel], [κ_soc[idx]], [y_soc[argmin(abs.(κgrid .- κ_soc[idx]))]]; color=colors[2], markerstrokecolor=:white, label=false)
        title!(plt[panel], "$(subtitles[panel]) V*=$(round(V[idx]; digits=3))")
        xlabel!(plt[panel], "κ")
        ylabel!(plt[panel], "W(κ | V*)")
        if panel == 1
            plot!(plt[panel]; legend=:topright)
        else
            plot!(plt[panel]; legend=:bottomleft)
        end
    end
    return plt
end

# Figure D: slices at representative κ
function plot_slices_kappa(data)
    κgrid = data.κgrid
    V = data.Vstar_sorted
    dec = data.W_dec_surface
    soc = data.W_soc_surface

    κ_vals = [κgrid[round(Int, 0.15*length(κgrid))],
              κgrid[round(Int, 0.5*length(κgrid))],
              κgrid[round(Int, 0.85*length(κgrid))]]
    idxs = map(κ -> argmin(abs.(κgrid .- κ)), κ_vals)

    plt = plot(layout=(3, 1), size=(950, 900), dpi=300, margin=5Plots.mm)
    subtitles = ["Low κ", "Mid κ", "High κ"]
    colors = (:black, :blue)
    for (panel, (κval, idx)) in enumerate(zip(κ_vals, idxs))
        y_dec = dec[:, idx]
        y_soc = soc[:, idx]
        plot!(plt[panel], V, y_dec; color=colors[1], linewidth=2.3, label="W_dec(V* | κ)")
        plot!(plt[panel], V, y_soc; color=colors[2], linewidth=2.3, label="W_soc(V* | κ)")
        title!(plt[panel], "$(subtitles[panel]) κ=$(round(κval; digits=3))")
        xlabel!(plt[panel], "V*")
        ylabel!(plt[panel], "W(V* | κ)")
        plot!(plt[panel]; legend=:topright)
    end
    return plt
end

# Figure E: pedagogical contour with annotations
function plot_pedagogical_contour(data; nlevels=10)
    clims, _ = shared_clims_levels(data.W_dec_surface, data.W_soc_surface; nlevels=nlevels)
    levels = collect(range(clims[1], clims[2]; length=nlevels))
    nan_fill = clims[1] - 0.1 * max(abs(clims[1]), 1.0)

    Wd = copy(data.W_dec_surface); map!(x -> isfinite(x) ? x : nan_fill, Wd, Wd)
    Ws = copy(data.W_soc_surface); map!(x -> isfinite(x) ? x : nan_fill, Ws, Ws)

    κgrid = data.κgrid
    V = data.Vstar_sorted

    plt = plot(layout=(1, 2), size=(1250, 460), dpi=300, margin=5Plots.mm)
    contourf!(plt[1], κgrid, V, Wd; c=:viridis, clims=clims, levels=levels,
              xlabel="κ", ylabel="V*", title="Decentralised welfare (pedagogical)",
              fillalpha=0.95, linealpha=0.6, legend=false, nan_color=:white)
    plot!(plt[1], data.κ_dec_ridge, V; color=:black, linewidth=3.0, label=false)
    annotate!(plt[1], (0.2, median(V)), text("Safe plateau", 9, :left))
    annotate!(plt[1], (0.45, 0.65), text("Welfare crater", 9, :left, :red))
    annotate!(plt[1], (0.8, 0.9), text("Danger zone", 9, :left, :purple))

    contourf!(plt[2], κgrid, V, Ws; c=:plasma, clims=clims, levels=levels,
              xlabel="κ", ylabel="V*", title="Planner welfare (pedagogical)",
              fillalpha=0.95, linealpha=0.6, legend=false, nan_color=:white)
    plot!(plt[2], data.κ_soc_ridge, V; color=:black, linewidth=3.0, label=false)
    annotate!(plt[2], (0.2, median(V)), text("Safe plateau", 9, :left))
    annotate!(plt[2], (0.45, 0.65), text("Welfare crater", 9, :left, :red))
    annotate!(plt[2], (0.8, 0.9), text("Danger zone", 9, :left, :purple))

    return plt
end

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

function main()
    Random.seed!(2025)
    mkpath("figs")

    # Match baseline parameters from fig4_welfare.jl
    p = Params(λ=0.65, σ=1.85, Θ=2.0, c0=1.0, hazard=StepHazard(0.6))
    α = 0.25
    β = 0.8

    data = compute_welfare_data(p; α=α, β=β)

    # Figure A
    plt_ridge = plot_externality_ridge(data)
    save_both(plt_ridge, "fig4_externality_kappa_ridge")

    # Figure B
    plt_diff = plot_welfare_difference(data)
    save_both(plt_diff, "fig4_welfare_difference_surface")

    # Figure C
    plt_slices_V = plot_slices_Vstar(data)
    save_both(plt_slices_V, "fig4_welfare_slices_Vstar")

    # Figure D
    plt_slices_k = plot_slices_kappa(data)
    save_both(plt_slices_k, "fig4_welfare_slices_kappa")

    # Figure E
    plt_pedagogical = plot_pedagogical_contour(data)
    save_both(plt_pedagogical, "fig4_welfare_contour_pedagogical")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
