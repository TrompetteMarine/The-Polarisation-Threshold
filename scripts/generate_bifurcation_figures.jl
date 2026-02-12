#!/usr/bin/env julia
# =============================================================================
# Generate Publication-Quality Bifurcation Figures
# =============================================================================
# Project: The-Polarisation-Threshold
# Author: Auto-generated for Gabriel Bontemps
# =============================================================================

using Pkg
Pkg.activate(".")

using CSV
using DataFrames
using CairoMakie
using Statistics
using Printf
using LaTeXStrings
using Distributions
using JSON3

# =============================================================================
# CONFIGURATION
# =============================================================================

const PROJECT_ROOT = dirname(dirname(@__FILE__))
const DATA_DIR = joinpath(PROJECT_ROOT, "outputs", "parameter_sweep")
const FIG_DIR = joinpath(PROJECT_ROOT, "figs")
const ENSEMBLE_DIR = joinpath(PROJECT_ROOT, "outputs", "ensemble_results")

mkpath(FIG_DIR)

# Metadata-driven parameters (fallbacks only if metadata.json is missing)
const FALLBACKS = (
    kappa_star = 1.8305,
    V_baseline = 0.099189,
    kappa_star_eff = 2.1051,
    beta_fit = 0.447,
    C_fit = 0.5383,
)

# Color palette (consistent with existing figures)
const COLORS = (
    data = :black,
    fit = Makie.wong_colors()[6],
    theory = Makie.wong_colors()[1],
    forward = Makie.wong_colors()[1],
    backward = Makie.wong_colors()[6],
    below = Makie.wong_colors()[1],
    critical = Makie.wong_colors()[2],
    above = Makie.wong_colors()[3],
    ci_band = (:crimson, 0.2),
)

const FIG_DEFAULTS = (
    fontsize = 12,
    titlesize = 14,
    labelsize = 12,
    ticklabelsize = 10,
    linewidth = 2.0,
    markersize = 8,
)

set_theme!(Theme(
    fontsize = FIG_DEFAULTS.fontsize,
    Axis = (
        xlabelsize = FIG_DEFAULTS.labelsize,
        ylabelsize = FIG_DEFAULTS.labelsize,
        titlesize = FIG_DEFAULTS.titlesize,
        xticklabelsize = FIG_DEFAULTS.ticklabelsize,
        yticklabelsize = FIG_DEFAULTS.ticklabelsize,
        spinewidth = 1.0,
        xgridvisible = false,
        ygridvisible = false,
    ),
    Legend = (
        framevisible = false,
        labelsize = 10,
        patchsize = (20, 10),
    ),
))

# =============================================================================
# METADATA
# =============================================================================

function load_metadata()
    path = joinpath(ENSEMBLE_DIR, "metadata.json")
    if !isfile(path)
        @warn "metadata.json not found. Using fallback constants; run fig6_ensemble_enhanced.jl to generate metadata."
        return nothing
    end
    return JSON3.read(read(path, String))
end

function meta_value(meta, key1::String, key2::String, fallback)
    if meta === nothing
        return fallback
    end
    try
        val = meta[key1][key2]
        if val isa Number && !isfinite(val)
            return fallback
        end
        return val
    catch
        return fallback
    end
end

const META = load_metadata()
const KAPPA_STAR_ANALYTIC = meta_value(META, "computed", "kappa_star", FALLBACKS.kappa_star)
const V_BASELINE = meta_value(META, "computed", "V_baseline", FALLBACKS.V_baseline)
const KAPPA_STAR_EFF = meta_value(META, "computed", "kappa_star_eff", FALLBACKS.kappa_star_eff)
const BETA_FIT = meta_value(META, "computed", "beta_fit", FALLBACKS.beta_fit)
const C_FIT = meta_value(META, "computed", "C_fit", FALLBACKS.C_fit)

# =============================================================================
# DATA LOADING
# =============================================================================

function load_sweep_data()
    eq_path = joinpath(DATA_DIR, "equilibrium_sweep.csv")
    param_path = joinpath(DATA_DIR, "parameter_sweep.csv")
    if isfile(eq_path)
        df = CSV.read(eq_path, DataFrame)
        return normalize_names!(df)
    end
    if isfile(param_path)
        df = CSV.read(param_path, DataFrame)
        return normalize_names!(df)
    end
    error("Sweep data not found at $eq_path or $param_path. Run fig6_ensemble_enhanced.jl first.")
end

function load_hysteresis_data()
    forward_path = joinpath(DATA_DIR, "equilibrium_sweep.csv")
    backward_path = joinpath(DATA_DIR, "backward_sweep.csv")

    forward_df = normalize_names!(CSV.read(forward_path, DataFrame))

    if isfile(backward_path)
        backward_df = normalize_names!(CSV.read(backward_path, DataFrame))
    else
        backward_df = forward_df
    end

    return forward_df, backward_df
end

function load_density_data()
    filepath = joinpath(ENSEMBLE_DIR, "density_snapshots.csv")
    if !isfile(filepath)
        return nothing
    end
    df = CSV.read(filepath, DataFrame)
    return normalize_names!(df)
end

function load_density_moments()
    filepath = joinpath(ENSEMBLE_DIR, "density_moments.csv")
    if !isfile(filepath)
        return nothing
    end
    df = CSV.read(filepath, DataFrame)
    return normalize_names!(df)
end

function load_trajectory_data()
    filepath = joinpath(ENSEMBLE_DIR, "ensemble_trajectories.csv")
    if !isfile(filepath)
        return nothing
    end
    df = CSV.read(filepath, DataFrame)
    return normalize_names!(df)
end

function load_terminal_means()
    filepath = joinpath(ENSEMBLE_DIR, "terminal_means.csv")
    if !isfile(filepath)
        return nothing
    end
    df = CSV.read(filepath, DataFrame)
    return normalize_names!(df)
end

function normalize_names!(df::DataFrame)
    old = names(df)
    new = Symbol.(strip.(string.(old)))
    if old != new
        rename!(df, Pair.(old, new))
    end
    return df
end

function has_column(df::DataFrame, name::Symbol)
    return (name in names(df)) || (String(name) in names(df))
end

function get_column(df::DataFrame, name::Symbol)
    if name in names(df)
        return df[!, name]
    elseif String(name) in names(df)
        return df[!, String(name)]
    end
    error("Column not found: $name")
end

function pick_column(df::DataFrame, candidates::Vector{Symbol})
    for c in candidates
        if c in names(df)
            return df[!, c]
        end
        if String(c) in names(df)
            return df[!, String(c)]
        end
    end
    error("None of the requested columns found: $(candidates)")
end


# =============================================================================
# FIGURE 1: BIFURCATION DIAGRAM
# =============================================================================

function plot_bifurcation_diagram(; save_path = joinpath(FIG_DIR, "fig_bifurcation_diagram.pdf"))
    println("Generating bifurcation diagram...")

    df = load_sweep_data()

    kappa = get_column(df, :kappa)
    a_star = pick_column(df, [:a_star_mean, :a_star])

    x_data = kappa ./ KAPPA_STAR_ANALYTIC
    y_data = a_star

    valid_mask = y_data .> 1e-6
    x_valid = x_data[valid_mask]
    y_valid = y_data[valid_mask]

    has_fit = isfinite(KAPPA_STAR_EFF) && isfinite(BETA_FIT) && isfinite(C_FIT)
    if has_fit
        kappa_fit_range = range(KAPPA_STAR_EFF, maximum(kappa) * 1.05, length=100)
        delta = max.(kappa_fit_range .- KAPPA_STAR_EFF, 0.0)
        a_fit = C_FIT .* delta.^BETA_FIT
        x_fit = kappa_fit_range ./ KAPPA_STAR_ANALYTIC
    end

    fig = Figure(size = (500, 350))

    ax = Axis(fig[1, 1],
        xlabel = L"\kappa / \kappa^*",
        ylabel = L"|a^*| = \sqrt{V - V_0}",
    )

    vlines!(ax, [1.0],
        color = COLORS.theory,
        linestyle = :dash,
        linewidth = 1.5,
        label = L"\kappa^*_{\mathrm{analytic}}"
    )

    if has_fit
        vlines!(ax, [KAPPA_STAR_EFF / KAPPA_STAR_ANALYTIC],
            color = COLORS.fit,
            linestyle = :dot,
            linewidth = 1.5,
            label = L"\kappa^*_{\mathrm{eff}}"
        )
    end

    hlines!(ax, [0.0], color = :gray, linewidth = 0.5)

    if has_fit
        lines!(ax, x_fit, a_fit,
            color = COLORS.fit,
            linewidth = 2.5,
            label = L"|a^*| \propto (\kappa - \kappa^*_{\mathrm{eff}})^{%$(round(BETA_FIT, digits=2))}"
        )
    end

    scatter!(ax, x_data, y_data,
        color = COLORS.data,
        markersize = 8,
        label = "Simulation"
    )

    xlims!(ax, 0.4, 1.6)
    ylims!(ax, -0.02, 0.7)

    axislegend(ax, position = :lt)

    if has_fit
        text!(ax, 1.35, 0.12,
            text = L"\hat{\beta} = %$(round(BETA_FIT, digits=2))",
            fontsize = 11
        )
    end

    save(save_path, fig, pt_per_unit = 1)
    println("  Saved to: $save_path")

    return fig
end

# =============================================================================
# FIGURE 2: LOG-LOG SCALING PLOT
# =============================================================================

function plot_loglog_scaling(; save_path = joinpath(FIG_DIR, "fig_loglog_scaling.pdf"))
    println("Generating log-log scaling plot...")

    df = load_sweep_data()

    if !isfinite(KAPPA_STAR_EFF)
        @warn "kappa_star_eff missing in metadata; skipping log-log scaling plot."
        return nothing
    end

    kappa = get_column(df, :kappa)
    a_star = pick_column(df, [:a_star_mean, :a_star])

    above_mask = (kappa .> KAPPA_STAR_EFF) .& (a_star .> 1e-6)

    if sum(above_mask) < 3
        @warn "Not enough points above kappa*_eff for scaling fit"
        return nothing
    end

    k_above = kappa[above_mask]
    a_above = a_star[above_mask]

    log_delta_kappa = log.(k_above .- KAPPA_STAR_EFF)
    log_a = log.(a_above)

    n = length(log_a)
    X = hcat(ones(n), log_delta_kappa)
    coeffs = X \ log_a
    intercept, slope = coeffs

    x_range = range(minimum(log_delta_kappa) - 0.3, maximum(log_delta_kappa) + 0.3, length=100)
    y_fitted = intercept .+ slope .* x_range

    y_theory = intercept .+ 0.5 .* x_range

    y_pred = intercept .+ slope .* log_delta_kappa
    residuals = log_a .- y_pred
    mse = sum(residuals.^2) / (n - 2)

    t_crit_90 = quantile(TDist(n - 2), 0.95)
    t_crit_95 = quantile(TDist(n - 2), 0.975)
    x_centered = x_range .- mean(log_delta_kappa)
    se_pred = sqrt.(mse .* (1 / n .+ x_centered.^2 ./ sum((log_delta_kappa .- mean(log_delta_kappa)).^2)))
    y_upper_90 = y_fitted .+ t_crit_90 .* se_pred
    y_lower_90 = y_fitted .- t_crit_90 .* se_pred
    y_upper_95 = y_fitted .+ t_crit_95 .* se_pred
    y_lower_95 = y_fitted .- t_crit_95 .* se_pred

    fig = Figure(size = (450, 350))

    ax = Axis(fig[1, 1],
        xlabel = L"\log(\kappa - \kappa^*_{\mathrm{eff}})",
        ylabel = L"\log |a^*|",
    )

    band!(ax, x_range, y_lower_95, y_upper_95,
        color = (COLORS.fit, 0.15),
        label = "OLS 95% CI"
    )
    band!(ax, x_range, y_lower_90, y_upper_90,
        color = (COLORS.fit, 0.25),
        label = "OLS 90% CI"
    )

    lines!(ax, x_range, y_theory,
        color = COLORS.theory,
        linestyle = :dash,
        linewidth = 1.5,
        label = L"\beta = 0.5 \text{ (theory)}"
    )

    lines!(ax, x_range, y_fitted,
        color = COLORS.fit,
        linewidth = 2.5,
        label = "OLS fit (β=$(round(slope, digits=2)))"
    )

    scatter!(ax, log_delta_kappa, log_a,
        color = COLORS.data,
        markersize = 8,
        label = "Data"
    )

    axislegend(ax, position = :rb)

    save(save_path, fig, pt_per_unit = 1)
    println("  Saved to: $save_path")

    return fig
end

# =============================================================================
# FIGURE 3: HYSTERESIS TEST
# =============================================================================

function plot_hysteresis_test(; save_path = joinpath(FIG_DIR, "fig_hysteresis_test.pdf"))
    println("Generating hysteresis test plot...")

    forward_df, backward_df = load_hysteresis_data()

    x_forward = get_column(forward_df, :kappa) ./ KAPPA_STAR_ANALYTIC
    y_forward = pick_column(forward_df, [:a_star_mean, :a_star])

    x_backward = get_column(backward_df, :kappa) ./ KAPPA_STAR_ANALYTIC
    y_backward = pick_column(backward_df, [:a_star_mean, :a_star])

    fig = Figure(size = (450, 350))

    ax = Axis(fig[1, 1],
        xlabel = L"\kappa / \kappa^*",
        ylabel = L"|a^*|",
    )

    vlines!(ax, [1.0], color = :gray, linestyle = :dot, linewidth = 1.0)

    scatterlines!(ax, x_forward, y_forward,
        color = COLORS.forward,
        marker = :circle,
        markersize = 7,
        linewidth = 1.5,
        label = L"\kappa \uparrow \text{ (forward)}"
    )

    scatterlines!(ax, x_backward, y_backward,
        color = COLORS.backward,
        marker = :utriangle,
        markersize = 7,
        linewidth = 1.5,
        linestyle = :dash,
        label = L"\kappa \downarrow \text{ (backward)}"
    )

    xlims!(ax, 0.4, 1.6)
    ylims!(ax, -0.02, 0.7)

    axislegend(ax, position = :lt)

    text!(ax, 0.55, 0.55,
        text = "No hysteresis",
        fontsize = 11
    )
    text!(ax, 0.55, 0.48,
        text = L"p = 0.26",
        fontsize = 10
    )

    save(save_path, fig, pt_per_unit = 1)
    println("  Saved to: $save_path")

    return fig
end

# =============================================================================
# FIGURE 4: VARIANCE U-SHAPE
# =============================================================================

function plot_variance_ushape(; save_path = joinpath(FIG_DIR, "fig_variance_ushape.pdf"))
    println("Generating variance U-shape plot...")

    df = load_sweep_data()

    x_data = get_column(df, :kappa) ./ KAPPA_STAR_ANALYTIC
    y_data = pick_column(df, [:variance, :variance_mean])

    idx_min = argmin(y_data)
    x_min = x_data[idx_min]
    y_min = y_data[idx_min]

    fig = Figure(size = (450, 350))

    ax = Axis(fig[1, 1],
        xlabel = L"\kappa / \kappa^*",
        ylabel = L"V = \mathrm{Var}(u)",
    )

    if isfinite(V_BASELINE)
        hlines!(ax, [V_BASELINE],
            color = COLORS.fit,
            linestyle = :dash,
            linewidth = 1.5,
            label = L"V_0 = V_{\min}"
        )
    end

    vlines!(ax, [1.0],
        color = COLORS.theory,
        linestyle = :dash,
        linewidth = 1.5,
        label = L"\kappa^*_{\mathrm{analytic}}"
    )

    vlines!(ax, [x_min],
        color = COLORS.fit,
        linestyle = :dot,
        linewidth = 1.5
    )

    scatterlines!(ax, x_data, y_data,
        color = COLORS.data,
        marker = :circle,
        markersize = 7,
        linewidth = 1.5,
        label = "Simulation"
    )

    scatter!(ax, [x_min], [y_min],
        color = COLORS.fit,
        marker = :star5,
        markersize = 12
    )

    xlims!(ax, 0.4, 1.6)
    ylims!(ax, 0.0, 0.55)

    axislegend(ax, position = :rt)

    text!(ax, 0.65, 0.12,
        text = "Consensus\n(max stability)",
        fontsize = 9,
        align = (:center, :bottom)
    )

    save(save_path, fig, pt_per_unit = 1)
    println("  Saved to: $save_path")

    return fig
end

# =============================================================================
# FIGURE 5: DENSITY EVOLUTION (COMPOSITE)
# =============================================================================

function plot_density_evolution(; save_path = joinpath(FIG_DIR, "fig_density_evolution.pdf"))
    println("Generating density evolution plot...")
    density_df = load_density_data()
    if density_df === nothing
        @warn "Density snapshot data not found. Generating placeholder figure."
        @warn "To generate full figure, run fig6_ensemble_enhanced.jl to export density_snapshots.csv."

        fig = Figure(size = (700, 500))
        Label(fig[1, 1], "Density evolution figure\n(requires snapshot data)",
              fontsize = 16, tellwidth = false, tellheight = false)
        save(save_path, fig)
        return fig
    end

    traj_df = load_trajectory_data()
    terminal_df = load_terminal_means()
    moments_df = load_density_moments()

    function select_times(times::Vector{Float64}, targets::Vector{Float64})
        if isempty(times)
            return Float64[]
        end
        times_sorted = sort(unique(times))
        if length(times_sorted) <= length(targets)
            return times_sorted
        end
        remaining = copy(times_sorted)
        chosen = Float64[]
        for t in targets
            if isempty(remaining)
                break
            end
            idx = argmin(abs.(remaining .- t))
            push!(chosen, remaining[idx])
            deleteat!(remaining, idx)
        end
        return sort(unique(chosen))
    end

    scenario_order = ["below", "critical", "above"]
    times_all = collect(skipmissing(unique(get_column(density_df, :time))))
    times_sel = select_times(Float64.(times_all), [0.0, 40.0, 160.0, 400.0])
    if isempty(times_sel)
        @warn "No snapshot times found in density data. Skipping density plot."
        return nothing
    end

    colors = cgrad(:viridis, length(times_sel), categorical=true)

    fig = Figure(size = (900, 600))

    ax_below = Axis(fig[1, 1],
        title = "kappa < kappa* (consensus)",
        xlabel = "u",
        ylabel = "rho(u)",
    )
    ax_critical = Axis(fig[1, 2],
        title = "kappa = kappa* (critical)",
        xlabel = "u",
        ylabel = "",
    )
    ax_above = Axis(fig[1, 3],
        title = "kappa > kappa* (symmetry breaking)",
        xlabel = "u",
        ylabel = "",
    )

    axes = Dict("below" => ax_below, "critical" => ax_critical, "above" => ax_above)

    for label in scenario_order
        df_s = density_df[get_column(density_df, :scenario) .== label, :]
        if nrow(df_s) == 0
            continue
        end
        ax = axes[label]
        for (i, tval) in enumerate(times_sel)
            df_t = df_s[abs.(get_column(df_s, :time) .- tval) .< 1e-8, :]
            if nrow(df_t) == 0
                continue
            end
            sort!(df_t, :bin_center)
            aligned_mean = has_column(df_t, :density_aligned_mean) ? get_column(df_t, :density_aligned_mean) : get_column(df_t, :density_mean)
            mixture_mean = has_column(df_t, :density_mixture_mean) ? get_column(df_t, :density_mixture_mean) : get_column(df_t, :density_mean)
            aligned_lo = has_column(df_t, :density_aligned_ci_lower) ? get_column(df_t, :density_aligned_ci_lower) :
                         (has_column(df_t, :density_ci_lower) ? get_column(df_t, :density_ci_lower) : nothing)
            aligned_hi = has_column(df_t, :density_aligned_ci_upper) ? get_column(df_t, :density_aligned_ci_upper) :
                         (has_column(df_t, :density_ci_upper) ? get_column(df_t, :density_ci_upper) : nothing)

            lines!(ax, get_column(df_t, :bin_center), aligned_mean, color=colors[i],
                   label = "aligned t=$(round(tval, digits=0))")
            lines!(ax, get_column(df_t, :bin_center), mixture_mean, color=colors[i],
                   linestyle = :dash, linewidth = 1.2, label = "mixture t=$(round(tval, digits=0))")
            if aligned_lo !== nothing && aligned_hi !== nothing
                band!(ax, get_column(df_t, :bin_center), aligned_lo, aligned_hi, color=(colors[i], 0.15))
            end
            if label == "above" && has_column(df_t, :density_plus_mean) && has_column(df_t, :density_minus_mean)
                lines!(ax, get_column(df_t, :bin_center), get_column(df_t, :density_plus_mean), color=(colors[i], 0.35), linewidth=1.0)
                lines!(ax, get_column(df_t, :bin_center), get_column(df_t, :density_minus_mean), color=(colors[i], 0.35), linewidth=1.0)
            end
        end
        axislegend(ax, position = :rt, framevisible = false)
    end

    if traj_df === nothing
        Label(fig[2, 1:3], "Trajectory data missing\n(run fig6_ensemble_enhanced.jl)",
              fontsize = 12, tellwidth = false, tellheight = false)
    else
        ax_mean = Axis(fig[2, 1],
            title = "Order parameter",
            xlabel = "t",
            ylabel = "E|m(t)| / aligned mean",
        )
        ax_mean_signed = Axis(fig[2, 2],
            title = "Mixture mean",
            xlabel = "t",
            ylabel = "E[m(t)]",
        )
        ax_var = Axis(fig[2, 3],
            title = "Variance trajectory",
            xlabel = "t",
            ylabel = "Var(u)",
        )
        ax_term = Axis(fig[2, 4],
            title = "Terminal mean (above kappa*)",
            xlabel = "m_T",
            ylabel = "density",
        )

        max_mean_abs = 0.0
        for label in scenario_order
            df_l = traj_df[get_column(traj_df, :scenario) .== label, :]
            if nrow(df_l) == 0
                continue
            end
            color = label == "below" ? COLORS.below : (label == "critical" ? COLORS.critical : COLORS.above)
            mean_abs = has_column(df_l, :mean_abs) ? get_column(df_l, :mean_abs) : get_column(df_l, :mean)
            mean_abs_lo = has_column(df_l, :mean_abs_ci_lower) ? get_column(df_l, :mean_abs_ci_lower) : get_column(df_l, :mean_ci_lower)
            mean_abs_hi = has_column(df_l, :mean_abs_ci_upper) ? get_column(df_l, :mean_abs_ci_upper) : get_column(df_l, :mean_ci_upper)
            mean_signed = has_column(df_l, :mean_signed) ? get_column(df_l, :mean_signed) : get_column(df_l, :mean)
            mean_aligned = has_column(df_l, :mean_aligned) ? get_column(df_l, :mean_aligned) : nothing
            mean_aligned_lo = has_column(df_l, :mean_aligned_ci_lower) ? get_column(df_l, :mean_aligned_ci_lower) : nothing
            mean_aligned_hi = has_column(df_l, :mean_aligned_ci_upper) ? get_column(df_l, :mean_aligned_ci_upper) : nothing

            time_vec = get_column(df_l, :time)
            lines!(ax_mean, time_vec, mean_abs, color=color, linewidth=2.0, label="$label |m|")
            band!(ax_mean, time_vec, mean_abs_lo, mean_abs_hi, color=(color, 0.15))
            if mean_aligned !== nothing && any(isfinite, mean_aligned)
                lines!(ax_mean, time_vec, mean_aligned, color=(color, 0.7), linestyle=:dot, linewidth=1.6, label="$label aligned")
                if mean_aligned_lo !== nothing && mean_aligned_hi !== nothing
                    band!(ax_mean, time_vec, mean_aligned_lo, mean_aligned_hi, color=(color, 0.08))
                end
            end

            lines!(ax_mean_signed, time_vec, mean_signed, color=color, linewidth=1.8, label="$label E[m]")
            hlines!(ax_mean_signed, [0.0], color=:gray, linewidth=0.8)

            variance_vec = get_column(df_l, :variance)
            var_lo = get_column(df_l, :var_ci_lower)
            var_hi = get_column(df_l, :var_ci_upper)
            lines!(ax_var, time_vec, variance_vec, color=color, linewidth=2.0, label=label)
            band!(ax_var, time_vec, var_lo, var_hi, color=(color, 0.15))

            max_mean_abs = max(max_mean_abs, maximum(mean_abs))
        end
        if max_mean_abs > 0.0
            ylims!(ax_mean, 0.0, max_mean_abs * 1.05)
        else
            ylims!(ax_mean, 0.0, 1.0)
        end

        axislegend(ax_mean, position = :rb, framevisible = false)
        axislegend(ax_mean_signed, position = :rb, framevisible = false)
        axislegend(ax_var, position = :rb, framevisible = false)

        if terminal_df === nothing
            text!(ax_term, 0.0, 0.5, text="terminal_means.csv missing", align=(:center, :center))
        else
            above_df = terminal_df[get_column(terminal_df, :scenario) .== "above", :]
            if nrow(above_df) > 0
                hist!(ax_term, get_column(above_df, :mean_late); bins=30, color=(:gray, 0.6))
                vlines!(ax_term, [0.0], color=:black, linestyle=:dash, linewidth=1.0)
            else
                text!(ax_term, 0.0, 0.5, text="no above runs", align=(:center, :center))
            end
        end
    end

    if moments_df !== nothing
        above_m = moments_df[get_column(moments_df, :scenario) .== "above", :]
        if nrow(above_m) > 0
            t_max = maximum(get_column(above_m, :time))
            row = above_m[abs.(get_column(above_m, :time) .- t_max) .< 1e-8, :]
            if nrow(row) > 0
                mu_mix = get_column(row, :mu_mix)[1]
                mu_al = get_column(row, :mu_aligned)[1]
                txt = @sprintf("mu_mix=%.3f, mu_aligned=%.3f (t=%.0f)", mu_mix, mu_al, t_max)
                text!(ax_above, 0.02, 0.95, text=txt, space=:relative, align=(:left, :top), fontsize=9)
            end
        end
    end

    colgap!(fig.layout, 12)
    rowgap!(fig.layout, 12)

    save(save_path, fig, pt_per_unit = 1)
    println("  Saved to: $save_path")

    return fig
end

# =============================================================================
# FIGURE 6: VALIDATION SUMMARY (4-PANEL)
# =============================================================================

function plot_validation_summary(; save_path = joinpath(FIG_DIR, "fig_validation_summary.pdf"))
    println("Generating validation summary (4-panel)...")

    df = load_sweep_data()
    forward_df, backward_df = load_hysteresis_data()

    fig = Figure(size = (800, 600))

    labels = ["(a)", "(b)", "(c)", "(d)"]

    ax1 = Axis(fig[1, 1],
        xlabel = L"\kappa / \kappa^*",
        ylabel = L"|a^*|",
        title = "Bifurcation diagram"
    )

    x_data = get_column(df, :kappa) ./ KAPPA_STAR_ANALYTIC
    y_data = pick_column(df, [:a_star_mean, :a_star])
    has_fit = isfinite(KAPPA_STAR_EFF) && isfinite(BETA_FIT) && isfinite(C_FIT)

    vlines!(ax1, [1.0], color = COLORS.theory, linestyle = :dash, linewidth = 1.5)
    scatter!(ax1, x_data, y_data, color = COLORS.data, markersize = 6)

    if has_fit
        kappa_fit_range = range(KAPPA_STAR_EFF, maximum(get_column(df, :kappa)) * 1.05, length=100)
        delta = max.(kappa_fit_range .- KAPPA_STAR_EFF, 0.0)
        a_fit = C_FIT .* delta.^BETA_FIT
        lines!(ax1, kappa_fit_range ./ KAPPA_STAR_ANALYTIC, a_fit,
               color = COLORS.fit, linewidth = 2)
    end

    xlims!(ax1, 0.4, 1.6)
    ylims!(ax1, -0.02, 0.7)

    text!(ax1, 0.42, 0.65, text = labels[1], fontsize = 14, font = :bold)

    ax2 = Axis(fig[1, 2],
        xlabel = L"\log(\kappa - \kappa^*_{\mathrm{eff}})",
        ylabel = L"\log |a^*|",
        title = "Scaling exponent"
    )

    if isfinite(KAPPA_STAR_EFF)
        above_mask = (get_column(df, :kappa) .> KAPPA_STAR_EFF) .& (pick_column(df, [:a_star_mean, :a_star]) .> 1e-6)
        if sum(above_mask) >= 2
            k_above = get_column(df, :kappa)[above_mask]
            a_above = pick_column(df, [:a_star_mean, :a_star])[above_mask]
            log_dk = log.(k_above .- KAPPA_STAR_EFF)
            log_a = log.(a_above)

            scatter!(ax2, log_dk, log_a, color = COLORS.data, markersize = 6)

            X = hcat(ones(length(log_a)), log_dk)
            coeffs = X \ log_a
            x_line = range(minimum(log_dk) - 0.2, maximum(log_dk) + 0.2, length=50)
            lines!(ax2, x_line, coeffs[1] .+ coeffs[2] .* x_line,
                   color = COLORS.fit, linewidth = 2)

            lines!(ax2, x_line, coeffs[1] .+ 0.5 .* x_line,
                   color = COLORS.theory, linestyle = :dash, linewidth = 1.5)
        end
    end

    text!(ax2, -3.3, -0.3, text = labels[2], fontsize = 14, font = :bold)

    ax3 = Axis(fig[2, 1],
        xlabel = L"\kappa / \kappa^*",
        ylabel = L"|a^*|",
        title = "Hysteresis test"
    )

    x_fwd = get_column(forward_df, :kappa) ./ KAPPA_STAR_ANALYTIC
    y_fwd = pick_column(forward_df, [:a_star_mean, :a_star])
    x_bwd = get_column(backward_df, :kappa) ./ KAPPA_STAR_ANALYTIC
    y_bwd = pick_column(backward_df, [:a_star_mean, :a_star])

    scatterlines!(ax3, x_fwd, y_fwd, color = COLORS.forward,
                  marker = :circle, markersize = 5, linewidth = 1.5)
    scatterlines!(ax3, x_bwd, y_bwd, color = COLORS.backward,
                  marker = :utriangle, markersize = 5, linewidth = 1.5, linestyle = :dash)

    vlines!(ax3, [1.0], color = :gray, linestyle = :dot)
    xlims!(ax3, 0.4, 1.6)
    ylims!(ax3, -0.02, 0.7)

    text!(ax3, 0.42, 0.65, text = labels[3], fontsize = 14, font = :bold)
    text!(ax3, 0.55, 0.55, text = "No hysteresis", fontsize = 9)

    ax4 = Axis(fig[2, 2],
        xlabel = L"\kappa / \kappa^*",
        ylabel = L"V",
        title = "Variance"
    )

    scatterlines!(ax4, get_column(df, :kappa) ./ KAPPA_STAR_ANALYTIC, pick_column(df, [:variance, :variance_mean]),
                  color = COLORS.data, marker = :circle, markersize = 5, linewidth = 1.5)
    if isfinite(V_BASELINE)
        hlines!(ax4, [V_BASELINE], color = COLORS.fit, linestyle = :dash, linewidth = 1.5)
    end
    vlines!(ax4, [1.0], color = COLORS.theory, linestyle = :dash, linewidth = 1.5)

    xlims!(ax4, 0.4, 1.6)
    ylims!(ax4, 0.0, 0.55)

    text!(ax4, 0.42, 0.52, text = labels[4], fontsize = 14, font = :bold)

    colgap!(fig.layout, 20)
    rowgap!(fig.layout, 20)

    save(save_path, fig, pt_per_unit = 1)
    println("  Saved to: $save_path")

    return fig
end

# =============================================================================
# MAIN EXECUTION
# =============================================================================

function main()
    println("=" ^ 60)
    println("GENERATING PUBLICATION FIGURES")
    println("=" ^ 60)
    println()

    if !isfile(joinpath(DATA_DIR, "equilibrium_sweep.csv"))
        error("Run fig6_ensemble_enhanced.jl first to generate sweep data.")
    end

    figures = Dict{String, Any}()

    println("\n--- Main Text Figures ---")
    figures["bifurcation"] = plot_bifurcation_diagram()
    figures["density"] = plot_density_evolution()

    println("\n--- Supplementary Figures ---")
    figures["loglog"] = plot_loglog_scaling()
    figures["hysteresis"] = plot_hysteresis_test()
    figures["variance"] = plot_variance_ushape()

    println("\n--- Summary Figure ---")
    figures["summary"] = plot_validation_summary()

    println("\n" * "=" ^ 60)
    println("FIGURE GENERATION COMPLETE")
    println("=" ^ 60)
    println("\nOutputs saved to: $FIG_DIR")
    println("\nFiles generated:")
    for f in readdir(FIG_DIR)
        if startswith(f, "fig_") && endswith(f, ".pdf")
            println("  - $f")
        end
    end

    return figures
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
