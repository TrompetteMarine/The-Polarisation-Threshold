#!/usr/bin/env julia
# ============================================================================
# Attractor Evolution Gallery
#
# Shows how attractors, phase portraits, and basins of attraction evolve
# as κ crosses the critical threshold κ*. Displays:
# 1. Phase portraits (a, ȧ) at multiple κ values
# 2. Basin of attraction evolution
# 3. Trajectory fates from different initial conditions
# ============================================================================

using Pkg; Pkg.activate(".")

using BeliefSim
using BeliefSim.Types, BeliefSim.Stats

# Load bifurcation module
include("../src/bifurcation/PhylogeneticDiagram.jl")
using .PhylogeneticDiagram

# Try CairoMakie for professional plotting
CAIRO_AVAILABLE = false
try
    using CairoMakie
    global CAIRO_AVAILABLE = true
catch
    @warn "CairoMakie not available, falling back to Plots.jl"
    using Plots
end

using Statistics, Random, Printf, LinearAlgebra

println("=" ^ 72)
println("ATTRACTOR EVOLUTION GALLERY")
println("=" ^ 72)

# ============================================================================
# Step 1: Setup and calibration
# ============================================================================

println("\n[1/4] Setting up parameters...")

p = Params(λ=1.0, σ=0.8, Θ=2.0, c0=0.5, hazard=StepHazard(0.6))
Vstar = estimate_Vstar(p; N=20_000, T=300.0, dt=0.01, burn_in=100.0, seed=2025)
κstar = critical_kappa(p; N=20_000, T=350.0, dt=0.01, burn_in=120.0, seed=2026)

@printf("   ✓ κ* = %.4f, V* = %.4f\n", κstar, Vstar)

nf = calibrate_normal_form(κstar, Vstar; λ=p.λ, σ=p.σ, b_default=0.5)

# ============================================================================
# Step 2: Phase portrait gallery
# ============================================================================

println("\n[2/4] Generating phase portraits...")

# κ values as fractions of κ*
κ_ratios = [0.5, 0.9, 0.99, 1.01, 1.2, 1.5]
κ_values = κ_ratios .* κstar

# Helper function: integrate trajectory
function integrate_trajectory(a0, κ, nf, T=50.0, dt=0.01)
    n_steps = round(Int, T / dt)
    a_traj = zeros(n_steps)
    da_traj = zeros(n_steps)

    a = a0
    for i in 1:n_steps
        a_traj[i] = a
        da = normal_form_rhs(a, κ, nf)
        da_traj[i] = da
        a = a + dt * da
    end

    return a_traj, da_traj
end

# Helper function: draw phase-line arrows (CairoMakie version)
function draw_phase_line_arrows_makie!(ax, κ, nf; x_min=-3.0, x_max=3.0, n_per_segment=5)
    # Get sorted equilibria
    eqs = sort(equilibria(κ, nf))

    # Build segment boundaries
    boundaries = [x_min, eqs..., x_max]

    # For each consecutive pair of boundaries
    for i in 1:(length(boundaries) - 1)
        xL = boundaries[i]
        xR = boundaries[i + 1]

        # Skip tiny segments
        if xR - xL <= 1e-6
            continue
        end

        # Place more arrows per segment with better spacing
        for j in 1:n_per_segment
            x = xL + (j / (n_per_segment + 1)) * (xR - xL)

            # Evaluate flow at this point
            f = normal_form_rhs(x, κ, nf)

            # Skip if near zero
            if abs(f) < 1e-6
                continue
            end

            # Determine arrow direction and length
            sgn = sign(f)
            dx = 0.5 * sgn  # Longer arrows for better visibility

            # Draw prominent horizontal arrow near ȧ = 0
            arrows!(ax, [x - dx/2], [0.0], [dx], [0.0],
                   linewidth = 2.5,
                   color = (:steelblue, 0.85),
                   lengthscale = 1.0)
        end
    end
end

# Helper function: draw phase-line arrows (Plots.jl version)
function draw_phase_line_arrows_plots!(plt, κ, nf; x_min=-3.0, x_max=3.0, n_per_segment=5)
    # Get sorted equilibria
    eqs = sort(equilibria(κ, nf))

    # Build segment boundaries
    boundaries = [x_min, eqs..., x_max]

    # For each consecutive pair of boundaries
    for i in 1:(length(boundaries) - 1)
        xL = boundaries[i]
        xR = boundaries[i + 1]

        # Skip tiny segments
        if xR - xL <= 1e-6
            continue
        end

        # Place more arrows per segment
        for j in 1:n_per_segment
            x = xL + (j / (n_per_segment + 1)) * (xR - xL)

            # Evaluate flow at this point
            f = normal_form_rhs(x, κ, nf)

            # Skip if near zero
            if abs(f) < 1e-6
                continue
            end

            # Determine arrow direction and length
            sgn = sign(f)
            dx = 0.5 * sgn  # Longer arrows

            # Draw prominent horizontal arrow near ȧ = 0
            quiver!(plt, [x], [0.0],
                   quiver = ([dx], [0.0]),
                   linecolor = :steelblue,
                   linewidth = 2.0,
                   arrow = :closed,
                   arrowsize = 0.12,
                   alpha = 0.85,
                   label = nothing)
        end
    end
end

mkpath("figs")

if CAIRO_AVAILABLE
    fig = Figure(size=(1600, 1000), fontsize=13, backgroundcolor=:white)

    for (idx, (κ, ratio)) in enumerate(zip(κ_values, κ_ratios))
        row = div(idx - 1, 3) + 1
        col = mod(idx - 1, 3) + 1

        # Add more vertical spacing between rows
        ax = Axis(fig[row, col],
            xlabel = "Amplitude a",
            ylabel = "Rate ȧ",
            title = @sprintf("κ/κ* = %.2f %s", ratio,
                             ratio < 1.0 ? "(pre-bifurcation)" :
                             ratio ≈ 1.0 ? "(at threshold)" : "(post-bifurcation)"),
            aspect = DataAspect(),
            xlabelsize = 13,
            ylabelsize = 13,
            titlesize = 14,
            titlecolor = ratio < 1.0 ? :darkblue : (ratio > 1.05 ? :darkred : :black)
        )

        # Nullcline ȧ = 0 (very subtle background reference)
        a_grid = range(-3, 3, length=200)
        da_nullcline = [normal_form_rhs(a, κ, nf) for a in a_grid]

        lines!(ax, a_grid, da_nullcline,
            color = (:gray80, 0.25), linewidth = 1.5, linestyle = :dot)

        # Horizontal reference line at ȧ = 0
        hlines!(ax, [0.0], color = (:black, 0.15), linewidth = 1.5)

        # Draw prominent phase-line arrows showing flow direction
        draw_phase_line_arrows_makie!(ax, κ, nf; x_min=-3.0, x_max=3.0, n_per_segment=5)

        # Equilibria with clear visual distinction
        eq_list = equilibria(κ, nf)
        for a_eq in eq_list
            is_stable = stability(a_eq, κ, nf)
            if is_stable
                # Stable: filled red circle
                scatter!(ax, [a_eq], [0.0],
                    markersize = 16, color = :red,
                    marker = :circle, strokewidth = 0)
            else
                # Unstable: hollow circle with thick black border
                scatter!(ax, [a_eq], [0.0],
                    markersize = 16, color = :white,
                    marker = :circle, strokewidth = 3,
                    strokecolor = :black)
            end
        end

        # Sample trajectories from different ICs with better visibility
        n_trajectories = 6
        colors = [:steelblue, :purple, :teal, :orange, :brown, :pink]
        for (traj_idx, _) in enumerate(1:n_trajectories)
            a0 = 5 * rand() - 2.5  # Random in [-2.5, 2.5]
            a_traj, da_traj = integrate_trajectory(a0, κ, nf, 40.0, 0.01)

            # Plot trajectory with moderate alpha
            lines!(ax, a_traj, da_traj,
                color = (colors[traj_idx], 0.5), linewidth = 2.5)

            # Mark starting point (green) and ending point (attractor)
            scatter!(ax, [a_traj[1]], [da_traj[1]],
                markersize = 7, color = :green, marker = :circle,
                strokewidth = 1, strokecolor = :darkgreen)

            # Mark ending point with arrow-like marker
            scatter!(ax, [a_traj[end]], [da_traj[end]],
                markersize = 6, color = (colors[traj_idx], 0.8),
                marker = :star5, strokewidth = 1, strokecolor = :black)
        end

        # Add legend only to first panel
        if idx == 1
            scatter!(ax, [NaN], [NaN], markersize = 16, color = :red,
                    label = "Stable equilibrium")
            scatter!(ax, [NaN], [NaN], markersize = 16, color = :white,
                    strokewidth = 3, strokecolor = :black,
                    label = "Unstable equilibrium")
            scatter!(ax, [NaN], [NaN], markersize = 7, color = :green,
                    label = "Initial condition")
            # Add flow direction indicator to legend
            arrows!(ax, [NaN], [NaN], [0.4], [0.0],
                   linewidth = 2.5, color = (:steelblue, 0.85),
                   lengthscale = 1.0, label = "Phase flow direction")
            axislegend(ax, position = :lt, framevisible = true,
                      labelsize = 11, backgroundcolor = (:white, 0.9))
        end

        xlims!(ax, -3.2, 3.2)
        ylims!(ax, -2.5, 2.5)
    end

    # Add spacing between subplots to prevent overlap
    colgap!(fig.layout, 25)
    rowgap!(fig.layout, 35)

    save("figs/phase_portrait_gallery.pdf", fig)
    save("figs/phase_portrait_gallery.png", fig, px_per_unit=3)

else
    # Fallback to Plots.jl
    using Plots
    gr()

    plots = []

    for (κ, ratio) in zip(κ_values, κ_ratios)
        plt = plot(
            xlabel = "Amplitude a",
            ylabel = "Rate ȧ",
            title = @sprintf("κ/κ* = %.2f", ratio),
            aspect_ratio = 1,
            xlims = (-3.5, 3.5),
            ylims = (-2, 2),
            legend = false,
            grid = false,
            fontfamily = "Computer Modern"
        )

        # Nullcline (very subtle background reference)
        a_grid = range(-3, 3, length=200)
        da_nullcline = [normal_form_rhs(a, κ, nf) for a in a_grid]
        plot!(plt, a_grid, da_nullcline,
            color = :gray85, linewidth = 1.5, linestyle = :dot, alpha = 0.3, label = nothing)

        # Horizontal reference line
        hline!(plt, [0.0], color = :black, linewidth = 0.5, alpha = 0.2, label = nothing)

        # Draw prominent phase-line arrows showing flow direction
        draw_phase_line_arrows_plots!(plt, κ, nf; x_min=-3.0, x_max=3.0, n_per_segment=5)

        # Equilibria
        eq_list = equilibria(κ, nf)
        for a_eq in eq_list
            is_stable = stability(a_eq, κ, nf)
            if is_stable
                scatter!(plt, [a_eq], [0.0],
                    markersize = 6, color = :red, label = nothing)
            else
                scatter!(plt, [a_eq], [0.0],
                    markersize = 6, color = :white,
                    markerstrokewidth = 2, markerstrokecolor = :black,
                    label = nothing)
            end
        end

        # Trajectories
        for _ in 1:8
            a0 = 6 * rand() - 3
            a_traj, da_traj = integrate_trajectory(a0, κ, nf, 30.0, 0.01)
            plot!(plt, a_traj, da_traj,
                color = :steelblue, alpha = 0.3, linewidth = 1.5, label = nothing)
        end

        push!(plots, plt)
    end

    final_plot = plot(plots..., layout=(2, 3), size=(1400, 900), dpi=300)
    savefig(final_plot, "figs/phase_portrait_gallery.pdf")
    savefig(final_plot, "figs/phase_portrait_gallery.png")
end

println("   ✓ Phase portrait gallery saved")

# ============================================================================
# Step 2b: Nullcline flow arrows (arrows along cubic curve)
# ============================================================================

println("\n[2b/4] Generating nullcline flow visualization...")

# Helper function: draw arrows along the cubic curve toward attractors
function draw_nullcline_flow_arrows_makie!(ax, κ, nf; n_arrows=6)
    # Get equilibria and identify stable ones
    eq_list = equilibria(κ, nf)
    stable_eq = filter(a -> stability(a, κ, nf), eq_list)

    # Define colors for different attractors
    # Sort stable equilibria to assign consistent colors
    stable_eq_sorted = sort(stable_eq)
    attractor_colors = Dict{Float64, Tuple{Symbol, Float64}}()
    if length(stable_eq_sorted) == 1
        # Single attractor (consensus)
        attractor_colors[stable_eq_sorted[1]] = (:purple, 0.9)
    elseif length(stable_eq_sorted) == 2
        # Two attractors (polarized states)
        attractor_colors[stable_eq_sorted[1]] = (:dodgerblue, 0.9)  # Negative
        attractor_colors[stable_eq_sorted[2]] = (:orangered, 0.9)   # Positive
    elseif length(stable_eq_sorted) >= 3
        # Multiple attractors (unusual case)
        colors_list = [:dodgerblue, :purple, :orangered, :green, :magenta]
        for (idx, seq) in enumerate(stable_eq_sorted)
            attractor_colors[seq] = (colors_list[min(idx, length(colors_list))], 0.9)
        end
    end

    # Define regions between equilibria
    eqs_sorted = sort(eq_list)
    boundaries = [-3.0, eqs_sorted..., 3.0]

    for i in 1:(length(boundaries) - 1)
        aL = boundaries[i]
        aR = boundaries[i + 1]

        if aR - aL <= 1e-6
            continue
        end

        # Sample points along this segment
        for j in 1:n_arrows
            a = aL + (j / (n_arrows + 1)) * (aR - aL)

            # Evaluate flow
            da_dt = normal_form_rhs(a, κ, nf)

            # Skip if very close to equilibrium
            if abs(da_dt) < 0.05
                continue
            end

            # Determine direction toward nearest stable equilibrium and get its color
            arrow_color = (:darkgreen, 0.9)  # Default color
            if isempty(stable_eq)
                # No stable equilibria, flow toward zero
                direction_a = -sign(a)
            else
                # Flow toward nearest stable equilibrium
                distances = [abs(a - seq) for seq in stable_eq]
                nearest_idx = argmin(distances)
                nearest_attractor = stable_eq[nearest_idx]
                direction_a = sign(nearest_attractor - a)

                # Get color for this attractor
                if haskey(attractor_colors, nearest_attractor)
                    arrow_color = attractor_colors[nearest_attractor]
                end
            end

            # Arrow components: move along the curve toward attractor
            # Horizontal component (larger for more visibility)
            arrow_da = direction_a * 0.35
            # Vertical component follows the curve
            a_end = a + arrow_da
            da_end = normal_form_rhs(a_end, κ, nf)
            arrow_dda = da_end - da_dt

            # Draw thick, salient arrow along the curve with color-coding
            arrows!(ax, [a], [da_dt], [arrow_da], [arrow_dda],
                   linewidth = 4.0,
                   color = arrow_color,
                   lengthscale = 1.0,
                   arrowsize = 15)
        end
    end
end

if CAIRO_AVAILABLE
    fig_nullcline = Figure(size=(1600, 1000), fontsize=13, backgroundcolor=:white)

    for (idx, (κ, ratio)) in enumerate(zip(κ_values, κ_ratios))
        row = div(idx - 1, 3) + 1
        col = mod(idx - 1, 3) + 1

        ax = Axis(fig_nullcline[row, col],
            xlabel = "Amplitude a",
            ylabel = "Rate ȧ",
            title = @sprintf("κ/κ* = %.2f %s", ratio,
                             ratio < 1.0 ? "(pre-bifurcation)" :
                             ratio ≈ 1.0 ? "(at threshold)" : "(post-bifurcation)"),
            aspect = DataAspect(),
            xlabelsize = 13,
            ylabelsize = 13,
            titlesize = 14,
            titlecolor = ratio < 1.0 ? :darkblue : (ratio > 1.05 ? :darkred : :black)
        )

        # Prominent nullcline curve
        a_grid = range(-3, 3, length=300)
        da_nullcline = [normal_form_rhs(a, κ, nf) for a in a_grid]

        lines!(ax, a_grid, da_nullcline,
            color = (:steelblue, 0.8), linewidth = 3.5, linestyle = :solid)

        # Horizontal reference line at ȧ = 0
        hlines!(ax, [0.0], color = (:black, 0.3), linewidth = 1.5, linestyle = :dash)

        # Draw flow arrows along the cubic curve
        draw_nullcline_flow_arrows_makie!(ax, κ, nf)

        # Equilibria with clear visual distinction
        eq_list = equilibria(κ, nf)
        for a_eq in eq_list
            is_stable = stability(a_eq, κ, nf)
            if is_stable
                # Stable: filled red circle
                scatter!(ax, [a_eq], [0.0],
                    markersize = 18, color = :red,
                    marker = :circle, strokewidth = 2,
                    strokecolor = :darkred)
            else
                # Unstable: hollow circle
                scatter!(ax, [a_eq], [0.0],
                    markersize = 18, color = :white,
                    marker = :circle, strokewidth = 3,
                    strokecolor = :black)
            end
        end

        # Add legend only to last panel (post-bifurcation with two attractors)
        if idx == 6
            lines!(ax, [NaN], [NaN], color = (:steelblue, 0.8), linewidth = 3.5,
                  label = "Nullcline ȧ = f(a)")
            scatter!(ax, [NaN], [NaN], markersize = 18, color = :red,
                    strokewidth = 2, strokecolor = :darkred,
                    label = "Stable equilibrium")
            scatter!(ax, [NaN], [NaN], markersize = 18, color = :white,
                    strokewidth = 3, strokecolor = :black,
                    label = "Unstable equilibrium")
            # Show color-coded flow arrows
            arrows!(ax, [NaN], [NaN], [0.3], [0.1],
                   linewidth = 4.0, color = (:dodgerblue, 0.9),
                   lengthscale = 1.0, arrowsize = 15,
                   label = "Flow → negative attractor")
            arrows!(ax, [NaN], [NaN], [0.3], [0.1],
                   linewidth = 4.0, color = (:orangered, 0.9),
                   lengthscale = 1.0, arrowsize = 15,
                   label = "Flow → positive attractor")
            axislegend(ax, position = :lt, framevisible = true,
                      labelsize = 10, backgroundcolor = (:white, 0.9))
        end

        xlims!(ax, -3.2, 3.2)
        ylims!(ax, -2.5, 2.5)
    end

    # Add spacing between subplots
    colgap!(fig_nullcline.layout, 25)
    rowgap!(fig_nullcline.layout, 35)

    save("figs/nullcline_flow_gallery.pdf", fig_nullcline)
    save("figs/nullcline_flow_gallery.png", fig_nullcline, px_per_unit=3)

    println("   ✓ Nullcline flow gallery saved")
else
    println("   ⚠ Skipping nullcline flow gallery (CairoMakie not available)")
end

# ============================================================================
# Step 3: Basin of attraction evolution
# ============================================================================

println("\n[3/5] Computing basin of attraction evolution...")

# Focus on κ > κ* where basins split
κ_basin_values = [1.05, 1.2, 1.5] .* κstar

function compute_basin(κ, nf; a_grid_size=200, T_integrate=100.0, dt=0.01)
    a_grid = range(-3, 3, length=a_grid_size)
    basin = zeros(Int, a_grid_size)

    for (i, a0) in enumerate(a_grid)
        a = a0
        n_steps = round(Int, T_integrate / dt)

        for _ in 1:n_steps
            da = normal_form_rhs(a, κ, nf)
            a = a + dt * da
        end

        # Classify final state
        if abs(a) < 0.1
            basin[i] = 0  # Consensus (unstable for κ > κ*)
        elseif a > 0.1
            basin[i] = 1  # Positive polarisation
        else
            basin[i] = -1  # Negative polarisation
        end
    end

    return a_grid, basin
end

if CAIRO_AVAILABLE
    fig_basin = Figure(size=(1500, 450), fontsize=14, backgroundcolor=:white)

    # Add overall title
    Label(fig_basin[0, :], "Basin of Attraction Evolution",
          fontsize = 18, font = :bold, color = :black)

    for (idx, κ) in enumerate(κ_basin_values)
        ax = Axis(fig_basin[1, idx],
            xlabel = "Initial amplitude a₀",
            ylabel = "Final attractor",
            title = @sprintf("κ/κ* = %.2f %s", κ/κstar,
                            κ/κstar < 1.05 ? "(emerging split)" :
                            κ/κstar < 1.3 ? "(clear split)" : "(well-separated)"),
            xlabelsize = 14,
            ylabelsize = 14,
            titlesize = 15,
            titlecolor = κ/κstar < 1.1 ? :darkgreen : (κ/κstar < 1.3 ? :darkorange : :darkred),
            xgridstyle = :dash,
            ygridstyle = :dash,
            xgridcolor = (:gray, 0.2),
            ygridcolor = (:gray, 0.2)
        )

        a_grid, basin = compute_basin(κ, nf; a_grid_size=300)

        # Color code by basin with clearer colors
        colors = [basin[i] == -1 ? (:blue, 0.7) :
                  (basin[i] == 1 ? (:red, 0.7) : (:gray50, 0.5)) for i in eachindex(basin)]

        scatter!(ax, a_grid, basin,
            color = colors, markersize = 4, strokewidth = 0)

        # Add horizontal reference lines
        hlines!(ax, [-1], color = :blue, linestyle = :solid, linewidth = 3,
                label = "Negative polarization")
        hlines!(ax, [0], color = :gray30, linestyle = :dash, linewidth = 2,
                label = "Consensus (unstable)")
        hlines!(ax, [1], color = :red, linestyle = :solid, linewidth = 3,
                label = "Positive polarization")

        # Add legend only to first panel
        if idx == 1
            axislegend(ax, position = :lt, framevisible = true,
                      labelsize = 11, backgroundcolor = (:white, 0.9))
        end

        # Add annotations about basin sizes
        n_neg = sum(basin .== -1)
        n_pos = sum(basin .== 1)
        n_zero = sum(basin .== 0)
        total = length(basin)

        text!(ax, 0, -1.35,
              text = @sprintf("%.1f%%", 100*n_neg/total),
              fontsize = 12, align = (:center, :center),
              color = :blue, font = :bold)
        text!(ax, 0, 1.35,
              text = @sprintf("%.1f%%", 100*n_pos/total),
              fontsize = 12, align = (:center, :center),
              color = :red, font = :bold)

        ylims!(ax, -1.6, 1.6)
        xlims!(ax, -3.2, 3.2)
    end

    # Add spacing between panels
    colgap!(fig_basin.layout, 30)

    save("figs/basin_evolution.pdf", fig_basin)
    save("figs/basin_evolution.png", fig_basin, px_per_unit=3)

else
    # Plots.jl fallback
    using Plots
    plots_basin = []

    for κ in κ_basin_values
        a_grid, basin = compute_basin(κ, nf; a_grid_size=300)

        colors = [basin[i] == -1 ? :blue : (basin[i] == 1 ? :red : :gray) for i in eachindex(basin)]

        plt = scatter(a_grid, basin,
            xlabel = "Initial amplitude a₀",
            ylabel = "Attractor",
            title = @sprintf("κ/κ* = %.2f", κ/κstar),
            color = colors, markersize = 2, alpha = 0.7,
            ylims = (-1.5, 1.5),
            legend = false,
            grid = false,
            fontfamily = "Computer Modern"
        )

        hline!(plt, [-1, 0, 1], color = :black, linestyle = :dash, linewidth = 1)

        push!(plots_basin, plt)
    end

    final_basin = plot(plots_basin..., layout=(1, 3), size=(1200, 400), dpi=300)
    savefig(final_basin, "figs/basin_evolution.pdf")
    savefig(final_basin, "figs/basin_evolution.png")
end

println("   ✓ Basin evolution saved")

# ============================================================================
# Step 4: Trajectory fates
# ============================================================================

println("\n[4/4] Generating trajectory fate comparison...")

function trajectory_panel(κ, nf, n_traj=10, T=100.0, dt=0.01)
    n_steps = round(Int, T / dt)
    t_grid = range(0, T, length=n_steps)

    trajectories = []

    for _ in 1:n_traj
        a0 = 4 * rand() - 2  # Random in [-2, 2]
        a_traj = zeros(n_steps)

        a = a0
        for i in 1:n_steps
            a_traj[i] = a
            da = normal_form_rhs(a, κ, nf)
            a = a + dt * da
        end

        push!(trajectories, (t_grid, a_traj))
    end

    return trajectories
end

κ_below = 0.8 * κstar
κ_above = 1.3 * κstar

traj_below = trajectory_panel(κ_below, nf, 10, 100.0, 0.01)
traj_above = trajectory_panel(κ_above, nf, 10, 100.0, 0.01)

if CAIRO_AVAILABLE
    fig_traj = Figure(size=(1400, 600), fontsize=14, backgroundcolor=:white)

    # Left panel: Below threshold - all trajectories converge to consensus
    ax1 = Axis(fig_traj[1, 1],
        xlabel = "Time t",
        ylabel = "Amplitude a(t)",
        title = @sprintf("Below threshold: κ = %.3f (%.1f%% of κ*)", κ_below, 100*κ_below/κstar),
        xlabelsize = 15,
        ylabelsize = 15,
        titlesize = 16,
        titlecolor = :darkblue,
        xgridstyle = :dash,
        ygridstyle = :dash,
        xgridcolor = (:gray, 0.3),
        ygridcolor = (:gray, 0.3)
    )

    # Plot trajectories with distinct colors
    colors_traj = [:steelblue, :purple, :teal, :orange, :brown, :pink, :olive, :navy, :maroon, :cyan]
    for (idx, (t, a)) in enumerate(traj_below)
        lines!(ax1, t, a, color = (colors_traj[idx], 0.7), linewidth = 2.5)
    end

    # Mark the single stable attractor (consensus at a=0)
    hlines!(ax1, [0], color = :red, linewidth = 4, linestyle = :solid,
            label = "Stable consensus (a = 0)")

    # Add shaded region showing basin of attraction (all initial conditions converge)
    text!(ax1, 50, 2.3, text = "All trajectories\nconverge to\nconsensus",
          fontsize = 13, align = (:center, :center),
          color = :darkblue, font = :bold)

    ylims!(ax1, -2.8, 2.8)
    axislegend(ax1, position = :rt, framevisible = true, backgroundcolor = (:white, 0.9))

    # Right panel: Above threshold - trajectories split to ±a*
    ax2 = Axis(fig_traj[1, 2],
        xlabel = "Time t",
        ylabel = "Amplitude a(t)",
        title = @sprintf("Above threshold: κ = %.3f (%.1f%% of κ*)", κ_above, 100*κ_above/κstar),
        xlabelsize = 15,
        ylabelsize = 15,
        titlesize = 16,
        titlecolor = :darkred,
        xgridstyle = :dash,
        ygridstyle = :dash,
        xgridcolor = (:gray, 0.3),
        ygridcolor = (:gray, 0.3)
    )

    for (idx, (t, a)) in enumerate(traj_above)
        lines!(ax2, t, a, color = (colors_traj[idx], 0.7), linewidth = 2.5)
    end

    # Mark the stable attractors (polarized states at ±a*)
    eq_above = equilibria(κ_above, nf)
    stable_eq = filter(a -> stability(a, κ_above, nf), eq_above)

    for a_eq in stable_eq
        if abs(a_eq) > 0.01  # Only mark non-zero attractors
            hlines!(ax2, [a_eq], color = :red, linewidth = 4, linestyle = :solid)
        else
            # Mark unstable consensus
            hlines!(ax2, [a_eq], color = :black, linewidth = 3, linestyle = :dash)
        end
    end

    # Add annotations
    if length(stable_eq) > 1
        positive_attractor = maximum(stable_eq)
        negative_attractor = minimum(stable_eq)
        text!(ax2, 50, positive_attractor + 0.4,
              text = "Stable polarized\nstate (+)",
              fontsize = 12, align = (:center, :center),
              color = :darkred)
        text!(ax2, 50, negative_attractor - 0.4,
              text = "Stable polarized\nstate (−)",
              fontsize = 12, align = (:center, :center),
              color = :darkred)
        text!(ax2, 50, 0.3,
              text = "Unstable\nconsensus",
              fontsize = 11, align = (:center, :bottom),
              color = :gray30)
    end

    ylims!(ax2, -2.8, 2.8)

    # Add overall title
    Label(fig_traj[0, :], "Trajectory Evolution: Consensus vs Polarization",
          fontsize = 18, font = :bold, color = :black)

    # Add spacing between panels
    colgap!(fig_traj.layout, 30)

    save("figs/trajectory_fates.pdf", fig_traj)
    save("figs/trajectory_fates.png", fig_traj, px_per_unit=3)

else
    # Plots.jl fallback
    using Plots

    plt1 = plot(
        xlabel = "Time t",
        ylabel = "Amplitude a(t)",
        title = @sprintf("Below: κ = %.3f", κ_below),
        ylims = (-3, 3),
        legend = false,
        grid = false,
        fontfamily = "Computer Modern"
    )

    for (t, a) in traj_below
        plot!(plt1, t, a, color = :steelblue, alpha = 0.6, linewidth = 2)
    end
    hline!(plt1, [0], color = :red, linewidth = 2, linestyle = :dash)

    plt2 = plot(
        xlabel = "Time t",
        ylabel = "Amplitude a(t)",
        title = @sprintf("Above: κ = %.3f", κ_above),
        ylims = (-3, 3),
        legend = false,
        grid = false,
        fontfamily = "Computer Modern"
    )

    for (t, a) in traj_above
        plot!(plt2, t, a, color = :steelblue, alpha = 0.6, linewidth = 2)
    end

    eq_above = equilibria(κ_above, nf)
    stable_eq = filter(a -> stability(a, κ_above, nf), eq_above)
    for a_eq in stable_eq
        hline!(plt2, [a_eq], color = :red, linewidth = 2, linestyle = :dash)
    end

    final_traj = plot(plt1, plt2, layout=(1, 2), size=(1200, 500), dpi=300)
    savefig(final_traj, "figs/trajectory_fates.pdf")
    savefig(final_traj, "figs/trajectory_fates.png")
end

println("   ✓ Trajectory fates saved")

# ============================================================================
# Summary
# ============================================================================

println("\n" * "="^72)
println("ATTRACTOR EVOLUTION ANALYSIS COMPLETE")
println("="^72)
println("\n✓ Figures saved:")
println("   • figs/phase_portrait_gallery.pdf")
println("   • figs/nullcline_flow_gallery.pdf")
println("   • figs/basin_evolution.pdf")
println("   • figs/trajectory_fates.pdf")
println("\n" * "="^72)
