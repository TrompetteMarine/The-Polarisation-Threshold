#!/usr/bin/env julia
# ============================================================================
# Phylogenetic Bifurcation Diagram Generator
#
# Generates the characteristic "tuning fork" bifurcation diagram showing how
# equilibria emerge as the social coupling κ crosses the critical threshold κ*.
# Analogous to classic logistic map diagrams, but with the clean pitchfork
# structure of this model.
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

using Statistics, Random, Printf

println("=" ^ 72)
println("PHYLOGENETIC BIFURCATION DIAGRAM GENERATOR")
println("=" ^ 72)

# ============================================================================
# Step 1: Calibrate micro-level parameters
# ============================================================================

println("\n[1/5] Calibrating micro-level parameters...")

# Micro-level parameters matching your typical setup
p = Params(λ=1.0, σ=0.8, Θ=2.0, c0=0.5, hazard=StepHazard(0.6))

# Estimate stationary dispersion at κ=0
println("   Estimating V* (stationary dispersion at κ=0)...")
Vstar = estimate_Vstar(p; N=20_000, T=300.0, dt=0.01, burn_in=100.0, seed=2025)

# Compute critical κ from spectral analysis
println("   Computing κ* (critical coupling)...")
κstar = critical_kappa(p; N=20_000, T=350.0, dt=0.01, burn_in=120.0, seed=2026)

@printf("   ✓ V* = %.4f\n", Vstar)
@printf("   ✓ κ* = %.4f\n", κstar)

# ============================================================================
# Step 2: Calibrate reduced normal form
# ============================================================================

println("\n[2/5] Calibrating reduced normal form...")

nf = calibrate_normal_form(κstar, Vstar; λ=p.λ, σ=p.σ, b_default=0.5)

@printf("   ✓ μ_slope = %.4f\n", nf.μ_slope)
@printf("   ✓ b (cubic coeff) = %.4f\n", nf.b)

# ============================================================================
# Step 3: Perform phylogenetic sweep
# ============================================================================

println("\n[3/5] Running phylogenetic parameter sweep...")
println("   Sweeping κ with 500 points, 30 ICs per point...")

result = phylogenetic_sweep(nf;
    κ_points=500,
    κ_max_factor=4.0,  # Extended range to see more structure
    n_ics=30,
    a_range=(-3.0, 3.0),
    T_burn=500.0,
    dt=0.01,
    cluster_tol=0.05,
    adaptive_burn=true,
    seed=2027
)

@printf("   ✓ Collected %d attractor points\n", length(result.κ_vals))

# Verify scaling exponent
β, C, r² = scaling_exponent(result; κ_range_factor=(1.05, 1.5))
@printf("   ✓ Scaling exponent β = %.3f (expected 0.5, R² = %.4f)\n", β, r²)

if abs(β - 0.5) > 0.1
    @warn "Scaling exponent deviates significantly from 0.5"
end

# ============================================================================
# Step 4: Create visualisation
# ============================================================================

println("\n[4/5] Generating bifurcation diagram...")

mkpath("figs")

if CAIRO_AVAILABLE
    # Professional CairoMakie visualization
    fig = Figure(size=(900, 650), fontsize=14, backgroundcolor=:white)

    # Main bifurcation diagram
    ax = Axis(fig[1, 1],
        xlabel = "Social coupling strength κ",
        ylabel = "Polarisation amplitude a*",
        title = "Phylogenetic Bifurcation Diagram: Emergence of Polarisation",
        xlabelsize = 16,
        ylabelsize = 16,
        titlesize = 18,
        xgridstyle = :dash,
        ygridstyle = :dash,
        xgridcolor = (:gray, 0.2),
        ygridcolor = (:gray, 0.2),
        backgroundcolor = :white
    )

    # Plot stable equilibria
    stable_idx = findall(result.stable_mask)
    scatter!(ax, result.κ_vals[stable_idx], result.a_vals[stable_idx],
        markersize = 3, color = (:steelblue, 0.7), label = "Stable equilibria")

    # Plot unstable equilibria
    unstable_idx = findall(.!result.stable_mask)
    if !isempty(unstable_idx)
        scatter!(ax, result.κ_vals[unstable_idx], result.a_vals[unstable_idx],
            markersize = 3, color = (:gray, 0.4), marker = :circle,
            label = "Unstable equilibria")
    end

    # Theoretical envelope ±√(μ/b)
    κ_theory, a_pos, a_neg = theoretical_envelope(range(κstar, 2*κstar, length=200), nf)
    if !isempty(κ_theory)
        lines!(ax, κ_theory, a_pos,
            color = :red, linestyle = :dash, linewidth = 3,
            label = "Theory: ±√(μ/b)")
        lines!(ax, κ_theory, a_neg,
            color = :red, linestyle = :dash, linewidth = 3)
    end

    # Critical line κ*
    vlines!(ax, [κstar],
        color = :black, linestyle = :dot, linewidth = 2.5,
        label = @sprintf("κ* = %.3f", κstar))

    # Regime annotations
    text!(ax, κstar - 0.15, 2.5,
        text = "Consensus\nRegime",
        fontsize = 12,
        align = (:right, :center),
        color = :black)

    text!(ax, κstar + 0.15, 2.5,
        text = "Polarised\nRegime",
        fontsize = 12,
        align = (:left, :center),
        color = :black)

    # Legend
    axislegend(ax, position = :lt, framevisible = false,
               labelsize = 12, bgcolor = (:white, 0.8))

    # Inset: Scaling verification (log-log plot)
    ax_inset = Axis(fig[1, 1],
        width = Relative(0.32),
        height = Relative(0.32),
        halign = 0.95,
        valign = 0.05,
        xlabel = "κ - κ*",
        ylabel = "|a*|",
        xlabelsize = 11,
        ylabelsize = 11,
        xscale = log10,
        yscale = log10,
        xgridvisible = false,
        ygridvisible = false,
        backgroundcolor = (:white, 0.95)
    )

    # Filter points for scaling plot
    above_threshold = findall((result.κ_vals .> κstar * 1.02) .&
                             (result.κ_vals .< κstar * 1.8) .&
                             (abs.(result.a_vals) .> 0.05) .&
                             result.stable_mask)

    if !isempty(above_threshold)
        scatter!(ax_inset,
            result.κ_vals[above_threshold] .- κstar,
            abs.(result.a_vals[above_threshold]),
            markersize = 2,
            color = (:steelblue, 0.5))

        # Reference line with β = 0.5
        x_ref = 10 .^ range(log10(0.02), log10(1.0), length=50)
        y_ref = C .* x_ref.^0.5
        lines!(ax_inset, x_ref, y_ref,
            color = :red, linewidth = 2.5,
            label = @sprintf("β = %.2f", β))

        axislegend(ax_inset, position = :rb, framevisible = false,
                  labelsize = 10)
    end

    # Save figure
    save("figs/phylogenetic_bifurcation.pdf", fig)
    save("figs/phylogenetic_bifurcation.png", fig, px_per_unit=3)

else
    # Fallback to Plots.jl
    using Plots
    gr()

    plt = plot(
        xlabel = "Social coupling κ",
        ylabel = "Polarisation amplitude a*",
        title = "Phylogenetic Bifurcation Diagram",
        legend = :topleft,
        size = (900, 650),
        dpi = 300,
        grid = false,
        background_color = :white,
        fontfamily = "Computer Modern"
    )

    # Stable points
    stable_idx = findall(result.stable_mask)
    scatter!(plt, result.κ_vals[stable_idx], result.a_vals[stable_idx],
        markersize = 2, color = :steelblue, alpha = 0.7,
        label = "Stable equilibria")

    # Unstable points
    unstable_idx = findall(.!result.stable_mask)
    if !isempty(unstable_idx)
        scatter!(plt, result.κ_vals[unstable_idx], result.a_vals[unstable_idx],
            markersize = 2, color = :gray, alpha = 0.4,
            markerstrokewidth = 0, label = "Unstable equilibria")
    end

    # Theoretical envelope
    κ_theory, a_pos, a_neg = theoretical_envelope(range(κstar, 2*κstar, length=200), nf)
    if !isempty(κ_theory)
        plot!(plt, κ_theory, a_pos,
            color = :red, linestyle = :dash, linewidth = 3,
            label = "Theory: ±√(μ/b)")
        plot!(plt, κ_theory, a_neg,
            color = :red, linestyle = :dash, linewidth = 3,
            label = nothing)
    end

    # Critical line
    vline!(plt, [κstar],
        color = :black, linestyle = :dot, linewidth = 2,
        label = @sprintf("κ* = %.3f", κstar))

    savefig(plt, "figs/phylogenetic_bifurcation.pdf")
    savefig(plt, "figs/phylogenetic_bifurcation.png")
end

# ============================================================================
# Step 5: Summary statistics
# ============================================================================

println("\n[5/5] Summary statistics:")
println("   " * "─"^60)
@printf("   Critical coupling:          κ* = %.4f\n", κstar)
@printf("   Stationary dispersion:      V* = %.4f\n", Vstar)
@printf("   Normal form μ-slope:        %.4f\n", nf.μ_slope)
@printf("   Cubic coefficient:          b  = %.4f\n", nf.b)
@printf("   Scaling exponent:           β  = %.3f ± 0.05\n", β)
@printf("   Regression R²:              %.4f\n", r²)
@printf("   Total attractor points:     %d\n", length(result.κ_vals))
@printf("   Stable points:              %d (%.1f%%)\n",
    sum(result.stable_mask),
    100 * sum(result.stable_mask) / length(result.stable_mask))
println("   " * "─"^60)

println("\n✓ Figures saved:")
println("   • figs/phylogenetic_bifurcation.pdf")
println("   • figs/phylogenetic_bifurcation.png")

println("\n" * "="^72)
println("ANALYSIS COMPLETE")
println("="^72)
