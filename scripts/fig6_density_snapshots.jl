#!/usr/bin/env julia
# =============================================================================
# Figure 6: Enhanced density snapshots with critical point and validation
# =============================================================================

using Pkg
Pkg.activate(".")

# Standard library
using Random
using Statistics
using Printf
using Dates
using LinearAlgebra

# External packages
using CSV
using DataFrames
using StatsBase

# Custom package (assumes BeliefSim is already in environment)
try
    using BeliefSim
    using BeliefSim.Types
    using BeliefSim.Stats
    using BeliefSim.Model: euler_maruyama_step!, reset_step!
    println("✓ BeliefSim loaded successfully")
catch e
    @error """
    Failed to load BeliefSim package: $e
    
    This appears to be a custom package for your project.
    Please ensure it's properly installed or available in your environment.
    """
    exit(1)
end

# Optional plotting backends
CAIRO_AVAILABLE = false
PLOTS_AVAILABLE = false

println("\nChecking plotting backends...")
try
    using CairoMakie
    global CAIRO_AVAILABLE = true
    println("✓ CairoMakie available (high-quality output)")
catch
    println("⚠ CairoMakie not available")
end

try
    using Plots
    global PLOTS_AVAILABLE = true
    if !CAIRO_AVAILABLE
        println("✓ Plots.jl available (fallback mode)")
    end
catch
    println("⚠ Plots.jl not available")
end

if !CAIRO_AVAILABLE && !PLOTS_AVAILABLE
    @error """
    No plotting backend available!
    Please install either CairoMakie or Plots:
      - Pkg.add("CairoMakie")  # Recommended
      - Pkg.add("Plots")       # Fallback
    """
    exit(1)
end

# =============================================================================
# SIMULATION FUNCTIONS
# =============================================================================

"""
    simulate_snapshots(p::Params; kwargs...)

Run agent-based simulation with enhanced tracking.

# Arguments
- `p::Params`: Model parameters
- `κ::Float64`: Social coupling strength
- `N::Int`: Number of agents
- `T::Float64`: Simulation time horizon
- `dt::Float64`: Time step
- `seed::Int`: Random seed
- `snapshot_times::Vector{Float64}`: Times to capture density snapshots
- `mt_stride::Int`: Stride for mean trajectory recording
- `track_variance::Bool=true`: Track variance over time
- `track_flux::Bool=false`: Track boundary flux (not implemented)

# Returns
- `snapshots`: Vector of density snapshots
- `mt_times`: Mean trajectory time points
- `mt_values`: Mean trajectory values
- `var_values`: Variance trajectory (if tracked)
"""
function simulate_snapshots(p::Params; κ::Float64, N::Int, T::Float64, dt::Float64,
                            seed::Int, snapshot_times::Vector{Float64}, mt_stride::Int,
                            track_variance::Bool=true, track_flux::Bool=false)
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
    
    # Additional tracking
    var_values = track_variance ? Float64[var(u)] : Float64[]
    
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
            track_variance && push!(var_values, var(u))
        end
    end

    return snapshots, mt_times, mt_values, var_values
end

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

"""
    histogram_density(data, edges)

Compute normalized probability density from histogram.

# Arguments
- `data::Vector{Float64}`: Raw data points
- `edges::Vector{Float64}`: Bin edges

# Returns
- `centers`: Bin centers
- `density`: Normalized density (integrates to 1)
"""
function histogram_density(data::Vector{Float64}, edges::Vector{Float64})
    hist = fit(Histogram, data, edges; closed=:left)
    weights = hist.weights
    total = sum(weights)
    binwidth = edges[2] - edges[1]
    density = total > 0 ? (weights ./ (total * binwidth)) : fill(0.0, length(weights))
    centers = midpoints(hist.edges[1])
    return centers, density
end

"""
    fit_exponential_rate(times, values; t_start, t_end)

Fit exponential growth model m(t) ≈ m₀ exp(λt) to extract growth rate λ.

This validates the linear stability analysis by measuring the empirical
growth rate of the mean trajectory in the early phase.

# Arguments
- `times::Vector{Float64}`: Time points
- `values::Vector{Float64}`: Mean trajectory values
- `t_start::Float64=10.0`: Start of fitting window
- `t_end::Float64=50.0`: End of fitting window

# Returns
- `λ_empirical`: Fitted growth rate
- `m0_empirical`: Initial amplitude
- `r_squared`: Quality of fit (R²)

# Notes
Performs linear regression on log-scale: log(|m|) = log(m₀) + λt
"""
function fit_exponential_rate(times::Vector{Float64}, values::Vector{Float64}; 
                              t_start::Float64=10.0, t_end::Float64=50.0)
    # Select fitting window
    idx = findall(t -> t_start <= t <= t_end, times)
    if length(idx) < 3
        return NaN, NaN, NaN  # Not enough points
    end
    
    t_fit = times[idx]
    m_fit = abs.(values[idx])
    
    # Remove zeros/negatives for log fitting
    valid = findall(m_fit .> 1e-10)
    if length(valid) < 3
        return NaN, NaN, NaN
    end
    
    t_fit = t_fit[valid]
    log_m = log.(m_fit[valid])
    
    # Linear regression on log scale: log(m) = log(m₀) + λt
    A = hcat(ones(length(t_fit)), t_fit)
    coeffs = A \ log_m
    
    λ_empirical = coeffs[2]
    m0_empirical = exp(coeffs[1])
    
    # Compute R²
    predictions = A * coeffs
    ss_res = sum((log_m .- predictions).^2)
    ss_tot = sum((log_m .- mean(log_m)).^2)
    r_squared = 1 - ss_res/ss_tot
    
    return λ_empirical, m0_empirical, r_squared
end

# =============================================================================
# MAIN EXECUTION
# =============================================================================

"""
    main()

Main execution function for Figure 6 generation.

Performs:
1. Parameter setup
2. Critical value estimation (V*, κ*)
3. Three simulations (below/at/above threshold)
4. Exponential growth rate validation
5. Data export (CSV)
6. Figure generation (PDF)
"""
function main()
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    
    # Baseline parameters matching manuscript
    λ = 0.85      # Private mean reversion rate
    σ = 0.8       # Noise strength
    Θ = 2.0       # Tolerance boundary
    c0 = 0.8      # Reset depth
    nu0 = 10.6    # Baseline hazard rate

    # Simulation settings
    N = 20000          # Number of agents
    T = 400.0          # Simulation horizon
    dt = 0.01          # Time step
    burn_in = 120.0    # Burn-in for V* estimation
    seed = 42          # Random seed

    # Analysis settings
    snapshot_times = [0.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, T]
    bins = 240
    mt_stride = max(1, Int(round(0.05 / dt)))  # Record every 0.05 time units

    # Output directories
    outdir = "outputs/fig6_density_snapshots_enhanced"
    figdir = "figs"
    mkpath(outdir)
    mkpath(figdir)

    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    
    p = Params(λ=λ, σ=σ, Θ=Θ, c0=c0, hazard=StepHazard(nu0))

    println("=" ^ 72)
    println("FIGURE 6: ENHANCED DENSITY SNAPSHOTS WITH CRITICAL POINT")
    println("=" ^ 72)
    @printf("\nModel Parameters:\n")
    @printf("  λ   = %.3f (private mean reversion)\n", λ)
    @printf("  σ   = %.3f (noise strength)\n", σ)
    @printf("  Θ   = %.3f (tolerance boundary)\n", Θ)
    @printf("  c₀  = %.3f (reset depth)\n", c0)
    @printf("  ν₀  = %.3f (baseline hazard rate)\n", nu0)
    
    @printf("\nSimulation Settings:\n")
    @printf("  N   = %d agents\n", N)
    @printf("  T   = %.1f time units\n", T)
    @printf("  dt  = %.3f\n", dt)
    @printf("  Burn-in = %.1f\n", burn_in)
    @printf("  Random seed = %d\n", seed)
    
    println("\nAnalysis Settings:")
    println("  Snapshot times: $(snapshot_times)")
    @printf("  Histogram bins: %d\n", bins)
    @printf("  Mean trajectory stride: %d steps (Δt = %.3f)\n", mt_stride, mt_stride * dt)

    # =========================================================================
    # CRITICAL VALUE ESTIMATION
    # =========================================================================
    
    println("\n" * "=" ^ 72)
    println("STEP 1: Estimating critical values")
    println("=" ^ 72)
    
    println("\nEstimating V* (stationary variance)...")
    Vstar = estimate_Vstar(p; N=N, T=350.0, dt=dt, burn_in=burn_in, seed=seed)
    @printf("✓ V* = %.6f\n", Vstar)
    
    println("\nComputing κ* (critical coupling)...")
    κstar = critical_kappa(p; Vstar=Vstar)
    @printf("✓ κ* = %.6f\n", κstar)

    # Define test values
    κ_minus = 0.8 * κstar     # Below threshold
    κ_critical = κstar        # At threshold
    κ_plus = 1.5 * κstar      # Above threshold
    
    println("\nTest scenarios:")
    @printf("  κ- = %.6f (%.0f%% of κ*) - Expected: Stable consensus\n", κ_minus, 80.0)
    @printf("  κ0 = %.6f (%.0f%% of κ*) - Expected: Critical slowing\n", κ_critical, 100.0)
    @printf("  κ+ = %.6f (%.0f%% of κ*) - Expected: Polarization\n", κ_plus, 150.0)

    # =========================================================================
    # SIMULATIONS
    # =========================================================================
    
    println("\n" * "=" ^ 72)
    println("STEP 2: Running agent-based simulations")
    println("=" ^ 72)
    
    println("\nScenario 1/3: Below threshold (κ = $(round(κ_minus, digits=4)))...")
    t_start = time()
    snaps_below, mt_t_below, mt_below, var_below = simulate_snapshots(
        p; κ=κ_minus, N=N, T=T, dt=dt,
        seed=seed + 1, snapshot_times=snapshot_times,
        mt_stride=mt_stride, track_variance=true
    )
    @printf("✓ Completed in %.1f seconds\n", time() - t_start)

    println("\nScenario 2/3: At critical point (κ = $(round(κ_critical, digits=4)))...")
    t_start = time()
    snaps_critical, mt_t_critical, mt_critical, var_critical = simulate_snapshots(
        p; κ=κ_critical, N=N, T=T, dt=dt,
        seed=seed + 2, snapshot_times=snapshot_times,
        mt_stride=mt_stride, track_variance=true
    )
    @printf("✓ Completed in %.1f seconds\n", time() - t_start)

    println("\nScenario 3/3: Above threshold (κ = $(round(κ_plus, digits=4)))...")
    t_start = time()
    snaps_above, mt_t_above, mt_above, var_above = simulate_snapshots(
        p; κ=κ_plus, N=N, T=T, dt=dt,
        seed=seed + 3, snapshot_times=snapshot_times,
        mt_stride=mt_stride, track_variance=true
    )
    @printf("✓ Completed in %.1f seconds\n", time() - t_start)

    # =========================================================================
    # EXPONENTIAL GROWTH RATE ANALYSIS
    # =========================================================================
    
    println("\n" * "=" ^ 72)
    println("STEP 3: Validating linear stability theory")
    println("=" ^ 72)
    
    println("\nFitting exponential growth rates (t ∈ [10, 50])...")
    
    λ_below, m0_below, r2_below = fit_exponential_rate(mt_t_below, mt_below)
    λ_critical, m0_critical, r2_critical = fit_exponential_rate(mt_t_critical, mt_critical)
    λ_above, m0_above, r2_above = fit_exponential_rate(mt_t_above, mt_above)
    
    println("\nResults:")
    println("  Below threshold (κ-):")
    @printf("    Empirical λ = %+.6f (R² = %.4f)\n", λ_below, r2_below)
    println("    Expected: λ < 0 (stable consensus)")
    status_below = λ_below < 0 ? "✓ PASS" : "✗ FAIL"
    println("    Status: $status_below")
    
    println("\n  At critical point (κ*):")
    @printf("    Empirical λ = %+.6f (R² = %.4f)\n", λ_critical, r2_critical)
    println("    Expected: λ ≈ 0 (critical slowing)")
    status_crit = abs(λ_critical) < 0.1 ? "✓ PASS" : "~ APPROXIMATE"
    println("    Status: $status_crit")
    
    println("\n  Above threshold (κ+):")
    @printf("    Empirical λ = %+.6f (R² = %.4f)\n", λ_above, r2_above)
    println("    Expected: λ > 0 (unstable, polarization)")
    status_above = λ_above > 0 ? "✓ PASS" : "✗ FAIL"
    println("    Status: $status_above")

    # =========================================================================
    # DATA EXPORT
    # =========================================================================
    
    println("\n" * "=" ^ 72)
    println("STEP 4: Exporting data")
    println("=" ^ 72)
    
    # Shared bin edges for all histograms
    all_vals = vcat(
        reduce(vcat, snaps_below), 
        reduce(vcat, snaps_critical),
        reduce(vcat, snaps_above)
    )
    umin = minimum(all_vals)
    umax = maximum(all_vals)
    pad = 0.05 * max(1e-6, umax - umin)
    edges = collect(range(umin - pad, umax + pad; length=bins + 1))
    
    # Mean trajectories
    println("\nSaving mean trajectories...")
    CSV.write(joinpath(outdir, "mt_below.csv"), 
              DataFrame(t=mt_t_below, m=mt_below, var=var_below))
    CSV.write(joinpath(outdir, "mt_critical.csv"), 
              DataFrame(t=mt_t_critical, m=mt_critical, var=var_critical))
    CSV.write(joinpath(outdir, "mt_above.csv"), 
              DataFrame(t=mt_t_above, m=mt_above, var=var_above))
    println("✓ Mean trajectories saved")

    # Growth rate analysis
    println("\nSaving growth rate analysis...")
    growth_df = DataFrame(
        condition = ["below", "critical", "above"],
        kappa = [κ_minus, κ_critical, κ_plus],
        kappa_ratio = [0.8, 1.0, 1.2],
        lambda_empirical = [λ_below, λ_critical, λ_above],
        r_squared = [r2_below, r2_critical, r2_above],
        m0 = [m0_below, m0_critical, m0_above],
        final_mean = [mt_below[end], mt_critical[end], mt_above[end]],
        final_var = [var_below[end], var_critical[end], var_above[end]]
    )
    CSV.write(joinpath(outdir, "growth_rates.csv"), growth_df)
    println("✓ Growth rate analysis saved")

    # Histograms
    println("\nSaving density snapshots...")
    for (label, snaps) in [("below", snaps_below), 
                           ("critical", snaps_critical), 
                           ("above", snaps_above)]
        for (idx, t) in enumerate(snapshot_times)
            centers, dens = histogram_density(snaps[idx], edges)
            tstr = replace(@sprintf("%.2f", t), "." => "p")
            path = joinpath(outdir, "hist_$(label)_t$(tstr).csv")
            CSV.write(path, DataFrame(bin_center=centers, density=dens))
        end
    end
    println("✓ Density snapshots saved ($(length(snapshot_times)) times × 3 scenarios)")

    # Summary file
    summary_path = joinpath(outdir, "summary.txt")
    println("\nWriting summary...")
    open(summary_path, "w") do io
        println(io, "Figure 6: Enhanced density snapshots with critical point validation")
        println(io, "Generated: $(Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))")
        println(io, "=" ^ 72)
        
        println(io, "\nMODEL PARAMETERS:")
        @printf(io, "  λ = %.4f (private mean reversion)\n", λ)
        @printf(io, "  σ = %.4f (noise strength)\n", σ)
        @printf(io, "  Θ = %.4f (tolerance boundary)\n", Θ)
        @printf(io, "  c₀ = %.4f (reset depth)\n", c0)
        @printf(io, "  ν₀ = %.4f (baseline hazard rate)\n", nu0)
        
        println(io, "\nSIMULATION SETTINGS:")
        @printf(io, "  N = %d agents\n", N)
        @printf(io, "  T = %.1f time units\n", T)
        @printf(io, "  dt = %.4f\n", dt)
        @printf(io, "  Burn-in = %.1f\n", burn_in)
        @printf(io, "  Random seeds: %d, %d, %d\n", seed+1, seed+2, seed+3)
        
        println(io, "\nCRITICAL VALUES:")
        @printf(io, "  V* = %.6f (stationary variance)\n", Vstar)
        @printf(io, "  κ* = %.6f (critical coupling)\n", κstar)
        
        println(io, "\nTEST SCENARIOS:")
        @printf(io, "  κ- = %.6f (80%% of κ*) - Below threshold\n", κ_minus)
        @printf(io, "  κ0 = %.6f (100%% of κ*) - At critical point\n", κ_critical)
        @printf(io, "  κ+ = %.6f (120%% of κ*) - Above threshold\n", κ_plus)
        
        println(io, "\nGROWTH RATE VALIDATION:")
        println(io, "  Fitting window: t ∈ [10.0, 50.0]")
        println(io, "  Model: m(t) ≈ m₀ exp(λt)")
        println(io, "")
        @printf(io, "  Below threshold:\n")
        @printf(io, "    λ = %+.6f, R² = %.4f\n", λ_below, r2_below)
        @printf(io, "    Status: %s\n", λ_below < 0 ? "STABLE" : "UNEXPECTED")
        println(io, "")
        @printf(io, "  At critical point:\n")
        @printf(io, "    λ = %+.6f, R² = %.4f\n", λ_critical, r2_critical)
        @printf(io, "    Status: %s\n", abs(λ_critical) < 0.1 ? "CRITICAL" : "NEAR-CRITICAL")
        println(io, "")
        @printf(io, "  Above threshold:\n")
        @printf(io, "    λ = %+.6f, R² = %.4f\n", λ_above, r2_above)
        @printf(io, "    Status: %s\n", λ_above > 0 ? "UNSTABLE" : "UNEXPECTED")
        
        println(io, "\nFINAL STATES (t = $(T)):")
        @printf(io, "  Below:    m = %+.4f, σ² = %.4f\n", mt_below[end], var_below[end])
        @printf(io, "  Critical: m = %+.4f, σ² = %.4f\n", mt_critical[end], var_critical[end])
        @printf(io, "  Above:    m = %+.4f, σ² = %.4f\n", mt_above[end], var_above[end])
        
        println(io, "\nDATA FILES:")
        println(io, "  Mean trajectories:")
        println(io, "    - mt_below.csv")
        println(io, "    - mt_critical.csv")
        println(io, "    - mt_above.csv")
        println(io, "  Growth rate analysis:")
        println(io, "    - growth_rates.csv")
        println(io, "  Density snapshots:")
        println(io, "    - hist_{below,critical,above}_t{time}.csv")
        println(io, "    - $(length(snapshot_times)) time points per scenario")
        println(io, "    - $bins bins uniformly spaced on [$(round(edges[1], digits=2)), $(round(edges[end], digits=2))]")
        
        println(io, "\nSNAPSHOT TIMES:")
        for (i, t) in enumerate(snapshot_times)
            println(io, "  $i. t = $t")
        end
    end
    println("✓ Summary saved to: $summary_path")

    # =========================================================================
    # FIGURE GENERATION
    # =========================================================================
    
    println("\n" * "=" ^ 72)
    println("STEP 5: Generating figures")
    println("=" ^ 72)
    
    if CAIRO_AVAILABLE
        println("\nUsing CairoMakie for high-quality output...")
        
        # Main figure: 4-row layout
        fig = Figure(size=(1200, 1400), fontsize=14, backgroundcolor=:white)
        
        # Row 1: Below threshold
        ax1 = Axis(fig[1, 1], xlabel="u", ylabel="ρ(u)", 
                   title="Below κ* (κ = $(round(κ_minus/κstar, digits=2))κ*)")
        
        # Row 2: At critical point  
        ax2 = Axis(fig[2, 1], xlabel="u", ylabel="ρ(u)", 
                   title="At κ* (critical slowing)")
        
        # Row 3: Above threshold
        ax3 = Axis(fig[3, 1], xlabel="u", ylabel="ρ(u)", 
                   title="Above κ* (κ = $(round(κ_plus/κstar, digits=2))κ*)")
        
        # Row 4: Mean trajectories
        ax4 = Axis(fig[4, 1], xlabel="t", ylabel="m(t)", 
                   title="Mean trajectory comparison")

        # Add tolerance boundaries
        for ax in [ax1, ax2, ax3]
            vlines!(ax, [Θ, -Θ], color=:gray, linestyle=:dash, alpha=0.3, linewidth=1.5)
            vlines!(ax, [c0*Θ, -c0*Θ], color=:lightgray, linestyle=:dot, alpha=0.3, linewidth=1.5)
        end

        # Plot density evolution
        colors = collect(cgrad(:viridis, length(snapshot_times), categorical=true))
        
        for (i, t) in enumerate(snapshot_times)
            # Below
            centers, dens = histogram_density(snaps_below[i], edges)
            lines!(ax1, centers, dens, color=colors[i], 
                   label=@sprintf("t=%.0f", t), linewidth=2)
            
            # Critical
            centers2, dens2 = histogram_density(snaps_critical[i], edges)
            lines!(ax2, centers2, dens2, color=colors[i], 
                   label=@sprintf("t=%.0f", t), linewidth=2)
            
            # Above
            centers3, dens3 = histogram_density(snaps_above[i], edges)
            lines!(ax3, centers3, dens3, color=colors[i], 
                   label=@sprintf("t=%.0f", t), linewidth=2)
        end

        # Mean trajectories
        lines!(ax4, mt_t_below, mt_below, color=:blue, linewidth=2.5, 
               label="κ⁻ (stable)")
        lines!(ax4, mt_t_critical, mt_critical, color=:green, linewidth=2.5, 
               label="κ* (critical)", linestyle=:dash)
        lines!(ax4, mt_t_above, mt_above, color=:red, linewidth=2.5, 
               label="κ⁺ (unstable)")
        
        # Add zero line
        hlines!(ax4, [0.0], color=:black, linestyle=:dot, alpha=0.5)

        # Add exponential fit visualization (if valid)
        if !isnan(λ_above) && r2_above > 0.8
            t_fit_range = 10.0:1.0:50.0
            m_fit_curve = m0_above .* exp.(λ_above .* t_fit_range)
            lines!(ax4, t_fit_range, m_fit_curve, color=:red, 
                   linestyle=:dashdot, alpha=0.5, linewidth=1.5,
                   label="exp. fit")
            
            # Add text annotation
            text!(ax4, 30.0, 0.8*maximum(abs.(mt_above)), 
                  text=@sprintf("λ ≈ %.4f", λ_above),
                  fontsize=12, color=:red)
        end

        # Legends
        axislegend(ax1; position=:rt, framevisible=false, labelsize=11)
        axislegend(ax2; position=:rt, framevisible=false, labelsize=11)
        axislegend(ax3; position=:rt, framevisible=false, labelsize=11)
        axislegend(ax4; position=:rb, framevisible=false, labelsize=12)

        # Save main figure
        fig_path = joinpath(figdir, "fig6_density_snapshots_enhanced.pdf")
        save(fig_path, fig)
        println("✓ Main figure saved: $fig_path")

        # Supplementary figure: Variance evolution
        fig_var = Figure(size=(800, 500), fontsize=14, backgroundcolor=:white)
        ax_var = Axis(fig_var[1, 1], xlabel="t", ylabel="Variance σ²(t)", 
                      title="Variance dynamics across threshold")
        
        lines!(ax_var, mt_t_below, var_below, color=:blue, linewidth=2.5, label="κ⁻")
        lines!(ax_var, mt_t_critical, var_critical, color=:green, linewidth=2.5, 
               label="κ*", linestyle=:dash)
        lines!(ax_var, mt_t_above, var_above, color=:red, linewidth=2.5, label="κ⁺")
        
        axislegend(ax_var; position=:rb, framevisible=false)
        
        var_path = joinpath(figdir, "fig6_variance_dynamics.pdf")
        save(var_path, fig_var)
        println("✓ Variance figure saved: $var_path")

    elseif PLOTS_AVAILABLE
        println("\nUsing Plots.jl (fallback mode)...")
        
        default(fontfamily="Computer Modern")
        colors = cgrad(:viridis, length(snapshot_times), categorical=true)

        plt1 = plot(; xlabel="u", ylabel="ρ(u)", title="Below κ*", legend=:right, size=(800, 300))
        for (i, t) in enumerate(snapshot_times)
            centers, dens = histogram_density(snaps_below[i], edges)
            plot!(plt1, centers, dens; color=colors[i], label=@sprintf("t=%.0f", t), linewidth=2)
        end

        plt2 = plot(; xlabel="u", ylabel="ρ(u)", title="At κ*", legend=:right, size=(800, 300))
        for (i, t) in enumerate(snapshot_times)
            centers, dens = histogram_density(snaps_critical[i], edges)
            plot!(plt2, centers, dens; color=colors[i], label=@sprintf("t=%.0f", t), linewidth=2)
        end

        plt3 = plot(; xlabel="u", ylabel="ρ(u)", title="Above κ*", legend=:right, size=(800, 300))
        for (i, t) in enumerate(snapshot_times)
            centers, dens = histogram_density(snaps_above[i], edges)
            plot!(plt3, centers, dens; color=colors[i], label=@sprintf("t=%.0f", t), linewidth=2)
        end

        plt4 = plot(mt_t_below, mt_below; xlabel="t", ylabel="m(t)", 
                    title="Mean trajectory", color=:blue, linewidth=2.0, label="κ⁻",
                    size=(800, 300))
        plot!(plt4, mt_t_critical, mt_critical; color=:green, linewidth=2.0, 
              label="κ*", linestyle=:dash)
        plot!(plt4, mt_t_above, mt_above; color=:red, linewidth=2.0, label="κ⁺")

        fig_path = joinpath(figdir, "fig6_density_snapshots_enhanced.pdf")
        fig = plot(plt1, plt2, plt3, plt4; layout=(4,1), size=(900, 1200))
        savefig(fig, fig_path)
        println("✓ Figure saved: $fig_path")
    end

    # =========================================================================
    # COMPLETION
    # =========================================================================
    
    println("\n" * "=" ^ 72)
    println("✓ FIGURE 6 GENERATION COMPLETE")
    println("=" ^ 72)
    println("\nOutputs:")
    println("  📁 Data directory: $outdir")
    println("  📄 Summary: $summary_path")
    println("  📊 Figures: $figdir")
    println("\nValidation results:")
    println("  Below threshold: $status_below")
    println("  At critical point: $status_crit")
    println("  Above threshold: $status_above")
    println("\n" * "=" ^ 72)
end

# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    try
        main()
    catch e
        @error "Script failed with error:" exception=(e, catch_backtrace())
        exit(1)
    end
end
