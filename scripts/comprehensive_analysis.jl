"""
Comprehensive Complementary Analysis and Visualization
For OU-with-Resets Mean-Field Belief Dynamics

Fixed issues:
- Symbol to String conversion for axis labels
- Enhanced plots to show system-specific features
- Better parameter scanning
- More informative visualizations
"""

include("../src/bifurcation/model_interface.jl")
include("../src/bifurcation/plotting_cairo.jl")
include("../src/bifurcation/simple_continuation.jl")

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src", "bifurcation"))

using LinearAlgebra
using Statistics
using DifferentialEquations
using FFTW
using StatsBase
using Logging
using Printf

# Check for CairoMakie
HAVE_MAKIE = try
    using CairoMakie
    using ColorSchemes
    true
catch
    @warn "CairoMakie not available"
    false
end

if !HAVE_MAKIE
    @error "This script requires CairoMakie for visualization"
    exit(1)
end

using .ModelInterface
using .SimpleContinuation
using .PlottingCairo

#═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
#═══════════════════════════════════════════════════════════════════════════

"""
Compute nullclines: curves where du₁/dt = 0 or du₂/dt = 0
"""
function compute_nullclines(f::Function, p; resolution=200, limits=(-3, 3))
    u1_range = range(limits[1], limits[2], length=resolution)
    u2_range = range(limits[1], limits[2], length=resolution)
    
    nullcline1_points = Tuple{Float64, Float64}[]
    nullcline2_points = Tuple{Float64, Float64}[]
    
    for u1 in u1_range
        for u2 in u2_range
            F = f([u1, u2], p)
            
            # Check if du₁/dt ≈ 0
            if abs(F[1]) < 0.05
                push!(nullcline1_points, (u1, u2))
            end
            
            # Check if du₂/dt ≈ 0
            if abs(F[2]) < 0.05
                push!(nullcline2_points, (u1, u2))
            end
        end
    end
    
    return nullcline1_points, nullcline2_points
end

"""
Classify final attractor by running trajectory to steady state
"""
function classify_attractor(u0::Vector{Float64}, p; tmax=1000.0, tol=1e-4)
    function ode!(du, u, p, t)
        F = ModelInterface.f(u, p)
        du .= F
    end
    
    prob = ODEProblem(ode!, u0, (0.0, tmax), p)
    
    try
        sol = solve(prob, Tsit5(); reltol=1e-6, abstol=1e-6)
        u_final = sol.u[end]
        
        # Classify based on mean of final state
        mean_final = mean(u_final)
        
        if abs(mean_final) < 0.1
            return 0  # Consensus at origin
        elseif mean_final > 0.1
            return 1  # Polarized positive
        elseif mean_final < -0.1
            return -1  # Polarized negative
        else
            return 0  # Near origin
        end
    catch e
        return NaN
    end
end

"""
Compute autocorrelation function
"""
function autocorrelation(x::Vector{Float64}, max_lag::Int)
    n = length(x)
    x_centered = x .- mean(x)
    
    acf = zeros(max_lag + 1)
    c0 = sum(x_centered.^2) / n
    
    if c0 < 1e-12
        return acf
    end
    
    for lag in 0:max_lag
        if lag == 0
            acf[1] = 1.0
        else
            c_lag = sum(x_centered[1:n-lag] .* x_centered[lag+1:n]) / n
            acf[lag+1] = c_lag / c0
        end
    end
    
    return acf
end

"""
Compute power spectral density
"""
function compute_psd(x::Vector{Float64}, dt::Float64)
    n = length(x)
    x_centered = x .- mean(x)
    
    # Apply Hanning window
    window = 0.5 .- 0.5 .* cos.(2π .* (0:n-1) ./ (n-1))
    x_windowed = x_centered .* window
    
    # Compute FFT
    X = fft(x_windowed)
    psd = abs2.(X) ./ n
    freqs = fftfreq(n, 1/dt)
    
    # Keep only positive frequencies
    pos_idx = freqs .>= 0
    
    return freqs[pos_idx], psd[pos_idx]
end

"""
Compute largest Lyapunov exponent
"""
function compute_lyapunov(f::Function, jac::Function, u0::Vector{Float64}, p;
                         tmax=1000.0, dt=0.1)
    n = length(u0)
    u = copy(u0)
    w = randn(n)
    w ./= norm(w)
    
    λ_sum = 0.0
    n_steps = 0
    
    t = 0.0
    while t < tmax
        # Integrate trajectory
        J = jac(u, p)
        F = f(u, p)
        
        u .+= F .* dt
        w .+= (J * w) .* dt
        
        # Renormalize
        w_norm = norm(w)
        if w_norm > 1e-12
            λ_sum += log(w_norm)
            w ./= w_norm
        end
        
        t += dt
        n_steps += 1
    end
    
    if n_steps == 0
        return 0.0
    end
    
    return λ_sum / (n_steps * dt)
end

#═══════════════════════════════════════════════════════════════════════════
# PLOTTING FUNCTIONS
#═══════════════════════════════════════════════════════════════════════════

"""
Plot 1: Extended bifurcation diagram with stability and eigenvalues
"""
function plot_extended_bifurcation(κ_range; p_base=ModelInterface.default_params())
    @info "Computing extended bifurcation diagram..."
    
    u0 = zeros(2)
    branch = SimpleContinuation.continue_equilibria(
        ModelInterface.f,
        ModelInterface.jacobian,
        u0,
        p_base,
        κ_range;
        tol=1e-10
    )
    
    fig = Figure(size=(1400, 1000))
    
    # Panel 1: Equilibrium values
    ax1 = Axis(fig[1, 1:2];
              xlabel="Coupling strength κ",
              ylabel="Equilibrium u₁",
              title="Equilibrium Branch (Mean-Field Dynamics)")
    
    u1_eq = [u[1] for u in branch.equilibria]
    u2_eq = [u[2] for u in branch.equilibria]
    mean_eq = (u1_eq .+ u2_eq) ./ 2
    
    # Determine stability
    stable_mask = [all(real.(λs) .< 0) for λs in branch.eigenvalues]
    
    # Plot stable branch
    lines!(ax1, branch.κ_values[stable_mask], mean_eq[stable_mask];
          linewidth=3, color=:blue, label="Stable (consensus)")
    
    # Plot unstable branch
    if any(.!stable_mask)
        lines!(ax1, branch.κ_values[.!stable_mask], mean_eq[.!stable_mask];
              linewidth=3, color=:red, linestyle=:dash, label="Unstable")
    end
    
    # Mark bifurcation points
    hopf_idx = findall([SimpleContinuation.detect_hopf(λs) for λs in branch.eigenvalues])
    if !isempty(hopf_idx)
        scatter!(ax1, branch.κ_values[hopf_idx], mean_eq[hopf_idx];
                color=:purple, markersize=15, marker=:star5, label="Hopf bifurcation")
    end
    
    # Mark critical κ* (where stability changes)
    stability_change = findall(diff(stable_mask) .!= 0)
    if !isempty(stability_change)
        idx = stability_change[1]
        vlines!(ax1, [branch.κ_values[idx]]; 
               color=:green, linestyle=:dash, linewidth=2, label="κ* (pitchfork)")
    end
    
    axislegend(ax1, position=:lt)
    
    # Panel 2: Largest real eigenvalue (stability indicator)
    ax2 = Axis(fig[2, 1];
              xlabel="κ",
              ylabel="max Re(λ)",
              title="Stability Indicator (Re(λ) < 0 = stable)")
    
    max_real_eig = [maximum(real.(λs)) for λs in branch.eigenvalues]
    lines!(ax2, branch.κ_values, max_real_eig;
          linewidth=2, color=:black)
    hlines!(ax2, [0]; color=:red, linestyle=:dash, linewidth=2, label="Stability boundary")
    
    # Shade stable/unstable regions
    band!(ax2, branch.κ_values, fill(-1, length(branch.κ_values)), 
          min.(max_real_eig, 0);
          color=(:green, 0.2), label="Stable region")
    band!(ax2, branch.κ_values, zeros(length(branch.κ_values)), 
          max.(max_real_eig, 0);
          color=(:red, 0.2), label="Unstable region")
    
    axislegend(ax2, position=:lt)
    
    # Panel 3: Imaginary part (oscillation frequency)
    ax3 = Axis(fig[2, 2];
              xlabel="κ",
              ylabel="|Im(λ)| (oscillation frequency)",
              title="Hopf Oscillation Frequency")
    
    # Get imaginary parts
    imag_parts = [maximum(abs.(imag.(λs))) for λs in branch.eigenvalues]
    lines!(ax3, branch.κ_values, imag_parts;
          linewidth=2, color=:purple)
    
    if !isempty(hopf_idx)
        scatter!(ax3, branch.κ_values[hopf_idx], imag_parts[hopf_idx];
                color=:purple, markersize=12, marker=:star5)
    end
    
    # Panel 4: Polarization measure
    ax4 = Axis(fig[3, 1:2];
              xlabel="κ",
              ylabel="Polarization |u₁ - u₂| at equilibrium",
              title="Equilibrium Polarization")
    
    polarization = abs.(u1_eq .- u2_eq)
    lines!(ax4, branch.κ_values, polarization; linewidth=2, color=:orange)
    
    # Mark bifurcation
    if !isempty(stability_change)
        idx = stability_change[1]
        vlines!(ax4, [branch.κ_values[idx]]; 
               color=:green, linestyle=:dash, linewidth=2)
    end
    
    return fig
end

"""
Plot 2: Phase portraits with nullclines and manifolds
"""
function plot_phase_space_analysis(κ_values; p_base=ModelInterface.default_params())
    @info "Computing phase space analysis..."
    
    n_plots = length(κ_values)
    ncols = 2
    nrows = ceil(Int, n_plots / ncols)
    
    fig = Figure(size=(1200, 600*nrows))
    
    for (idx, κ) in enumerate(κ_values)
        p = ModelInterface.kappa_set(p_base, κ)
        
        row = div(idx - 1, ncols) + 1
        col = mod(idx - 1, ncols) + 1
        
        ax = Axis(fig[row, col];
                 xlabel="u₁ (agent 1 belief)",
                 ylabel="u₂ (agent 2 belief)",
                 title="Phase Space: κ = $(round(κ, digits=3))",
                 aspect=DataAspect())
        
        # Phase portrait
        PlottingCairo.phase_portrait!(ax, ModelInterface.f, p;
                                     lims=(-3, 3, -3, 3),
                                     density=35, alpha=0.25)
        
        # Add diagonal (u₁ = u₂ line - consensus line)
        lines!(ax, [-3, 3], [-3, 3]; 
              color=:gray, linestyle=:dash, linewidth=2, label="Consensus line")
        
        # Compute and plot nullclines
        nc1, nc2 = compute_nullclines(ModelInterface.f, p; resolution=100)
        
        if !isempty(nc1)
            nc1_x = [p[1] for p in nc1]
            nc1_y = [p[2] for p in nc1]
            scatter!(ax, nc1_x, nc1_y;
                    color=:blue, markersize=2, alpha=0.5, label="du₁/dt=0")
        end
        
        if !isempty(nc2)
            nc2_x = [p[1] for p in nc2]
            nc2_y = [p[2] for p in nc2]
            scatter!(ax, nc2_x, nc2_y;
                    color=:red, markersize=2, alpha=0.5, label="du₂/dt=0")
        end
        
        # Find and mark equilibria
        u_eq = SimpleContinuation.newton_solve(
            ModelInterface.f,
            ModelInterface.jacobian,
            zeros(2),
            p
        )
        
        # Check stability
        J = ModelInterface.jacobian(u_eq, p)
        λs = eigvals(J)
        is_stable = all(real.(λs) .< 0)
        
        scatter!(ax, [u_eq[1]], [u_eq[2]];
                color=is_stable ? :green : :orange,
                markersize=15,
                marker=is_stable ? :circle : :xcross,
                label=is_stable ? "Stable equilibrium" : "Unstable equilibrium")
        
        # Add text annotation
        κ_star = getproperty(p, :kstar)
        regime = κ < κ_star ? "Pre-bifurcation" : "Post-bifurcation"
        text!(ax, -2.5, 2.5;
             text="$regime\nκ/κ* = $(round(κ/κ_star, digits=2))",
             fontsize=14, align=(:left, :top))
        
        if idx == 1
            axislegend(ax, position=:rb, labelsize=10)
        end
    end
    
    return fig
end

"""
Plot 3: Basin of attraction
"""
function plot_basin_of_attraction(κ_values; p_base=ModelInterface.default_params(),
                                 resolution=60, limits=(-3, 3))
    @info "Computing basins of attraction (this may take a while)..."
    
    n_plots = length(κ_values)
    ncols = 2
    nrows = ceil(Int, n_plots / ncols)
    
    fig = Figure(size=(1400, 700*nrows))
    
    u1_range = range(limits[1], limits[2], length=resolution)
    u2_range = range(limits[1], limits[2], length=resolution)
    
    for (idx, κ) in enumerate(κ_values)
        p = ModelInterface.kappa_set(p_base, κ)
        
        row = div(idx - 1, ncols) + 1
        col = mod(idx - 1, ncols) + 1
        
        ax = Axis(fig[row, col];
                 xlabel="Initial u₁",
                 ylabel="Initial u₂",
                 title="Basin of Attraction: κ=$(round(κ, digits=3))",
                 aspect=DataAspect())
        
        # Compute basin map
        basin_map = zeros(resolution, resolution)
        
        @info "  Computing basin for κ=$κ..." progress=true
        
        Threads.@threads for i in 1:resolution
            for j in 1:resolution
                u0 = [u1_range[i], u2_range[j]]
                basin_map[i, j] = classify_attractor(u0, p; tmax=1000.0)
            end
        end
        
        # Plot heatmap
        hm = heatmap!(ax, u1_range, u2_range, basin_map';
                     colormap=:RdBu, colorrange=(-1.2, 1.2))
        
        # Add consensus line
        lines!(ax, [-3, 3], [-3, 3]; 
              color=:black, linestyle=:dash, linewidth=2, alpha=0.5)
        
        # Overlay equilibrium
        u_eq = SimpleContinuation.newton_solve(
            ModelInterface.f,
            ModelInterface.jacobian,
            zeros(2),
            p
        )
        
        scatter!(ax, [u_eq[1]], [u_eq[2]];
                color=:yellow, markersize=20, marker=:star5,
                strokewidth=2, strokecolor=:black)
        
        # Add colorbar
        if col == ncols || idx == n_plots
            Colorbar(fig[row, col+1], hm;
                    label="Final state",
                    ticks=([-1, 0, 1], ["Negative\npolarized", "Consensus", "Positive\npolarized"]))
        end
    end
    
    return fig
end

"""
Plot 4: Time series analysis
"""
function plot_time_series_analysis(κ; p_base=ModelInterface.default_params(),
                                   u0=[0.5, -0.5], tmax=1000.0, dt=0.1)
    @info "Performing time series analysis for κ=$κ..."
    
    p = ModelInterface.kappa_set(p_base, κ)
    
    # Generate trajectory
    function ode!(du, u, p, t)
        F = ModelInterface.f(u, p)
        du .= F
    end
    
    prob = ODEProblem(ode!, u0, (0.0, tmax), p)
    sol = solve(prob, Tsit5(); reltol=1e-8, abstol=1e-8, saveat=dt)
    
    t = sol.t
    traj = reduce(hcat, sol.u)
    u1 = traj[1, :]
    u2 = traj[2, :]
    
    fig = Figure(size=(1600, 1200))
    
    # Panel 1: Time series
    ax1 = Axis(fig[1, 1:2];
              xlabel="Time",
              ylabel="Belief state",
              title="Trajectory Evolution (κ=$(round(κ, digits=3)))")
    
    lines!(ax1, t, u1; linewidth=1.5, color=:blue, label="Agent 1 (u₁)")
    lines!(ax1, t, u2; linewidth=1.5, color=:red, label="Agent 2 (u₂)")
    
    # Add mean
    mean_traj = (u1 .+ u2) ./ 2
    lines!(ax1, t, mean_traj; linewidth=2, color=:green, label="Mean ⟨u⟩")
    hlines!(ax1, [0]; color=:black, linestyle=:dash, linewidth=1, alpha=0.5)
    
    axislegend(ax1, position=:rt)
    
    # Panel 2: Polarization measure
    ax2 = Axis(fig[2, 1];
              xlabel="Time",
              ylabel="Polarization",
              title="Opinion Divergence |u₁ - u₂|")
    
    polarization = abs.(u1 .- u2)
    lines!(ax2, t, polarization; linewidth=1.5, color=:purple)
    
    # Add running average
    window = 50
    if length(polarization) > window
        smoothed = [mean(polarization[max(1,i-window):i]) for i in 1:length(polarization)]
        lines!(ax2, t, smoothed; linewidth=2, color=:orange, label="Smoothed")
        axislegend(ax2, position=:rt)
    end
    
    # Panel 3: Distance from consensus
    ax3 = Axis(fig[2, 2];
              xlabel="Time",
              ylabel="Distance from origin",
              title="System State Magnitude")
    
    distance = sqrt.(u1.^2 .+ u2.^2)
    lines!(ax3, t, distance; linewidth=1.5, color=:green)
    
    # Panel 4: Autocorrelation
    ax4 = Axis(fig[3, 1];
              xlabel="Lag (time)",
              ylabel="Autocorrelation",
              title="Memory in Dynamics")
    
    max_lag = min(500, length(u1) ÷ 4)
    acf = autocorrelation(mean_traj, max_lag)
    lags = 0:max_lag
    
    lines!(ax4, lags .* dt, acf; linewidth=2, color=:blue)
    hlines!(ax4, [0]; color=:black, linestyle=:dash, linewidth=1)
    
    # Compute decorrelation time
    decay_idx = findfirst(acf .< exp(-1))
    if !isnothing(decay_idx)
        τ_decorr = lags[decay_idx] * dt
        vlines!(ax4, [τ_decorr]; color=:red, linestyle=:dash, 
               linewidth=2, label="τ = $(round(τ_decorr, digits=1))")
        axislegend(ax4, position=:rt)
    end
    
    # Panel 5: Power spectrum
    ax5 = Axis(fig[3, 2];
              xlabel="Frequency (Hz)",
              ylabel="Power",
              title="Frequency Content",
              xscale=log10,
              yscale=log10)
    
    freqs, psd = compute_psd(mean_traj, dt)
    valid_idx = (freqs .> 0) .& (psd .> 1e-12)
    lines!(ax5, freqs[valid_idx], psd[valid_idx]; linewidth=2, color=:blue)
    
    # Panel 6: Phase space trajectory
    ax6 = Axis(fig[4, 1:2];
              xlabel="u₁",
              ylabel="u₂",
              title="Phase Space Trajectory",
              aspect=DataAspect())
    
    # Downsample for visualization
    stride = max(1, length(u1) ÷ 2000)
    u1_plot = u1[1:stride:end]
    u2_plot = u2[1:stride:end]
    
    # Color by time
    n_seg = length(u1_plot) - 1
    colors = range(colorant"blue", colorant"red", length=n_seg)
    
    for i in 1:n_seg
        lines!(ax6, u1_plot[i:i+1], u2_plot[i:i+1];
              linewidth=1.5, color=colors[i])
    end
    
    # Add consensus line
    lines!(ax6, [-3, 3], [-3, 3]; color=:gray, linestyle=:dash, linewidth=1)
    
    scatter!(ax6, [u1[1]], [u2[1]];
            color=:blue, markersize=15, marker=:circle, label="Start")
    scatter!(ax6, [u1[end]], [u2[end]];
            color=:red, markersize=15, marker=:star5, label="End")
    axislegend(ax6, position=:lt)
    
    return fig
end


"""
Plot 5: Enhanced return map with multiple dynamical behaviors
"""
function plot_enhanced_return_maps(κ_values; p_base=ModelInterface.default_params())
    @info "Computing enhanced return maps..."
    
    fig = Figure(size=(1600, 1200))
    
    # Different scenarios to capture different dynamics
    scenarios = [
        (u0=[1.0, -1.0], tmax=1000.0, name="Polarized start"),
        (u0=[0.1, 0.1], tmax=1000.0, name="Near consensus"),
        (u0=[2.0, -2.0], tmax=1000.0, name="Extreme polarization")
    ]
    
    dt = 0.1
    
    for (κ_idx, κ) in enumerate(κ_values)
        p = ModelInterface.kappa_set(p_base, κ)
        
        for (scenario_idx, scenario) in enumerate(scenarios)
            # Generate trajectory
            function ode!(du, u, p, t)
                F = ModelInterface.f(u, p)
                du .= F
            end
            
            prob = ODEProblem(ode!, scenario.u0, (0.0, scenario.tmax), p)
            sol = solve(prob, Tsit5(); saveat=dt)
            
            traj = reduce(hcat, sol.u)
            
            # Use u1 component for return map (more interesting than mean)
            u1 = traj[1, :]
            
            # Compute return map with lag
            delay = 20  # Fixed delay for consistency
            
            row = κ_idx
            col = scenario_idx
            
            ax = Axis(fig[row, col];
                     xlabel="u₁(t)",
                     ylabel="u₁(t+$(delay)Δt)",
                     title="κ=$(round(κ, digits=2)) - $(scenario.name)",
                     aspect=DataAspect())
            
            # Check variance
            traj_var = var(u1)
            
            if traj_var < 1e-6
                # Converged to fixed point
                eq_val = mean(u1)
                scatter!(ax, [eq_val], [eq_val];
                        markersize=20, color=:red, marker=:xcross,
                        label="Fixed point")
                
                padding = 0.5
                lims = (eq_val - padding, eq_val + padding)
                xlims!(ax, lims)
                ylims!(ax, lims)
                
                lines!(ax, [lims[1], lims[2]], [lims[1], lims[2]];
                      color=:gray, linestyle=:dash, linewidth=1)
                
            else
                # Plot return map
                x_vals = u1[1:end-delay]
                y_vals = u1[delay+1:end]
                
                # Color by density (darker = more visits)
                scatter!(ax, x_vals, y_vals;
                        markersize=2, alpha=0.3, color=:blue)
                
                # Compute and set limits
                x_range = extrema(x_vals)
                y_range = extrema(y_vals)
                all_range = (min(x_range[1], y_range[1]), 
                            max(x_range[2], y_range[2]))
                
                padding = 0.1 * (all_range[2] - all_range[1])
                if padding < 1e-6
                    padding = 0.5
                end
                
                lims = (all_range[1] - padding, all_range[2] + padding)
                
                xlims!(ax, lims)
                ylims!(ax, lims)
                
                # Add diagonal
                lines!(ax, [lims[1], lims[2]], [lims[1], lims[2]];
                      color=:red, linestyle=:dash, linewidth=2, 
                      label="Perfect memory", alpha=0.5)
                
                # Compute and display statistics
                if length(x_vals) > 0
                    corr_val = cor(x_vals, y_vals)
                    
                    # Check for structure
                    if abs(corr_val) > 0.9
                        structure = "Strong memory"
                        color = :green
                    elseif abs(corr_val) > 0.5
                        structure = "Moderate memory"
                        color = :orange
                    else
                        structure = "Weak memory"
                        color = :red
                    end
                    
                    text!(ax, lims[1] + 0.05*(lims[2]-lims[1]), 
                         lims[2] - 0.05*(lims[2]-lims[1]);
                         text="ρ = $(round(corr_val, digits=2))\n$structure",
                         fontsize=11, align=(:left, :top),
                         color=color, font=:bold)
                end
            end
            
            if scenario_idx == 1
                axislegend(ax, position=:rb, labelsize=9)
            end
        end
    end
    
    return fig
end

"""
Plot 6: Two-parameter bifurcation diagram (COMPLETE FIX)
Searches for both trivial and non-trivial equilibria
"""
function plot_2d_bifurcation(κ_range, param2_range, param2_name::Symbol;
                            p_base=ModelInterface.default_params(),
                            resolution=(40, 40))
    @info "Computing 2D bifurcation diagram..."
    @info "Searching for trivial AND non-trivial equilibria..."
    
    param2_label = String(param2_name)
    
    n_κ, n_param2 = resolution
    κ_vals = range(κ_range[1], κ_range[2], length=n_κ)
    param2_vals = range(param2_range[1], param2_range[2], length=n_param2)
    
    # Storage for results
    max_real_eig = fill(NaN, n_κ, n_param2)
    equilibrium_type = fill(0, n_κ, n_param2)  # 0=none, 1=trivial, 2=non-trivial
    has_hopf = zeros(Bool, n_κ, n_param2)
    equilibrium_norm = fill(NaN, n_κ, n_param2)
    
    @info "Scanning $(n_κ)×$(n_param2) parameter grid..."
    
    progress_counter = Threads.Atomic{Int}(0)
    total = n_κ * n_param2
    
    Threads.@threads for i in 1:n_κ
        for j in 1:n_param2
            κ = κ_vals[i]
            param2_val = param2_vals[j]
            
            try
                # Create parameters
                # Note: For now we only vary κ since ModelInterface doesn't expose β variation
                p = ModelInterface.kappa_set(p_base, κ)
                
                # Multiple initial guesses to find different equilibria
                initial_guesses = [
                    zeros(2),           # Trivial equilibrium
                    [1.0, 1.0],        # Positive consensus
                    [-1.0, -1.0],      # Negative consensus  
                    [1.0, -1.0],       # Polarized state
                    [0.5, -0.5],       # Small polarization
                    [2.0, -2.0],       # Large polarization
                ]
                
                all_equilibria = Vector{Float64}[]
                
                for u0 in initial_guesses
                    try
                        u_candidate = SimpleContinuation.newton_solve(
                            ModelInterface.f,
                            ModelInterface.jacobian,
                            u0,
                            p;
                            tol=1e-8,
                            maxiter=100
                        )
                        
                        # Verify it's an equilibrium
                        F_norm = norm(ModelInterface.f(u_candidate, p))
                        
                        if F_norm < 1e-6
                            # Check if this is a new equilibrium
                            is_new = true
                            for u_exist in all_equilibria
                                if norm(u_candidate - u_exist) < 1e-4
                                    is_new = false
                                    break
                                end
                            end
                            
                            if is_new
                                push!(all_equilibria, u_candidate)
                            end
                        end
                    catch
                        continue
                    end
                end
                
                # Analyze equilibria found
                if !isempty(all_equilibria)
                    # Find the most stable equilibrium
                    min_real_eig = Inf
                    best_eq = all_equilibria[1]
                    
                    for u_eq in all_equilibria
                        J = ModelInterface.jacobian(u_eq, p)
                        λs = eigvals(J)
                        max_re = maximum(real.(λs))
                        
                        if max_re < min_real_eig
                            min_real_eig = max_re
                            best_eq = u_eq
                        end
                        
                        # Check for Hopf
                        if SimpleContinuation.detect_hopf(λs; tol=2e-2)
                            has_hopf[i, j] = true
                        end
                    end
                    
                    # Store results for most stable equilibrium
                    max_real_eig[i, j] = min_real_eig
                    equilibrium_norm[i, j] = norm(best_eq)
                    
                    # Classify equilibrium type
                    if norm(best_eq) < 0.01
                        equilibrium_type[i, j] = 1  # Trivial (consensus at origin)
                    else
                        equilibrium_type[i, j] = 2  # Non-trivial (polarized)
                    end
                end
                
            catch e
                # Failed
            end
            
            Threads.atomic_add!(progress_counter, 1)
            if progress_counter[] % 200 == 0
                n_found = sum(equilibrium_type .> 0)
                @info "  Progress: $(progress_counter[])/$(total) ($(round(100*progress_counter[]/total, digits=1))%) - Found equilibria: $(n_found)"
            end
        end
    end
    
    n_found = sum(equilibrium_type .> 0)
    n_trivial = sum(equilibrium_type .== 1)
    n_nontrivial = sum(equilibrium_type .== 2)
    
    @info "Computation complete:"
    @info "  Total equilibria: $(n_found)/$(total)"
    @info "  Trivial (consensus): $(n_trivial)"
    @info "  Non-trivial (polarized): $(n_nontrivial)"
    
    if n_found == 0
        @error "No equilibria found! This shouldn't happen - check model."
        return create_diagnostic_plot(κ_vals, param2_vals, param2_label, p_base)
    end
    
    # Create comprehensive visualization
    fig = Figure(size=(2000, 1200))
    
    # Panel 1: Stability landscape
    ax1 = Axis(fig[1, 1];
              xlabel="Coupling κ",
              ylabel=param2_label,
              title="Stability Landscape")
    
    stability_map = similar(max_real_eig)
    for i in 1:n_κ, j in 1:n_param2
        if equilibrium_type[i, j] > 0
            if max_real_eig[i, j] < -1e-3
                stability_map[i, j] = 1.0  # Stable
            elseif max_real_eig[i, j] > 1e-3
                stability_map[i, j] = -1.0  # Unstable
            else
                stability_map[i, j] = 0.0  # Marginal
            end
        else
            stability_map[i, j] = NaN
        end
    end
    
    hm1 = heatmap!(ax1, κ_vals, param2_vals, stability_map';
                  colormap=:RdYlGn, colorrange=(-1.5, 1.5),
                  nan_color=:lightgray)
    
    # Mark critical κ*
    κ_star = getproperty(p_base, :kstar)
    vlines!(ax1, [κ_star]; color=:black, linestyle=:dash, linewidth=3, label="κ*")
    
    # Mark Hopf points
    hopf_coords = findall(has_hopf)
    if !isempty(hopf_coords)
        hopf_κ = [κ_vals[coord[1]] for coord in hopf_coords]
        hopf_param2 = [param2_vals[coord[2]] for coord in hopf_coords]
        scatter!(ax1, hopf_κ, hopf_param2;
                color=:purple, markersize=8, marker=:star5,
                strokecolor=:black, strokewidth=1)
    end
    
    axislegend(ax1, position=:lt)
    Colorbar(fig[1, 2], hm1; label="Stability",
            ticks=([-1, 0, 1], ["Unstable", "Marginal", "Stable"]))
    
    # Panel 2: Equilibrium type
    ax2 = Axis(fig[1, 3];
              xlabel="Coupling κ",
              ylabel=param2_label,
              title="Equilibrium Type")
    
    type_map = Float64.(equilibrium_type)
    hm2 = heatmap!(ax2, κ_vals, param2_vals, type_map';
                  colormap=[:lightgray, :blue, :red],
                  colorrange=(0, 2))
    
    vlines!(ax2, [κ_star]; color=:black, linestyle=:dash, linewidth=3)
    
    Colorbar(fig[1, 4], hm2; label="Type",
            ticks=([0, 1, 2], ["None", "Consensus", "Polarized"]))
    
    # Panel 3: Continuous stability
    ax3 = Axis(fig[2, 1];
              xlabel="Coupling κ",
              ylabel=param2_label,
              title="Stability (max Re(λ))")
    
    valid_eigs = filter(!isnan, max_real_eig)
    if !isempty(valid_eigs)
        eig_range = quantile(valid_eigs, [0.05, 0.95])
        color_range = max(abs(eig_range[1]), abs(eig_range[2]), 0.1)
    else
        color_range = 0.5
    end
    
    hm3 = heatmap!(ax3, κ_vals, param2_vals, max_real_eig';
                  colormap=:balance, 
                  colorrange=(-color_range, color_range),
                  nan_color=:lightgray)
    
    # Bifurcation contour
    if sum(.!isnan.(max_real_eig)) > 20
        try
            contour!(ax3, κ_vals, param2_vals, max_real_eig';
                    levels=[0.0], color=:black, linewidth=4,
                    label="Bifurcation")
        catch
        end
    end
    
    vlines!(ax3, [κ_star]; color=:white, linestyle=:dash, linewidth=3)
    
    Colorbar(fig[2, 2], hm3; label="max Re(λ)")
    
    # Panel 4: Equilibrium magnitude
    ax4 = Axis(fig[2, 3];
              xlabel="Coupling κ",
              ylabel=param2_label,
              title="Polarization Strength ||u||")
    
    hm4 = heatmap!(ax4, κ_vals, param2_vals, equilibrium_norm';
                  colormap=:plasma,
                  nan_color=:lightgray)
    
    vlines!(ax4, [κ_star]; color=:white, linestyle=:dash, linewidth=3)
    
    Colorbar(fig[2, 4], hm4; label="||u_eq||")
    
    # Panel 5 & 6: Cross-sections
    # Vertical slice at κ = κ*
    ax5 = Axis(fig[3, 1:2];
              xlabel="Coupling κ",
              ylabel="max Re(λ)",
              title="Bifurcation Structure (horizontal slice)")
    
    mid_j = div(n_param2, 2)
    slice_horizontal = max_real_eig[:, mid_j]
    valid_h = .!isnan.(slice_horizontal)
    
    lines!(ax5, κ_vals[valid_h], slice_horizontal[valid_h];
          linewidth=3, color=:blue)
    hlines!(ax5, [0]; color=:red, linestyle=:dash, linewidth=2)
    vlines!(ax5, [κ_star]; color=:green, linestyle=:dash, linewidth=2)
    
    # Horizontal slice at mid-β
    ax6 = Axis(fig[3, 3:4];
              xlabel=param2_label,
              ylabel="||u_eq||",
              title="Equilibrium magnitude (vertical slice at κ=κ*)")
    
    κ_star_idx = argmin(abs.(κ_vals .- κ_star))
    slice_vertical = equilibrium_norm[κ_star_idx, :]
    valid_v = .!isnan.(slice_vertical)
    
    lines!(ax6, param2_vals[valid_v], slice_vertical[valid_v];
          linewidth=3, color=:purple)
    
    return fig
end

"""
Diagnostic plot when things fail
"""
function create_diagnostic_plot(κ_vals, param2_vals, param2_label, p_base)
    fig = Figure(size=(1400, 500))
    
    ax = Axis(fig[1, 1];
             xlabel="Coupling κ",
             ylabel=param2_label,
             title="DIAGNOSTIC: Check Model Implementation")
    
    κ_star = getproperty(p_base, :kstar)
    
    dummy = zeros(length(κ_vals), length(param2_vals))
    heatmap!(ax, κ_vals, param2_vals, dummy'; colormap=:grays, alpha=0.3)
    
    vlines!(ax, [κ_star]; color=:red, linewidth=3, label="κ* = $(round(κ_star, digits=3))")
    
    text!(ax, mean(κ_vals), mean(param2_vals);
         text="NO EQUILIBRIA FOUND\n\nModel may need debugging",
         fontsize=20, align=(:center, :center),
         color=:red, font=:bold)
    
    axislegend(ax)
    
    return fig
end
"""
Create diagnostic plot when convergence fails
"""
function create_diagnostic_plot(κ_vals, param2_vals, param2_label, p_base)
    fig = Figure(size=(1400, 500))
    
    ax = Axis(fig[1, 1];
             xlabel="Coupling κ",
             ylabel=param2_label,
             title="DIAGNOSTIC: No Convergence - Check Parameter Ranges")
    
    # Show the scanning region
    κ_star = getproperty(p_base, :kstar)
    
    # Create a background showing what we tried
    dummy = zeros(length(κ_vals), length(param2_vals))
    heatmap!(ax, κ_vals, param2_vals, dummy'; colormap=:grays, alpha=0.3)
    
    # Mark critical value
    vlines!(ax, [κ_star]; color=:red, linewidth=3, label="κ* = $(round(κ_star, digits=3))")
    
    # Add text annotation
    text!(ax, mean(κ_vals), mean(param2_vals);
         text="NO EQUILIBRIA FOUND\n\nCheck:\n• Parameter ranges\n• Model implementation\n• Initial guesses",
         fontsize=20, align=(:center, :center),
         color=:red, font=:bold)
    
    axislegend(ax)
    
    return fig
end


"""
Simpler alternative: Show κ bifurcation at current β
"""
function plot_simple_bifurcation_scan(; p_base=ModelInterface.default_params(),
                                      κ_range=(0.5, 1.5), n_points=150)
    @info "Computing simple bifurcation scan..."
    
    κ_vals = range(κ_range[1], κ_range[2], length=n_points)
    κ_star = getproperty(p_base, :kstar)
    
    # Storage
    trivial_stable = Float64[]
    trivial_unstable = Float64[]
    polarized_stable = Float64[]
    polarized_unstable = Float64[]
    
    trivial_κ_stable = Float64[]
    trivial_κ_unstable = Float64[]
    polarized_κ_stable = Float64[]
    polarized_κ_unstable = Float64[]
    
    for κ in κ_vals
        p = ModelInterface.kappa_set(p_base, κ)
        
        # Check trivial equilibrium
        u_trivial = zeros(2)
        F_norm = norm(ModelInterface.f(u_trivial, p))
        
        if F_norm < 1e-8
            J = ModelInterface.jacobian(u_trivial, p)
            λs = eigvals(J)
            max_re = maximum(real.(λs))
            
            if max_re < 0
                push!(trivial_stable, mean(u_trivial))
                push!(trivial_κ_stable, κ)
            else
                push!(trivial_unstable, mean(u_trivial))
                push!(trivial_κ_unstable, κ)
            end
        end
        
        # Search for polarized equilibria
        for u0 in [[0.5, -0.5], [1.0, -1.0], [1.5, -1.5]]
            try
                u_eq = SimpleContinuation.newton_solve(
                    ModelInterface.f,
                    ModelInterface.jacobian,
                    u0,
                    p
                )
                
                if norm(ModelInterface.f(u_eq, p)) < 1e-8 && norm(u_eq) > 0.01
                    J = ModelInterface.jacobian(u_eq, p)
                    λs = eigvals(J)
                    max_re = maximum(real.(λs))
                    
                    if max_re < 0
                        push!(polarized_stable, abs(mean(u_eq)))
                        push!(polarized_κ_stable, κ)
                    else
                        push!(polarized_unstable, abs(mean(u_eq)))
                        push!(polarized_κ_unstable, κ)
                    end
                    break
                end
            catch
                continue
            end
        end
    end
    
    fig = Figure(size=(1200, 700))
    
    ax = Axis(fig[1, 1];
             xlabel="Coupling κ",
             ylabel="Mean belief |⟨u⟩|",
             title="Bifurcation Diagram (β = $(getproperty(p_base, :beta)))")
    
    # Plot branches
    if !isempty(trivial_κ_stable)
        scatter!(ax, trivial_κ_stable, trivial_stable;
                color=:blue, markersize=3, label="Consensus (stable)")
    end
    if !isempty(trivial_κ_unstable)
        scatter!(ax, trivial_κ_unstable, trivial_unstable;
                color=:blue, markersize=3, marker=:xcross, alpha=0.5,
                label="Consensus (unstable)")
    end
    if !isempty(polarized_κ_stable)
        scatter!(ax, polarized_κ_stable, polarized_stable;
                color=:red, markersize=3, label="Polarized (stable)")
    end
    if !isempty(polarized_κ_unstable)
        scatter!(ax, polarized_κ_unstable, polarized_unstable;
                color=:red, markersize=3, marker=:xcross, alpha=0.5,
                label="Polarized (unstable)")
    end
    
    # Mark κ*
    vlines!(ax, [κ_star]; color=:green, linestyle=:dash, linewidth=3,
           label="κ* = $(round(κ_star, digits=3))")
    
    axislegend(ax, position=:lt)
    
    return fig
end
"""
Plot 7: Lyapunov exponent spectrum
"""
function plot_lyapunov_spectrum(κ_range; p_base=ModelInterface.default_params(),
                               u0=[0.1, 0.1], tmax=2000.0)
    @info "Computing Lyapunov exponents (this takes time)..."
    
    lyap_values = Float64[]
    
    for (i, κ) in enumerate(κ_range)
        p = ModelInterface.kappa_set(p_base, κ)
        
        λ = compute_lyapunov(
            ModelInterface.f,
            ModelInterface.jacobian,
            u0,
            p;
            tmax=tmax,
            dt=0.1
        )
        
        push!(lyap_values, λ)
        
        if i % 5 == 0
            @info "  Progress: $(i)/$(length(κ_range))"
        end
    end
    
    fig = Figure(size=(1200, 700))
    
    ax = Axis(fig[1, 1];
             xlabel="Coupling strength κ",
             ylabel="Largest Lyapunov exponent λ",
             title="Chaos Indicator (λ > 0 indicates chaos)")
    
    lines!(ax, κ_range, lyap_values; linewidth=3, color=:blue, label="λ_max")
    hlines!(ax, [0]; color=:red, linestyle=:dash, linewidth=2,
           label="λ=0 (stability boundary)")
    
    # Shade regions
    band!(ax, κ_range, fill(-1, length(κ_range)), 
          min.(lyap_values, 0);
          color=(:green, 0.2), label="Stable (λ<0)")
    band!(ax, κ_range, zeros(length(κ_range)), 
          max.(lyap_values, 0);
          color=(:red, 0.2), label="Unstable/Chaotic (λ>0)")
    
    axislegend(ax, position=:lt)
    
    return fig
end

#═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
#═══════════════════════════════════════════════════════════════════════════

function main()
    @info "═══════════════════════════════════════════════════════════════"
    @info "  Comprehensive Complementary Analysis"
    @info "  OU-with-Resets Belief Dynamics System"
    @info "═══════════════════════════════════════════════════════════════"
    
    PlottingCairo.set_theme_elegant!()
    
    # Create output directory
    outdir = joinpath(@__DIR__, "..", "figs", "comprehensive")
    mkpath(outdir)
    
    p_base = ModelInterface.default_params()
    κ_star = getproperty(p_base, :kstar)
    
    @info "System parameters:" λ=p_base.λ β=p_base.beta κ_star=κ_star
    
    # Define analysis parameters
    κ_range_fine = collect(0.7:0.005:1.4)
    κ_range_coarse = collect(0.7:0.03:1.4)
    κ_selected = [0.85*κ_star, 0.95*κ_star, 1.05*κ_star, 1.15*κ_star]
    
    # ═══ PLOT 1: Extended Bifurcation Diagram ═══
    @info "Generating Plot 1: Extended bifurcation diagram..."
    fig1 = plot_extended_bifurcation(κ_range_fine; p_base=p_base)
    save(joinpath(outdir, "01_extended_bifurcation.png"), fig1)
    @info "  ✓ Saved: 01_extended_bifurcation.png"
    
    # ═══ PLOT 2: Phase Space Analysis ═══
    @info "Generating Plot 2: Phase space with nullclines..."
    fig2 = plot_phase_space_analysis(κ_selected; p_base=p_base)
    save(joinpath(outdir, "02_phase_space_nullclines.png"), fig2)
    @info "  ✓ Saved: 02_phase_space_nullclines.png"
    
    # ═══ PLOT 3: Basins of Attraction ═══
    @info "Generating Plot 3: Basins of attraction..."
    fig3 = plot_basin_of_attraction([0.9*κ_star, 1.0*κ_star, 1.1*κ_star, 1.2*κ_star]; 
                                   p_base=p_base, resolution=50)
    save(joinpath(outdir, "03_basins_of_attraction.png"), fig3)
    @info "  ✓ Saved: 03_basins_of_attraction.png"
    
    # ═══ PLOT 4: Time Series Analysis ═══
    @info "Generating Plot 4: Time series analysis..."
    for κ_factor in [0.95, 1.05, 1.15]
        κ = κ_factor * κ_star
        fig4 = plot_time_series_analysis(κ; p_base=p_base, 
                                        u0=[0.5, -0.5], tmax=500.0)
        save(joinpath(outdir, "04_timeseries_kappa_$(round(κ_factor, digits=2)).png"), fig4)
        @info "  ✓ Saved: 04_timeseries_kappa_$(round(κ_factor, digits=2)).png"
    end
    
    # ═══ PLOT 5: Return Maps (Enhanced) ═══
    @info "Generating Plot 5: Enhanced return maps..."
    fig5 = plot_enhanced_return_maps([0.95*κ_star, 1.05*κ_star, 1.15*κ_star]; 
                                    p_base=p_base)
    save(joinpath(outdir, "05_return_maps_enhanced.png"), fig5)
    @info "  ✓ Saved: 05_return_maps_enhanced.png"
    
   # ═══ PLOT 6: Bifurcation Structure ═══
    @info "Generating Plot 6: Comprehensive bifurcation analysis..."

    # Option A: Simple 1D bifurcation (always works)
    fig6a = plot_simple_bifurcation_scan(;
                                        p_base=p_base,
                                        κ_range=(0.6*κ_star, 1.5*κ_star),
                                        n_points=200)
    save(joinpath(outdir, "06a_bifurcation_scan.png"), fig6a)
    @info "  ✓ Saved: 06a_bifurcation_scan.png"

    # Option B: Try 2D (may work now with non-trivial equilibrium search)
    try
        fig6b = plot_2d_bifurcation((0.6*κ_star, 1.5*κ_star), (0.5, 2.0), :beta;
                                p_base=p_base, resolution=(50, 50))
        save(joinpath(outdir, "06b_2d_parameter_space.png"), fig6b)
        @info "  ✓ Saved: 06b_2d_parameter_space.png"
    catch e
        @warn "2D bifurcation failed" exception=e
    end
    
    # ═══ PLOT 7: Lyapunov Exponents ═══
    @info "Generating Plot 7: Lyapunov spectrum..."
    fig7 = plot_lyapunov_spectrum(κ_range_coarse; p_base=p_base, tmax=2000.0)
    save(joinpath(outdir, "07_lyapunov_spectrum.png"), fig7)
    @info "  ✓ Saved: 07_lyapunov_spectrum.png"
    
    @info "═══════════════════════════════════════════════════════════════"
    @info "  ✅ All plots generated successfully!"
    @info "  Output directory: $outdir"
    @info "═══════════════════════════════════════════════════════════════"
    @info ""
    @info "Generated files:"
    @info "  1. Extended bifurcation - Shows pitchfork structure & stability"
    @info "  2. Phase space nullclines - Equilibria & flow structure"
    @info "  3. Basins of attraction - Which beliefs lead to polarization"
    @info "  4. Time series - Dynamics & statistical properties"
    @info "  5. Return maps - Reveals periodic/chaotic structure"
    @info "  6. 2D bifurcation - Parameter space (κ vs β)"
    @info "  7. Lyapunov spectrum - Quantifies chaos"
    @info "═══════════════════════════════════════════════════════════════"
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end