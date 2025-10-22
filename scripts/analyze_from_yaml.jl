"""
analyze_from_yaml.jl

Complete bifurcation analysis from YAML configuration.
Self-contained with all dependencies from the project.

Usage:
    julia --project=. scripts/analyze_from_yaml.jl configs/example_sweep.yaml
    julia --project=. scripts/analyze_from_yaml.jl --all  # Process all configs
"""

#═══════════════════════════════════════════════════════════════════════════
# IMPORTS - Only system dependencies
#═══════════════════════════════════════════════════════════════════════════

using Pkg
Pkg.activate(".")

using LinearAlgebra
using Statistics
using DifferentialEquations
using FFTW
using StatsBase
using Logging
using Printf
using Dates

# Import simulation toolkit for estimating V*, g, and κ*
using BeliefSim
using BeliefSim.Types: Params, StepHazard, LogisticHazard
using BeliefSim.Stats: estimate_Vstar, estimate_g, critical_kappa

# Load YAML
try
    using YAML
catch
    @info "Installing YAML.jl..."
    Pkg.add("YAML")
    using YAML
end

# Load plotting backend
try
    using CairoMakie
    using ColorSchemes
catch e
    @error "CairoMakie not available. Install with: Pkg.add(\"CairoMakie\")"
    exit(1)
end

# Load bifurcation modules
include(joinpath(@__DIR__, "..", "src", "bifurcation", "model_interface.jl"))
include(joinpath(@__DIR__, "..", "src", "bifurcation", "simple_continuation.jl"))
include(joinpath(@__DIR__, "..", "src", "bifurcation", "plotting_cairo.jl"))

using .ModelInterface
using .SimpleContinuation
using .PlottingCairo

#═══════════════════════════════════════════════════════════════════════════
# YAML CONFIGURATION LOADER
#═══════════════════════════════════════════════════════════════════════════

"""
Load and validate YAML configuration
"""
function load_yaml_config(filepath::String)
    if !isfile(filepath)
        throw(ArgumentError("Config file not found: $filepath"))
    end
    
    cfg = YAML.load_file(filepath)
    
    # Validate required fields
    required_fields = ["params", "N", "T", "dt", "output_dir"]
    for field in required_fields
        if !haskey(cfg, field)
            throw(ArgumentError("Missing required field: $field"))
        end
    end
    
    # Extract and validate parameters
    params = cfg["params"]
    required_params = ["lambda", "sigma", "theta", "c0", "hazard"]
    for param in required_params
        if !haskey(params, param)
            throw(ArgumentError("Missing required parameter: $param"))
        end
    end
    
    return cfg
end

"""
Convert YAML parameters to ModelInterface format
"""
function yaml_to_model_params(cfg::Dict)
    params = cfg["params"]

    # Extract core parameters
    λ = Float64(params["lambda"])
    σ = Float64(params["sigma"])
    Θ = Float64(params["theta"])
    c0 = Float64(params["c0"])

    hazard_cfg = params["hazard"]
    hazard = begin
        kind = lowercase(String(hazard_cfg["kind"]))
        if kind == "step"
            StepHazard(Float64(hazard_cfg["nu0"]))
        elseif kind == "logistic"
            LogisticHazard(Float64(hazard_cfg["numax"]), Float64(hazard_cfg["beta"]))
        else
            error("Unsupported hazard kind: $(hazard_cfg["kind"]) — use 'step' or 'logistic'.")
        end
    end

    seed = get(cfg, "seed", 0)
    N = cfg["N"]
    T = cfg["T"]
    dt = cfg["dt"]
    burn_in = cfg["burn_in"]

    p_sim = Params(; λ=λ, σ=σ, Θ=Θ, c0=c0, hazard=hazard)

    @info "  Estimating stationary dispersion V* via Monte Carlo" N=N T=T dt=dt burn_in=burn_in seed=seed
    Vstar = estimate_Vstar(p_sim; N=N, T=T, dt=dt, burn_in=burn_in, seed=seed)

    @info "  Estimating odd-mode decay g at κ=0"
    g_est = estimate_g(p_sim;
                       N=max(N, 20_000),
                       T=min(max(T, 40.0), 120.0),
                       dt=min(dt, 0.01),
                       seed=seed)

    κ_star = critical_kappa(p_sim; Vstar=Vstar, g=g_est, N=N, T=T, dt=dt, burn_in=burn_in, seed=seed)

    # Normalize cubic term so that polarization amplitude ~ sqrt(κ - κ*) scaled by V*
    β = 1.0 / max(Vstar, eps())
    p_model = (
        λ=λ,
        σ=σ,
        Θ=Θ,
        c0=c0,
        Vstar=Vstar,
        g=g_est,
        beta=β,
        kappa=0.0,
        kstar=κ_star,
        seed=seed,
    )

    meta = (
        params=p_sim,
        hazard=hazard,
        Vstar=Vstar,
        g=g_est,
        κ_star=κ_star,
    )

    return p_model, meta
end

#═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
#═══════════════════════════════════════════════════════════════════════════

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
Order parameter for polarization: half the difference between the two agents.
Falls back to mean if dimensionality ≠ 2.
"""
polarization_coordinate(u::AbstractVector) = length(u) == 2 ? 0.5 * (u[1] - u[2]) : mean(u)

polarization_amplitude(u::AbstractVector) = abs(polarization_coordinate(u))

"""
Classify attractor by running trajectory
"""
function classify_attractor(u0::Vector{Float64}, p; tmax=500.0)
    function ode!(du, u, p, t)
        F = ModelInterface.f(u, p)
        du .= F
    end
    
    prob = ODEProblem(ode!, u0, (0.0, tmax), p)
    
    try
        sol = solve(prob, Tsit5(); reltol=1e-6, abstol=1e-6)
        u_final = sol.u[end]
        pol_final = polarization_coordinate(u_final)

        if abs(pol_final) < 0.05
            return 0  # Consensus
        elseif pol_final > 0.05
            return 1  # Positive polarization
        elseif pol_final < -0.05
            return -1  # Negative polarization
        else
            return 0
        end
    catch
        return NaN
    end
end

"""
Compute Lyapunov exponent
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
        J = jac(u, p)
        F = f(u, p)
        
        u .+= F .* dt
        w .+= (J * w) .* dt
        
        w_norm = norm(w)
        if w_norm > 1e-12
            λ_sum += log(w_norm)
            w ./= w_norm
        end
        
        t += dt
        n_steps += 1
    end
    
    return n_steps > 0 ? λ_sum / (n_steps * dt) : 0.0
end

#═══════════════════════════════════════════════════════════════════════════
# PLOTTING FUNCTIONS
#═══════════════════════════════════════════════════════════════════════════

"""
Plot 1: Extended bifurcation diagram
"""
function plot_bifurcation(κ_range, p_base)
    @info "  Computing bifurcation diagram..."
    
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
    
    # Extract data
    u1_eq = [u[1] for u in branch.equilibria]
    u2_eq = [u[2] for u in branch.equilibria]
    mean_eq = (u1_eq .+ u2_eq) ./ 2
    polarization = 0.5 .* abs.(u1_eq .- u2_eq)
    
    stable_mask = [all(real.(λs) .< 0) for λs in branch.eigenvalues]
    max_real_eig = [maximum(real.(λs)) for λs in branch.eigenvalues]
    imag_parts = [maximum(abs.(imag.(λs))) for λs in branch.eigenvalues]
    
    # Panel 1: Mean equilibrium
    ax1 = Axis(fig[1, 1:2];
              xlabel="Coupling strength κ",
              ylabel="Mean belief ⟨u⟩",
              title="Bifurcation Diagram: Consensus → Polarization")
    
    lines!(ax1, branch.κ_values[stable_mask], mean_eq[stable_mask];
          linewidth=3, color=:blue, label="Stable")
    
    if any(.!stable_mask)
        lines!(ax1, branch.κ_values[.!stable_mask], mean_eq[.!stable_mask];
              linewidth=3, color=:red, linestyle=:dash, label="Unstable")
    end
    
    # Mark critical point
    stability_changes = findall(diff(stable_mask) .!= 0)
    if !isempty(stability_changes)
        idx = stability_changes[1]
        vlines!(ax1, [branch.κ_values[idx]]; 
               color=:green, linestyle=:dash, linewidth=2, label="κ* (bifurcation)")
    end
    
    axislegend(ax1, position=:lt)
    
    # Panel 2: Stability indicator
    ax2 = Axis(fig[2, 1];
              xlabel="κ",
              ylabel="max Re(λ)",
              title="Stability (negative = stable)")
    
    lines!(ax2, branch.κ_values, max_real_eig; linewidth=2, color=:black)
    hlines!(ax2, [0]; color=:red, linestyle=:dash, linewidth=2)
    
    band!(ax2, branch.κ_values, fill(-1, length(max_real_eig)), 
          min.(max_real_eig, 0); color=(:green, 0.2))
    band!(ax2, branch.κ_values, zeros(length(max_real_eig)), 
          max.(max_real_eig, 0); color=(:red, 0.2))
    
    # Panel 3: Oscillation frequency
    ax3 = Axis(fig[2, 2];
              xlabel="κ",
              ylabel="|Im(λ)| (Hz)",
              title="Oscillation Frequency")
    
    lines!(ax3, branch.κ_values, imag_parts; linewidth=2, color=:purple)
    
    # Panel 4: Polarization
    ax4 = Axis(fig[3, 1:2];
              xlabel="κ",
              ylabel="Polarization |u₁ - u₂|/2",
              title="Opinion Divergence")
    
    lines!(ax4, branch.κ_values, polarization; linewidth=2, color=:orange)
    
    if !isempty(stability_changes)
        idx = stability_changes[1]
        vlines!(ax4, [branch.κ_values[idx]]; 
               color=:green, linestyle=:dash, linewidth=2)
    end
    
    return fig
end

"""
Plot 2: Phase portraits
"""
function plot_phase_portraits(κ_values, p_base)
    @info "  Computing phase portraits..."
    
    n = length(κ_values)
    fig = Figure(size=(1200, 600 * ceil(Int, n/2)))
    
    for (idx, κ) in enumerate(κ_values)
        p = ModelInterface.kappa_set(p_base, κ)
        
        row = div(idx - 1, 2) + 1
        col = mod(idx - 1, 2) + 1
        
        ax = Axis(fig[row, col];
                 xlabel="u₁ (agent 1)",
                 ylabel="u₂ (agent 2)",
                 title="Phase Space: κ = $(round(κ, digits=3))",
                 aspect=DataAspect())
        
        # Vector field
        PlottingCairo.phase_portrait!(ax, ModelInterface.f, p;
                                     lims=(-3, 3, -3, 3),
                                     density=35, alpha=0.25)
        
        # Consensus line
        lines!(ax, [-3, 3], [-3, 3]; 
              color=:gray, linestyle=:dash, linewidth=2)
        
        # Equilibrium
        try
            u_eq = SimpleContinuation.newton_solve(
                ModelInterface.f, ModelInterface.jacobian, zeros(2), p
            )
            
            J = ModelInterface.jacobian(u_eq, p)
            λs = eigvals(J)
            is_stable = all(real.(λs) .< 0)
            
            scatter!(ax, [u_eq[1]], [u_eq[2]];
                    color=is_stable ? :green : :orange,
                    markersize=15,
                    marker=is_stable ? :circle : :xcross)
        catch
        end
    end
    
    return fig
end

"""
Plot 3: Basins of attraction
"""
function plot_basins(κ_values, p_base; resolution=50)
    @info "  Computing basins of attraction..."
    
    n = length(κ_values)
    fig = Figure(size=(1400, 700 * ceil(Int, n/2)))
    
    u1_range = range(-3, 3, length=resolution)
    u2_range = range(-3, 3, length=resolution)
    
    for (idx, κ) in enumerate(κ_values)
        p = ModelInterface.kappa_set(p_base, κ)
        
        row = div(idx - 1, 2) + 1
        col = mod(idx - 1, 2) + 1
        
        ax = Axis(fig[row, col];
                 xlabel="Initial u₁",
                 ylabel="Initial u₂",
                 title="Basin: κ=$(round(κ, digits=3))",
                 aspect=DataAspect())
        
        basin_map = zeros(resolution, resolution)
        
        @info "    κ=$κ..."
        Threads.@threads for i in 1:resolution
            for j in 1:resolution
                u0 = [u1_range[i], u2_range[j]]
                basin_map[i, j] = classify_attractor(u0, p; tmax=300.0)
            end
        end
        
        hm = heatmap!(ax, u1_range, u2_range, basin_map';
                     colormap=:RdBu, colorrange=(-1.2, 1.2))
        
        lines!(ax, [-3, 3], [-3, 3]; color=:black, linestyle=:dash, linewidth=1)
        
        if col == 2 || idx == n
            Colorbar(fig[row, col+1], hm; label="Attractor",
                    ticks=([-1, 0, 1], ["Neg.", "Consensus", "Pos."]))
        end
    end
    
    return fig
end

"""
Plot 4: Time series analysis
"""
function plot_timeseries(κ, p_base; u0=[0.5, -0.5], tmax=500.0, dt=0.1)
    @info "  Analyzing time series for κ=$κ..."
    
    p = ModelInterface.kappa_set(p_base, κ)
    
    function ode!(du, u, p, t)
        F = ModelInterface.f(u, p)
        du .= F
    end
    
    prob = ODEProblem(ode!, u0, (0.0, tmax), p)
    sol = solve(prob, Tsit5(); saveat=dt)
    
    t = sol.t
    traj = reduce(hcat, sol.u)
    u1, u2 = traj[1, :], traj[2, :]
    mean_traj = (u1 .+ u2) ./ 2
    pol_traj = 0.5 .* (u1 .- u2)
    
    fig = Figure(size=(1600, 1200))
    
    # Panel 1: Time series
    ax1 = Axis(fig[1, 1:2];
              xlabel="Time", ylabel="Belief",
              title="Evolution: κ=$(round(κ, digits=3))")
    
    lines!(ax1, t, u1; linewidth=1.5, color=:blue, label="Agent 1")
    lines!(ax1, t, u2; linewidth=1.5, color=:red, label="Agent 2")
    lines!(ax1, t, mean_traj; linewidth=2, color=:green, label="Mean")
    hlines!(ax1, [0]; color=:black, linestyle=:dash, linewidth=1)
    axislegend(ax1, position=:rt)
    
    # Panel 2: Polarization
    ax2 = Axis(fig[2, 1]; xlabel="Time", ylabel="|u₁ - u₂|/2",
              title="Polarization")

    polarization = abs.(pol_traj)
    lines!(ax2, t, polarization; linewidth=1.5, color=:purple)
    
    # Panel 3: Distance from origin
    ax3 = Axis(fig[2, 2]; xlabel="Time", ylabel="||u||",
              title="State Magnitude")
    
    distance = sqrt.(u1.^2 .+ u2.^2)
    lines!(ax3, t, distance; linewidth=1.5, color=:green)
    
    # Panel 4: Autocorrelation
    ax4 = Axis(fig[3, 1]; xlabel="Lag", ylabel="ACF",
              title="Autocorrelation")
    
    max_lag = min(500, length(pol_traj) ÷ 4)
    acf = autocorrelation(pol_traj, max_lag)
    lines!(ax4, (0:max_lag) .* dt, acf; linewidth=2, color=:blue)
    hlines!(ax4, [0]; color=:black, linestyle=:dash, linewidth=1)
    
    # Panel 5: Power spectrum
    ax5 = Axis(fig[3, 2]; xlabel="Frequency", ylabel="Power",
              title="Spectrum", xscale=log10, yscale=log10)
    
    freqs, psd = compute_psd(pol_traj, dt)
    valid = (freqs .> 0) .& (psd .> 1e-12)
    lines!(ax5, freqs[valid], psd[valid]; linewidth=2, color=:blue)
    
    # Panel 6: Phase space
    ax6 = Axis(fig[4, 1:2]; xlabel="u₁", ylabel="u₂",
              title="Phase Trajectory", aspect=DataAspect())
    
    stride = max(1, length(u1) ÷ 2000)
    u1_plot = u1[1:stride:end]
    u2_plot = u2[1:stride:end]
    
    n_seg = length(u1_plot) - 1
    colors = range(colorant"blue", colorant"red", length=n_seg)
    
    for i in 1:n_seg
        lines!(ax6, u1_plot[i:i+1], u2_plot[i:i+1];
              linewidth=1.5, color=colors[i])
    end
    
    lines!(ax6, [-3, 3], [-3, 3]; color=:gray, linestyle=:dash, linewidth=1)
    scatter!(ax6, [u1[1]], [u2[1]]; color=:blue, markersize=15, label="Start")
    scatter!(ax6, [u1[end]], [u2[end]]; color=:red, markersize=15, label="End")
    axislegend(ax6, position=:lt)
    
    return fig
end

"""
Plot 5: Return maps
"""
function plot_return_maps(κ_values, p_base)
    @info "  Computing return maps..."
    
    n_κ = length(κ_values)
    fig = Figure(size=(1600, 1200))
    
    scenarios = [
        (u0=[1.0, -1.0], tmax=800.0, name="Polarized"),
        (u0=[0.1, 0.1], tmax=800.0, name="Consensus"),
        (u0=[2.0, -2.0], tmax=800.0, name="Extreme")
    ]
    
    dt = 0.1
    delay = 20
    
    for (κ_idx, κ) in enumerate(κ_values)
        p = ModelInterface.kappa_set(p_base, κ)
        
        for (s_idx, scen) in enumerate(scenarios)
            function ode!(du, u, p, t)
                F = ModelInterface.f(u, p)
                du .= F
            end
            
            prob = ODEProblem(ode!, scen.u0, (0.0, scen.tmax), p)
            sol = solve(prob, Tsit5(); saveat=dt)
            
            traj = reduce(hcat, sol.u)
            u1 = traj[1, :]
            
            ax = Axis(fig[κ_idx, s_idx];
                     xlabel="u₁(t)", ylabel="u₁(t+τ)",
                     title="κ=$(round(κ, digits=2)) - $(scen.name)",
                     aspect=DataAspect())
            
            if var(u1) < 1e-6
                eq = mean(u1)
                scatter!(ax, [eq], [eq]; markersize=20, color=:red, marker=:xcross)
                padding = 0.5
                lims = (eq - padding, eq + padding)
                xlims!(ax, lims)
                ylims!(ax, lims)
            else
                x_vals = u1[1:end-delay]
                y_vals = u1[delay+1:end]
                
                scatter!(ax, x_vals, y_vals; markersize=2, alpha=0.3, color=:blue)
                
                all_vals = [x_vals; y_vals]
                lims = extrema(all_vals)
                padding = 0.1 * (lims[2] - lims[1])
                plot_lims = (lims[1] - padding, lims[2] + padding)
                
                lines!(ax, [plot_lims[1], plot_lims[2]], [plot_lims[1], plot_lims[2]];
                      color=:red, linestyle=:dash, linewidth=2)
                
                xlims!(ax, plot_lims)
                ylims!(ax, plot_lims)
                
                corr_val = cor(x_vals, y_vals)
                text!(ax, plot_lims[1] + 0.05*(plot_lims[2]-plot_lims[1]),
                     plot_lims[2] - 0.05*(plot_lims[2]-plot_lims[1]);
                     text="ρ=$(round(corr_val, digits=2))",
                     fontsize=12, align=(:left, :top))
            end
        end
    end
    
    return fig
end

"""
Plot 6: Parameter scan
"""
function plot_parameter_scan(κ_range, p_base; n_points=150)
    @info "  Computing parameter scan..."
    
    κ_vals = range(κ_range[1], κ_range[2], length=n_points)
    κ_star = getproperty(p_base, :kstar)
    
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
        
        # Trivial equilibrium
        u_trivial = zeros(2)
        if norm(ModelInterface.f(u_trivial, p)) < 1e-8
            J = ModelInterface.jacobian(u_trivial, p)
            λs = eigvals(J)
            max_re = maximum(real.(λs))
            
            if max_re < 0
                push!(trivial_stable, 0.0)
                push!(trivial_κ_stable, κ)
            else
                push!(trivial_unstable, 0.0)
                push!(trivial_κ_unstable, κ)
            end
        end
        
        # Polarized equilibria
        for u0 in [[0.5, -0.5], [1.0, -1.0]]
            try
                u_eq = SimpleContinuation.newton_solve(
                    ModelInterface.f, ModelInterface.jacobian, u0, p
                )
                
                if norm(ModelInterface.f(u_eq, p)) < 1e-8 && norm(u_eq) > 0.01
                    J = ModelInterface.jacobian(u_eq, p)
                    λs = eigvals(J)
                    max_re = maximum(real.(λs))
                    
                    if max_re < 0
                        push!(polarized_stable, polarization_amplitude(u_eq))
                        push!(polarized_κ_stable, κ)
                    else
                        push!(polarized_unstable, polarization_amplitude(u_eq))
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
             xlabel="Coupling κ", ylabel="Polarization |u₁ - u₂|/2",
             title="Bifurcation Structure")
    
    if !isempty(trivial_κ_stable)
        scatter!(ax, trivial_κ_stable, trivial_stable;
                color=:blue, markersize=3, label="Consensus (stable)")
    end
    if !isempty(trivial_κ_unstable)
        scatter!(ax, trivial_κ_unstable, trivial_unstable;
                color=:blue, markersize=3, marker=:xcross, alpha=0.5)
    end
    if !isempty(polarized_κ_stable)
        scatter!(ax, polarized_κ_stable, polarized_stable;
                color=:red, markersize=3, label="Polarized (stable)")
    end
    if !isempty(polarized_κ_unstable)
        scatter!(ax, polarized_κ_unstable, polarized_unstable;
                color=:red, markersize=3, marker=:xcross, alpha=0.5)
    end
    
    vlines!(ax, [κ_star]; color=:green, linestyle=:dash, linewidth=3,
           label="κ* = $(round(κ_star, digits=3))")
    
    axislegend(ax, position=:lt)
    
    return fig
end

"""
Plot 7: Lyapunov spectrum
"""
function plot_lyapunov(κ_range, p_base; tmax=2000.0)
    @info "  Computing Lyapunov exponents..."
    
    κ_vals = collect(κ_range)
    lyap_values = Float64[]
    
    u0 = [0.1, 0.1]
    
    for (i, κ) in enumerate(κ_vals)
        p = ModelInterface.kappa_set(p_base, κ)
        
        λ = compute_lyapunov(
            ModelInterface.f, ModelInterface.jacobian, u0, p;
            tmax=tmax, dt=0.1
        )
        
        push!(lyap_values, λ)
        
        if i % 5 == 0
            @info "    Progress: $(i)/$(length(κ_vals))"
        end
    end
    
    fig = Figure(size=(1200, 700))
    
    ax = Axis(fig[1, 1];
             xlabel="Coupling κ", ylabel="Lyapunov exponent λ",
             title="Chaos Indicator")
    
    lines!(ax, κ_vals, lyap_values; linewidth=3, color=:blue)
    hlines!(ax, [0]; color=:red, linestyle=:dash, linewidth=2)
    
    band!(ax, κ_vals, fill(-1, length(lyap_values)),
          min.(lyap_values, 0); color=(:green, 0.2))
    band!(ax, κ_vals, zeros(length(lyap_values)),
          max.(lyap_values, 0); color=(:red, 0.2))
    
    return fig
end

#═══════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS PIPELINE
#═══════════════════════════════════════════════════════════════════════════

"""
Run complete analysis from YAML config
"""
function analyze_config(config_path::String)
    @info "═══════════════════════════════════════════════════════════════"
    @info "  Comprehensive Analysis from YAML"
    @info "═══════════════════════════════════════════════════════════════"
    @info "Config: $config_path"

    @info ""

    # Load configuration
    cfg = load_yaml_config(config_path)
    config_name = get(cfg, "name", splitext(basename(config_path))[1])

    @info "Configuration: $config_name"
    @info "  λ = $(cfg["params"]["lambda"])"
    @info "  σ = $(cfg["params"]["sigma"])"
    @info "  Θ = $(cfg["params"]["theta"])"
    @info "  c₀ = $(cfg["params"]["c0"])"
    @info ""
    
    # Convert to model parameters
    p_base, meta = yaml_to_model_params(cfg)
    κ_star = getproperty(p_base, :kstar)

    results = Dict{String, String}()
    results["kappa_star"] = string(κ_star)
    results["Vstar"] = string(p_base.Vstar)
    results["g"] = string(p_base.g)

    @info "Model parameters:"
    @info "  λ = $(p_base.λ)"
    @info "  σ = $(p_base.σ)"
    @info "  V* ≈ $(round(p_base.Vstar, digits=5))"
    @info "  g ≈ $(round(p_base.g, digits=5))"
    @info "  β = $(p_base.beta)"
    @info "  κ* (theory) = $(round(κ_star, digits=5))"
    @info ""
    
    # Determine κ range
    if haskey(cfg, "sweep")
        κ_min = cfg["sweep"]["kappa_from"]
        κ_max = κ_star * cfg["sweep"]["kappa_to_factor_of_kstar"]
        n_points = cfg["sweep"]["points"]
    else
        κ_min = 0.6 * κ_star
        κ_max = 1.5 * κ_star
        n_points = 100
    end
    
    @info "Analysis range: κ ∈ [$(round(κ_min, digits=3)), $(round(κ_max, digits=3))]"
    @info ""
    
    # Set up output
    outdir = joinpath(cfg["output_dir"], "comprehensive_analysis_$config_name")
    mkpath(outdir)
    @info "Output directory: $outdir"
    @info ""
    
    # Set theme
    PlottingCairo.set_theme_elegant!()
    
    # Generate plots
    κ_range_fine = range(κ_min, κ_max, length=n_points)
    κ_range_coarse = range(κ_min, κ_max, length=max(20, div(n_points, 5)))
    κ_selected = [0.85*κ_star, 0.95*κ_star, 1.05*κ_star, 1.15*κ_star]
    
    # Plot 1: Bifurcation
    try
        @info "Generating Plot 1: Bifurcation diagram"
        fig = plot_bifurcation(κ_range_fine, p_base)
        filename = "01_bifurcation.png"
        save(joinpath(outdir, filename), fig)
        results["bifurcation"] = filename
        @info "  ✓ Saved: $filename"
    catch e
        @error "  ✗ Failed" exception=e
    end
    
    # Plot 2: Phase portraits
    try
        @info "Generating Plot 2: Phase portraits"
        fig = plot_phase_portraits(κ_selected, p_base)
        filename = "02_phase_portraits.png"
        save(joinpath(outdir, filename), fig)
        results["phase_portraits"] = filename
        @info "  ✓ Saved: $filename"
    catch e
        @error "  ✗ Failed" exception=e
    end
    
    # Plot 3: Basins
    try
        @info "Generating Plot 3: Basins of attraction"
        κ_basin = [0.9*κ_star, 1.0*κ_star, 1.1*κ_star, 1.2*κ_star]
        fig = plot_basins(κ_basin, p_base; resolution=40)
        filename = "03_basins.png"
        save(joinpath(outdir, filename), fig)
        results["basins"] = filename
        @info "  ✓ Saved: $filename"
    catch e
        @error "  ✗ Failed" exception=e
    end
    
    # Plot 4: Time series
    try
        @info "Generating Plot 4: Time series analysis"
        dt = cfg["dt"]
        T = min(cfg["T"], 500.0)
        
        for κ_factor in [0.95, 1.05, 1.15]
            κ = κ_factor * κ_star
            fig = plot_timeseries(κ, p_base; tmax=T, dt=dt)
            filename = "04_timeseries_k$(round(κ_factor, digits=2)).png"
            save(joinpath(outdir, filename), fig)
        end
        results["timeseries"] = "04_timeseries_*.png"
        @info "  ✓ Saved: 04_timeseries_*.png"
    catch e
        @error "  ✗ Failed" exception=e
    end
    
    # Plot 5: Return maps
    try
        @info "Generating Plot 5: Return maps"
        κ_return = [0.95*κ_star, 1.05*κ_star, 1.15*κ_star]
        fig = plot_return_maps(κ_return, p_base)
        filename = "05_return_maps.png"
        save(joinpath(outdir, filename), fig)
        results["return_maps"] = filename
        @info "  ✓ Saved: $filename"
    catch e
        @error "  ✗ Failed" exception=e
    end
    
    # Plot 6: Parameter scan
    try
        @info "Generating Plot 6: Parameter scan"
        fig = plot_parameter_scan((κ_min, κ_max), p_base; n_points=n_points)
        filename = "06_parameter_scan.png"
        save(joinpath(outdir, filename), fig)
        results["parameter_scan"] = filename
        @info "  ✓ Saved: $filename"
    catch e
        @error "  ✗ Failed" exception=e
    end
    
    # Plot 7: Lyapunov
    try
        @info "Generating Plot 7: Lyapunov spectrum"
        fig = plot_lyapunov(κ_range_coarse, p_base; tmax=min(2000.0, cfg["T"]))
        filename = "07_lyapunov.png"
        save(joinpath(outdir, filename), fig)
        results["lyapunov"] = filename
        @info "  ✓ Saved: $filename"
    catch e
        @error "  ✗ Failed" exception=e
    end
    
    # Generate summary
    summary_path = joinpath(outdir, "summary.txt")
    open(summary_path, "w") do io
        println(io, "═══════════════════════════════════════════════════════════════")
        println(io, "Comprehensive Bifurcation Analysis")
        println(io, "═══════════════════════════════════════════════════════════════")
        println(io, "")
        println(io, "Configuration: $config_name")
        println(io, "Generated: $(now())")
        println(io, "")
        println(io, "Parameters:")
        println(io, "  λ = $(cfg["params"]["lambda"])")
        println(io, "  σ = $(cfg["params"]["sigma"])")
        println(io, "  Θ = $(cfg["params"]["theta"])")
        println(io, "  c₀ = $(cfg["params"]["c0"])")
        println(io, "")
        println(io, "Model (normal form calibration):")
        println(io, "  λ = $(p_base.λ)")
        println(io, "  σ = $(p_base.σ)")
        println(io, "  Θ = $(p_base.Θ)")
        println(io, "  c₀ = $(p_base.c0)")
        println(io, "  Hazard = $(meta.hazard)")
        println(io, "  V* ≈ $(round(p_base.Vstar, digits=6))")
        println(io, "  g ≈ $(round(p_base.g, digits=6))")
        println(io, "  β = $(p_base.beta)")
        println(io, "  κ* = $(round(κ_star, digits=6))")
        println(io, "  Theory: κ* = g σ² / (2 λ V*)")
        theory_val = p_base.g * p_base.σ^2 / (2 * p_base.λ * p_base.Vstar)
        println(io, "         = $(round(theory_val, digits=6)) (numerical check)")
        println(io, "")
        println(io, "Analysis Range:")
        println(io, "  κ ∈ [$(round(κ_min, digits=3)), $(round(κ_max, digits=3))]")
        println(io, "  Points: $n_points")
        println(io, "")
        println(io, "Generated Plots:")
        for (key, file) in sort(collect(results))
            println(io, "  $key: $file")
        end
        println(io, "")
        println(io, "═══════════════════════════════════════════════════════════════")
    end
    
    @info ""
    @info "═══════════════════════════════════════════════════════════════"
    @info "  ✅ Analysis Complete!"
    @info "═══════════════════════════════════════════════════════════════"
    @info "Output: $outdir"
    @info "Summary: $summary_path"
    @info "Plots generated: $(length(results))"
    @info "═══════════════════════════════════════════════════════════════"
    
    return results
end

"""
Process all YAML files in a directory
"""
function analyze_all(config_dir::String="configs")
    @info "Scanning directory: $config_dir"
    
    if !isdir(config_dir)
        @error "Directory not found: $config_dir"
        return Dict()
    end
    
    yaml_files = filter(f -> endswith(lowercase(f), ".yaml") || endswith(lowercase(f), ".yml"),
                       readdir(config_dir, join=true))
    
    if isempty(yaml_files)
        @error "No YAML files found in $config_dir"
        return Dict()
    end
    
    @info "Found $(length(yaml_files)) configuration(s)"
    
    all_results = Dict{String, Any}()
    
    for config_file in yaml_files
        config_name = splitext(basename(config_file))[1]
        @info ""
        @info "Processing: $config_name"
        
        try
            results = analyze_config(config_file)
            all_results[config_name] = results
        catch e
            @error "Failed to process $config_name" exception=(e, catch_backtrace())
            all_results[config_name] = :failed
        end
    end
    
    @info ""
    @info "═══════════════════════════════════════════════════════════════"
    @info "  ✅ Batch Processing Complete"
    @info "═══════════════════════════════════════════════════════════════"
    @info "Processed: $(length(all_results)) configurations"
    @info "  Successful: $(sum(values(all_results) .!= :failed))"
    @info "  Failed: $(sum(values(all_results) .== :failed))"
    @info "═══════════════════════════════════════════════════════════════"
    
    return all_results
end

#═══════════════════════════════════════════════════════════════════════════
# COMMAND LINE INTERFACE
#═══════════════════════════════════════════════════════════════════════════

function main()
    if length(ARGS) == 0
        println("""
        Usage:
          julia --project=. scripts/analyze_from_yaml.jl <config.yaml>
          julia --project=. scripts/analyze_from_yaml.jl --all [config_dir]
        
        Examples:
          julia --project=. scripts/analyze_from_yaml.jl configs/example_sweep.yaml
          julia --project=. scripts/analyze_from_yaml.jl --all
          julia --project=. scripts/analyze_from_yaml.jl --all configs/
        """)
        return
    end
    
    if ARGS[1] == "--all"
        config_dir = length(ARGS) >= 2 ? ARGS[2] : "configs"
        analyze_all(config_dir)
    else
        config_path = ARGS[1]
        analyze_config(config_path)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end