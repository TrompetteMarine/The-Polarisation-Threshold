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
using Base.Threads

# Import only the simulation submodules we need (without triggering BifurcationKit)
module BeliefAnalysisCore
    include(joinpath(@__DIR__, "..", "src", "Utils.jl"))
    include(joinpath(@__DIR__, "..", "src", "Types.jl"))
    include(joinpath(@__DIR__, "..", "src", "Hazard.jl"))
    include(joinpath(@__DIR__, "..", "src", "Model.jl"))
    include(joinpath(@__DIR__, "..", "src", "Stats.jl"))
end

"""
Map requested κ/κ* ratios to κ values inside [κ_min, κ_max].
Falls back to an evenly spaced grid if κ* <= 0 or if the requested
ratios do not intersect the interval.
"""
function kappa_from_ratios(κ_star::Real, κ_min::Real, κ_max::Real,
                           ratios::AbstractVector{<:Real}; fallback_count::Int=length(ratios))
    if !isfinite(κ_star) || κ_star <= 0
        return collect(range(κ_min, κ_max; length=max(fallback_count, 2)))
    end

    raw = κ_star .* collect(ratios)
    span = maximum(abs.([κ_star, κ_min, κ_max]))
    tol = max(1e-9, 1e-6 * span)
    mask = (raw .>= κ_min - tol) .& (raw .<= κ_max + tol)
    selected = sort(unique(raw[mask]))

    if isempty(selected)
        return collect(range(κ_min, κ_max; length=max(fallback_count, 2)))
    end

    return selected
end

function ratio_ticks(κ_min::Real, κ_max::Real, κ_star::Real)
    if !isfinite(κ_star) || κ_star <= 0
        return nothing
    end

    r_min = max(0.0, floor(κ_min / κ_star * 2) / 2)
    r_max = ceil(max(κ_max / κ_star, 0.0) * 2) / 2
    ratios = collect(r_min:0.5:r_max)
    ticks = κ_star .* ratios
    labels = [string(round(r, digits=2), " κ*") for r in ratios]
    return (ticks, labels)
end

using .BeliefAnalysisCore.Types: Params, StepHazard, LogisticHazard
using .BeliefAnalysisCore.Stats: estimate_Vstar, estimate_g, critical_kappa

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
include(joinpath(@__DIR__, "..", "src", "bifurcation", "plotting_cairo.jl"))

using .ModelInterface
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

    # Calibrate cubic nonlinearity so that |u₁-u₂|/2 ≈ √((κ-κ*)/κ*) for κ > κ*
    σ_sq = max(σ^2, eps())
    prefactor = (2 * λ * Vstar) / σ_sq
    β = prefactor * max(κ_star, eps())
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

        amp_ref = ModelInterface.polarization_amplitude(p)
        tol = amp_ref > 0 ? max(0.02, 0.1 * amp_ref) : 0.02

        if abs(pol_final) < tol
            return 0  # Consensus
        elseif pol_final > tol
            return 1  # Positive polarization
        elseif pol_final < -tol
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

"""
Pre-compute equilibrium diagnostics across a κ-grid.
Returns consensus stability and polarized branch information for plotting.
"""
function equilibrium_diagnostics(κ_values::AbstractVector{<:Real}, p_base)
    κ_vec = collect(float.(κ_values))
    n = length(κ_vec)

    consensus_max = Vector{Float64}(undef, n)
    consensus_imag = Vector{Float64}(undef, n)
    polarized_amp = zeros(Float64, n)
    polarized_exists = falses(n)
    polarized_max = fill(NaN, n)
    polarized_imag = fill(NaN, n)

    for (i, κ) in enumerate(κ_vec)
        p = ModelInterface.kappa_set(p_base, κ)

        J_cons = ModelInterface.jacobian(zeros(2), p)
        eig_cons = eigvals(J_cons)
        consensus_max[i] = maximum(real.(eig_cons))
        consensus_imag[i] = maximum(abs.(imag.(eig_cons)))

        amp = ModelInterface.polarization_amplitude(p)
        if amp > 0
            polarized_exists[i] = true
            polarized_amp[i] = amp
            u_pol = [amp, -amp]
            eig_pol = eigvals(ModelInterface.jacobian(u_pol, p))
            polarized_max[i] = maximum(real.(eig_pol))
            polarized_imag[i] = maximum(abs.(imag.(eig_pol)))
        end
    end

    return (; κ=κ_vec,
            consensus_max=consensus_max,
            consensus_imag=consensus_imag,
            polarized_amp=polarized_amp,
            polarized_exists=polarized_exists,
            polarized_max=polarized_max,
            polarized_imag=polarized_imag)
end

function format_ratio_label(κ::Real, κ_star::Real; digits_ratio::Int=2, digits_kappa::Int=3)
    if isfinite(κ_star) && κ_star > 0
        ratio = κ / κ_star
        return @sprintf("κ = %.*f (κ/κ* = %.*f)", digits_kappa, κ, digits_ratio, ratio)
    else
        return @sprintf("κ = %.*f", digits_kappa, κ)
    end
end

#═══════════════════════════════════════════════════════════════════════════
# PLOTTING FUNCTIONS
#═══════════════════════════════════════════════════════════════════════════

"""
Plot 1: Extended bifurcation diagram
"""
function plot_bifurcation(κ_range, p_base)
    @info "  Computing bifurcation diagram..."

    κ_vals = collect(κ_range)
    diag = equilibrium_diagnostics(κ_vals, p_base)
    κ_star = getproperty(p_base, :kstar)

    fig = Figure(size=(1400, 1000))

    consensus_mean = zeros(length(κ_vals))
    stable_consensus = diag.consensus_max .< 0

    polarized_mask = diag.polarized_exists .& (.!isnan.(diag.polarized_max))
    stable_polarized = polarized_mask .& (diag.polarized_max .< 0)

    # Panel 1: Mean equilibrium
    ax1 = Axis(fig[1, 1:2];
              xlabel="Coupling strength κ",
              ylabel="Mean belief ⟨u⟩",
              title="Bifurcation Diagram: Consensus → Polarization")

    if any(stable_consensus)
        lines!(ax1, κ_vals[stable_consensus], consensus_mean[stable_consensus];
              linewidth=3, color=:blue, label="Consensus (stable)")
    end
    if any(.!stable_consensus)
        lines!(ax1, κ_vals[.!stable_consensus], consensus_mean[.!stable_consensus];
              linewidth=3, color=:red, linestyle=:dash, label="Consensus (unstable)")
    end

    vlines!(ax1, [κ_star]; color=:green, linestyle=:dash, linewidth=2,
            label="κ* = $(round(κ_star, digits=4))")
    ticks_main = ratio_ticks(minimum(κ_vals), maximum(κ_vals), κ_star)
    ticks_main === nothing || (ax1.xticks = ticks_main)
    axislegend(ax1, position=:lt)

    # Panel 2: Stability indicator
    ax2 = Axis(fig[2, 1];
              xlabel="κ",
              ylabel="max Re(λ)",
              title="Stability (negative = stable)")

    lines!(ax2, κ_vals, diag.consensus_max; linewidth=2, color=:blue, label="Consensus")
    if any(polarized_mask)
        lines!(ax2, κ_vals[polarized_mask], diag.polarized_max[polarized_mask];
              linewidth=2, color=:orange, label="Polarized")
    end
    hlines!(ax2, [0]; color=:black, linestyle=:dash, linewidth=1.5)
    axislegend(ax2, position=:lt)

    # Panel 3: Oscillation frequency
    ax3 = Axis(fig[2, 2];
              xlabel="κ",
              ylabel="|Im(λ)|",
              title="Oscillation Frequency")

    lines!(ax3, κ_vals, diag.consensus_imag; linewidth=2, color=:blue)
    if any(polarized_mask)
        lines!(ax3, κ_vals[polarized_mask], diag.polarized_imag[polarized_mask];
              linewidth=2, color=:orange)
    end

    # Panel 4: Polarization amplitude
    ax4 = Axis(fig[3, 1:2];
              xlabel="κ",
              ylabel="Polarization |u₁ - u₂|/2",
              title="Opinion Divergence")

    lines!(ax4, κ_vals, zeros(length(κ_vals)); color=:gray70, linewidth=1, linestyle=:dash)
    if any(stable_polarized)
        lines!(ax4, κ_vals[stable_polarized], diag.polarized_amp[stable_polarized];
              linewidth=3, color=:orange, label="Polarized (stable)")
        lines!(ax4, κ_vals[stable_polarized], -diag.polarized_amp[stable_polarized];
              linewidth=3, color=:orange)
    end
    if any(polarized_mask .& (.!stable_polarized))
        mask = polarized_mask .& (.!stable_polarized)
        lines!(ax4, κ_vals[mask], diag.polarized_amp[mask];
              linewidth=3, color=:red, linestyle=:dot, label="Polarized (unstable)")
        lines!(ax4, κ_vals[mask], -diag.polarized_amp[mask];
              linewidth=3, color=:red, linestyle=:dot)
    end

    vlines!(ax4, [κ_star]; color=:green, linestyle=:dash, linewidth=2)
    ticks_div = ratio_ticks(minimum(κ_vals), maximum(κ_vals), κ_star)
    ticks_div === nothing || (ax4.xticks = ticks_div)
    axislegend(ax4, position=:lt)

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
        
        κ_star = getproperty(p, :kstar)
        title_str = "Phase Space: " * format_ratio_label(κ, κ_star)
        ax = Axis(fig[row, col];
                 xlabel="u₁ (agent 1)",
                 ylabel="u₂ (agent 2)",
                 title=title_str,
                 aspect=DataAspect())
        
        # Vector field
        PlottingCairo.phase_portrait!(ax, ModelInterface.f, p;
                                     lims=(-3, 3, -3, 3),
                                     density=35, alpha=0.25)
        
        # Consensus line
        lines!(ax, [-3, 3], [-3, 3]; 
              color=:gray, linestyle=:dash, linewidth=2)
        
        # Consensus equilibrium
        u_cons = zeros(2)
        eig_cons = eigvals(ModelInterface.jacobian(u_cons, p))
        stable_cons = all(real.(eig_cons) .< 0)
        scatter!(ax, [u_cons[1]], [u_cons[2]];
                color=stable_cons ? :green : :orange,
                markersize=14,
                marker=stable_cons ? :circle : :xcross,
                label=stable_cons ? (idx == 1 ? "Consensus" : nothing)
                                  : (idx == 1 ? "Consensus (unstable)" : nothing))

        # Polarized equilibria
        pol_idx = 0
        for u_pol in ModelInterface.polarized_equilibria(p)
            pol_idx += 1
            eig_pol = eigvals(ModelInterface.jacobian(u_pol, p))
            stable_pol = all(real.(eig_pol) .< 0)
            scatter!(ax, [u_pol[1]], [u_pol[2]];
                    color=stable_pol ? :red : :purple,
                    markersize=14,
                    marker=stable_pol ? :rect : :utriangle,
                    label=stable_pol ? (pol_idx == 1 ? "Polarized" : nothing)
                                     : (pol_idx == 1 ? "Polarized (unstable)" : nothing))
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
        
        κ_star = getproperty(p, :kstar)
        ax = Axis(fig[row, col];
                 xlabel="Initial u₁",
                 ylabel="Initial u₂",
                 title="Basin: " * format_ratio_label(κ, κ_star),
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

    label_ratio = format_ratio_label(κ, getproperty(p, :kstar); digits_ratio=3)

    # Panel 1: Time series
    ax1 = Axis(fig[1, 1:2];
              xlabel="Time", ylabel="Belief",
              title="Evolution: " * label_ratio)
    
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
            
            label_ratio = format_ratio_label(κ, getproperty(p, :kstar); digits_ratio=3, digits_kappa=2)
            ax = Axis(fig[κ_idx, s_idx];
                     xlabel="u₁(t)", ylabel="u₁(t+τ)",
                     title=label_ratio * " — $(scen.name)",
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
    
    κ_vals = collect(range(κ_range[1], κ_range[2], length=n_points))
    κ_star = getproperty(p_base, :kstar)
    diag = equilibrium_diagnostics(κ_vals, p_base)

    consensus_stable_mask = diag.consensus_max .< 0
    consensus_unstable_mask = .!consensus_stable_mask

    polarized_mask = diag.polarized_exists .& (.!isnan.(diag.polarized_max))
    polarized_stable_mask = polarized_mask .& (diag.polarized_max .< 0)
    polarized_unstable_mask = polarized_mask .& (.!polarized_stable_mask)

    fig = Figure(size=(1200, 720))

    ax = Axis(fig[1, 1];
             xlabel="Coupling κ", ylabel="Polarization |u₁ - u₂|/2",
             title="Bifurcation Structure")

    if any(consensus_stable_mask)
        lines!(ax, κ_vals[consensus_stable_mask], zeros(sum(consensus_stable_mask));
               color=:steelblue, linewidth=3, label="Consensus (stable)")
    end
    if any(consensus_unstable_mask)
        lines!(ax, κ_vals[consensus_unstable_mask], zeros(sum(consensus_unstable_mask));
               color=:steelblue, linewidth=3, linestyle=:dash,
               label="Consensus (unstable)")
    end
    if any(polarized_stable_mask)
        lines!(ax, κ_vals[polarized_stable_mask], diag.polarized_amp[polarized_stable_mask];
               color=:firebrick, linewidth=3, label="Polarized (stable)")
        lines!(ax, κ_vals[polarized_stable_mask], -diag.polarized_amp[polarized_stable_mask];
               color=:firebrick, linewidth=3)
    end
    if any(polarized_unstable_mask)
        lines!(ax, κ_vals[polarized_unstable_mask], diag.polarized_amp[polarized_unstable_mask];
               color=:darkorange, linewidth=3, linestyle=:dot,
               label="Polarized (unstable)")
        lines!(ax, κ_vals[polarized_unstable_mask], -diag.polarized_amp[polarized_unstable_mask];
               color=:darkorange, linewidth=3, linestyle=:dot)
    end

    vlines!(ax, [κ_star]; color=:seagreen, linestyle=:dash, linewidth=3,
           label="κ* = $(round(κ_star, digits=4))")

    ticks = ratio_ticks(minimum(κ_vals), maximum(κ_vals), κ_star)
    ticks === nothing || (ax.xticks = ticks)

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

    κ_star = getproperty(p_base, :kstar)
    ticks = ratio_ticks(minimum(κ_vals), maximum(κ_vals), κ_star)
    ticks === nothing || (ax.xticks = ticks)
    
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
        sweep_cfg = cfg["sweep"]
        n_points = get(sweep_cfg, "points", 121)

        if haskey(sweep_cfg, "kappa_from_factor_of_kstar") && isfinite(κ_star) && κ_star > 0
            κ_min = κ_star * Float64(sweep_cfg["kappa_from_factor_of_kstar"])
        else
            κ_min = Float64(get(sweep_cfg, "kappa_from", max(0.0, 0.5 * κ_star)))
        end

        if haskey(sweep_cfg, "kappa_to")
            κ_max = Float64(sweep_cfg["kappa_to"])
        else
            factor = Float64(get(sweep_cfg, "kappa_to_factor_of_kstar", 3.0))
            κ_max = (isfinite(κ_star) && κ_star > 0) ? κ_star * factor : κ_min + max(abs(κ_min), 1.0)
        end
    else
        κ_min = max(0.0, 0.4 * κ_star)
        κ_max = (isfinite(κ_star) && κ_star > 0) ? 2.5 * κ_star : κ_min + 1.0
        n_points = 121
    end

    if κ_max <= κ_min
        bump = max(abs(κ_star) * 0.25, 1e-3)
        κ_max = κ_min + bump
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
    κ_range_fine = range(κ_min, κ_max; length=n_points)
    κ_range_coarse = range(κ_min, κ_max; length=max(20, div(n_points, 5)))
    κ_selected = kappa_from_ratios(κ_star, κ_min, κ_max, [0.5, 0.9, 1.1, 1.6]; fallback_count=4)
    
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
        κ_basin = kappa_from_ratios(κ_star, κ_min, κ_max, [0.6, 0.9, 1.1, 1.5]; fallback_count=4)
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
        
        κ_times = kappa_from_ratios(κ_star, κ_min, κ_max, [0.75, 1.05, 1.4]; fallback_count=3)
        for κ in κ_times
            fig = plot_timeseries(κ, p_base; tmax=T, dt=dt)
            ratio_tag = isfinite(κ_star) && κ_star > 0 ? @sprintf("%0.2f", κ / κ_star) : @sprintf("%0.2f", κ)
            filename = "04_timeseries_ratio_$(replace(ratio_tag, "." => "p")).png"
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
        κ_return = kappa_from_ratios(κ_star, κ_min, κ_max, [0.8, 1.05, 1.4]; fallback_count=3)
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
        if isfinite(κ_star) && κ_star > 0
            println(io, "  κ/κ* ∈ [$(round(κ_min/κ_star, digits=3)), $(round(κ_max/κ_star, digits=3))]")
        end
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