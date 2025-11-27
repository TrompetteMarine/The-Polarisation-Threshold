module BeliefSim

# Import standard library first
using Statistics, Random, LinearAlgebra, Printf
using Plots

# Set default backend without pyplot
ENV["GKSwstype"] = "100"

# Core modules in dependency order
include("Utils.jl")
include("Types.jl")
include("Hazard.jl")
include("Model.jl")
include("Stats.jl")      # Stats before Simulate (Stats is used more broadly)
include("Network.jl")    # Network needs Model
include("Plotting.jl")
include("OUResets.jl")

# Re-export all submodules
using .Utils
using .Types
using .Hazard
using .Model
using .Stats
using .Network
using .Plotting
using .OUResets

# Export modules
export Utils, Types, Hazard, Model, Stats, Network, Plotting

# Export commonly used functions
export Params, StepHazard, LogisticHazard
export estimate_Vstar, critical_kappa, sweep_kappa, pitchfork_fit
export simulate_network
export plot_bifurcation, plot_vector_field, plot_orbit
export OUResets
export simulate_single_path, stationary_density, leading_odd_eigenvalue
export order_parameter, compute_welfare_curves

# Optional: Load bifurcation analysis if available
const BIFURCATION_AVAILABLE = Ref(false)

function __init__()
    try
        @eval using CairoMakie
        @eval using BifurcationKit
        BIFURCATION_AVAILABLE[] = true
        @info "Bifurcation analysis features enabled"
    catch
        # Silently continue without bifurcation features
    end
end

end # module