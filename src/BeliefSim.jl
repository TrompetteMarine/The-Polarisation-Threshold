module BeliefSim

using Statistics, Random, LinearAlgebra
using Plots

# Set default backend without pyplot
ENV["GKSwstype"] = "100"

# Core modules first
include("Utils.jl")
include("Types.jl")
include("Hazard.jl")
include("Model.jl")
include("Stats.jl")
include("Plotting.jl")

# Re-export all submodules
using .Utils
using .Types
using .Hazard
using .Model
using .Stats
using .Plotting

export Utils, Types, Hazard, Model, Stats, Plotting

end # module
