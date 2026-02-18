#!/usr/bin/env julia
using Pkg
Pkg.activate(".")

using Random
using Statistics
using LinearAlgebra
using Printf
using Dates
using CSV
using DataFrames
using Distributions
using HypothesisTests
using StatsBase
using JSON3
using LibGit2
using Distributed

using BeliefSim
using BeliefSim.Types
using BeliefSim.Stats
using BeliefSim.Hazard: ν

# Legacy dependencies
include(joinpath(@__DIR__, "ensemble_utils.jl"))
include(joinpath(@__DIR__, "statistical_tests.jl"))
include(joinpath(@__DIR__, "visualization.jl"))
using .EnsembleUtils
using .StatisticalTests
using .Visualization

# New modular components
include(joinpath(@__DIR__, "config.jl"))
include(joinpath(@__DIR__, "modules", "IOUtils.jl"))
include(joinpath(@__DIR__, "modules", "ThresholdEstimation.jl"))
include(joinpath(@__DIR__, "modules", "EnsembleRunner.jl"))
include(joinpath(@__DIR__, "modules", "BranchClassification.jl"))
include(joinpath(@__DIR__, "modules", "OrderParameter.jl"))
include(joinpath(@__DIR__, "modules", "ScalingRegression.jl"))
include(joinpath(@__DIR__, "modules", "BifurcationTests.jl"))
include(joinpath(@__DIR__, "modules", "DensityAnalysis.jl"))
include(joinpath(@__DIR__, "modules", "SweepPipeline.jl"))
include(joinpath(@__DIR__, "modules", "Reporting.jl"))

using .Config
using .IOUtils
using .ThresholdEstimation
using .EnsembleRunner
using .BranchClassification
using .OrderParameter
using .ScalingRegression
using .BifurcationTests
using .DensityAnalysis
using .SweepPipeline
using .Reporting

"""
    driver_main()

Thin driver that delegates to the current fig6_ensemble_enhanced pipeline.
The modular pipeline is staged for gradual migration while preserving outputs.
"""
function driver_main()
    cfg = Config.parse_config(ARGS)
    Config.setup_output_dirs(cfg)

    # Delegate to the monolithic pipeline for now to preserve behavior.
    include(joinpath(@__DIR__, "fig6_ensemble_enhanced.jl"))
    if isdefined(Main, :main)
        Base.invokelatest(Main.main)
    else
        error("fig6_ensemble_enhanced.jl did not define main().")
    end
end

driver_main()
