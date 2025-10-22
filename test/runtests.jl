using Test
using BeliefSim
using BeliefSim.Types
using BeliefSim.Stats
using BeliefSim.Model
using BeliefSim.Simulate

@testset "BeliefSim.jl" begin
    @testset "Module loading" begin
        @test isdefined(BeliefSim, :Types)
        @test isdefined(BeliefSim, :Stats)
        @test isdefined(BeliefSim, :Model)
        @test isdefined(BeliefSim, :Simulate)
        @test isdefined(BeliefSim, :Plotting)
        @test isdefined(BeliefSim, :Network)
    end

    @testset "Parameter construction" begin
        p = Params(λ=1.0, σ=0.8, Θ=2.0, c0=0.5, hazard=StepHazard(0.5))
        @test p.λ == 1.0
        @test p.σ == 0.8
        @test p.Θ == 2.0
        @test p.c0 == 0.5
        @test p.hazard isa StepHazard
    end

    @testset "V* estimation" begin
        p = Params(λ=1.0, σ=0.8, Θ=2.0, c0=0.5, hazard=StepHazard(0.5))
        Vstar = estimate_Vstar(p; N=5_000, T=50.0, dt=0.02, 
                              burn_in=10.0, seed=2025)
        @test Vstar > 0.0
        @test isfinite(Vstar)
    end

    @testset "Critical κ estimation" begin
        p = Params(λ=1.0, σ=0.8, Θ=2.0, c0=0.5, hazard=StepHazard(0.5))
        Vstar = estimate_Vstar(p; N=5_000, T=50.0, dt=0.02, 
                              burn_in=10.0, seed=2025)
        κcrit = critical_kappa(p; Vstar=Vstar)
        @test κcrit > 0.0
        @test isfinite(κcrit)
    end

    @testset "κ sweep" begin
        p = Params(λ=1.0, σ=0.8, Θ=2.0, c0=0.5, hazard=StepHazard(0.5))
        κgrid = [0.0, 0.5, 1.0]
        res = sweep_kappa(p, κgrid; N=2_000, T=20.0, dt=0.02, 
                         burn_in=5.0, seed=2025)
        @test length(res.κ) == 3
        @test length(res.amp) == 3
        @test length(res.V) == 3
        @test all(isfinite, res.amp)
        @test all(isfinite, res.V)
    end
end

# Include bifurcation tests only if available
if isdefined(Main, :BifurcationKit)
    include("test_bifurcation.jl")
else
    @warn "BifurcationKit not available, skipping bifurcation tests"
end