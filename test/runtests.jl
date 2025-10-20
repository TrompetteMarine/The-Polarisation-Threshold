using Test
using BeliefSim
using BeliefSim.Types
using BeliefSim.Stats

@testset "BeliefSim smoke tests" begin
    p = Params(λ=1.0, σ=0.8, Θ=2.0, c0=0.5, hazard=StepHazard(0.5))
    Vstar = estimate_Vstar(p; N=5_000, T=50.0, dt=0.02, burn_in=10.0, seed=2025)
    @test Vstar > 0.0
    κcrit = critical_kappa(p; Vstar=Vstar)
    @test κcrit > 0.0
end
