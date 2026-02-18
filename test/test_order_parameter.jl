using Test

include(joinpath(@__DIR__, "..", "scripts", "modules", "BranchClassification.jl"))
include(joinpath(@__DIR__, "..", "scripts", "modules", "OrderParameter.jl"))
using .OrderParameter

@testset "OrderParameter" begin
    mc, mse = correct_amplitude(0.3, 0.01, 0.1, 0.005)
    @test isapprox(mc, sqrt(0.08); atol=1e-10)
    @test isfinite(mse) && mse > 0

    mc2, _ = correct_amplitude(0.05, 0.01, 0.1, 0.005)
    @test mc2 == 0.0

    kstar = 5.0
    kappas = vcat(kstar .* (1 .- [0.3, 0.2, 0.15, 0.1]), kstar .* (1 .+ [0.05, 0.1]))
    m_abs = [0.05, 0.06, 0.055, 0.052, 0.2, 0.3]
    m0, m0_se, n_used = estimate_baseline_m0(kappas, m_abs, kstar)
    @test n_used >= 2
    @test m0 > 0
end
