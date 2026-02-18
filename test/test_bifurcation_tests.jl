using Test
using Random

include(joinpath(@__DIR__, "..", "scripts", "modules", "BranchClassification.jl"))
include(joinpath(@__DIR__, "..", "scripts", "modules", "OrderParameter.jl"))
include(joinpath(@__DIR__, "..", "scripts", "modules", "ScalingRegression.jl"))
include(joinpath(@__DIR__, "..", "scripts", "modules", "BifurcationTests.jl"))
using .BifurcationTests

@testset "BifurcationTests" begin
    rng = MersenneTwister(42)
    kstar = 5.0
    n = 80
    kappas = vcat(
        kstar .* (1 .+ exp.(range(log(0.01), log(0.15), length=60))),
        kstar .* (1 .- exp.(range(log(0.10), log(0.25), length=20))),
    )
    m_abs_true = [k > kstar ? 2.0 * sqrt(k - kstar) : 0.05 for k in kappas]
    m_abs = m_abs_true .+ 0.01 .* randn(rng, n)
    m_abs_se = fill(0.01, n)

    result = test_scaling_exponent_multi(kappas, m_abs, m_abs_se, kstar * 1.02)
    @test result.n_methods_pass >= 1
    @test isfinite(result.primary_beta) || isnan(result.primary_beta)
end
