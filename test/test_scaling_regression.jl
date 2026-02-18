using Test
using Random

include(joinpath(@__DIR__, "..", "scripts", "modules", "ScalingRegression.jl"))
using .ScalingRegression

@testset "ScalingRegression" begin
    x_test = log.(collect(0.01:0.01:0.15))
    y_test = log(2.0) .+ 0.5 .* x_test
    fit = fit_scaling_ols(x_test, y_test)
    @test abs(fit.beta_hat - 0.5) < 1e-10
    @test fit.r2 > 0.9999

    rng = MersenneTwister(42)
    kstar_true = 5.0
    n = 60
    deltas = exp.(range(log(0.01), log(0.15), length=n))
    kappas = kstar_true .* (1 .+ deltas)
    m_true = 2.0 .* (kappas .- kstar_true) .^ 0.5
    m_obs = m_true .* exp.(0.02 .* randn(rng, n))
    pf = fit_scaling_profile(kappas, m_obs, kstar_true * 1.03)
    @test pf.converged
    @test abs(pf.kappa_star_eff - kstar_true) / kstar_true < 0.05
    @test pf.beta_ci[1] <= 0.5 <= pf.beta_ci[2]

    bf = fit_scaling_bootstrap(kappas, m_obs, kstar_true * 1.02; n_boot=500, rng=rng)
    @test bf.n_successful >= 100
    @test bf.beta_ci[1] <= 0.5 <= bf.beta_ci[2]

    # Cook's D detects outlier
    y_out = copy(y_test)
    y_out[1] = 10.0
    fit_out = fit_scaling_ols(x_test, y_out)
    @test argmax(fit_out.cooks_d) == 1
end
