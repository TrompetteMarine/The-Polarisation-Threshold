using Test
using Distributions

include(joinpath(@__DIR__, "..", "scripts", "modules", "DensityAnalysis.jl"))
using .DensityAnalysis

@testset "DensityAnalysis" begin
    x = collect(-2.0:0.1:2.0)
    dx = x[2] - x[1]
    density = pdf.(Normal(0, 1), x)
    density ./= sum(density) * dx
    moments = compute_density_moments(x, density)
    @test abs(moments.mean) < 0.05
    @test abs(moments.var - 1.0) < 0.1
    @test run_density_sanity_checks(x, density; tol=0.05)
end
