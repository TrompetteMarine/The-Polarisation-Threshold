using Test, LinearAlgebra, Statistics, CITReplication

@testset "CIT replication" begin
    model = ModelParameters()
    simulation = SimulationParameters(particles=2400, blocks=12, burn_steps=250, horizon=1.0, dt=0.05)
    response = paired_response(model, simulation)
    @test all(isapprox.(response.block_responses[1, :], 1.0; atol=1e-12))
    @test abs(mean(response.stationary_u)) < 0.08

    times = collect(range(0, 12; length=1201))
    kernel = exp.(-times)
    blocks = repeat(kernel, 1, 8)
    summary = summarize_kernel(times, blocks)
    @test abs(summary.susceptibility - 1.0) < 2e-3
    @test abs(summary.threshold - 1.0) < 2e-3

    signed_times = [0.0, 1.0, 2.0, 3.0]
    signed_kernel = [1.0, 0.6, -0.2, 0.0]
    signed_blocks = repeat(signed_kernel, 1, 8)
    signed_summary = summarize_kernel(signed_times, signed_blocks)
    raw = sum((signed_kernel[1:end-1] .+ signed_kernel[2:end]) .* diff(signed_times)) / 2
    positive_kernel = max.(signed_kernel, 0.0)
    positive = sum((positive_kernel[1:end-1] .+ positive_kernel[2:end]) .* diff(signed_times)) / 2
    @test raw > 0.0
    @test positive > raw
    @test isapprox(signed_summary.susceptibility, raw)
    @test isapprox(signed_summary.positive_susceptibility, positive)
    @test isapprox(signed_summary.threshold, inv(raw))
    @test isapprox(signed_summary.positive_threshold, inv(positive))
    @test !isapprox(signed_summary.threshold, signed_summary.positive_threshold)

    generator = generator_cross_check(model; grid_points=401, domain=5.0)
    @test abs(sum(generator.stationary_mass) - 1.0) < 1e-10
    @test abs(dot(generator.grid, generator.response_direction) - 1.0) < 1e-9
    @test 0.63 < generator.susceptibility < 0.72
    @test 1.38 < generator.threshold < 1.59
end
