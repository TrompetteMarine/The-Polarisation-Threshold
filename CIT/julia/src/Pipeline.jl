module Pipeline

using CSV, DataFrames, JSON3
using ..Types: ModelParameters, SimulationParameters
using ..MonteCarlo: paired_response
using ..KernelStatistics: summarize_kernel
using ..GeneratorCheck: generator_cross_check
using ..CITPlotting: main_figure

function run_pipeline(output_dir::AbstractString="outputs"; publication::Bool=false)
    model = ModelParameters()
    simulation = publication ? SimulationParameters() : SimulationParameters(
        particles=12_000,
        blocks=24,
        burn_steps=700,
        burn_dt=0.01,
        horizon=5.0,
        dt=0.025,
        delta=0.02,
    )
    response = paired_response(model, simulation)
    summary = summarize_kernel(response.times, response.block_responses)
    generator = generator_cross_check(model; grid_points=publication ? 801 : 401)

    mkpath(output_dir)
    CSV.write(joinpath(output_dir, "response_kernel.csv"), DataFrame(
        t=response.times,
        k_hat=summary.mean,
        lower95=summary.lower,
        upper95=summary.upper,
    ))
    main_figure(response, summary, generator, joinpath(output_dir, "figure_numerical.png"))

    report = Dict(
        "mc_susceptibility" => summary.susceptibility,
        "mc_positive_susceptibility" => summary.positive_susceptibility,
        "mc_threshold" => summary.threshold,
        "generator_susceptibility" => generator.susceptibility,
        "generator_threshold" => generator.threshold,
        "k0" => summary.mean[1],
        "support_95" => summary.support_95,
        "support_997" => summary.support_997,
    )
    open(joinpath(output_dir, "metrics.json"), "w") do io
        JSON3.pretty(io, report)
    end
    report
end

end
