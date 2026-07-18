module Pipeline

using DelimitedFiles, Printf
using ..Types: ModelParameters, SimulationParameters
using ..MonteCarlo: paired_response
using ..KernelStatistics: summarize_kernel
using ..GeneratorCheck: generator_cross_check
using ..CITPlotting: main_figure

function write_metrics(path, report)
    keys_sorted = sort(collect(keys(report)))
    open(path,"w") do io
        println(io,"{")
        for (i,key) in enumerate(keys_sorted)
            comma = i < length(keys_sorted) ? "," : ""
            println(io,@sprintf("  \"%s\": %.12g%s",key,report[key],comma))
        end
        println(io,"}")
    end
end

function run_pipeline(output_dir::AbstractString="outputs"; publication::Bool=false)
    model = ModelParameters()
    simulation = publication ? SimulationParameters() : SimulationParameters(
        particles=12_000, blocks=24, burn_steps=700, burn_dt=0.01,
        horizon=5.0, dt=0.025, delta=0.02,
    )
    response = paired_response(model, simulation)
    summary = summarize_kernel(response.times, response.block_responses)
    generator = generator_cross_check(model; grid_points=publication ? 801 : 401)

    mkpath(output_dir)
    table = hcat(response.times, summary.mean, summary.lower, summary.upper)
    open(joinpath(output_dir,"response_kernel.csv"),"w") do io
        println(io,"t,k_hat,lower95,upper95")
        writedlm(io,table,',')
    end
    main_figure(response,summary,generator,joinpath(output_dir,"figure_numerical.svg"))

    report = Dict(
        "mc_susceptibility"=>summary.susceptibility,
        "mc_positive_susceptibility"=>summary.positive_susceptibility,
        "mc_threshold"=>summary.threshold,
        "generator_susceptibility"=>generator.susceptibility,
        "generator_threshold"=>generator.threshold,
        "k0"=>summary.mean[1],
        "support_95"=>summary.support_95,
        "support_997"=>summary.support_997,
    )
    write_metrics(joinpath(output_dir,"metrics.json"),report)
    report
end

end
