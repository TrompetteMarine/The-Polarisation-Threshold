module CITReplication

include("Types.jl")
include("Model.jl")
include("MonteCarlo.jl")
include("Statistics.jl")
include("Generator.jl")
include("Plotting.jl")
include("Pipeline.jl")

using .Types
using .Model
using .MonteCarlo
using .KernelStatistics
using .GeneratorCheck
using .CITPlotting
using .Pipeline

export ModelParameters, SimulationParameters, ResponseResult, KernelSummary, GeneratorResult
export paired_response, summarize_kernel, laplace_transform, generator_cross_check, run_pipeline

end
