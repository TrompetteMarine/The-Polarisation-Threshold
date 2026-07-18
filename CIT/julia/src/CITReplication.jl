module CITReplication

include("Types.jl")
include("Model.jl")
include("MonteCarlo.jl")
include("Statistics.jl")
include("Generator.jl")

using .Types
using .Model
using .MonteCarlo
using .KernelStatistics
using .GeneratorCheck

export ModelParameters, SimulationParameters, ResponseResult, KernelSummary, GeneratorResult
export paired_response, summarize_kernel, laplace_transform, generator_cross_check

end
