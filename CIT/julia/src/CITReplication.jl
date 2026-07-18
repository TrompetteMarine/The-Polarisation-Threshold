module CITReplication

include("Types.jl")
include("Model.jl")
include("MonteCarlo.jl")
include("Statistics.jl")
include("Generator.jl")

using .Types: ModelParameters, SimulationParameters
using .MonteCarlo: ResponseResult, paired_response
using .KernelStatistics: KernelSummary, summarize_kernel, laplace_transform
using .GeneratorCheck: GeneratorResult, generator_cross_check

export ModelParameters, SimulationParameters, ResponseResult, KernelSummary, GeneratorResult
export paired_response, summarize_kernel, laplace_transform, generator_cross_check

end
