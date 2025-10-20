module Hazard

using ..Types: AbstractHazard, StepHazard, LogisticHazard

export ν

@inline function ν(h::StepHazard, u::Float64, Θ::Float64)
    return abs(u) >= Θ ? h.ν0 : 0.0
end

@inline function ν(h::LogisticHazard, u::Float64, Θ::Float64)
    return h.νmax / (1 + exp(-h.β*(abs(u) - Θ)))
end

function ν(h::AbstractHazard, u::Float64, Θ::Float64)
    error("Unsupported hazard type: $(typeof(h))")
end

end # module
