module Types

abstract type AbstractHazard end

# Step function hazard: ν(u) = ν0 if |u| >= Θ else 0
struct StepHazard <: AbstractHazard
    ν0::Float64
end

# Logistic hazard: ν(u) = νmax / (1 + exp(-β (|u| - Θ)))
struct LogisticHazard <: AbstractHazard
    νmax::Float64
    β::Float64
end

Base.show(io::IO, h::StepHazard) = print(io, "StepHazard(ν0=$(h.ν0))")
Base.show(io::IO, h::LogisticHazard) = print(io, "LogisticHazard(νmax=$(h.νmax), β=$(h.β))")

# Parameter bundle
struct Params{H<:AbstractHazard}
    λ::Float64      # mean reversion rate
    σ::Float64      # diffusion intensity
    Θ::Float64      # dissonance threshold
    c0::Float64     # reset contraction in (0,1)
    hazard::H       # hazard object
end

# Add keyword constructor
function Params(; λ::Float64, σ::Float64, Θ::Float64, c0::Float64, hazard::AbstractHazard)
    Params(λ, σ, Θ, c0, hazard)
end

# Results from kappa parameter sweep
struct KappaSweepResult
    κ::Vector{Float64}    # kappa values
    amp::Vector{Float64}  # amplitude values
    V::Vector{Float64}    # variance values
end

export Params, StepHazard, LogisticHazard, KappaSweepResult

end # module
