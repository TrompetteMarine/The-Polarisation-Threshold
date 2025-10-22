module ModelInterface

using LinearAlgebra
using ForwardDiff
using Statistics

export f!, f, default_params, jacobian, kappa_get, kappa_set

"Default parameters for deterministic mean-field normal form."
function default_params()
    return (; λ = 1.0, beta = 1.0, kappa = 0.99, kstar = 1.0)
end

"Out-of-place model built from the in-place version."
function f(u::AbstractVector, p)
    du = similar(u)
    f!(du, u, p)
    return du
end

"""In-place vector field implementing the cubic mean-field normal form.
The dynamics approximate the deterministic skeleton of the OU-with-resets
model: each component relaxes to zero at rate λ, is driven by the population
mean with strength κ, and saturates through a cubic self-interaction with
coefficient β.
"""
function f!(du, u, p)
    hasproperty(p, :λ) || throw(ArgumentError("Parameter object must provide :λ"))
    λ = getproperty(p, :λ)
    β = hasproperty(p, :beta) ? getproperty(p, :beta) : 1.0
    κ = kappa_get(p)
    g = mean(u)
    @inbounds for i in eachindex(u)
        du[i] = -λ * u[i] + κ * g - β * u[i]^3
    end
    return nothing
end

"ForwardDiff Jacobian for the vector field."
jacobian(u::AbstractVector, p) = ForwardDiff.jacobian(x -> f(x, p), u)

function kappa_get(p)
    hasproperty(p, :kappa) || throw(ArgumentError("Parameter object must provide :kappa"))
    return getproperty(p, :kappa)
end

kappa_set(p::NamedTuple, κ) = merge(p, (; kappa = κ,))

function kappa_set(p, κ)
    hasproperty(p, :kappa) || throw(ArgumentError("Parameter object must provide :kappa"))
    newp = deepcopy(p)
    Base.setproperty!(newp, :kappa, κ)
    return newp
end

end # module
