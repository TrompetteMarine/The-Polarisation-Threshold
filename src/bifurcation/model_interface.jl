module ModelInterface

using LinearAlgebra
using ForwardDiff
using Statistics

export f!, f, default_params, jacobian, kappa_get, kappa_set
export polarization_amplitude, polarized_equilibria

"Default parameters for deterministic mean-field normal form."
function default_params()
    return (
        λ = 1.0,
        σ = 0.8,
        Θ = 2.0,
        c0 = 0.5,
        Vstar = 1.0,
        g = 1.0,
        beta = 1.0,
        kappa = 0.0,
        kstar = 1.0,
    )
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
    κ_star = hasproperty(p, :kstar) ? getproperty(p, :kstar) : λ
    σ = hasproperty(p, :σ) ? getproperty(p, :σ) : 1.0
    Vstar = hasproperty(p, :Vstar) ? getproperty(p, :Vstar) : 1.0

    σ_sq = max(σ^2, eps())

    if length(u) == 2
        m = (u[1] + u[2]) / 2
        a = (u[1] - u[2]) / 2

        dm_dt = -λ * m
        prefactor = (2 * λ * Vstar) / σ_sq
        growth = prefactor * (κ - κ_star)
        da_dt = growth * a - β * a^3

        du[1] = dm_dt + da_dt
        du[2] = dm_dt - da_dt
    else
        m = mean(u)
        dm_dt = -λ * m
        prefactor = (2 * λ * Vstar) / σ_sq
        growth = prefactor * (κ - κ_star)
        @inbounds for i in eachindex(u)
            deviation = u[i] - m
            du[i] = dm_dt + growth * deviation - β * deviation^3
        end
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

function polarization_growth(p)
    hasproperty(p, :λ) || throw(ArgumentError("Parameter object must provide :λ"))
    hasproperty(p, :σ) || throw(ArgumentError("Parameter object must provide :σ"))
    hasproperty(p, :Vstar) || throw(ArgumentError("Parameter object must provide :Vstar"))
    hasproperty(p, :kstar) || throw(ArgumentError("Parameter object must provide :kstar"))

    λ = getproperty(p, :λ)
    σ = getproperty(p, :σ)
    Vstar = getproperty(p, :Vstar)
    κ = kappa_get(p)
    κ_star = getproperty(p, :kstar)
    σ_sq = max(abs2(σ), eps())

    prefactor = (2 * λ * Vstar) / σ_sq
    return prefactor * (κ - κ_star)
end

function polarization_amplitude(p)
    growth = polarization_growth(p)
    β = hasproperty(p, :beta) ? getproperty(p, :beta) : 1.0

    if growth <= 0 || β <= 0
        return 0.0
    end

    return sqrt(max(growth / β, 0.0))
end

function polarized_equilibria(p)
    amp = polarization_amplitude(p)

    if amp <= 0
        return Vector{Vector{Float64}}()
    end

    return [[amp, -amp], [-amp, amp]]
end

end # module
