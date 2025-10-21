module Bifurcation

using LinearAlgebra
using DifferentialEquations: ODEProblem, solve
using BifurcationKit
using Parameters

import ModelInterface

export continue_equilibria, detect_pitchfork, detect_hopf,
       continue_cycles_from_hopf, find_homoclinic_near_saddle, relaxation_rate,
       EquilibriumContinuation

struct EquilibriumContinuation{P,T}
    kappa::Vector{Float64}
    param::Vector{P}
    branch::Vector{T}
end

function _newton(u0, p; tol=1e-10, maxiter=25)
    u = copy(u0)
    for _ in 1:maxiter
        F = ModelInterface.f(u, p)
        if norm(F) < tol
            break
        end
        J = ModelInterface.jacobian(u, p)
        δ = -J \ F
        u .= u .+ δ
        if norm(δ) < tol
            break
        end
    end
    return u
end

function continue_equilibria(u0, p; κ_grid=collect(0.7:0.005:1.4), tol=1e-10, maxiter=25)
    params = Vector{typeof(p)}()
    branch = Vector{NamedTuple{(:u,), Tuple{Vector{eltype(u0)}}}}()
    kappas = Float64[]
    u_guess = copy(u0)
    for κ in κ_grid
        pκ = ModelInterface.kappa_set(p, κ)
        u_guess = _newton(u_guess, pκ; tol=tol, maxiter=maxiter)
        push!(params, pκ)
        push!(branch, (; u = copy(u_guess)))
        push!(kappas, κ)
    end
    return EquilibriumContinuation(kappas, params, branch)
end

function detect_pitchfork(u, p; tol=1e-6)
    λ = eigvals(ModelInterface.jacobian(u, p))
    return any(abs(real(λi)) < tol && abs(imag(λi)) < tol for λi in λ)
end

function detect_hopf(u, p; tol=1e-3)
    λ = eigvals(ModelInterface.jacobian(u, p))
    return any(abs(real(λi)) < tol && abs(imag(λi)) > 10tol for λi in λ)
end

function continue_cycles_from_hopf(br::EquilibriumContinuation, idx; tmax=200.0, perturbation=1e-3)
    p = br.param[idx]
    ueq = br.branch[idx].u
    dim = length(ueq)
    perturb = normalize(randn(dim)) * perturbation
    u0 = ueq .+ perturb
    function vf!(du, u, _, t)
        ModelInterface.f!(du, u, p)
    end
    prob = ODEProblem(vf!, u0, (0.0, tmax), nothing)
    sol = solve(prob; reltol=1e-9, abstol=1e-9)
    orbit = [copy(s) for s in sol.u]
    κ = ModelInterface.kappa_get(p)
    return (; orbit=orbit, param=fill(κ, length(orbit)), T=sol.t[end])
end

function find_homoclinic_near_saddle(u_s, p; tmax=2000.0, tol=1e-2)
    J = ModelInterface.jacobian(u_s, p)
    λ, V = eigen(J)
    idx = argmax(real.(λ))
    v = normalize(real.(V[:, idx]))
    function shoot(sign)
        u0 = u_s .+ sign * 1e-5 * v
        function vf!(du, u, _, t)
            ModelInterface.f!(du, u, p)
        end
        prob = ODEProblem(vf!, u0, (0.0, tmax), nothing)
        sol = solve(prob; reltol=1e-9, abstol=1e-9, save_everystep=false)
        mins = map(u -> norm(u .- u_s), sol.u)
        return minimum(mins)
    end
    return min(shoot(+1), shoot(-1)) < tol
end

relaxation_rate(u_star, p) = maximum(real.(eigvals(ModelInterface.jacobian(u_star, p))))

end # module
