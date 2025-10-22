module SimpleContinuation

using LinearAlgebra
using DifferentialEquations

export continue_equilibria, detect_hopf, find_limit_cycle

"""
Simple equilibrium continuation by Newton iteration
"""
struct ContinuationBranch
    κ_values::Vector{Float64}
    equilibria::Vector{Vector{Float64}}
    eigenvalues::Vector{Vector{ComplexF64}}
end

function continue_equilibria(f::Function, jac::Function, u0::Vector{Float64}, 
                            p_base, κ_values::AbstractVector{Float64};
                            tol=1e-10, maxiter=50)
    equilibria = Vector{Vector{Float64}}()
    eigenvalues = Vector{Vector{ComplexF64}}()
    
    u_guess = copy(u0)
    
    for κ in κ_values
        # Update parameter
        p = merge(p_base, (kappa=κ,))
        
        # Newton iteration
        u_eq = newton_solve(f, jac, u_guess, p; tol=tol, maxiter=maxiter)
        
        # Compute eigenvalues at equilibrium
        J = jac(u_eq, p)
        λ = eigvals(J)
        
        push!(equilibria, copy(u_eq))
        push!(eigenvalues, λ)
        
        # Use previous solution as next guess
        u_guess = u_eq
    end
    
    return ContinuationBranch(collect(κ_values), equilibria, eigenvalues)
end

function newton_solve(f::Function, jac::Function, u0::Vector{Float64}, p;
                     tol=1e-10, maxiter=50)
    u = copy(u0)
    
    for iter in 1:maxiter
        F = f(u, p)
        
        if norm(F) < tol
            return u
        end
        
        J = jac(u, p)
        
        # Newton step
        try
            δu = -J \ F
            u .+= δu
            
            if norm(δu) < tol
                return u
            end
        catch e
            @warn "Newton iteration failed" iteration=iter exception=e
            return u
        end
    end
    
    @warn "Newton did not converge" norm_residual=norm(f(u, p))
    return u
end

"""
Detect Hopf bifurcation: pair of pure imaginary eigenvalues
"""
function detect_hopf(eigenvalues::Vector{ComplexF64}; tol=1e-2)
    for λ in eigenvalues
        # Check if real part is near zero and imaginary part is significant
        if abs(real(λ)) < tol && abs(imag(λ)) > tol
            return true
        end
    end
    return false
end

"""
Find limit cycle near Hopf bifurcation using direct integration
"""
function find_limit_cycle(f::Function, u_eq::Vector{Float64}, p;
                         perturbation=1e-3, tmax=500.0, 
                         detect_period=true)
    # Perturb equilibrium
    u0 = u_eq .+ perturbation .* randn(length(u_eq))
    
    # Define ODE
    function ode!(du, u, p, t)
        F = f(u, p)
        du .= F
    end
    
    # Solve
    prob = ODEProblem(ode!, u0, (0.0, tmax), p)
    sol = solve(prob, Tsit5(); reltol=1e-9, abstol=1e-9, saveat=0.1)
    
    if !detect_period
        return sol
    end
    
    # Detect period from last portion of trajectory
    n_detect = min(1000, length(sol.t))
    t_detect = sol.t[end-n_detect+1:end]
    u_detect = reduce(hcat, sol.u[end-n_detect+1:end])
    
    # Use first component for period detection
    signal = u_detect[1, :]
    
    # Simple peak detection
    period = estimate_period_from_peaks(t_detect, signal)
    
    return (sol=sol, period=period, is_periodic=(period > 0))
end

function estimate_period_from_peaks(t::Vector{Float64}, signal::Vector{Float64})
    # Find peaks (local maxima)
    peaks = Int[]
    for i in 2:length(signal)-1
        if signal[i] > signal[i-1] && signal[i] > signal[i+1]
            push!(peaks, i)
        end
    end
    
    if length(peaks) < 2
        return 0.0
    end
    
    # Compute mean inter-peak interval
    intervals = diff(t[peaks])
    
    if isempty(intervals)
        return 0.0
    end
    
    # Use median to be robust to outliers
    return median(intervals)
end

end # module