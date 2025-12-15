#!/usr/bin/env julia
"""
Quick test of welfare contours with small grid (5×5)
"""

using BeliefSim
using BeliefSim.OUResets
using BeliefSim.Types
using Plots
using Printf

# Test parameters
params = Params(
    λ = 0.65,
    σ = 1.15,
    Θ = 0.87,
    c0 = 0.50,
    hazard = StepHazard(0.5)
)

# Small grid for quick test
theta_grid = collect(range(0.6, 1.0; length=5))
c0_grid = collect(range(0.3, 0.7; length=5))

println("Testing welfare surface computation with 5×5 grid...")
println("θ range: [$(theta_grid[1]), $(theta_grid[end])]")
println("c₀ range: [$(c0_grid[1]), $(c0_grid[end])]")

# Test a single point
theta_test = theta_grid[3]
c0_test = c0_grid[3]

println("\nTesting point (θ=$(theta_test), c₀=$(c0_test)):")

# Compute V*
V = compute_stationary_variance(theta_test, c0_test, params; N=2000, T=50.0, seed=42)
@printf("  V* = %.4f\n", V)

if isfinite(V) && V > 0
    # Compute spectral quantities
    p_test = Params(λ=params.λ, σ=params.σ, Θ=theta_test, c0=c0_test, hazard=params.hazard)
    lambda10, lambda1_dot = compute_lambda1_and_derivative(V, p_test; M=201)
    @printf("  λ₁₀ = %.5f, λ̇₁₀ = %.5f\n", lambda10, lambda1_dot)

    if isfinite(lambda10) && isfinite(lambda1_dot) && abs(lambda1_dot) > 1e-12
        kappa_star = -lambda10 / lambda1_dot
        @printf("  κ* = %.4f\n", kappa_star)

        # Compute welfare loss
        L_dec = welfare_loss(V, theta_test, p_test; regime=:dec)
        L_soc = welfare_loss(V, theta_test, p_test; regime=:soc)
        @printf("  L_dec(V,θ) = %.6f\n", L_dec)
        @printf("  L_soc(V,θ) = %.6f\n", L_soc)

        if isfinite(L_dec) && isfinite(L_soc)
            println("✓ Single point test passed!")
        else
            println("✗ Welfare loss is not finite")
        end
    else
        println("✗ Invalid eigenvalues")
    end
else
    println("✗ Invalid variance")
end

println("\n✓ Quick test completed successfully!")
