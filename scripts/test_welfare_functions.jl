#!/usr/bin/env julia
"""
Test script for bifurcation-based welfare analysis functions

This script verifies that the new welfare analysis functions work correctly:
- compute_stationary_variance
- compute_lambda1_and_derivative
- compute_b_cubic
- welfare_loss
"""

using BeliefSim
using BeliefSim.OUResets
using BeliefSim.Types
using Printf

function main()
    println("="^70)
    println("Testing Bifurcation-Based Welfare Analysis Functions")
    println("="^70)

    # Define baseline parameters
    p = Params(
        λ = 0.65,
        σ = 1.15,
        Θ = 0.87,
        c0 = 0.50,
        hazard = StepHazard(0.5)
    )

    println("\nBaseline parameters:")
    println("  λ = $(p.λ), σ = $(p.σ), Θ = $(p.Θ), c₀ = $(p.c0)")
    println("  Hazard: ", p.hazard)

    # Test 1: compute_stationary_variance
    println("\n" * "─"^70)
    println("Test 1: compute_stationary_variance")
    println("─"^70)

    theta_test = 0.87
    c0_test = 0.50

    println("Computing V* for (θ=$(theta_test), c₀=$(c0_test))...")
    V_star = compute_stationary_variance(theta_test, c0_test, p; N=5_000, T=100.0, seed=42)
    @printf("  V* = %.4f\n", V_star)

    if isfinite(V_star) && V_star > 0
        println("  ✓ Valid variance computed")
    else
        println("  ✗ Invalid variance: $V_star")
    end

    # Test 2: compute_lambda1_and_derivative
    println("\n" * "─"^70)
    println("Test 2: compute_lambda1_and_derivative")
    println("─"^70)

    println("Computing λ₁₀(V) and λ̇₁₀(V) for V=$(V_star)...")
    lambda10, lambda1_dot = compute_lambda1_and_derivative(V_star, p; δκ=0.01, L=5.0, M=301)
    @printf("  λ₁₀(V) = %.6f\n", lambda10)
    @printf("  λ̇₁₀(V) = %.6f\n", lambda1_dot)

    if isfinite(lambda10) && isfinite(lambda1_dot)
        κ_star = -lambda10 / lambda1_dot
        @printf("  → κ*(V) ≈ %.4f\n", κ_star)
        println("  ✓ Valid eigenvalue and derivative computed")
    else
        println("  ✗ Invalid eigenvalue or derivative")
    end

    # Test 3: compute_b_cubic
    println("\n" * "─"^70)
    println("Test 3: compute_b_cubic")
    println("─"^70)

    println("Computing cubic coefficient b(V) for V=$(V_star)...")
    b_V = compute_b_cubic(V_star, p; b_default=0.5)
    @printf("  b(V) = %.4f\n", b_V)

    if b_V > 0
        println("  ✓ Valid cubic coefficient (b > 0)")
    else
        println("  ✗ Invalid cubic coefficient: $b_V")
    end

    # Test 4: welfare_loss
    println("\n" * "─"^70)
    println("Test 4: welfare_loss")
    println("─"^70)

    println("Computing welfare loss L(V, θ) for V=$(V_star), θ=$(theta_test)...")
    L_dec = welfare_loss(V_star, theta_test, p; regime=:dec)
    L_soc = welfare_loss(V_star, theta_test, p; regime=:soc)
    @printf("  L_dec(V,θ) = %.6f\n", L_dec)
    @printf("  L_soc(V,θ) = %.6f\n", L_soc)

    if isfinite(L_dec) && isfinite(L_soc)
        println("  ✓ Valid welfare loss computed")
    else
        println("  ✗ Invalid welfare loss: L_dec=$L_dec, L_soc=$L_soc")
    end

    # Test 5: Sweep over different V values
    println("\n" * "─"^70)
    println("Test 5: Welfare loss over V range")
    println("─"^70)

    println("Computing L(V) for different dispersion levels...")
    V_grid = [0.6, 0.7, 0.8, 0.9, 1.0]

    println("\n  V      |   L_dec   |   L_soc  |  κ*(V)   | λ₁₀(V)   | λ̇₁₀(V)")
    println("  " * "─"^74)

    for V in V_grid
        Ld = welfare_loss(V, theta_test, p; regime=:dec)
        Ls = welfare_loss(V, theta_test, p; regime=:soc)
        λ10, λ1d = compute_lambda1_and_derivative(V, p)
        κs = -λ10 / λ1d
        @printf("  %.2f   | %9.6f | %9.6f | %8.4f | %8.5f | %8.5f\n", V, Ld, Ls, κs, λ10, λ1d)
    end

    println("\n" * "="^70)
    println("All tests completed!")
    println("="^70)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
