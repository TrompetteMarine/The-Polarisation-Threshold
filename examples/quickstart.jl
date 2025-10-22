using BeliefSim
using BeliefSim.Types: Params, StepHazard
using BeliefSim.Stats: estimate_Vstar, critical_kappa, sweep_kappa
using Printf

# Parameters (feel free to tweak)
p = Params(λ=1.0, σ=0.8, Θ=2.0, c0=0.5, hazard=StepHazard(0.5))

println("Estimating V*...")
Vstar = estimate_Vstar(p; N=200, T=30.0, dt=0.01, burn_in=10.0, seed=123)

κcrit = critical_kappa(p; Vstar=Vstar) # canonical g ≈ λ
println("Estimated V* = ", Vstar)
println("Estimated κ* ≈ ", κcrit)

# Sweep κ for a pitchfork-like diagram
println("\nRunning κ sweep...")
κgrid = range(0.0, stop=2.0*κcrit, length=21) |> collect
res = sweep_kappa(p, κgrid; N=200, T=30.0, dt=0.01, burn_in=10.0, seed=42)

println("\nResults:")
for (κ, a, V) in zip(res.κ, res.amp, res.V)
    @printf("κ = %.4f   |ḡ| ≈ %.4f   V ≈ %.4f\n", κ, a, V)
end

println("\n✓ Quickstart complete!")