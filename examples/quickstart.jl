using BeliefSim
using BeliefSim.Types
using BeliefSim.Stats
using BeliefSim.Simulate

# Parameters (feel free to tweak)
p = Params(λ=1.0, σ=0.8, Θ=2.0, c0=0.5, hazard=StepHazard(0.5))

Vstar = estimate_Vstar(p; N=20_000, T=300.0, dt=0.01, burn_in=100.0, seed=123)
κcrit = critical_kappa(p; Vstar=Vstar) # canonical g ≈ λ
println("Estimated V* = ", Vstar)
println("Estimated κ* ≈ ", κcrit)

# Sweep κ for a pitchfork-like diagram
κgrid = range(0.0, stop=2.0*κcrit, length=21) |> collect
res = BeliefSim.Stats.sweep_kappa(p, κgrid; N=20_000, T=300.0, dt=0.01, burn_in=100.0, seed=42)
for (κ, a, V) in zip(res.κ, res.amp, res.V)
    @printf("κ = %6.3f   |ḡ| ≈ %.4f   V ≈ %.4f\n", κ, a, V)
end
