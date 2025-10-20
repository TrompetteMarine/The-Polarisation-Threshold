using BeliefSim
using BeliefSim.Types
using BeliefSim.Stats
using BeliefSim.Simulate

# Parameters close to manuscript defaults
p = Params(λ=1.0, σ=1.0, Θ=2.5, c0=0.6, hazard=StepHazard(0.5))

Vstar = estimate_Vstar(p; N=30_000, T=400.0, dt=0.01, burn_in=120.0, seed=1)
κcrit = critical_kappa(p; Vstar=Vstar)
println("V* ≈ ", Vstar, "  ⇒ κ* ≈ ", κcrit)

κgrid = collect(range(0.0, stop=2.2*κcrit, length=25))
res = sweep_kappa(p, κgrid; N=30_000, T=350.0, dt=0.01, burn_in=120.0, seed=2)

open("bifurcation.tsv", "w") do io
    println(io, "kappa	amp	variance")
    for (κ, a, V) in zip(res.κ, res.amp, res.V)
        println(io, string(κ, '\t', a, '\t', V))
    end
end
println("Wrote bifurcation.tsv")
