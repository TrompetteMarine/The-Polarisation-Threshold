using BeliefSim
using BeliefSim.Types
using BeliefSim.Network
using Graphs

# Star graph demo: hubs tip first
N = 2000
g = star_graph(N)  # one hub, many leaves

p = Params(λ=1.0, σ=0.8, Θ=2.0, c0=0.5, hazard=StepHazard(0.5))

out = simulate_network(g, p; κ=0.3, T=200.0, dt=0.01, burn_in=50.0, seed=2025)

println("Final ḡ ≈ ", out.gbar, "   V ≈ ", out.V)
