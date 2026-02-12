#!/usr/bin/env julia
using BeliefSim
using BeliefSim.OUResets
using BeliefSim.Types
using BeliefSim.Plotting
using Plots
using Random

"""
Generate Figure 1: (a) single-agent OU-with-resets sample path with jump markers;
(b) stationary density comparison between pure OU and OU+resets.
"""
function main()
    Random.seed!(2025)

    # Baseline parameters
    p = Params(λ=0.85, σ=1.15, Θ=2.0, c0=0.80, hazard=StepHazard(10.0))
    T = 300.0
    dt = 0.01

    # Simulate sample paths
    t, x_resets, reset_idx = simulate_single_path(p; T=T, dt=dt, seed=2025, with_resets=true)

    mkpath("figs")

    # Panel A: path with reset markers
    plt_path = plot(; xlabel="t", ylabel="u_t", title="OU with stochastic resets",
                    legend=:topright, size=(900, 400), dpi=300)
    hline!(plt_path, [p.Θ, -p.Θ]; color=:gray, linestyle=:dash, label="±Θ")
    plot!(plt_path, t, x_resets; color=:black, label="u_t")
    scatter!(plt_path, t[reset_idx], x_resets[reset_idx]; color=:red, markersize=4,
             label="reset to c₀u")
    savefig(plt_path, "figs/fig1_ou_resets_path.pdf")

    # Panel B: stationary densities (use longer T for smoother estimate)
    centers_resets, dens_resets = stationary_density(p; T=600.0, dt=dt, burn_in=120.0,
                                                    nbins=90, seed=7, with_resets=true)
    centers_pure, dens_pure = stationary_density(p; T=600.0, dt=dt, burn_in=120.0,
                                                nbins=90, seed=7, with_resets=false)

    plt_density = plot(; xlabel="u", ylabel="ρ̄(u)", title="Stationary density",
                        legend=:topright, size=(700, 400), dpi=300)
    plot!(plt_density, centers_pure, dens_pure; color=:blue, linewidth=2,
          label="Pure OU")
    plot!(plt_density, centers_resets, dens_resets; color=:red, linewidth=2,
          label="OU + resets")
    vline!(plt_density, [p.Θ, -p.Θ]; color=:gray, linestyle=:dash, label=nothing)

    mkpath("figs")
    savefig(plt_density, "figs/fig1_ou_resets_density.pdf")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
