#!/usr/bin/env julia
using BeliefSim
using BeliefSim.Types
using BeliefSim.OUResets
using Plots
using Random

"""
Figure 4: decentralised vs social welfare across κ.
"""
function main()
    Random.seed!(2025)

    p = Params(λ=1.0, σ=0.8, Θ=2.0, c0=0.5, hazard=StepHazard(0.6))
    κgrid = collect(range(0.0, 2.2, length=28))

    welfare = compute_welfare_curves(p, κgrid; α=0.25, β=0.8, N=8_000,
                                     T=260.0, dt=0.01, seed=99)

    κ_dec_idx = argmax(welfare.W_dec)
    κ_soc_idx = argmax(welfare.W_soc)

    mkpath("figs")
    plt = plot(; xlabel="κ", ylabel="W(κ)", title="Welfare comparison",
               legend=:bottomright, size=(750, 420), dpi=300)
    plot!(plt, κgrid, welfare.W_dec; color=:black, linewidth=2.5, label="Decentralised")
    plot!(plt, κgrid, welfare.W_soc; color=:blue, linewidth=2.5, label="Planner")

    scatter!(plt, [κgrid[κ_dec_idx]], [welfare.W_dec[κ_dec_idx]];
             color=:black, markersize=6, label="κ^dec")
    scatter!(plt, [κgrid[κ_soc_idx]], [welfare.W_soc[κ_soc_idx]];
             color=:blue, markersize=6, label="κ^soc")

    # Shade externality wedge
    κ_lo = min(κgrid[κ_dec_idx], κgrid[κ_soc_idx])
    κ_hi = max(κgrid[κ_dec_idx], κgrid[κ_soc_idx])
    band = range(κ_lo, κ_hi; length=50)
    fill_upper = similar(band)
    fill_lower = similar(band)
    for (i, κ) in enumerate(band)
        fill_upper[i] = max(welfare.W_dec[κ_dec_idx], welfare.W_soc[κ_soc_idx])
        fill_lower[i] = min(welfare.W_dec[κ_dec_idx], welfare.W_soc[κ_soc_idx])
    end
    plot!(plt, band, fill_upper; fillrange=fill_lower, fillalpha=0.15,
          color=:purple, label="Externality wedge")

    savefig(plt, "figs/fig4_welfare.pdf")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
