#!/usr/bin/env julia
using BeliefSim
using BeliefSim.Types
using BeliefSim.Stats
using BeliefSim.OUResets
using Plots
using Random

"""
Figure 3: bifurcation diagram for the stationary order parameter a*(κ).
"""
function main()
    Random.seed!(2025)

    p = Params(λ=1.0, σ=0.8, Θ=2.0, c0=0.5, hazard=StepHazard(0.6))
    κstar = critical_kappa(p; N=15_000, T=350.0, dt=0.01, burn_in=120.0, seed=11)
    κmax = 1.6 * κstar
    κgrid = collect(range(0.0, κmax, length=24))

    # Symmetric sweep using built-in routine (returns |mean|)
    res = sweep_kappa(p, κgrid; N=15_000, T=350.0, dt=0.01, burn_in=120.0, seed=18, max_abs=500.0)

    # Signed branches using biased initial conditions
    pos_branch = similar(κgrid)
    neg_branch = similar(κgrid)
    for (i, κ) in enumerate(κgrid)
        pos_branch[i] = order_parameter(p; κ=κ, bias=1e-2, N=8_000, T=220.0, dt=0.01, seed=100 + i, max_abs=250.0)
        neg_branch[i] = order_parameter(p; κ=κ, bias=-1e-2, N=8_000, T=220.0, dt=0.01, seed=300 + i, max_abs=250.0)
    end

    mkpath("figs")
    plt = plot(; xlabel="κ", ylabel="a*(κ)", title="Pitchfork bifurcation",
               legend=:topleft, size=(750, 450), dpi=300)
    mask_pos = isfinite.(pos_branch)
    mask_neg = isfinite.(neg_branch)
    mask_res = isfinite.(res.amp)

    plot!(plt, κgrid[mask_pos], pos_branch[mask_pos]; color=:red, linewidth=2.5, label="Stable branch +")
    plot!(plt, κgrid[mask_neg], neg_branch[mask_neg]; color=:blue, linewidth=2.5, label="Stable branch -")
    plot!(plt, κgrid, zeros(length(κgrid)); color=:black, linestyle=:dash, label="Central branch")
    scatter!(plt, κgrid[mask_res], res.amp[mask_res]; color=:black, markershape=:diamond, label="|mean(u)| sweep")
    vline!(plt, [κstar]; color=:gray, linestyle=:dash, label="κ*")

    savefig(plt, "figs/fig3_bifurcation.pdf")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
