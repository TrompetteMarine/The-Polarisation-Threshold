#!/usr/bin/env julia
using BeliefSim
using BeliefSim.OUResets
using BeliefSim.Types
using Plots
using Random

"""
Figure 2: leading odd eigenvalue λ₁(κ) for the linearised generator.
"""
function main()
    Random.seed!(25)

    p = Params(λ=0.65, σ=1.15, Θ=0.87, c0=0.50, hazard=StepHazard(0.5))
    κgrid = collect(range(0.0, 2.2, length=40))
    λ1 = similar(κgrid)

    for (i, κ) in enumerate(κgrid)
        λ1[i], _ = leading_odd_eigenvalue(p; κ=κ, L=5.0, M=301)
    end

    # Locate κ* where λ₁ crosses zero via linear interpolation
    κstar = NaN
    for i in 1:length(κgrid)-1
        if λ1[i] <= 0 && λ1[i+1] > 0
            w = -λ1[i] / (λ1[i+1] - λ1[i])
            κstar = κgrid[i] + w * (κgrid[i+1] - κgrid[i])
            break
        end
    end

    mkpath("figs")
    plt = plot(κgrid, λ1; xlabel="κ", ylabel="λ₁(κ)", legend=:topright,
               title="Leading odd eigenvalue", linewidth=3, color=:black,
               size=(700, 420), dpi=300, label="λ₁")
    hline!(plt, [0.0]; color=:gray, linestyle=:dash, label="Neutral")

    if !isnan(κstar)
        vline!(plt, [κstar]; color=:red, linestyle=:dash, label="κ*")
        scatter!(plt, [κstar], [0.0]; color=:red, label=nothing)
    end

    savefig(plt, "figs/fig2_eigen_kappa.pdf")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
