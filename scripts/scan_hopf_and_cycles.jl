push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using LinearAlgebra
using Logging
using CairoMakie
using DifferentialEquations
using ModelInterface
using Bifurcation
using Plotting

mkpath(joinpath(@__DIR__, "..", "figs"))

p = ModelInterface.default_params()
u0 = zeros(2)

κ_grid = collect(0.7:0.002:1.4)
br = Bifurcation.continue_equilibria(u0, p; κ_grid=κ_grid)

hopf_idx = Int[]
for i in eachindex(br.param)
    ueq = br.branch[i].u
    par = br.param[i]
    if Bifurcation.detect_hopf(ueq, par)
        push!(hopf_idx, i)
    end
end

if !isempty(hopf_idx)
    idx = hopf_idx[1]
    po = Bifurcation.continue_cycles_from_hopf(br, idx)
    Plotting.set_theme_elegant!()
    fig = Figure(resolution=(900, 600))
    ax = Axis(fig[1,1]; xlabel="u₁", ylabel="u₂", title="Limit cycle from Hopf")
    par = br.param[idx]
    Plotting.phase_portrait!(ax, ModelInterface.f, par; density=45, alpha=0.25)
    traj = hcat(po.orbit...)
    lines!(ax, traj[1, :], traj[2, :]; linewidth=2, label="Periodic orbit")
    axislegend(ax, position=:lt)
    outfile = joinpath(@__DIR__, "..", "figs", "hopf_limit_cycle")
    Plotting.savefig_smart(fig, outfile)
else
    @info "No Hopf point detected on scanned branch."
end
