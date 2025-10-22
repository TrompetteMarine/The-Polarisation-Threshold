

include("../src/bifurcation/model_interface.jl")
include("../src/bifurcation/plotting_cairo.jl")
include("../src/bifurcation/bifurcation_core.jl")

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src", "bifurcation"))

using LinearAlgebra
using Logging
using CairoMakie
using DifferentialEquations

try
    using .ModelInterface
    using .BifurcationCore
    using .PlottingCairo
catch e
    @error "Failed to load bifurcation modules" exception=e
    exit(1)
end

function main()
    mkpath(joinpath(@__DIR__, "..", "figs"))

    p = ModelInterface.default_params()
    u0 = zeros(2)

    κ_grid = collect(0.7:0.002:1.4)
    @info "Computing equilibrium continuation..."
    br = BifurcationCore.continue_equilibria(u0, p; κ_grid=κ_grid)

    hopf_idx = Int[]
    @info "Detecting Hopf bifurcations..."
    for i in eachindex(br.param)
        ueq = br.branch[i].u
        par = br.param[i]
        if BifurcationCore.detect_hopf(ueq, par)
            push!(hopf_idx, i)
        end
    end

    if !isempty(hopf_idx)
        @info "Found Hopf bifurcation at index $(hopf_idx[1])"
        idx = hopf_idx[1]
        po = BifurcationCore.continue_cycles_from_hopf(br, idx)
        
        PlottingCairo.set_theme_elegant!()
        fig = Figure(size=(900, 600))
        ax = Axis(fig[1,1]; xlabel="u₁", ylabel="u₂", 
                  title="Limit cycle from Hopf")
        
        par = br.param[idx]
        PlottingCairo.phase_portrait!(ax, ModelInterface.f, par; 
                                     density=45, alpha=0.25)
        
        traj = hcat(po.orbit...)
        lines!(ax, traj[1, :], traj[2, :]; linewidth=2, label="Periodic orbit")
        axislegend(ax, position=:lt)
        
        outfile = joinpath(@__DIR__, "..", "figs", "hopf_limit_cycle")
        PlottingCairo.savefig_smart(fig, outfile)
        @info "Saved limit cycle plot"
    else
        @info "No Hopf point detected on scanned branch."
    end
end

main()