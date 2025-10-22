# Check if required packages are available
include("../src/bifurcation/model_interface.jl")
include("../src/bifurcation/plotting_cairo.jl")
try
    using CairoMakie
catch e
    println("❌ Error: CairoMakie is required for this script.")
    println("Install it with:")
    println("  julia --project=. -e 'using Pkg; Pkg.add(\"CairoMakie\")'")
    exit(1)
end

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src", "bifurcation"))

using ArgParse
using LinearAlgebra

try
    using .ModelInterface
    using .PlottingCairo
catch e
    @error "Failed to load bifurcation modules" exception=e
    println("\nMake sure you've run: ./scripts/setup_environment.sh")
    exit(1)
end

function main()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--kappas"
            help = "comma-separated κ values"
            arg_type = String
            default = "0.79,0.81,0.99,1.01,1.19,1.21"
        "--lims"
            help = "xmin,xmax,ymin,ymax"
            arg_type = String
            default = "-3,3,-3,3"
    end
    args = parse_args(s)
    
    κs = parse.(Float64, split(args["kappas"], ","))
    limv = parse.(Float64, split(args["lims"], ","))
    lims = (limv[1], limv[2], limv[3], limv[4])

    PlottingCairo.set_theme_elegant!()
    mkpath(joinpath(@__DIR__, "..", "figs"))
    p0 = ModelInterface.default_params()

    for κ in κs
        p = ModelInterface.kappa_set(p0, κ)
        fig = Figure(size=(900, 600))
        ratio = round(κ / getproperty(p, :kstar); digits=2)
        ax = Axis(fig[1,1]; xlabel="u₁", ylabel="u₂", 
                  title="Phase space (κ/κ* = $ratio)")
        PlottingCairo.phase_portrait!(ax, ModelInterface.f, p; 
                                     lims=lims, density=45, alpha=0.35)

        β = hasproperty(p, :beta) ? getproperty(p, :beta) : 1.0
        κcrit = getproperty(p, :kstar)
        
        stable_pts = nothing
        unstable_pts = nothing
        if κ > κcrit + 1e-8
            amp = sqrt(max(0, (κ - κcrit) / max(β, eps())))
            stable_pts = [amp amp; -amp -amp]
            unstable_pts = [0.0 0.0]
        else
            stable_pts = [0.0 0.0]
            unstable_pts = zeros(0, 2)
        end
        
        PlottingCairo.mark_points!(ax; stable=stable_pts, unstable=unstable_pts)
        axislegend(ax, position=:lt)
        
        outfile = joinpath(@__DIR__, "..", "figs", 
                          "phase_kappa_$(round(κ; digits=3))")
        PlottingCairo.savefig_smart(fig, outfile)
    end
    
    @info "Phase portraits saved to figs/"
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end