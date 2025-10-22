include("../src/bifurcation/model_interface.jl")
include("../src/bifurcation/plotting_cairo.jl")
include("../src/bifurcation/simple_continuation.jl")

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src", "bifurcation"))

using LinearAlgebra
using Logging
using Statistics

# Check for CairoMakie
HAVE_MAKIE = try
    using CairoMakie
    true
catch
    false
end

try
    using .ModelInterface
    using .SimpleContinuation 
    HAVE_MAKIE && (using .PlottingCairo)
catch e
    @error "Failed to load required modules" exception=e
    println("\nMake sure bifurcation modules exist in src/bifurcation/")
    exit(1)
end

function main()
    @info "Starting Hopf bifurcation analysis (BifurcationKit-free version)"
    
    # Setup
    p_base = ModelInterface.default_params()
    u0 = zeros(2)
    κ_grid = collect(0.0:0.005:2.0)
    
    @info "Computing equilibrium continuation..." n_points=length(κ_grid)
    
    # Continuation
    branch = SimpleContinuation.continue_equilibria(
        ModelInterface.f,
        ModelInterface.jacobian,
        u0,
        p_base,
        κ_grid;
        tol=1e-10
    )
    
    # Detect Hopf bifurcations
    @info "Detecting Hopf bifurcations..."
    hopf_indices = Int[]
    
    for (i, λs) in enumerate(branch.eigenvalues)
        if SimpleContinuation.detect_hopf(λs; tol=1e-2)
            push!(hopf_indices, i)
            κ_hopf = branch.κ_values[i]
            @info "Found Hopf bifurcation" κ=κ_hopf index=i
        end
    end
    
    if isempty(hopf_indices)
        @info "No Hopf bifurcations detected in scanned range"
        return
    end
    
    # Analyze first Hopf point
    idx = hopf_indices[1]
    κ_hopf = branch.κ_values[idx]
    u_hopf = branch.equilibria[idx]
    
    @info "Analyzing limit cycle near Hopf point" κ=κ_hopf
    
    # Find limit cycle
    p_hopf = ModelInterface.kappa_set(p_base, κ_hopf)
    result = SimpleContinuation.find_limit_cycle(
        ModelInterface.f,
        u_hopf,
        p_hopf;
        perturbation=1e-2,
        tmax=500.0
    )
    
    if result.is_periodic
        @info "Found limit cycle" period=result.period
    else
        @warn "Could not detect periodic orbit"
    end
    
    # Plot if CairoMakie available
    if HAVE_MAKIE
        @info "Generating plots..."
        mkpath(joinpath(@__DIR__, "..", "figs"))
        
        PlottingCairo.set_theme_elegant!()
        
        # Bifurcation diagram
        fig1 = Figure(size=(900, 600))
        ax1 = Axis(fig1[1,1]; 
                  xlabel="κ", 
                  ylabel="u₁ (equilibrium)",
                  title="Equilibrium Branch")
        
        u1_eq = [u[1] for u in branch.equilibria]
        lines!(ax1, branch.κ_values, u1_eq; linewidth=2, color=:blue)
        
        # Mark Hopf points
        for idx in hopf_indices
            scatter!(ax1, [branch.κ_values[idx]], [branch.equilibria[idx][1]];
                    color=:red, markersize=10, label="Hopf")
        end
        
        outfile1 = joinpath(@__DIR__, "..", "figs", "hopf_bifurcation")
        PlottingCairo.savefig_smart(fig1, outfile1)
        
        # Phase portrait with limit cycle
        if result.is_periodic
            fig2 = Figure(size=(900, 600))
            ax2 = Axis(fig2[1,1]; 
                      xlabel="u₁", 
                      ylabel="u₂",
                      title="Limit Cycle at κ = $(round(κ_hopf, digits=3))")
            
            # Vector field
            PlottingCairo.phase_portrait!(ax2, ModelInterface.f, p_hopf; 
                                         density=30, alpha=0.25)
            
            # Limit cycle trajectory (plot last period)
            n_plot = min(500, length(result.sol.t))
            traj = reduce(hcat, result.sol.u[end-n_plot+1:end])
            lines!(ax2, traj[1, :], traj[2, :]; 
                  linewidth=3, color=:red, label="Limit Cycle")
            
            # Equilibrium
            scatter!(ax2, [u_hopf[1]], [u_hopf[2]]; 
                    color=:black, markersize=8, label="Unstable Eq.")
            
            axislegend(ax2, position=:lt)
            
            outfile2 = joinpath(@__DIR__, "..", "figs", "hopf_limit_cycle")
            PlottingCairo.savefig_smart(fig2, outfile2)
            
            @info "Plots saved to figs/"
        end
    else
        @info "CairoMakie not available, skipping plots"
    end
    
    @info "Analysis complete"
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end