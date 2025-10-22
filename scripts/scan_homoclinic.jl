include("../src/bifurcation/model_interface.jl")
include("../src/bifurcation/plotting_cairo.jl")
include("../src/bifurcation/simple_continuation.jl")

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src", "bifurcation"))

using Logging
using LinearAlgebra
using DifferentialEquations
using Statistics

# Check for CairoMakie
HAVE_MAKIE = try
    using CairoMakie
    true
catch
    @warn "CairoMakie not available, plots will be disabled"
    false
end

try
    using .ModelInterface
    using .SimpleContinuation
    HAVE_MAKIE && (using .PlottingCairo)
catch e
    @error "Failed to load bifurcation modules" exception=e
    exit(1)
end

"""
Result structure for homoclinic analysis
"""
struct HomoclinicResult
    found::Bool
    min_dist::Float64
    trajectory::Union{Nothing, Matrix{Float64}}  # 2×N matrix
    time::Union{Nothing, Vector{Float64}}
    return_time::Union{Nothing, Float64}  # Time when closest approach occurs
    saddle_point::Vector{Float64}
end

"""
Enhanced homoclinic orbit finder with trajectory storage
"""
function find_homoclinic(f::Function, jac::Function, u_saddle::Vector{Float64}, p;
                        tmax=2000.0, tol=1e-2, store_trajectory=false)
    
    # Get eigenvectors at saddle
    J = jac(u_saddle, p)
    λs, V = eigen(J)
    
    # Find unstable direction (positive real eigenvalue)
    unstable_idx = findmax(real.(λs))[2]
    
    if real(λs[unstable_idx]) <= 0
        return HomoclinicResult(false, Inf, nothing, nothing, nothing, u_saddle)
    end
    
    v_unstable = real.(V[:, unstable_idx])
    v_unstable ./= norm(v_unstable)
    
    # Try shooting in both directions along unstable manifold
    min_dist = Inf
    best_trajectory = nothing
    best_time = nothing
    return_time = nothing
    
    for sign in [1, -1]
        u0 = u_saddle .+ sign * 1e-5 .* v_unstable
        
        # Define ODE
        function ode!(du, u, p, t)
            F = f(u, p)
            du .= F
        end
        
        # Integrate
        saveat = store_trajectory ? 0.5 : nothing
        prob = ODEProblem(ode!, u0, (0.0, tmax), p)
        
        try
            sol = solve(prob, Tsit5(); 
                       reltol=1e-9, abstol=1e-9, 
                       saveat=saveat)
            
            # Find minimum distance to saddle
            min_dist_this = Inf
            closest_idx = 1
            
            for (i, u) in enumerate(sol.u)
                dist = norm(u .- u_saddle)
                if dist < min_dist_this
                    min_dist_this = dist
                    closest_idx = i
                end
            end
            
            # Update best result
            if min_dist_this < min_dist
                min_dist = min_dist_this
                return_time = sol.t[closest_idx]
                
                if store_trajectory
                    best_trajectory = reduce(hcat, sol.u)
                    best_time = sol.t
                end
            end
        catch e
            @warn "Integration failed" direction=sign exception=e
        end
    end
    
    found = min_dist < tol
    
    return HomoclinicResult(
        found, 
        min_dist, 
        best_trajectory, 
        best_time, 
        return_time,
        u_saddle
    )
end

"""
Find all equilibria in the system for heteroclinic detection
"""
function find_equilibria(f::Function, jac::Function, p;
                        search_radius=3.0, n_initial=20)
    equilibria = Vector{Float64}[]
    
    # Try multiple initial guesses
    for _ in 1:n_initial
        u0 = search_radius .* randn(2)
        
        # Newton solve
        u_eq = SimpleContinuation.newton_solve(f, jac, u0, p; tol=1e-10, maxiter=100)
        
        # Check if this is a new equilibrium
        is_new = true
        for u_exist in equilibria
            if norm(u_eq .- u_exist) < 1e-6
                is_new = false
                break
            end
        end
        
        # Verify it's actually an equilibrium
        if is_new && norm(f(u_eq, p)) < 1e-8
            push!(equilibria, u_eq)
        end
    end
    
    return equilibria
end

"""
Check for heteroclinic connections between different equilibria
"""
function find_heteroclinic(f::Function, jac::Function, 
                          u_from::Vector{Float64}, u_to::Vector{Float64}, p;
                          tmax=2000.0, tol=5e-2)
    
    # Get unstable direction from u_from
    J_from = jac(u_from, p)
    λs, V = eigen(J_from)
    unstable_idx = findmax(real.(λs))[2]
    
    if real(λs[unstable_idx]) <= 0
        return (found=false, min_dist=Inf, trajectory=nothing, time=nothing)
    end
    
    v_unstable = real.(V[:, unstable_idx])
    v_unstable ./= norm(v_unstable)
    
    # Shoot from u_from
    best_dist = Inf
    best_traj = nothing
    best_time = nothing
    
    for sign in [1, -1]
        u0 = u_from .+ sign * 1e-5 .* v_unstable
        
        function ode!(du, u, p, t)
            F = f(u, p)
            du .= F
        end
        
        prob = ODEProblem(ode!, u0, (0.0, tmax), p)
        
        try
            sol = solve(prob, Tsit5(); reltol=1e-9, abstol=1e-9, saveat=0.5)
            
            # Find minimum distance to target equilibrium
            min_dist_to_target = minimum([norm(u .- u_to) for u in sol.u])
            
            if min_dist_to_target < best_dist
                best_dist = min_dist_to_target
                best_traj = reduce(hcat, sol.u)
                best_time = sol.t
            end
        catch e
            @warn "Heteroclinic integration failed" exception=e
        end
    end
    
    return (
        found=(best_dist < tol),
        min_dist=best_dist,
        trajectory=best_traj,
        time=best_time
    )
end

"""
Main analysis and plotting function
"""
function main()
    @info "═══════════════════════════════════════════════════════"
    @info "  Homoclinic/Heteroclinic Bifurcation Analysis"
    @info "═══════════════════════════════════════════════════════"
    
    p_base = ModelInterface.default_params()
    u_saddle = zeros(2)
    
    # Scan parameters
    κ_range = 0.85:0.01:1.35
    @info "Scanning κ range" range=(first(κ_range), last(κ_range)) n_points=length(κ_range)
    
    # Storage for bifurcation diagram
    κ_values = Float64[]
    min_distances = Float64[]
    homoclinic_κ = Float64[]
    homoclinic_results = HomoclinicResult[]
    
    # Scan for homoclinic orbits
    @info "Phase 1: Scanning for homoclinic orbits..."
    for κ in κ_range
        p = ModelInterface.kappa_set(p_base, κ)
        
        # Store trajectory if this might be interesting
        store_traj = (κ >= 0.9 && κ <= 1.3)
        
        result = find_homoclinic(
            ModelInterface.f,
            ModelInterface.jacobian,
            u_saddle,
            p;
            tmax=3000.0,
            tol=1e-2,
            store_trajectory=store_traj
        )
        
        push!(κ_values, κ)
        push!(min_distances, result.min_dist)
        
        if result.found
            @info "Homoclinic orbit detected!" κ=κ min_distance=result.min_dist
            push!(homoclinic_κ, κ)
            push!(homoclinic_results, result)
        end
    end
    
    # Search for heteroclinic connections at selected κ values
    @info "Phase 2: Searching for heteroclinic connections..."
    heteroclinic_data = []
    
    for κ in [0.95, 1.0, 1.05, 1.1, 1.15, 1.2]
        p = ModelInterface.kappa_set(p_base, κ)
        
        # Find all equilibria
        equilibria = find_equilibria(ModelInterface.f, ModelInterface.jacobian, p)
        
        @info "Found equilibria at κ=$κ" n_equilibria=length(equilibria)
        
        # Check for heteroclinic connections
        for i in 1:length(equilibria)
            for j in 1:length(equilibria)
                if i != j
                    result = find_heteroclinic(
                        ModelInterface.f,
                        ModelInterface.jacobian,
                        equilibria[i],
                        equilibria[j],
                        p;
                        tol=5e-2
                    )
                    
                    if result.found
                        @info "Heteroclinic connection!" κ=κ from=i to=j distance=result.min_dist
                        push!(heteroclinic_data, (κ=κ, from=i, to=j, result=result))
                    end
                end
            end
        end
    end
    
    # Generate plots
    if HAVE_MAKIE && !isempty(homoclinic_results)
        @info "Phase 3: Generating visualizations..."
        mkpath(joinpath(@__DIR__, "..", "figs"))
        
        PlottingCairo.set_theme_elegant!()
        
        # Plot 1: Bifurcation diagram (minimum distance vs κ)
        fig1 = Figure(size=(1000, 600))
        ax1 = Axis(fig1[1,1];
                  xlabel="Coupling parameter κ",
                  ylabel="Minimum distance to saddle (log scale)",
                  title="Homoclinic Bifurcation Diagram",
                  yscale=log10)
        
        # Plot distance curve
        lines!(ax1, κ_values, min_distances .+ 1e-10; 
              linewidth=2, color=:blue, label="Min. distance")
        
        # Mark homoclinic orbits
        if !isempty(homoclinic_κ)
            scatter!(ax1, homoclinic_κ, 
                    [r.min_dist for r in homoclinic_results] .+ 1e-10;
                    color=:red, markersize=12, marker=:star5,
                    label="Homoclinic orbit")
        end
        
        # Add threshold line
        hlines!(ax1, [1e-2]; color=:red, linestyle=:dash, 
               linewidth=2, label="Detection threshold")
        
        axislegend(ax1, position=:rt)
        
        save(joinpath(@__DIR__, "..", "figs", "homoclinic_bifurcation.png"), fig1)
        @info "Saved: homoclinic_bifurcation.png"
        
        # Plot 2: Phase portraits with homoclinic orbits
        n_plots = min(4, length(homoclinic_results))
        if n_plots > 0
            fig2 = Figure(size=(1200, 800))
            
            for (idx, result) in enumerate(homoclinic_results[1:n_plots])
                κ = homoclinic_κ[idx]
                p = ModelInterface.kappa_set(p_base, κ)
                
                row = div(idx - 1, 2) + 1
                col = mod(idx - 1, 2) + 1
                
                ax = Axis(fig2[row, col];
                         xlabel="u₁",
                         ylabel="u₂",
                         title="κ = $(round(κ, digits=3))",
                         aspect=DataAspect())
                
                # Phase portrait
                PlottingCairo.phase_portrait!(ax, ModelInterface.f, p;
                                             lims=(-3, 3, -3, 3),
                                             density=30, alpha=0.2)
                
                # Plot homoclinic trajectory
                if !isnothing(result.trajectory)
                    traj = result.trajectory
                    lines!(ax, traj[1, :], traj[2, :];
                          linewidth=3, color=:red, label="Homoclinic orbit")
                end
                
                # Mark saddle point
                scatter!(ax, [result.saddle_point[1]], [result.saddle_point[2]];
                        color=:black, markersize=15, marker=:circle,
                        label="Saddle")
                
                # Add info text
                text!(ax, -2.5, 2.5;
                     text="d_min = $(round(result.min_dist, digits=4))",
                     fontsize=12)
                
                axislegend(ax, position=:rb)
            end
            
            save(joinpath(@__DIR__, "..", "figs", "homoclinic_orbits.png"), fig2)
            @info "Saved: homoclinic_orbits.png"
        end
        
        # Plot 3: Individual detailed homoclinic orbit (best example)
        if !isempty(homoclinic_results)
            best_idx = argmin([r.min_dist for r in homoclinic_results])
            best_result = homoclinic_results[best_idx]
            κ_best = homoclinic_κ[best_idx]
            p_best = ModelInterface.kappa_set(p_base, κ_best)
            
            fig3 = Figure(size=(1400, 600))
            
            # Left: Phase portrait
            ax_phase = Axis(fig3[1, 1];
                           xlabel="u₁",
                           ylabel="u₂",
                           title="Homoclinic Orbit (κ = $(round(κ_best, digits=4)))",
                           aspect=DataAspect())
            
            PlottingCairo.phase_portrait!(ax_phase, ModelInterface.f, p_best;
                                         lims=(-3, 3, -3, 3),
                                         density=40, alpha=0.25)
            
            if !isnothing(best_result.trajectory)
                traj = best_result.trajectory
                
                # Color-code trajectory by time
                n_seg = size(traj, 2) - 1
                colors = range(colorant"blue", colorant"red", length=n_seg)
                
                for i in 1:n_seg
                    lines!(ax_phase, 
                          traj[1, i:i+1], 
                          traj[2, i:i+1];
                          linewidth=3, 
                          color=colors[i])
                end
                
                # Mark start and end
                scatter!(ax_phase, [traj[1, 1]], [traj[2, 1]];
                        color=:blue, markersize=15, marker=:circle,
                        label="Start")
                scatter!(ax_phase, [traj[1, end]], [traj[2, end]];
                        color=:red, markersize=15, marker=:star5,
                        label="Return")
            end
            
            scatter!(ax_phase, [best_result.saddle_point[1]], 
                    [best_result.saddle_point[2]];
                    color=:black, markersize=20, marker=:xcross,
                    label="Saddle", markerspace=:pixel)
            
            axislegend(ax_phase, position=:lt)
            
            # Right: Time series
            ax_time = Axis(fig3[1, 2];
                          xlabel="Time",
                          ylabel="State variables",
                          title="Trajectory Evolution")
            
            if !isnothing(best_result.time)
                lines!(ax_time, best_result.time, best_result.trajectory[1, :];
                      linewidth=2, color=:blue, label="u₁")
                lines!(ax_time, best_result.time, best_result.trajectory[2, :];
                      linewidth=2, color=:red, label="u₂")
                
                # Mark return time
                if !isnothing(best_result.return_time)
                    vlines!(ax_time, [best_result.return_time];
                           color=:green, linestyle=:dash, linewidth=2,
                           label="Closest approach")
                end
                
                axislegend(ax_time, position=:rt)
            end
            
            save(joinpath(@__DIR__, "..", "figs", "homoclinic_detailed.png"), fig3)
            @info "Saved: homoclinic_detailed.png"
        end
        
        # Plot 4: Heteroclinic connections (if any found)
        if !isempty(heteroclinic_data)
            fig4 = Figure(size=(1200, 800))
            
            n_het = min(4, length(heteroclinic_data))
            for (idx, het) in enumerate(heteroclinic_data[1:n_het])
                row = div(idx - 1, 2) + 1
                col = mod(idx - 1, 2) + 1
                
                p = ModelInterface.kappa_set(p_base, het.κ)
                
                ax = Axis(fig4[row, col];
                         xlabel="u₁",
                         ylabel="u₂",
                         title="Heteroclinic (κ=$(round(het.κ, digits=3)))",
                         aspect=DataAspect())
                
                PlottingCairo.phase_portrait!(ax, ModelInterface.f, p;
                                             lims=(-4, 4, -4, 4),
                                             density=30, alpha=0.2)
                
                # Plot trajectory
                if !isnothing(het.result.trajectory)
                    traj = het.result.trajectory
                    lines!(ax, traj[1, :], traj[2, :];
                          linewidth=3, color=:purple, label="Heteroclinic")
                    
                    # Mark endpoints
                    scatter!(ax, [traj[1, 1]], [traj[2, 1]];
                            color=:blue, markersize=12, label="From")
                    scatter!(ax, [traj[1, end]], [traj[2, end]];
                            color=:red, markersize=12, label="To")
                end
                
                axislegend(ax, position=:rb)
            end
            
            save(joinpath(@__DIR__, "..", "figs", "heteroclinic_connections.png"), fig4)
            @info "Saved: heteroclinic_connections.png"
        end
        
        @info "All plots saved to figs/"
    elseif !HAVE_MAKIE
        @warn "CairoMakie not available - skipping visualization"
    else
        @warn "No homoclinic orbits found - skipping detailed plots"
    end
    
    # Summary
    @info "═══════════════════════════════════════════════════════"
    @info "  Analysis Summary"
    @info "═══════════════════════════════════════════════════════"
    @info "Scanned κ range" n_points=length(κ_values)
    @info "Homoclinic orbits found" count=length(homoclinic_κ)
    
    if !isempty(homoclinic_κ)
        @info "Homoclinic κ values" values=round.(homoclinic_κ, digits=4)
        best_idx = argmin([r.min_dist for r in homoclinic_results])
        @info "Best homoclinic" κ=homoclinic_κ[best_idx] distance=homoclinic_results[best_idx].min_dist
    end
    
    @info "Heteroclinic connections found" count=length(heteroclinic_data)
    
    @info "═══════════════════════════════════════════════════════"
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end