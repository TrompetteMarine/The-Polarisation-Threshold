module Plotting

using Plots, FFTW, Statistics
using ..Types: KappaSweepResult, Params

# Export all plotting functions explicitly
export plot_bifurcation, plot_vector_field, plot_orbit, plot_phase_diagram

# Common plotting settings
const PLOT_SETTINGS = Dict(
    :fontfamily => "Computer Modern",
    :linewidth => 2,
    :grid => false,
    :background_color => :white,
    :foreground_color => :black,
    :legendfontsize => 8,
    :guidefontsize => 10,
    :tickfontsize => 9,
    :size => (800, 500),
    :dpi => 300
)

# Initialize plotting defaults without theme
const PLOT_DEFAULTS = Dict(
    :fontfamily => "Computer Modern",
    :linewidth => 2,
    :grid => false,
    :background_color => :white,
    :foreground_color => :black,
    :legendfontsize => 8,
    :guidefontsize => 10,
    :tickfontsize => 9,
    :size => (800, 500),
    :dpi => 300
)

# Apply defaults to each plot individually
function create_base_plot(;kwargs...)
    plot(; merge(PLOT_DEFAULTS, Dict(kwargs...))...)
end

function make_plot_attributes(;title="")
    return Dict(
        :xlabel => "Coupling strength κ",
        :ylabel => "Mean field amplitude |⟨u⟩|",
        :title => title,
        :legend => :topleft,
        :background_color => :white,
        :foreground_color => :black,
        :grid => false,
        :fontfamily => "Computer Modern",
        :size => (800, 500),
        :dpi => 300
    )
end

function plot_bifurcation(res::KappaSweepResult; κstar=nothing, title="Bifurcation Diagram",
                         extend_factor=2.0, num_points=2000)
    # Extended range for theoretical curves
    κ_min = 0.0  # Start from origin
    κ_max = maximum(res.κ) * extend_factor
    κ_extended = range(κ_min, κ_max, length=num_points)
    
    # Create plot with attributes
    plt = plot(;make_plot_attributes(title=title)...)
    
    if !isnothing(κstar)
        # Pre-bifurcation stable branch
        κ_pre = κ_extended[κ_extended .≤ κstar]
        plot!(plt, κ_pre, zeros(length(κ_pre)), 
              color=:black, linewidth=3, label="Stable")
        
        # Post-bifurcation branches
        κ_post = κ_extended[κ_extended .> κstar]
        stable_branches = @. sqrt((κ_post - κstar) / κstar)
        
        # Unstable middle branch
        plot!(plt, κ_post, zeros(length(κ_post)), 
              color=:black, linestyle=:dash, linewidth=2, label="Unstable")
        
        # Stable polarized branches
        plot!(plt, κ_post, stable_branches, 
              color=:black, linewidth=3, label=nothing)
        plot!(plt, κ_post, -stable_branches, 
              color=:black, linewidth=3, label=nothing)
        
        # Critical point and stability indicators
        scatter!(plt, [κstar], [0], 
                color=:red, markersize=6, label="κ*")
        vline!([κstar], color=:red, linestyle=:dash, 
               alpha=0.3, label=nothing)
        
        # Add stability labels
        annotate!(plt, [
            (κstar/2, 0.1, text("Stable consensus", 8, :blue)),
            (κstar*1.5, stable_branches[end]/2, text("Stable polarization", 8, :blue))
        ])
        
        # Add theoretical scaling curve
        κ_theory = range(κstar, κ_max, length=100)  # Use κ_max instead of κmax
        scaling = @. sqrt((κ_theory - κstar)/κstar)
        plot!(plt, κ_theory, scaling, 
              color=:red, linestyle=:dot, linewidth=1,
              label="∝√(κ-κ*)")
    end
    
    return plt
end

# Helper function to plot circle
function plot_circle!(plt, center::Tuple{Real,Real}, radius::Real; kwargs...)
    θ = range(0, 2π, length=200)
    x = center[1] .+ radius .* cos.(θ)
    y = center[2] .+ radius .* sin.(θ)
    plot!(plt, x, y; kwargs...)
end

function plot_vector_field(p::Params, κ::Float64;
                         xlims=(-3,3), ylims=(-3,3),
                         nx=40, ny=40)
    plt = plot(xlabel="u₁", ylabel="u₂",
              title="Phase space (κ/κ* = $(round(κ/p.λ, digits=2)))",
              aspect_ratio=1)
    
    # Add stability regions
    if κ > p.λ
        r = sqrt((κ - p.λ) / κ)
        
        # Stable manifold using parametric circle
        θ = range(0, 2π, length=200)
        circle_x = r .* cos.(θ)
        circle_y = r .* sin.(θ)
        
        # Fill stability regions
        plot!(plt, circle_x, circle_y,
              fillrange=0, fillalpha=0.1, fillcolor=:blue,
              linecolor=:red, linewidth=2, 
              label="Stable manifold")
        
        # Fixed points
        scatter!(plt, [0], [0], color=:black, markersize=6, label="Unstable")
        scatter!(plt, [r, -r], [0, 0], color=:red, markersize=6, label="Stable")
        
        # Add flow direction indicators
        arrow_length = r/5
        for θi in range(0, 2π, length=8)
            x, y = r*cos(θi), r*sin(θi)
            quiver!([x], [y], 
                   quiver=([arrow_length*cos(θi)], [arrow_length*sin(θi)]),
                   color=:red, alpha=0.5)
        end
    else
        # Consensus state
        scatter!(plt, [0], [0], color=:red, markersize=6, label="Stable")
        # Replace circle! with plot_circle!
        plot_circle!(plt, (0,0), p.λ/κ, 
                    linecolor=:blue, linestyle=:dash, 
                    label="Convergence region")
    end
    
    # Vector field
    x = range(xlims..., length=nx)
    y = range(ylims..., length=ny)
    u = zeros(2)
    
    # Compute and normalize vector field
    quiver_x = zeros(nx, ny)
    quiver_y = zeros(nx, ny)
    
    for (i,xi) in enumerate(x), (j,yi) in enumerate(y)
        u[1] = xi
        u[2] = yi
        g = mean(u)
        du = [-p.λ * u[1] + κ * g,
              -p.λ * u[2] + κ * g]
        mag = sqrt(sum(abs2, du))
        quiver_x[i,j] = du[1]/mag
        quiver_y[i,j] = du[2]/mag
    end
    
    quiver!(plt, repeat(x, 1, ny), repeat(y', nx, 1),
            quiver=(quiver_x, quiver_y),
            color=:gray, alpha=0.3, label=nothing)
    
    return plt
end

# Plot orbit diagram with envelope
function plot_orbit(t::AbstractVector{<:Real}, u::AbstractVector{<:Real}; 
                   show_envelope::Bool=true, title="Time Evolution")
    plt = plot(; PLOT_SETTINGS..., 
               xlabel="Time", ylabel="u", 
               title=title)
    
    # Main trajectory
    plot!(plt, t, u, color=:black, label="Trajectory")
    
    if show_envelope
        # Compute envelope using Hilbert transform
        analytical = hilbert(u)
        envelope = abs.(analytical)
        
        # Add envelope
        plot!(plt, t, envelope, color=:red, alpha=0.3, label="Envelope")
        plot!(plt, t, -envelope, color=:red, alpha=0.3, label=nothing)
    end
    
    return plt
end

# Plot phase space diagram
function plot_phase_diagram(u::AbstractVector{<:Real}, du::AbstractVector{<:Real}; 
                          title="Phase Space")
    plot(; PLOT_SETTINGS...,
         xlabel="u", ylabel="du/dt",
         title=title,
         aspect_ratio=1,
         legend=:topright)
    
    scatter!(u, du, color=:black, alpha=0.5, 
            markersize=2, label=nothing)
end

# Helper functions
function hilbert(x::AbstractVector{<:Real})
    N = length(x)
    X = fft(x)
    H = zeros(ComplexF64, N)
    freq = fftfreq(N)
    H[freq .> 0] .= 2 .* X[freq .> 0]
    H[freq .< 0] .= 0
    H[freq .== 0] .= X[freq .== 0]
    return ifft(H)
end

end # module
