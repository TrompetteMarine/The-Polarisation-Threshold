module Plotting

using Plots, FFTW, Statistics, LaTeXStrings
using ..Types: KappaSweepResult, Params

export plot_bifurcation, plot_vector_field, plot_orbit, plot_phase_diagram

# Common plotting settings
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

function create_base_plot(;kwargs...)
    plot(; merge(PLOT_DEFAULTS, Dict(kwargs...))...)
end

function plot_bifurcation(res::KappaSweepResult; κstar=nothing,
                         title="Bifurcation Diagram",
                         extend_factor=2.0, num_points=5000)
    # Create plot
    plt = plot(
        xlabel="Coupling strength κ",
        ylabel="Mean field amplitude |⟨u⟩|",
        title=title,
        legend=:topleft,
        background_color=:white,
        foreground_color=:black,
        grid=false,
        fontfamily="Computer Modern",
        size=(800, 500),
        dpi=300
    )
    
    # Plot data points
    scatter!(plt, res.κ, res.amp, 
            color=:blue, alpha=0.6, markersize=4,
            label="Simulation")
    
    if !isnothing(κstar)
        # Extended range for theoretical curves
        κ_min = 0.0
        κ_max = maximum(res.κ) * extend_factor
        κ_extended = range(κ_min, κ_max, length=num_points)
        
        # Pre-bifurcation stable branch
        κ_pre = κ_extended[κ_extended .≤ κstar]
        plot!(plt, κ_pre, zeros(length(κ_pre)),
              color=:black, linewidth=3.5, alpha=0.9, label="Stable")

        # Post-bifurcation branches with refined grid near κ* for smoothness
        κ_post = κ_extended[κ_extended .> κstar]
        # Compute stable branches with smooth square root
        stable_branches = @. sqrt(max(0, (κ_post - κstar) / κstar))

        # Unstable middle branch
        plot!(plt, κ_post, zeros(length(κ_post)),
              color=:black, linestyle=:dash, linewidth=2.5, alpha=0.6, label="Unstable")

        # Stable polarized branches with professional styling
        plot!(plt, κ_post, stable_branches,
              color=:black, linewidth=3.5, alpha=0.9, label=nothing)
        plot!(plt, κ_post, -stable_branches,
              color=:black, linewidth=3.5, alpha=0.9, label=nothing)
        
        # Critical point with professional marker
        scatter!(plt, [κstar], [0],
                color=:red, markersize=8, markerstrokewidth=2,
                markerstrokecolor=:darkred, label="κ*")
        vline!(plt, [κstar], color=:red, linestyle=:dash,
               linewidth=2.0, alpha=0.4, label=nothing)
    end
    
    return plt
end

function plot_vector_field(p::Params, κ::Float64;
                         xlims=(-3,3), ylims=(-3,3),
                         nx=20, ny=20)
    plt = plot(
        xlabel="u₁", ylabel="u₂",
        title="Phase space (κ = $(round(κ, digits=3)))",
        aspect_ratio=1,
        xlims=xlims, ylims=ylims,
        legend=:topright
    )
    
    # Create grid
    x = range(xlims..., length=nx)
    y = range(ylims..., length=ny)
    
    # Compute vector field
    for xi in x, yi in y
        u = [xi, yi]
        g = mean(u)
        du = [-p.λ * u[1] + κ * g,
              -p.λ * u[2] + κ * g]
        
        mag = sqrt(sum(abs2, du))
        if mag > 1e-10
            du_norm = du ./ mag
            # Plot small arrow
            quiver!([xi], [yi], 
                   quiver=([0.1*du_norm[1]], [0.1*du_norm[2]]),
                   color=:gray, alpha=0.5, label=nothing)
        end
    end
    
    # Add fixed points if κ > λ (approximate threshold)
    if κ > p.λ
        r = sqrt(max(0, (κ - p.λ) / κ))
        scatter!(plt, [0], [0], color=:black, markersize=6, label="Unstable")
        scatter!(plt, [r, -r], [0, 0], color=:red, markersize=6, label="Stable")
    else
        scatter!(plt, [0], [0], color=:red, markersize=6, label="Stable")
    end
    
    return plt
end

function plot_orbit(t::AbstractVector{<:Real}, u::AbstractVector{<:Real}; 
                   show_envelope::Bool=true, title="Time Evolution")
    plt = plot(
        xlabel="Time", ylabel="u", 
        title=title,
        legend=:topright,
        size=(800, 500)
    )
    
    # Main trajectory
    plot!(plt, t, u, color=:black, linewidth=1, label="Trajectory")
    
    if show_envelope && length(u) > 100
        # Compute envelope using Hilbert transform
        try
            analytical = hilbert(u)
            envelope = abs.(analytical)
            
            # Downsample for clarity
            stride = max(1, div(length(t), 1000))
            t_env = t[1:stride:end]
            env_pos = envelope[1:stride:end]
            
            # Add envelope
            plot!(plt, t_env, env_pos, color=:red, alpha=0.3, 
                 linewidth=2, label="Envelope")
            plot!(plt, t_env, -env_pos, color=:red, alpha=0.3, 
                 linewidth=2, label=nothing)
        catch e
            @warn "Failed to compute envelope" exception=e
        end
    end
    
    return plt
end

function plot_phase_diagram(u::AbstractVector{<:Real}, du::AbstractVector{<:Real}; 
                          title="Phase Space")
    plt = plot(
        xlabel="u", ylabel="du/dt",
        title=title,
        aspect_ratio=1,
        legend=:topright
    )
    
    scatter!(plt, u, du, color=:black, alpha=0.3, 
            markersize=2, markerstrokewidth=0, label=nothing)
    
    return plt
end

# Helper function for Hilbert transform
function hilbert(x::AbstractVector{<:Real})
    N = length(x)
    X = fft(x)
    H = zeros(ComplexF64, N)
    
    # Create Hilbert filter in frequency domain
    H[1] = X[1]  # DC component
    if N % 2 == 0
        H[div(N,2)+1] = X[div(N,2)+1]  # Nyquist for even N
        H[2:div(N,2)] = 2 .* X[2:div(N,2)]
    else
        H[2:div(N+1,2)] = 2 .* X[2:div(N+1,2)]
    end
    
    return ifft(H)
end

end # module
