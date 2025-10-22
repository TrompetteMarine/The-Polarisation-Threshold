module PlottingCairo

using CairoMakie
using LinearAlgebra
using Colors
using Statistics

export set_theme_elegant!, phase_portrait!, mark_points!, savefig_smart

function set_theme_elegant!()
    set_theme!(Theme(
        fontsize = 16,
        Axis = (xlabelsize=16, ylabelsize=16, xticklabelsize=12, yticklabelsize=12,
                xgridstyle=:dash, ygridstyle=:dash, xgridcolor=:gray80, ygridcolor=:gray80,
                spinewidth=1.2),
        Legend = (framevisible=false,)
    ))
end

function _as_matrix(points)
    if points === nothing
        return zeros(Float64, 0, 2)
    elseif isa(points, AbstractMatrix)
        size(points, 2) == 2 || throw(ArgumentError("Points must have two columns."))
        return Matrix{Float64}(points)
    elseif isa(points, AbstractVector)
        if length(points) == 0
            return zeros(Float64, 0, 2)
        elseif length(points) == 2
            return reshape(Float64.(points), 1, 2)
        elseif eltype(points) <: AbstractVector
            mat = hcat(points...)
            size(mat, 1) == 2 || throw(ArgumentError("Nested points must be 2-vectors."))
            return permutedims(Float64.(mat))
        end
    end
    throw(ArgumentError("Unsupported point container."))
end

function phase_portrait!(ax, f, p; lims = (-3,3,-3,3), density=45, alpha=0.35)
    xs = range(lims[1], lims[2], length=density)
    ys = range(lims[3], lims[4], length=density)
    X = [x for y in ys, x in xs]
    Y = [y for y in ys, x in xs]
    U = similar(X, Float64)
    V = similar(Y, Float64)
    for i in eachindex(ys), j in eachindex(xs)
        u = [X[i,j], Y[i,j]]
        du = f(u, p)
        n = max(norm(du), 1e-12)
        U[i,j] = du[1] / n
        V[i,j] = du[2] / n
    end
    quiver!(ax, X, Y, U, V; arrowsize=6, linewidth=1, color=:gray40, alpha=alpha)
    xlims!(ax, lims[1], lims[2])
    ylims!(ax, lims[3], lims[4])
    ax.aspect = DataAspect()
    return ax
end

function mark_points!(ax; stable=nothing, unstable=nothing)
    spts = _as_matrix(stable)
    if size(spts, 1) > 0
        scatter!(ax, spts[:,1], spts[:,2]; color=:red, markersize=14, label="Stable")
    end
    upts = _as_matrix(unstable)
    if size(upts, 1) > 0
        scatter!(ax, upts[:,1], upts[:,2]; color=:black, markersize=10, label="Unstable")
    end
    return ax
end

function savefig_smart(fig, path::AbstractString)
    dir = dirname(path)
    dir == "" && (dir = ".")
    mkpath(dir)
    save(path * ".png", fig; px_per_unit=2)
    save(path * ".pdf", fig)
end

end # module
