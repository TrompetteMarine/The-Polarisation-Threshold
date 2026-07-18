module GeneratorCheck

using LinearAlgebra, SparseArrays
using ..Types: ModelParameters, validate

struct GeneratorResult
    grid::Vector{Float64}
    stationary_mass::Vector{Float64}
    response_direction::Vector{Float64}
    poisson_solution::Vector{Float64}
    susceptibility::Float64
    threshold::Float64
end

function build_generator(p::ModelParameters; grid_points::Int=801, domain::Float64=5.0)
    validate(p)
    grid_points >= 51 && isodd(grid_points) || throw(ArgumentError("grid_points must be odd and at least 51"))
    grid = collect(range(-domain, domain; length=grid_points))
    dx = grid[2] - grid[1]
    diffusion = 0.5 * p.diffusion^2
    Q = spzeros(Float64, grid_points, grid_points)
    for (i,u) in enumerate(grid)
        drift = -p.mean_reversion * u
        rate_plus = diffusion / dx^2 + max(drift, 0.0) / dx
        rate_minus = diffusion / dx^2 + max(-drift, 0.0) / dx
        Q[i, i < grid_points ? i+1 : i-1] += rate_plus
        Q[i, i > 1 ? i-1 : i+1] += rate_minus
        if abs(u) > p.tolerance
            target = sign(u) * p.reset_fraction * p.tolerance
            j = argmin(abs.(grid .- target))
            j != i && (Q[i,j] += p.active_intensity)
        end
        Q[i,i] = -sum(Q[i,:])
    end
    grid, Q
end

function stationary_mass(Q::SparseMatrixCSC{Float64,Int})
    n = size(Q,1)
    A = Matrix(transpose(Q)); b = zeros(n)
    A[end,:] .= 1.0; b[end] = 1.0
    mass = max.(A \ b, 0.0)
    mass ./ sum(mass)
end

function translation_direction(grid, mass)
    dx = grid[2] - grid[1]
    direction = similar(mass)
    direction[2:end-1] .= -(mass[3:end] .- mass[1:end-2]) ./ (2 * dx)
    direction[1] = -(mass[2]-mass[1]) / dx
    direction[end] = -(mass[end]-mass[end-1]) / dx
    direction .-= mass .* sum(direction)
    normalization = dot(grid, direction)
    abs(normalization) > 1e-12 || error("response direction normalization failed")
    direction ./ normalization
end

function solve_poisson(Q, direction)
    n = size(Q,1)
    A = Matrix(-transpose(Q)); b = copy(direction)
    A[end,:] .= 1.0; b[end] = 0.0
    A \ b
end

function generator_cross_check(p::ModelParameters; grid_points::Int=801, domain::Float64=5.0)
    grid, Q = build_generator(p; grid_points, domain)
    mass = stationary_mass(Q)
    direction = translation_direction(grid, mass)
    poisson = solve_poisson(Q, direction)
    susceptibility = dot(grid, poisson)
    GeneratorResult(grid, mass, direction, poisson, susceptibility, inv(susceptibility))
end

end
