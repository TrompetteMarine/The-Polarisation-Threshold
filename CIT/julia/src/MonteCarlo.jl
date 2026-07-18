module MonteCarlo

using Random, Statistics
using ..Types: ModelParameters, SimulationParameters, validate
using ..Model: one_step!

struct ResponseResult
    times::Vector{Float64}
    block_responses::Matrix{Float64}
    stationary_u::Vector{Float64}
    stationary_duration::Vector{Float64}
end

function simulate_stationary(p::ModelParameters, s::SimulationParameters)
    validate(p); validate(s)
    rng = MersenneTwister(s.seed)
    u = 1.3 .* randn(rng, s.particles)
    duration = zeros(s.particles)
    normal = similar(u); uniform = similar(u)
    for _ in 1:s.burn_steps
        randn!(rng, normal); rand!(rng, uniform)
        one_step!(u, duration, normal, uniform, s.burn_dt, p)
    end
    u, duration, rng
end

function paired_response(p::ModelParameters, s::SimulationParameters)
    u, duration, rng = simulate_stationary(p, s)
    block_size = div(s.particles, s.blocks)
    n = block_size * s.blocks
    u = copy(@view u[1:n]); duration = copy(@view duration[1:n])
    uplus = u .+ s.delta; uminus = u .- s.delta
    aplus = copy(duration); aminus = copy(duration)
    steps = round(Int, s.horizon / s.dt)
    block_responses = Matrix{Float64}(undef, steps + 1, s.blocks)
    diff = (uplus .- uminus) ./ (2 * s.delta)
    block_responses[1, :] .= vec(mean(reshape(diff, block_size, s.blocks); dims=1))
    normal = zeros(n); uniform = zeros(n)
    for step in 1:steps
        randn!(rng, normal); rand!(rng, uniform)
        one_step!(uplus, aplus, normal, uniform, s.dt, p)
        one_step!(uminus, aminus, normal, uniform, s.dt, p)
        @. diff = (uplus - uminus) / (2 * s.delta)
        block_responses[step + 1, :] .= vec(mean(reshape(diff, block_size, s.blocks); dims=1))
    end
    ResponseResult(collect(range(0.0, s.horizon; length=steps+1)), block_responses, u, duration)
end

end
