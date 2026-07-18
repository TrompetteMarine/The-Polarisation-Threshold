module Types

Base.@kwdef struct ModelParameters
    mean_reversion::Float64 = 1.0
    diffusion::Float64 = 1.0
    tolerance::Float64 = 1.0
    reset_fraction::Float64 = 0.5
    active_intensity::Float64 = 2.0
end

Base.@kwdef struct SimulationParameters
    particles::Int = 60_000
    blocks::Int = 60
    burn_steps::Int = 3_000
    burn_dt::Float64 = 0.01
    horizon::Float64 = 8.0
    dt::Float64 = 0.02
    delta::Float64 = 0.02
    seed::Int = 20_260_603
end

function validate(p::ModelParameters)
    p.mean_reversion > 0 || throw(ArgumentError("mean_reversion must be positive"))
    p.diffusion > 0 || throw(ArgumentError("diffusion must be positive"))
    p.tolerance > 0 || throw(ArgumentError("tolerance must be positive"))
    0 < p.reset_fraction < 1 || throw(ArgumentError("reset_fraction must lie in (0,1)"))
    p.active_intensity >= 0 || throw(ArgumentError("active_intensity must be non-negative"))
    p
end

function validate(s::SimulationParameters)
    s.particles > 0 || throw(ArgumentError("particles must be positive"))
    s.blocks > 1 || throw(ArgumentError("blocks must exceed one"))
    s.particles >= s.blocks || throw(ArgumentError("particles must be at least blocks"))
    minimum((s.burn_steps, s.burn_dt, s.horizon, s.dt, s.delta)) > 0 || throw(ArgumentError("time and finite-difference settings must be positive"))
    s
end

end
