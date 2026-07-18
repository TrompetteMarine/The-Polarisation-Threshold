module Model

using ..Types: ModelParameters

function one_step!(u::Vector{Float64}, duration::Vector{Float64}, normal::Vector{Float64}, uniform::Vector{Float64}, dt::Float64, p::ModelParameters)
    @. u += -p.mean_reversion * u * dt + p.diffusion * sqrt(dt) * normal
    @inbounds for i in eachindex(u)
        if abs(u[i]) > p.tolerance
            duration[i] += dt
            if uniform[i] < p.active_intensity * dt
                u[i] = sign(u[i]) * p.reset_fraction * p.tolerance
                duration[i] = 0.0
            end
        end
    end
    nothing
end

end
