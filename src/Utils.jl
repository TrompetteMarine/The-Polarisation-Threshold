module Utils

export safe_array_slice, validate_simulation_params, safe_time_range

"""
Safe array operations and validation functions
"""
struct SimulationParams
    total_steps::Int
    burn_in_idx::Int
    array_size::Int
    dt::Float64
end

function safe_array_slice(arr::AbstractArray, start_idx::Int, end_idx::Int)
    n = length(arr)
    safe_start = clamp(start_idx, 1, n)
    safe_end = clamp(end_idx, safe_start, n)
    return @view arr[safe_start:safe_end]
end

function validate_simulation_params(; T::Float64, dt::Float64, burn_in::Float64, N::Int)
    if T <= 0 || dt <= 0 || burn_in < 0
        throw(ArgumentError("Time parameters must be positive"))
    end
    
    total_steps = floor(Int, T/dt)
    burn_in_steps = floor(Int, burn_in/dt)
    
    if burn_in_steps >= total_steps
        throw(ArgumentError("burn_in must be less than T"))
    end
    
    return SimulationParams(
        total_steps,
        clamp(burn_in_steps + 1, 1, total_steps),
        N,
        dt
    )
end

function safe_time_range(start::Float64, stop::Float64, n::Int)
    return range(start, stop, length=n)
end

end # module
