module Validation

export safe_time_index, validate_simulation_params, safe_array_slice

"""
Safely convert time values to indices with bounds checking
"""
function safe_time_index(t::Float64, dt::Float64, max_steps::Int)::Int
    idx = floor(Int, t/dt)
    return clamp(idx, 1, max_steps)
end

"""
Validate and bound simulation parameters
"""
function validate_simulation_params(;T::Float64, dt::Float64, burn_in::Float64, N::Int)
    if T <= 0 || dt <= 0 || burn_in < 0
        throw(ArgumentError("Time parameters must be positive"))
    end
    
    total_steps = floor(Int, T/dt)
    burn_in_steps = floor(Int, burn_in/dt)
    
    if burn_in_steps >= total_steps
        throw(ArgumentError("burn_in must be less than T"))
    end
    
    return (
        total_steps = total_steps,
        burn_in_idx = min(burn_in_steps + 1, total_steps),
        dt = dt,
        array_size = N
    )
end

"""
Safely slice array with bounds checking
"""
function safe_array_slice(arr::AbstractArray, start_idx::Int, end_idx::Int)
    n = length(arr)
    safe_start = clamp(start_idx, 1, n)
    safe_end = clamp(end_idx, safe_start, n)
    return @view arr[safe_start:safe_end]
end

end # module
