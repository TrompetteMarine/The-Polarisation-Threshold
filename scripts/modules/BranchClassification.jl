module BranchClassification

using Statistics

export late_window_indices,
       compute_branch_signs

"""
    late_window_indices(time_grid; late_window_fraction=0.2, min_window=50.0) -> Vector{Int}

Return indices for the late-time window used for branch classification.
"""
function late_window_indices(
    time_grid::Vector{Float64};
    late_window_fraction::Float64 = 0.2,
    min_window::Float64 = 50.0,
)::Vector{Int}
    T_sim = time_grid[end]
    window = min(T_sim, max(late_window_fraction * T_sim, min_window))
    idx = findall(t -> t >= T_sim - window, time_grid)
    if isempty(idx)
        return [length(time_grid)]
    end
    return idx
end

"""
    compute_branch_signs(mean_traj, time_grid; kwargs...) -> (signs, mean_late, decided, tau_used, decided_share, plus_share, minus_share, undecided_share)

Classify runs into ± branches using late-window mean and a threshold τ.
τ is computed from the median late-window standard deviation (NOT SE).
"""
function compute_branch_signs(
    mean_traj::Matrix{Float64},
    time_grid::Vector{Float64};
    late_window_fraction::Float64 = 0.2,
    min_window::Float64 = 50.0,
    tau::Union{Nothing, Float64} = nothing,
    tau_mult::Float64 = 3.0,
    min_decided_share::Float64 = 0.1,
    tau_floor::Float64 = 1e-6,
)
    idx = late_window_indices(time_grid; late_window_fraction=late_window_fraction, min_window=min_window)
    n_late = length(idx)
    mean_late = [mean(mean_traj[i, idx]) for i in 1:size(mean_traj, 1)]
    sd_late = n_late > 1 ? [std(mean_traj[i, idx]; corrected=true) for i in 1:size(mean_traj, 1)] :
              fill(0.0, size(mean_traj, 1))

    tau_scale = tau === nothing ? tau_mult * median(sd_late) : tau
    if !isfinite(tau_scale) || tau_scale <= 0.0
        tau_scale = tau_floor
    end
    tau_used = max(tau_floor, tau_scale)

    decided = abs.(mean_late) .>= tau_used
    decided_share = mean(decided)
    if decided_share < min_decided_share
        decided .= false
        decided_share = 0.0
    end

    signs = sign.(mean_late)
    for i in eachindex(signs)
        if !decided[i] || signs[i] == 0
            signs[i] = 1.0
        end
    end

    decided_count = count(decided)
    plus_share = count(i -> decided[i] && signs[i] > 0, eachindex(signs)) / max(1, decided_count)
    minus_share = count(i -> decided[i] && signs[i] < 0, eachindex(signs)) / max(1, decided_count)
    undecided_share = 1.0 - decided_share

    return Int.(signs), mean_late, decided, tau_used, decided_share, plus_share, minus_share, undecided_share
end

end # module
