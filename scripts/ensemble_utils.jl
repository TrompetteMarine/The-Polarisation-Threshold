module EnsembleUtils

using Random
using Statistics
using Distributed

using BeliefSim
using BeliefSim.Types
using BeliefSim.Model: euler_maruyama_step!, reset_step!

export EnsembleResults, simulate_snapshots_stats, run_ensemble_simulation

struct EnsembleResults
    mean_trajectories::Matrix{Float64}      # n_ensemble x n_timepoints
    var_trajectories::Matrix{Float64}
    skew_trajectories::Matrix{Float64}
    kurt_trajectories::Matrix{Float64}
    snapshots::Vector{Vector{Vector{Float64}}}  # [n_ensemble][n_times][N]
    time_grid::Vector{Float64}
    snapshot_times::Vector{Float64}
    seeds::Vector{Int}
    n_agents::Int
end

function sample_skew_kurt(x::Vector{Float64})
    n = length(x)
    if n == 0
        return 0.0, 3.0
    end
    mu = mean(x)
    m2 = 0.0
    m3 = 0.0
    m4 = 0.0
    for xi in x
        d = xi - mu
        d2 = d * d
        m2 += d2
        m3 += d2 * d
        m4 += d2 * d2
    end
    m2 /= n
    if m2 <= 0.0
        return 0.0, 3.0
    end
    m3 /= n
    m4 /= n
    skew = m3 / (m2^(3 / 2))
    kurt = m4 / (m2^2)
    return skew, kurt
end

"""
    simulate_snapshots_stats(p::Params; kwargs...)

Run one simulation and collect snapshots, mean/variance trajectories, and moments.
"""
function simulate_snapshots_stats(
    p::Params;
    kappa::Float64,
    N::Int,
    T::Float64,
    dt::Float64,
    seed::Int,
    snapshot_times::Vector{Float64},
    mt_stride::Int,
    store_snapshots::Bool = true,
    track_moments::Bool = true,
    x_init::Union{Nothing, Vector{Float64}} = nothing,
)
    Random.seed!(seed)

    steps = Int(round(T / dt))
    time_grid = collect(0.0:dt:T)

    # Initial condition: OU stationary variance (no bias), unless overridden.
    u = if x_init === nothing
        randn(N) .* (p.σ / sqrt(2 * p.λ))
    else
        length(x_init) == N || error("x_init length $(length(x_init)) != N=$N")
        copy(x_init)
    end

    snapshot_indices = [clamp(Int(round(t / dt)) + 1, 1, length(time_grid)) for t in snapshot_times]
    snapshots = store_snapshots ? Vector{Vector{Float64}}(undef, length(snapshot_indices)) :
                Vector{Vector{Float64}}()

    next_snap = 1
    if store_snapshots && snapshot_indices[next_snap] == 1
        snapshots[next_snap] = copy(u)
        next_snap += 1
    end

    mt_times = Float64[0.0]
    mt_values = Float64[mean(u)]
    var_values = Float64[var(u)]
    if track_moments
        skew0, kurt0 = sample_skew_kurt(u)
        skew_values = Float64[skew0]
        kurt_values = Float64[kurt0]
    else
        skew_values = Float64[]
        kurt_values = Float64[]
    end

    for step in 1:steps
        gbar = mean(u)
        euler_maruyama_step!(u, kappa, gbar, p, dt)
        reset_step!(u, p, dt)

        idx = step + 1
        if store_snapshots && next_snap <= length(snapshot_indices) && idx == snapshot_indices[next_snap]
            snapshots[next_snap] = copy(u)
            next_snap += 1
        end

        if step % mt_stride == 0
            push!(mt_times, step * dt)
            push!(mt_values, mean(u))
            push!(var_values, var(u))
            if track_moments
                skew, kurt = sample_skew_kurt(u)
                push!(skew_values, skew)
                push!(kurt_values, kurt)
            end
        end
    end

    return snapshots, mt_times, mt_values, var_values, skew_values, kurt_values
end

"""
    run_ensemble_simulation(p::Params; kwargs...) -> EnsembleResults

Run multiple realizations and aggregate trajectories and snapshots.
"""
function run_ensemble_simulation(
    p::Params;
    kappa::Float64,
    n_ensemble::Int = 10,
    N::Int = 20000,
    T::Float64 = 400.0,
    dt::Float64 = 0.01,
    base_seed::Int = 42,
    snapshot_times::Vector{Float64} = [0.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 400.0],
    mt_stride::Int = 5,
    store_snapshots::Bool = true,
    track_moments::Bool = true,
    parallel::Bool = false,
    x_init::Union{Nothing, Vector{Float64}} = nothing,
)
    steps = Int(round(T / dt))
    n_timepoints = Int(floor(steps / mt_stride)) + 1

    mean_traj = Matrix{Float64}(undef, n_ensemble, n_timepoints)
    var_traj = Matrix{Float64}(undef, n_ensemble, n_timepoints)
    skew_traj = Matrix{Float64}(undef, n_ensemble, n_timepoints)
    kurt_traj = Matrix{Float64}(undef, n_ensemble, n_timepoints)

    seeds = [base_seed + (i - 1) * 1000 for i in 1:n_ensemble]
    snapshots = store_snapshots ? Vector{Vector{Vector{Float64}}}(undef, n_ensemble) :
                Vector{Vector{Vector{Float64}}}()

    time_grid_ref = collect(0:mt_stride:steps) .* dt

    if parallel && Threads.nthreads() > 1
        Threads.@threads for i in 1:n_ensemble
            snaps, mt_times, mt_values, var_values, skew_values, kurt_values =
                simulate_snapshots_stats(
                    p;
                    kappa=kappa,
                    N=N,
                    T=T,
                    dt=dt,
                    seed=seeds[i],
                    snapshot_times=snapshot_times,
                    mt_stride=mt_stride,
                    store_snapshots=store_snapshots,
                    track_moments=track_moments,
                    x_init=x_init,
                )

            mean_traj[i, :] .= mt_values
            var_traj[i, :] .= var_values
            if track_moments
                skew_traj[i, :] .= skew_values
                kurt_traj[i, :] .= kurt_values
            else
                skew_traj[i, :] .= NaN
                kurt_traj[i, :] .= NaN
            end

            if store_snapshots
                snapshots[i] = snaps
            end

        end
    elseif parallel && Distributed.nprocs() > 1
        results = Distributed.pmap(1:n_ensemble) do i
            simulate_snapshots_stats(
                p;
                kappa=kappa,
                N=N,
                T=T,
                dt=dt,
                seed=seeds[i],
                snapshot_times=snapshot_times,
                mt_stride=mt_stride,
                store_snapshots=store_snapshots,
                track_moments=track_moments,
                x_init=x_init,
            )
        end

        for i in 1:n_ensemble
            snaps, mt_times, mt_values, var_values, skew_values, kurt_values = results[i]

            mean_traj[i, :] .= mt_values
            var_traj[i, :] .= var_values
            if track_moments
                skew_traj[i, :] .= skew_values
                kurt_traj[i, :] .= kurt_values
            else
                skew_traj[i, :] .= NaN
                kurt_traj[i, :] .= NaN
            end

            if store_snapshots
                snapshots[i] = snaps
            end
        end
    else
        for i in 1:n_ensemble
            snaps, mt_times, mt_values, var_values, skew_values, kurt_values =
            simulate_snapshots_stats(
                p;
                kappa=kappa,
                N=N,
                T=T,
                dt=dt,
                seed=seeds[i],
                snapshot_times=snapshot_times,
                mt_stride=mt_stride,
                store_snapshots=store_snapshots,
                track_moments=track_moments,
                x_init=x_init,
            )

            mean_traj[i, :] .= mt_values
            var_traj[i, :] .= var_values
            if track_moments
                skew_traj[i, :] .= skew_values
                kurt_traj[i, :] .= kurt_values
            else
                skew_traj[i, :] .= NaN
                kurt_traj[i, :] .= NaN
            end

            if store_snapshots
                snapshots[i] = snaps
            end

        end
    end

    if length(time_grid_ref) != n_timepoints
        error("Mean trajectory length mismatch: expected $n_timepoints, got $(length(time_grid_ref))")
    end

    return EnsembleResults(
        mean_traj,
        var_traj,
        skew_traj,
        kurt_traj,
        snapshots,
        time_grid_ref,
        snapshot_times,
        seeds,
        N,
    )
end

end # module
