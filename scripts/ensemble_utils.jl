module EnsembleUtils

using Random
using Statistics
using Distributed

using BeliefSim
using BeliefSim.Types
using BeliefSim.Model: euler_maruyama_step!, reset_step!
using BeliefSim.Hazard: ν

const _CUDA_IMPORTED = let
    ok = false
    if Base.find_package("CUDA") !== nothing
        try
            @eval import CUDA
            ok = true
        catch
            ok = false
        end
    end
    ok
end

export EnsembleResults, simulate_snapshots_stats, run_ensemble_simulation, gpu_backend_ready

function gpu_backend_ready()
    _CUDA_IMPORTED || return false
    try
        return CUDA.functional()
    catch
        return false
    end
end

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

@inline function _normalize_device(device)::Symbol
    return Symbol(lowercase(String(device)))
end

function _store_snapshot_batch!(
    snapshots::Vector{Vector{Vector{Float64}}},
    u_host::Matrix{Float64},
    snapshot_idx::Int,
    row_lo::Int,
    row_hi::Int,
)
    b = row_hi - row_lo + 1
    @inbounds for j in 1:b
        snapshots[row_lo + j - 1][snapshot_idx] = vec(u_host[:, j])
    end
    return nothing
end

function _write_gpu_moment_column!(
    mean_traj::Matrix{Float64},
    var_traj::Matrix{Float64},
    skew_traj::Matrix{Float64},
    kurt_traj::Matrix{Float64},
    u,
    col::Int,
    row_lo::Int,
    row_hi::Int,
    track_moments::Bool,
)
    inv_n = 1.0 / size(u, 1)
    m1 = vec(Array(CUDA.sum(u; dims=1))) .* inv_n
    raw2 = vec(Array(CUDA.sum(u .* u; dims=1))) .* inv_n
    varv = max.(raw2 .- m1 .^ 2, 0.0)
    mean_traj[row_lo:row_hi, col] .= m1
    var_traj[row_lo:row_hi, col] .= varv

    if track_moments
        raw3 = vec(Array(CUDA.sum(u .* u .* u; dims=1))) .* inv_n
        raw4 = vec(Array(CUDA.sum(u .* u .* u .* u; dims=1))) .* inv_n
        m3 = raw3 .- 3.0 .* m1 .* raw2 .+ 2.0 .* (m1 .^ 3)
        m4 = raw4 .- 4.0 .* m1 .* raw3 .+ 6.0 .* (m1 .^ 2) .* raw2 .- 3.0 .* (m1 .^ 4)

        skew = similar(m1)
        kurt = similar(m1)
        @inbounds for i in eachindex(m1)
            if varv[i] <= 0.0 || !isfinite(varv[i])
                skew[i] = 0.0
                kurt[i] = 3.0
            else
                denom = varv[i]
                skew[i] = m3[i] / (denom^(3 / 2))
                kurt[i] = m4[i] / (denom^2)
            end
        end
        skew_traj[row_lo:row_hi, col] .= skew
        kurt_traj[row_lo:row_hi, col] .= kurt
    else
        skew_traj[row_lo:row_hi, col] .= NaN
        kurt_traj[row_lo:row_hi, col] .= NaN
    end

    return nothing
end

if _CUDA_IMPORTED

function _run_ensemble_simulation_gpu(
    p::Params;
    kappa::Float64,
    n_ensemble::Int,
    N::Int,
    T::Float64,
    dt::Float64,
    base_seed::Int,
    snapshot_times::Vector{Float64},
    mt_stride::Int,
    store_snapshots::Bool,
    track_moments::Bool,
    x_init::Union{Nothing, Vector{Float64}},
    gpu_batch_size::Int,
)
    gpu_backend_ready() || error("CUDA backend not functional on this system.")

    steps = Int(round(T / dt))
    n_timepoints = Int(floor(steps / mt_stride)) + 1
    time_grid_full = collect(0.0:dt:T)
    time_grid_ref = collect(0:mt_stride:steps) .* dt

    mean_traj = Matrix{Float64}(undef, n_ensemble, n_timepoints)
    var_traj = Matrix{Float64}(undef, n_ensemble, n_timepoints)
    skew_traj = Matrix{Float64}(undef, n_ensemble, n_timepoints)
    kurt_traj = Matrix{Float64}(undef, n_ensemble, n_timepoints)

    seeds = [base_seed + (i - 1) * 1000 for i in 1:n_ensemble]
    snapshots = store_snapshots ? [Vector{Vector{Float64}}(undef, length(snapshot_times)) for _ in 1:n_ensemble] :
                Vector{Vector{Vector{Float64}}}()

    snapshot_indices = [clamp(Int(round(t / dt)) + 1, 1, length(time_grid_full)) for t in snapshot_times]
    bsz = max(1, gpu_batch_size)

    λ = p.λ
    σ = p.σ
    Θ = p.Θ
    c0 = p.c0
    sigma_sqrt_dt = σ * sqrt(dt)
    init_scale = σ / sqrt(2 * λ)

    for row_lo in 1:bsz:n_ensemble
        row_hi = min(n_ensemble, row_lo + bsz - 1)
        b = row_hi - row_lo + 1

        CUDA.seed!(base_seed + (row_lo - 1) * 1000)
        u = if x_init === nothing
            CUDA.randn(Float64, N, b) .* init_scale
        else
            length(x_init) == N || error("x_init length $(length(x_init)) != N=$N")
            x0 = reshape(x_init, N, 1)
            CUDA.CuArray(repeat(x0, 1, b))
        end

        next_snap = 1
        if store_snapshots && !isempty(snapshot_indices) && snapshot_indices[next_snap] == 1
            _store_snapshot_batch!(snapshots, Array(u), next_snap, row_lo, row_hi)
            next_snap += 1
        end

        col = 1
        _write_gpu_moment_column!(mean_traj, var_traj, skew_traj, kurt_traj, u, col, row_lo, row_hi, track_moments)
        col += 1

        for step in 1:steps
            gbar = CUDA.sum(u; dims=1) ./ N
            ξ = CUDA.randn(Float64, size(u))
            @. u = u + ((-λ * u + kappa * gbar) * dt + sigma_sqrt_dt * ξ)

            rates = ν.(Ref(p.hazard), u, Θ)
            probs = @. 1.0 - exp(-rates * dt)
            draws = CUDA.rand(Float64, size(u))
            @. u = ifelse(draws < probs, c0 * u, u)

            idx = step + 1
            if store_snapshots && next_snap <= length(snapshot_indices) && idx == snapshot_indices[next_snap]
                _store_snapshot_batch!(snapshots, Array(u), next_snap, row_lo, row_hi)
                next_snap += 1
            end

            if step % mt_stride == 0
                _write_gpu_moment_column!(mean_traj, var_traj, skew_traj, kurt_traj, u, col, row_lo, row_hi, track_moments)
                col += 1
            end
        end
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

end # _CUDA_IMPORTED

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
    device::Symbol = :cpu,
    gpu_batch_size::Int = 16,
)
    dev = _normalize_device(device)
    if dev in (:gpu, :auto)
        if _CUDA_IMPORTED && gpu_backend_ready()
            return _run_ensemble_simulation_gpu(
                p;
                kappa=kappa,
                n_ensemble=n_ensemble,
                N=N,
                T=T,
                dt=dt,
                base_seed=base_seed,
                snapshot_times=snapshot_times,
                mt_stride=mt_stride,
                store_snapshots=store_snapshots,
                track_moments=track_moments,
                x_init=x_init,
                gpu_batch_size=gpu_batch_size,
            )
        elseif dev == :gpu
            @warn "GPU requested but CUDA backend is unavailable; falling back to CPU."
        end
    end

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
