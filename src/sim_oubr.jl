module SimOUBR

using Random
using Statistics

using BeliefSim.Types: Params
using BeliefSim.Model: euler_maruyama_step!

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

export boundary_reset_first!, boundary_reset_interp!, simulate_path_oubr, run_ensemble_oubr, gpu_backend_ready

function gpu_backend_ready()
    _CUDA_IMPORTED || return false
    try
        return CUDA.functional()
    catch
        return false
    end
end

@inline function boundary_reset_first!(u::Vector{Float64}, p::Params)
    theta = p.Θ
    c0 = p.c0
    @inbounds for i in eachindex(u)
        if abs(u[i]) >= theta
            s = sign(u[i])
            s = s == 0 ? 1.0 : s
            u[i] = c0 * s * theta
        end
    end
    return nothing
end

@inline function boundary_reset_interp!(u::Vector{Float64}, u_prev::Vector{Float64}, p::Params)
    theta = p.Θ
    c0 = p.c0
    @inbounds for i in eachindex(u)
        if abs(u[i]) >= theta
            if abs(u_prev[i]) < theta
                s = sign(u[i])
                s = s == 0 ? sign(u_prev[i]) : s
                s = s == 0 ? 1.0 : s
                u[i] = c0 * s * theta
            else
                s = sign(u[i])
                s = s == 0 ? 1.0 : s
                u[i] = c0 * s * theta
            end
        end
    end
    return nothing
end

function _init_state(p::Params, N::Int, init_mode::Symbol, epsilon::Float64, ordered_value::Float64)
    u = randn(N) .* (p.σ / sqrt(2 * p.λ))
    if init_mode == :odd
        n1 = div(N, 2)
        @inbounds u[1:n1] .+= epsilon
        @inbounds u[(n1 + 1):end] .-= epsilon
    elseif init_mode == :ordered_plus
        fill!(u, ordered_value)
    elseif init_mode == :ordered_minus
        fill!(u, -ordered_value)
    end
    return u
end

"""
    simulate_path_oubr(p; kappa, N, T, dt, seed, impl, mt_stride, init_mode, epsilon, ordered_value)

Run a single OU-BR path and return time grid, mean trajectory, mean-square trajectory,
variance trajectory, and final state.
"""
function simulate_path_oubr(
    p::Params;
    kappa::Float64,
    N::Int,
    T::Float64,
    dt::Float64,
    seed::Int,
    impl::Symbol,
    mt_stride::Int,
    init_mode::Symbol,
    epsilon::Float64,
    ordered_value::Float64,
)
    Random.seed!(seed)
    steps = Int(round(T / dt))
    n_time = Int(floor(steps / mt_stride)) + 1
    time_grid = collect(0:mt_stride:steps) .* dt

    u = _init_state(p, N, init_mode, epsilon, ordered_value)
    u_prev = similar(u)

    mean_traj = Vector{Float64}(undef, n_time)
    mean_sq_traj = Vector{Float64}(undef, n_time)
    var_traj = Vector{Float64}(undef, n_time)
    mean_traj[1] = mean(u)
    mean_sq_traj[1] = mean(u .^ 2)
    var_traj[1] = var(u)

    idx = 2
    for step in 1:steps
        gbar = mean(u)
        copyto!(u_prev, u)
        euler_maruyama_step!(u, kappa, gbar, p, dt)
        if impl == :first_crossing
            boundary_reset_first!(u, p)
        elseif impl == :interp
            boundary_reset_interp!(u, u_prev, p)
        else
            error("Unknown boundary reset impl: $impl")
        end

        if step % mt_stride == 0
            mean_traj[idx] = mean(u)
            mean_sq_traj[idx] = mean(u .^ 2)
            var_traj[idx] = var(u)
            idx += 1
        end
    end

    return (time_grid=time_grid, mean_traj=mean_traj, mean_sq_traj=mean_sq_traj,
            var_traj=var_traj, final_state=u)
end

"""
    run_ensemble_oubr(p; kappa, N, T, dt, n_ensemble, seed_base, impl, mt_stride,
                     init_mode, epsilon, ordered_value)

Run an ensemble of OU-BR simulations. Returns mean trajectories and mean-square
trajectories across runs, plus final states for diagnostics.
"""
function run_ensemble_oubr(
    p::Params;
    kappa::Float64,
    N::Int,
    T::Float64,
    dt::Float64,
    n_ensemble::Int,
    seed_base::Int,
    impl::Symbol,
    mt_stride::Int,
    init_mode::Symbol,
    epsilon::Float64,
    ordered_value::Float64,
)
    steps = Int(round(T / dt))
    n_time = Int(floor(steps / mt_stride)) + 1
    mean_traj = Matrix{Float64}(undef, n_ensemble, n_time)
    mean_sq_traj = Matrix{Float64}(undef, n_ensemble, n_time)
    var_traj = Matrix{Float64}(undef, n_ensemble, n_time)
    finals = Vector{Vector{Float64}}(undef, n_ensemble)
    time_grid_ref = collect(0:mt_stride:steps) .* dt

    for i in 1:n_ensemble
        seed = seed_base + (i - 1) * 1000
        res = simulate_path_oubr(
            p;
            kappa=kappa,
            N=N,
            T=T,
            dt=dt,
            seed=seed,
            impl=impl,
            mt_stride=mt_stride,
            init_mode=init_mode,
            epsilon=epsilon,
            ordered_value=ordered_value,
        )
        mean_traj[i, :] .= res.mean_traj
        mean_sq_traj[i, :] .= res.mean_sq_traj
        var_traj[i, :] .= res.var_traj
        finals[i] = res.final_state
    end

    return (time_grid=time_grid_ref, mean_traj=mean_traj, mean_sq_traj=mean_sq_traj,
            var_traj=var_traj, finals=finals)
end

if _CUDA_IMPORTED

function _write_gpu_batch_stats!(
    mean_traj::Matrix{Float64},
    mean_sq_traj::Matrix{Float64},
    var_traj::Matrix{Float64},
    u::CUDA.CuArray{Float64,2},
    row_lo::Int,
    row_hi::Int,
    col::Int,
)
    inv_n = 1.0 / size(u, 1)
    m = vec(Array(CUDA.sum(u; dims=1))) .* inv_n
    ms = vec(Array(CUDA.sum(u .* u; dims=1))) .* inv_n
    v = max.(ms .- m .^ 2, 0.0)
    mean_traj[row_lo:row_hi, col] .= m
    mean_sq_traj[row_lo:row_hi, col] .= ms
    var_traj[row_lo:row_hi, col] .= v
    return nothing
end

function _boundary_reset_first_gpu!(u::CUDA.CuArray{Float64,2}, theta::Float64, c0::Float64)
    s = sign.(u)
    s = ifelse.(s .== 0.0, 1.0, s)
    mask = abs.(u) .>= theta
    @. u = ifelse(mask, c0 * theta * s, u)
    return nothing
end

function _boundary_reset_interp_gpu!(
    u::CUDA.CuArray{Float64,2},
    u_prev::CUDA.CuArray{Float64,2},
    theta::Float64,
    c0::Float64,
)
    mask = abs.(u) .>= theta
    crossing = abs.(u_prev) .< theta
    su = sign.(u)
    sp = sign.(u_prev)
    s_cross = ifelse.(su .== 0.0, sp, su)
    s_cross = ifelse.(s_cross .== 0.0, 1.0, s_cross)
    s_nocross = ifelse.(su .== 0.0, 1.0, su)
    s = ifelse.(crossing, s_cross, s_nocross)
    @. u = ifelse(mask, c0 * theta * s, u)
    return nothing
end

"""
    run_ensemble_oubr_gpu(p; kwargs...)

Optional CUDA backend for OU-BR ensembles. The return schema matches `run_ensemble_oubr`.
If CUDA is unavailable at runtime, this function throws and callers should fallback to CPU.
"""
function run_ensemble_oubr_gpu(
    p::Params;
    kappa::Float64,
    N::Int,
    T::Float64,
    dt::Float64,
    n_ensemble::Int,
    seed_base::Int,
    impl::Symbol,
    mt_stride::Int,
    init_mode::Symbol,
    epsilon::Float64,
    ordered_value::Float64,
    batch_size::Int = 32,
)
    gpu_backend_ready() || error("CUDA backend not functional on this system.")
    impl in (:first_crossing, :interp) || error("Unknown boundary reset impl: $impl")

    steps = Int(round(T / dt))
    n_time = Int(floor(steps / mt_stride)) + 1
    mean_traj = Matrix{Float64}(undef, n_ensemble, n_time)
    mean_sq_traj = Matrix{Float64}(undef, n_ensemble, n_time)
    var_traj = Matrix{Float64}(undef, n_ensemble, n_time)
    finals = Vector{Vector{Float64}}(undef, n_ensemble)
    time_grid_ref = collect(0:mt_stride:steps) .* dt

    λ = p.λ
    σ = p.σ
    θ = p.Θ
    c0 = p.c0
    sigma_sqrt_dt = σ * sqrt(dt)
    init_scale = σ / sqrt(2 * λ)

    bsz = max(1, batch_size)
    for row_lo in 1:bsz:n_ensemble
        row_hi = min(n_ensemble, row_lo + bsz - 1)
        b = row_hi - row_lo + 1

        CUDA.seed!(seed_base + (row_lo - 1) * 1000)
        u = CUDA.randn(Float64, N, b) .* init_scale
        if init_mode == :odd
            n1 = div(N, 2)
            @views u[1:n1, :] .+= epsilon
            @views u[(n1 + 1):end, :] .-= epsilon
        elseif init_mode == :ordered_plus
            CUDA.fill!(u, ordered_value)
        elseif init_mode == :ordered_minus
            CUDA.fill!(u, -ordered_value)
        end
        u_prev = impl == :interp ? similar(u) : CUDA.zeros(Float64, 0, 0)

        time_col = 1
        _write_gpu_batch_stats!(mean_traj, mean_sq_traj, var_traj, u, row_lo, row_hi, time_col)
        time_col += 1

        for step in 1:steps
            if impl == :interp
                copyto!(u_prev, u)
            end

            gbar = CUDA.sum(u; dims=1) ./ N
            ξ = CUDA.randn(Float64, size(u))
            @. u = u + ((-λ * u + kappa * gbar) * dt + sigma_sqrt_dt * ξ)

            if impl == :first_crossing
                _boundary_reset_first_gpu!(u, θ, c0)
            else
                _boundary_reset_interp_gpu!(u, u_prev, θ, c0)
            end

            if step % mt_stride == 0
                _write_gpu_batch_stats!(mean_traj, mean_sq_traj, var_traj, u, row_lo, row_hi, time_col)
                time_col += 1
            end
        end

        u_host = Array(u)
        @inbounds for j in 1:b
            finals[row_lo + j - 1] = vec(u_host[:, j])
        end
    end

    return (time_grid=time_grid_ref, mean_traj=mean_traj, mean_sq_traj=mean_sq_traj,
            var_traj=var_traj, finals=finals)
end

end # _CUDA_IMPORTED

end # module
