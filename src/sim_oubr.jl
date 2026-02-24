module SimOUBR

using Random
using Statistics

using BeliefSim.Types: Params
using BeliefSim.Model: euler_maruyama_step!

export boundary_reset_first!, boundary_reset_interp!, simulate_path_oubr, run_ensemble_oubr

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

end # module
