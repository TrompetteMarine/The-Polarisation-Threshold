module EnsembleRunner

using Distributed
using Dates
using Statistics

using ..IOUtils
using ..EnsembleUtils: run_ensemble_simulation

export configure_parallelism,
       setup_distributed,
       run_ensemble_with_adaptive_T,
       get_adaptive_T,
       parallel_backend_label

"""
    configure_parallelism(; mode, target_threads, target_workers, project_root, script_dir) -> Bool

Configure parallel execution mode.
"""
function configure_parallelism(; mode::Symbol = :auto,
    target_threads::Int = 1,
    target_workers::Int = 1,
    project_root::String,
    script_dir::String)
    if mode == :off
        return false
    end
    if mode == :distributed
        setup_distributed(target_workers; project_root=project_root, script_dir=script_dir)
        return Distributed.nprocs() > 1
    end

    nthreads = Threads.nthreads()
    if nthreads > 1
        if nthreads < target_threads
            @warn "Julia has $nthreads threads; target is $target_threads. Consider JULIA_NUM_THREADS=$target_threads for faster runs."
        end
        return true
    end

    if mode == :threads
        @warn "Julia is single-threaded. For maximum speed, start with: JULIA_NUM_THREADS=$target_threads"
        return false
    end

    setup_distributed(target_workers; project_root=project_root, script_dir=script_dir)
    return Distributed.nprocs() > 1
end

"""
    setup_distributed(target_workers; project_root, script_dir)

Attach distributed workers and preload ensemble utilities.
"""
function setup_distributed(target_workers::Int; project_root::String, script_dir::String)
    if target_workers < 1
        return
    end
    if Distributed.nprocs() < target_workers + 1
        addprocs(target_workers + 1 - Distributed.nprocs())
    end
    for w in Distributed.workers()
        Distributed.remotecall_eval(Main, w, :(begin
            using Pkg
            Pkg.activate($project_root)
            using Random
            using Statistics
            using BeliefSim
            using BeliefSim.Types
            using BeliefSim.Model: euler_maruyama_step!, reset_step!
            include(joinpath($script_dir, "ensemble_utils.jl"))
            using .EnsembleUtils
        end))
    end
    return
end

"""
    parallel_backend_label(parallel_ensemble) -> String

Label for the active parallel backend.
"""
function parallel_backend_label(parallel_ensemble::Bool)::String
    if !parallel_ensemble
        return "serial"
    end
    if Threads.nthreads() > 1
        return "threads"
    end
    if Distributed.nprocs() > 1
        return "distributed"
    end
    return "serial"
end

"""
    get_adaptive_T(ratio, base_T; T_max=1200.0) -> Float64

Adaptive time horizon based on distance from critical ratio.
"""
function get_adaptive_T(ratio::Float64, base_T::Float64; T_max::Float64 = 1200.0)::Float64
    delta = max(abs(ratio - 1.0), 1e-4)
    scale = clamp(0.05 / delta, 1.0, 10.0)
    return min(T_max, base_T * scale)
end

"""
    run_ensemble_with_adaptive_T(p; kappa, base_T, ratio, kwargs...) -> EnsembleResults

Run ensemble simulation with adaptive T.
"""
function run_ensemble_with_adaptive_T(
    p;
    kappa::Float64,
    ratio::Float64,
    base_T::Float64,
    T_max::Float64,
    kwargs...,
)
    T_adapt = get_adaptive_T(ratio, base_T; T_max=T_max)
    return run_ensemble_simulation(p; kappa=kappa, T=T_adapt, kwargs...)
end

end
