module SweepPipeline

using Statistics
using DataFrames

using ..EnsembleUtils: run_ensemble_simulation
using ..BranchClassification: compute_branch_signs
using ..OrderParameter: compute_m_abs_star, correct_amplitude
using ..EnsembleRunner: get_adaptive_T

export run_forward_sweep,
       run_backward_sweep,
       build_dense_sweep_ratios,
       build_scaling_grid,
       build_scaling_baseline_grid,
       run_scaling_grid,
       run_scaling_point

"""
    build_dense_sweep_ratios(kappa_star_B, kappa_star_ref, sweep_ratios, sweep_ratio_cap)

Construct a near-critical dense sweep grid.
"""
function build_dense_sweep_ratios(
    kappa_star_B::Float64,
    kappa_star_ref::Float64,
    sweep_ratios::Vector{Float64},
    sweep_ratio_cap::Float64,
)
    deltas_above = [1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2, 3.2e-2, 6.4e-2, 1e-1, 0.15, 0.20]
    deltas_below = [-1e-1, -6.4e-2, -3.2e-2, -1.6e-2, -8e-3, -4e-3]
    kappas_near = kappa_star_B .* (1 .+ vcat(deltas_below, deltas_above))
    ratios_near = kappas_near ./ kappa_star_ref

    ratios = vcat(sweep_ratios, ratios_near)
    ratios = filter(r -> r > 0.0 && r <= sweep_ratio_cap, ratios)
    return sort(unique(ratios))
end

"""
    build_scaling_grid(kappa_star; delta_min, delta_max, n_target)

Geometric grid above kappa*.
"""
function build_scaling_grid(
    kappa_star::Float64;
    delta_min::Float64,
    delta_max::Float64,
    n_target::Int,
)
    deltas = exp.(range(log(delta_min), log(delta_max), length=n_target))
    kappas = kappa_star .* (1 .+ deltas)
    return kappas, deltas
end

"""
    build_scaling_baseline_grid(kappa_star; delta_min=0.05, delta_max=0.25, n_sub=24)

Subcritical baseline grid for m0 estimation.
"""
function build_scaling_baseline_grid(
    kappa_star::Float64;
    delta_min::Float64 = 0.05,
    delta_max::Float64 = 0.25,
    n_sub::Int = 24,
)
    deltas = -exp.(range(log(delta_min), log(delta_max), length=n_sub))
    kappas = kappa_star .* (1 .+ deltas)
    return kappas, deltas
end

"""
    run_scaling_point(p, kappa; ...)

Run a single kappa point for scaling regression.
"""
function run_scaling_point(
    p,
    kappa::Float64;
    kappa_star::Float64,
    ratio::Float64,
    N::Int,
    T::Float64,
    dt::Float64,
    n_ensemble::Int,
    base_seed::Int,
    snapshot_times::Vector{Float64},
    mt_stride::Int,
    parallel::Bool,
    m0::Float64 = 0.0,
    m0_se::Float64 = NaN,
)
    res = run_ensemble_simulation(
        p;
        kappa=kappa,
        n_ensemble=n_ensemble,
        N=N,
        T=T,
        dt=dt,
        burn_in=0.0,
        base_seed=base_seed,
        snapshot_times=snapshot_times,
        mt_stride=mt_stride,
        parallel=parallel,
        store_snapshots=false,
    )

    mbar, m_abs_star, m_abs_ci, _ = compute_m_abs_star(res.mean_trajectories, res.time_grid)
    m_abs_se = n_ensemble > 1 ? m_abs_ci / 1.96 : NaN
    m_corr, m_corr_se = correct_amplitude(m_abs_star, m_abs_se, m0, m0_se)

    signs, mean_late, decided, tau_used, decided_share, plus_share, minus_share, undecided_share =
        compute_branch_signs(res.mean_trajectories, res.time_grid)

    delta_abs = kappa - kappa_star
    delta_rel = delta_abs / kappa_star
    imbalance = plus_share - minus_share

    return (
        kappa = kappa,
        delta_abs = delta_abs,
        delta_rel = delta_rel,
        m_abs_star = m_abs_star,
        m_abs_ci = m_abs_ci,
        m_abs_se = m_abs_se,
        m_corr = m_corr,
        m_corr_se = m_corr_se,
        converged = true,
        decided_share = decided_share,
        p_plus = plus_share,
        p_minus = minus_share,
        imbalance = imbalance,
        n_ensemble = n_ensemble,
        tau_used = tau_used,
        ratio = ratio,
    )
end

"""
    run_scaling_grid(p, kappas; ...)

Run scaling grid and return a DataFrame.
"""
function run_scaling_grid(
    p,
    kappas::Vector{Float64};
    kappa_star::Float64,
    N::Int,
    T::Float64,
    dt::Float64,
    n_ensemble::Int,
    base_seed::Int,
    snapshot_times::Vector{Float64},
    mt_stride::Int,
    parallel::Bool,
    m0::Float64 = 0.0,
    m0_se::Float64 = NaN,
)
    rows = NamedTuple[]
    for (i, κ) in enumerate(kappas)
        ratio = κ / kappa_star
        row = run_scaling_point(
            p, κ;
            kappa_star=kappa_star,
            ratio=ratio,
            N=N,
            T=T,
            dt=dt,
            n_ensemble=n_ensemble,
            base_seed=base_seed + i * 101,
            snapshot_times=snapshot_times,
            mt_stride=mt_stride,
            parallel=parallel,
            m0=m0,
            m0_se=m0_se,
        )
        push!(rows, row)
    end
    return DataFrame(rows)
end

"""
    run_forward_sweep(p, ratios; ...)

Run a forward kappa sweep.
"""
function run_forward_sweep(
    p,
    ratios::Vector{Float64};
    kappa_star::Float64,
    N::Int,
    T::Float64,
    dt::Float64,
    n_ensemble::Int,
    base_seed::Int,
    snapshot_times::Vector{Float64},
    mt_stride::Int,
    parallel::Bool,
    T_max::Float64,
)
    rows = NamedTuple[]
    for (i, ratio) in enumerate(ratios)
        kappa = ratio * kappa_star
        T_adapt = get_adaptive_T(ratio, T; T_max=T_max)
        res = run_ensemble_simulation(
            p;
            kappa=kappa,
            n_ensemble=n_ensemble,
            N=N,
            T=T_adapt,
            dt=dt,
            burn_in=0.0,
            base_seed=base_seed + i * 123,
            snapshot_times=snapshot_times,
            mt_stride=mt_stride,
            parallel=parallel,
            store_snapshots=false,
        )
        _, m_abs_star, m_abs_ci, _ = compute_m_abs_star(res.mean_trajectories, res.time_grid)
        push!(rows, (kappa=kappa, ratio=ratio, m_abs_star=m_abs_star, m_abs_ci=m_abs_ci))
    end
    return DataFrame(rows)
end

"""
    run_backward_sweep(p, ratios; ...)

Run a backward kappa sweep.
"""
function run_backward_sweep(
    p,
    ratios::Vector{Float64};
    kappa_star::Float64,
    N::Int,
    T::Float64,
    dt::Float64,
    n_ensemble::Int,
    base_seed::Int,
    snapshot_times::Vector{Float64},
    mt_stride::Int,
    parallel::Bool,
    T_max::Float64,
)
    return run_forward_sweep(
        p, reverse(ratios);
        kappa_star=kappa_star,
        N=N,
        T=T,
        dt=dt,
        n_ensemble=n_ensemble,
        base_seed=base_seed,
        snapshot_times=snapshot_times,
        mt_stride=mt_stride,
        parallel=parallel,
        T_max=T_max,
    )
end

end
