module ThresholdEstimation

using LinearAlgebra
using Statistics

using BeliefSim
using BeliefSim.Types
using BeliefSim.Hazard: ν

using ..EnsembleUtils: run_ensemble_simulation

export compute_kappa_star_B,
       growth_scan_kappa_A,
       estimate_kappa_star_from_scan,
       build_A0_operator,
       stationary_density_from_A0,
       odd_subspace_matrix

"""
    trapezoid_weights(x) -> Vector{Float64}

Trapezoid rule weights for uniform grid `x`.
"""
function trapezoid_weights(x::Vector{Float64})::Vector{Float64}
    n = length(x)
    @assert n >= 2 "Need at least two grid points for trapezoid weights."
    h = x[2] - x[1]
    w = fill(h, n)
    w[1] *= 0.5
    w[end] *= 0.5
    return w
end

"""
    build_A0_operator(p; L, M, boundary=:reflecting)

Finite-difference operator for the linearised Fokker-Planck.
"""
function build_A0_operator(p::Params; L::Float64, M::Int, boundary::Symbol = :reflecting)
    @assert isodd(M) "M should be odd so that x=0 is on the grid."
    x = collect(range(-L, L, length=M))
    h = x[2] - x[1]
    σ2 = p.σ^2 / 2
    A = zeros(Float64, M, M)

    function add_gain!(row::Int, xi::Float64)
        y = xi / p.c0
        if y < -L || y > L
            return
        end
        rate = (1 / p.c0) * ν(p.hazard, y, p.Θ)
        if rate == 0.0
            return
        end
        pos = (y - x[1]) / h + 1
        j = clamp(Int(floor(pos)), 1, M - 1)
        w = pos - j
        A[row, j] += rate * (1 - w)
        A[row, j + 1] += rate * w
    end

    for i in 2:(M - 1)
        xi = x[i]
        rate = ν(p.hazard, xi, p.Θ)

        A[i, i] += p.λ
        A[i, i-1] += -p.λ * xi / (2h)
        A[i, i+1] += p.λ * xi / (2h)

        A[i, i-1] += σ2 / h^2
        A[i, i] += -2σ2 / h^2
        A[i, i+1] += σ2 / h^2

        A[i, i] += -rate
        add_gain!(i, xi)
    end

    if boundary == :reflecting
        for i in (1, M)
            xi = x[i]
            rate = ν(p.hazard, xi, p.Θ)
            A[i, i] += p.λ
            A[i, i] += -σ2 / h^2
            if i == 1
                A[i, i+1] += σ2 / h^2
            else
                A[i, i-1] += σ2 / h^2
            end
            A[i, i] += -rate
            add_gain!(i, xi)
        end
    else
        A[1, 1] = -10.0
        A[M, M] = -10.0
    end

    return A, x
end

"""
    stationary_density_from_A0(A, x) -> (rho, weights)

Stationary density for the linearised operator.
"""
function stationary_density_from_A0(A::Matrix{Float64}, x::Vector{Float64})
    M = length(x)
    w = trapezoid_weights(x)
    A2 = copy(A)
    b = zeros(Float64, M)
    mid = div(M + 1, 2)
    A2[mid, :] .= w
    b[mid] = 1.0
    rho = A2 \ b
    rho .= 0.5 .* (rho .+ reverse(rho))
    rho ./= sum(rho .* w)
    return rho, w
end

"""
    odd_subspace_matrix(A) -> (Aodd, idx_pos)

Odd subspace restriction of operator A.
"""
function odd_subspace_matrix(A::Matrix{Float64})
    M = size(A, 1)
    mid = div(M + 1, 2)
    idx_pos = (mid + 1):M
    n = length(idx_pos)
    Aodd = zeros(Float64, n, n)
    for (col, j) in enumerate(idx_pos)
        f = zeros(Float64, M)
        f[j] = 1.0
        j_mirror = mid - (j - mid)
        f[j_mirror] = -1.0
        g = A * f
        Aodd[:, col] = g[idx_pos]
    end
    return Aodd, idx_pos
end

"""
    compute_kappa_star_B(p; L, M, boundary) -> NamedTuple

Rank-one susceptibility for theoretical kappa*.
"""
function compute_kappa_star_B(p::Params; L::Float64, M::Int, boundary::Symbol = :reflecting)
    A0, x = build_A0_operator(p; L=L, M=M, boundary=boundary)
    rho, w = stationary_density_from_A0(A0, x)
    h = x[2] - x[1]

    b = zeros(Float64, length(x))
    for i in 2:(length(x) - 1)
        b[i] = -(rho[i + 1] - rho[i - 1]) / (2h)
    end
    b[1] = -(rho[2] - rho[1]) / h
    b[end] = -(rho[end] - rho[end - 1]) / h

    Aodd, idx_pos = odd_subspace_matrix(A0)
    b_odd = b[idx_pos]
    x_pos = x[idx_pos]
    w_pos = w[idx_pos]
    a_odd = 2 .* x_pos .* w_pos

    v = (-Aodd) \ b_odd
    Phi0 = dot(a_odd, v)
    kappa_B = 1 / Phi0

    return (
        kappa_B = kappa_B,
        Phi0 = Phi0,
        x = x,
        rho = rho,
        Aodd = Aodd,
        b_odd = b_odd,
        a_odd = a_odd,
    )
end

"""
    fit_log_growth(t, y, window)

Fit log-growth slope in a time window.
"""
function fit_log_growth(t::Vector{Float64}, y::Vector{Float64}, window::Tuple{Float64, Float64})
    t1, t2 = window
    mask = (t .>= t1) .& (t .<= t2)
    tt = t[mask]
    yy = y[mask]
    yy = log.(yy .+ 1e-12)
    good = isfinite.(yy)
    tt = tt[good]
    yy = yy[good]
    if length(tt) < 3
        return (slope=NaN, intercept=NaN, r2=NaN, se=NaN, n=length(tt))
    end
    X = hcat(ones(length(tt)), tt)
    coeffs = X \ yy
    yhat = X * coeffs
    ss_res = sum((yy .- yhat).^2)
    ss_tot = sum((yy .- mean(yy)).^2)
    r2 = ss_tot > 0 ? 1 - ss_res / ss_tot : NaN
    se = sqrt(ss_res / max(1, length(tt) - 2) / sum((tt .- mean(tt)).^2))
    return (slope=coeffs[2], intercept=coeffs[1], r2=r2, se=se, n=length(tt))
end

"""
    growth_scan_kappa_A(p, kappa_grid; ...)

Empirical growth scan for kappa*_A.
"""
function growth_scan_kappa_A(
    p::Params,
    kappa_grid::Vector{Float64};
    N::Int,
    T::Float64,
    dt::Float64,
    n_ensemble::Int,
    base_seed::Int,
    snapshot_times::Vector{Float64},
    mt_stride::Int,
    window::Tuple{Float64, Float64},
    n_boot::Int,
    parallel::Bool,
)
    rows = Vector{NamedTuple}()
    boot_slopes = Matrix{Float64}(undef, n_boot, length(kappa_grid))

    for (k_idx, κ) in enumerate(kappa_grid)
        res = run_ensemble_simulation(
            p;
            kappa=κ,
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
        t = res.time_grid
        mean_abs = vec(mean(abs.(res.mean_trajectories); dims=1))
        fit = fit_log_growth(t, mean_abs, window)
        push!(rows, (kappa=κ, lambda_hat=fit.slope, r2=fit.r2, se=fit.se, n=fit.n))

        if n_boot > 0
            for b in 1:n_boot
                sample_idx = rand(1:n_ensemble, n_ensemble)
                mean_boot = vec(mean(abs.(res.mean_trajectories[sample_idx, :]); dims=1))
                boot_fit = fit_log_growth(t, mean_boot, window)
                boot_slopes[b, k_idx] = boot_fit.slope
            end
        end
    end

    return rows, boot_slopes
end

"""
    estimate_kappa_star_from_scan(kappa_grid, slopes)

Estimate kappa* by zero-crossing of slopes.
"""
function estimate_kappa_star_from_scan(kappa_grid::Vector{Float64}, slopes::Vector{Float64})
    idx = sortperm(kappa_grid)
    κ = kappa_grid[idx]
    s = slopes[idx]
    for i in 1:(length(κ) - 1)
        if s[i] == 0
            return κ[i]
        elseif s[i] * s[i + 1] < 0
            t = abs(s[i]) / (abs(s[i]) + abs(s[i + 1]))
            return κ[i] * (1 - t) + κ[i + 1] * t
        end
    end
    return κ[argmin(abs.(s))]
end

end
