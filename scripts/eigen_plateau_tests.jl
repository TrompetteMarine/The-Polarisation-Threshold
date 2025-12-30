#!/usr/bin/env julia
# ============================================================
# eigen_plateau_tests.jl
#
# Plateau + branch-switching diagnostics for lambda1(kappa).
#
# Usage:
#   julia --project=. scripts/eigen_plateau_tests.jl
# ============================================================

using BeliefSim
using BeliefSim.OUResets
using BeliefSim.Types
using LinearAlgebra
using Statistics
using Random
using Printf
using Dates
using DelimitedFiles
using Logging
using Plots

# For code search (optional):
#   rg -n "A\[1, 1\] = -10.0|A\[M, M\] = -10.0|max\(|min\(|clamp\(" src/OUResets.jl

# ----------------------------
# Configuration
# ----------------------------
const FAST_MODE = false

const KAPPA_MIN = 0.0
const KAPPA_MAX = FAST_MODE ? 20.0 : 30.0
const KAPPA_N   = FAST_MODE ? 200 : 400

const HAZARD_LIST = [7.0, 10.0, 10.5, 12.5]
const L_LIST = FAST_MODE ? [10.0, 12.0] : [10.0, 12.0, 15.0, 20.0]
const M_LIST = FAST_MODE ? [201, 401] : [201, 401, 601, 801]

const RESIDUAL_MODE   = :matrix   # :matrix, :operator, :none
const RESIDUAL_THRESH = 1e-6

const TAIL_FRAC         = 0.2
const TAIL_FLAT_TOL     = 1e-3
const PLATEAU_TOL       = 1e-6
const PLATEAU_RUN       = 8
const NONCONV_TAIL_FRAC = 0.2

const BOUNDARY_FRAC        = 0.05
const BOUNDARY_MASS_THRESH = 0.2
const BOUNDARY_EIGENVALUE  = -10.0

const DO_SPECTRUM_TESTS = true
const DO_EIGENVECTOR_TESTS = true
const TOPK = 10
const SPECTRUM_KAPPAS = FAST_MODE ? [10.0, 20.0] : [8.0, 10.0, 11.0, 20.0]
const EIGENVECTOR_KAPPAS = FAST_MODE ? [11.0] : [11.0, 20.0]
const SPECTRUM_ALL_LM = !FAST_MODE
const SPECTRUM_LM = (10.0, 401)

const OUT_ROOT = joinpath("figs", "eigen_diag", "plateau_tests")
mkpath(OUT_ROOT)

# ----------------------------
# Utilities
# ----------------------------
function tag_float(x::Real)
    return replace(string(x), "." => "p")
end

function default_params(hazard)
    return Params(λ=0.85, σ=0.80, Θ=2.00, c0=0.80, hazard=hazard)
end

function hazard_intensity_probe(p::Params; expected_nu0::Float64)
    h = p.hazard
    println("hazard type: ", typeof(h))
    println("hazard value: ", h)
    nu0_at_0 = ν(h, 0.0, p.Θ)
    nu0_at_100 = ν(h, 100.0, p.Θ)
    println(@sprintf("nu(0) = %.6f, nu(100) = %.6f", nu0_at_0, nu0_at_100))
    @assert abs(nu0_at_0) < 1e-12 "nu(0) expected to be 0 for StepHazard"
    @assert abs(nu0_at_100 - expected_nu0) < 1e-9 "hazard intensity mismatch"
end

function safe_leading_odd_eigenvalue(p; kappa::Float64, L::Float64, M::Int,
                                     return_diag::Bool=false,
                                     return_mats::Bool=false,
                                     return_ops::Bool=false)
    try
        return leading_odd_eigenvalue(
            p;
            κ=kappa,
            L=L,
            M=M,
            return_diag=return_diag,
            return_mats=return_mats,
            return_ops=return_ops
        )
    catch err
        @warn "leading_odd_eigenvalue failed" kappa L M err
        return (NaN, nothing)
    end
end

function find_zero_crossings(kappa_grid::Vector{Float64}, lambda1::Vector{Float64})
    xs = Float64[]
    n = length(kappa_grid)
    for i in 1:(n-1)
        a, b = lambda1[i], lambda1[i+1]
        if !isfinite(a) || !isfinite(b)
            continue
        end
        if a == 0.0
            push!(xs, kappa_grid[i])
        elseif (a < 0 && b > 0) || (a > 0 && b < 0)
            w = -a / (b - a)
            push!(xs, kappa_grid[i] + w * (kappa_grid[i+1] - kappa_grid[i]))
        end
    end
    return xs
end

function central_diff(kappa_grid::Vector{Float64}, lambda1::Vector{Float64})
    n = length(kappa_grid)
    d = fill(NaN, n)
    for i in 2:(n-1)
        if !isfinite(lambda1[i-1]) || !isfinite(lambda1[i+1])
            continue
        end
        h = kappa_grid[i+1] - kappa_grid[i-1]
        d[i] = (lambda1[i+1] - lambda1[i-1]) / h
    end
    if isfinite(lambda1[1]) && isfinite(lambda1[2])
        d[1] = (lambda1[2] - lambda1[1]) / (kappa_grid[2] - kappa_grid[1])
    end
    if isfinite(lambda1[end-1]) && isfinite(lambda1[end])
        d[end] = (lambda1[end] - lambda1[end-1]) / (kappa_grid[end] - kappa_grid[end-1])
    end
    return d
end

function residual_check_from_diag(diag)
    if RESIDUAL_MODE == :none || diag === nothing
        return NaN
    end
    x = diag.x
    if x === nothing
        return NaN
    end
    lambda = diag.eigval
    if RESIDUAL_MODE == :matrix
        A = diag.A
        if A === nothing
            return NaN
        end
        Mmat = diag.Mmat
        Mx = Mmat === nothing ? x : Mmat * x
        return norm(A * x - lambda * Mx) / max(norm(Mx), eps())
    elseif RESIDUAL_MODE == :operator
        apply_A = diag.apply_A
        if apply_A === nothing
            return NaN
        end
        apply_M = diag.apply_M
        Mx = apply_M === nothing ? x : apply_M(x)
        return norm(apply_A(x) - lambda * Mx) / max(norm(Mx), eps())
    end
    return NaN
end

function tail_indices(n::Int, tail_frac::Float64)
    if n <= 0
        return 1:0
    end
    tail_n = max(1, Int(ceil(tail_frac * n)))
    if n <= tail_n
        return 1:n
    end
    return (n - tail_n + 1):n
end

function plateau_onset_index(lambda1::Vector{Float64}, plateau_level::Float64;
                             tol::Float64, consecutive::Int)
    if !isfinite(plateau_level)
        return 0
    end
    run = 0
    for i in eachindex(lambda1)
        v = lambda1[i]
        if isfinite(v) && abs(v - plateau_level) <= tol
            run += 1
            if run >= consecutive
                return i - consecutive + 1
            end
        else
            run = 0
        end
    end
    return 0
end

function compute_plateau_stats(kappa_grid, lambda1, converged, residuals;
                               tail_frac, flat_tol, plateau_tol, plateau_run,
                               nonconv_frac, resid_thresh)
    idx = tail_indices(length(lambda1), tail_frac)
    tail_vals = lambda1[idx]
    tail_valid = filter(isfinite, tail_vals)

    plateau_level = isempty(tail_valid) ? NaN : median(tail_valid)
    tail_std = length(tail_valid) >= 2 ? std(tail_valid) : NaN

    exact_const_tail = !isempty(tail_valid) && all(x -> x == tail_valid[1], tail_valid)
    tail_flattening = length(tail_valid) >= 2 && (maximum(tail_valid) - minimum(tail_valid) < flat_tol)

    onset_idx = plateau_onset_index(lambda1, plateau_level; tol=plateau_tol, consecutive=plateau_run)
    onset_kappa = onset_idx > 0 ? kappa_grid[onset_idx] : NaN

    tail_conv = converged[idx]
    tail_count = length(tail_conv)
    nonconv_tail_frac = tail_count == 0 ? 0.0 : count(x -> !x, tail_conv) / tail_count
    nonconvergence_tail = nonconv_tail_frac > nonconv_frac

    tail_res = residuals[idx]
    tail_res_valid = filter(isfinite, tail_res)
    resid_tail_max = isempty(tail_res_valid) ? NaN : maximum(tail_res_valid)
    resid_tail_median = isempty(tail_res_valid) ? NaN : median(tail_res_valid)
    residual_spike_tail = !isempty(tail_res_valid) && resid_tail_max > resid_thresh

    plateau_flag = tail_flattening || exact_const_tail || nonconvergence_tail || residual_spike_tail
    hard_clamp = exact_const_tail && isfinite(tail_std) && tail_std <= eps()
    plateau_is_integer = isfinite(plateau_level) && abs(plateau_level - round(plateau_level)) < 1e-12

    return (
        plateau_level = plateau_level,
        plateau_onset_idx = onset_idx,
        plateau_onset_kappa = onset_kappa,
        tail_std = tail_std,
        tail_flattening = tail_flattening,
        exact_const_tail = exact_const_tail,
        hard_clamp = hard_clamp,
        plateau_is_integer = plateau_is_integer,
        nonconv_tail_frac = nonconv_tail_frac,
        nonconvergence_tail = nonconvergence_tail,
        resid_tail_max = resid_tail_max,
        resid_tail_median = resid_tail_median,
        residual_spike_tail = residual_spike_tail,
        plateau_flag = plateau_flag
    )
end

function write_curve_csv(outpath, kappa_grid, lambda1, dlambda,
                         converged, niter, solver_info, solver_resid, residual_check)
    header = [
        "kappa", "lambda1", "dlambda",
        "converged", "niter", "solver_info",
        "solver_resid", "residual_check"
    ]
    open(outpath, "w") do io
        writedlm(io, permutedims(header), ',')
        for i in eachindex(kappa_grid)
            writedlm(io, permutedims(Any[
                kappa_grid[i], lambda1[i], dlambda[i],
                converged[i] ? 1 : 0,
                niter[i],
                solver_info[i],
                solver_resid[i],
                residual_check[i]
            ]), ',')
        end
    end
end

function save_lambda_plot(kappa_grid, lambda1, dlambda, crossings;
                          L, M, nu0, outpath,
                          plateau_onset=NaN, plateau_level=NaN,
                          kappa_peak=NaN, lambda_peak=NaN,
                          bad_conv_idx=Int[], bad_resid_idx=Int[])
    plt = plot(
        kappa_grid, lambda1;
        xlabel="kappa",
        ylabel="lambda1(kappa)",
        title=@sprintf("lambda1 (h=%g, L=%.1f, M=%d)", nu0, L, M),
        linewidth=3,
        color=:black,
        legend=:topright,
        size=(850, 480),
        dpi=300,
        label="lambda1"
    )
    hline!(plt, [0.0]; color=:gray, linestyle=:dash, label="0")

    for (j, kx) in enumerate(crossings)
        vline!(plt, [kx]; color=:red, linestyle=:dash, label=(j == 1 ? "crossings" : nothing))
        scatter!(plt, [kx], [0.0]; color=:red, label=nothing)
    end

    if isfinite(kappa_peak) && isfinite(lambda_peak)
        scatter!(plt, [kappa_peak], [lambda_peak]; color=:green, marker=:star5, label="peak")
    end

    if isfinite(plateau_onset)
        vline!(plt, [plateau_onset]; color=:orange, linestyle=:dash, label="plateau_onset")
    end
    if isfinite(plateau_level)
        hline!(plt, [plateau_level]; color=:orange, linestyle=:dot, label="plateau_level")
    end

    if !isempty(bad_conv_idx)
        scatter!(plt, kappa_grid[bad_conv_idx], lambda1[bad_conv_idx];
                 color=:orange, marker=:circle, label="nonconverged")
    end

    if !isempty(bad_resid_idx)
        scatter!(plt, kappa_grid[bad_resid_idx], lambda1[bad_resid_idx];
                 color=:purple, marker=:diamond,
                 label=@sprintf("resid>%.1e", RESIDUAL_THRESH))
    end

    plot!(plt, kappa_grid, dlambda; linewidth=2, linestyle=:dot, color=:blue, label="d lambda1 / d kappa")
    savefig(plt, outpath)
end

function save_residual_plot(kappa_grid, residuals; L, M, nu0, outpath)
    resid_plot = [ (isfinite(r) && r > 0) ? r : NaN for r in residuals ]
    plt = plot(
        kappa_grid, resid_plot;
        xlabel="kappa",
        ylabel="||A*x - lambda*M*x|| / ||M*x||",
        title=@sprintf("residual (h=%g, L=%.1f, M=%d)", nu0, L, M),
        linewidth=2,
        color=:black,
        legend=:topright,
        yscale=:log10,
        size=(850, 480),
        dpi=300,
        label="residual"
    )
    if isfinite(RESIDUAL_THRESH) && RESIDUAL_THRESH > 0
        hline!(plt, [RESIDUAL_THRESH]; color=:red, linestyle=:dash,
               label=@sprintf("thresh %.1e", RESIDUAL_THRESH))
    end
    savefig(plt, outpath)
end

function save_spectrum_plot(vals, kappa::Float64, nu0::Float64, p::Params;
                            L, M, outpath)
    idx = 1:length(vals)
    plt = plot(
        idx, real.(vals);
        xlabel="odd eigenvalue index",
        ylabel="Re(lambda)",
        title=@sprintf("odd spectrum (kappa=%.2f, h=%g, L=%.1f, M=%d)", kappa, nu0, L, M),
        linewidth=2,
        marker=:circle,
        color=:black,
        legend=:topright,
        size=(850, 480),
        dpi=300,
        label="Re(lambda)"
    )
    hline!(plt, [-nu0]; color=:red, linestyle=:dash, label=@sprintf("-nu0=%.2f", -nu0))
    hline!(plt, [p.λ - kappa]; color=:blue, linestyle=:dot, label=@sprintf("lambda-kappa=%.2f", p.λ - kappa))
    savefig(plt, outpath)
end

function save_eigenvector_plot(xgrid, vec, kappa::Float64, nu0::Float64;
                               L, M, outpath, Theta)
    plt = plot(
        xgrid, real.(vec);
        xlabel="u",
        ylabel="eigenvector (real)",
        title=@sprintf("leading odd eigenvector (kappa=%.2f, h=%g, L=%.1f, M=%d)", kappa, nu0, L, M),
        linewidth=2,
        color=:black,
        legend=:topright,
        size=(850, 480),
        dpi=300,
        label="v"
    )
    vline!(plt, [-Theta, Theta]; color=:red, linestyle=:dash, label="Theta")
    vline!(plt, [-L, L]; color=:gray, linestyle=:dot, label="boundary")
    savefig(plt, outpath)
end

function topk_odd_spectrum_from_matrix(A, xgrid; corr_tol::Float64, k::Int)
    eig = eigen(A)
    vals = eig.values
    vecs = eig.vectors

    x_vec = xgrid ./ norm(xgrid)
    weights = zeros(Float64, length(vals))
    @inbounds for i in eachindex(vals)
        v = vecs[:, i]
        weights[i] = abs(real(dot(v, x_vec)))
    end

    idxs = findall(w -> w > corr_tol, weights)
    if isempty(idxs)
        return (Float64[], Matrix{eltype(vecs)}(undef, length(xgrid), 0), Float64[])
    end

    odd_vals = vals[idxs]
    odd_vecs = vecs[:, idxs]
    odd_weights = weights[idxs]

    order = sortperm(real.(odd_vals), rev=true)
    odd_vals = odd_vals[order]
    odd_vecs = odd_vecs[:, order]
    odd_weights = odd_weights[order]

    if k > 0 && length(odd_vals) > k
        odd_vals = odd_vals[1:k]
        odd_vecs = odd_vecs[:, 1:k]
        odd_weights = odd_weights[1:k]
    end

    return (odd_vals, odd_vecs, odd_weights)
end

function compute_eigenvector_metrics(xgrid, vec, Theta)
    w = abs2.(vec)
    total = sum(w)
    if total == 0
        return (mass_outside=NaN, boundary_mass=NaN, mean_abs_u=NaN)
    end
    outside = sum(w[abs.(xgrid) .>= Theta]) / total

    n = length(xgrid)
    nb = max(1, Int(ceil(BOUNDARY_FRAC * n)))
    idx = vcat(1:nb, (n - nb + 1):n)
    boundary = sum(w[idx]) / total

    mean_abs_u = sum(abs.(xgrid) .* w) / total

    return (mass_outside=outside, boundary_mass=boundary, mean_abs_u=mean_abs_u)
end

# ----------------------------
# Main test runner
# ----------------------------
function main()
    Random.seed!(25)
    kappa_grid = collect(range(KAPPA_MIN, KAPPA_MAX, length=KAPPA_N))
    dk = length(kappa_grid) >= 2 ? kappa_grid[2] - kappa_grid[1] : 1.0
    onset_tol = max(PLATEAU_TOL, 2.0 * dk)

    summary_rows = Vector{Dict{String, Any}}()
    eigenvector_rows = Vector{Dict{String, Any}}()
    spectrum_rows = Vector{Dict{String, Any}}()

    for nu0 in HAZARD_LIST
        hazard = StepHazard(nu0)
        p = default_params(hazard)

        @info "Hazard intensity probe" nu0
        hazard_intensity_probe(p; expected_nu0=nu0)

        for L in L_LIST
            for M in M_LIST
                @info "Plateau test" nu0 L M

                lambda1 = fill(NaN, length(kappa_grid))
                converged = fill(false, length(kappa_grid))
                niter = fill(0, length(kappa_grid))
                solver_info = fill("", length(kappa_grid))
                solver_resid = fill(NaN, length(kappa_grid))
                residual_check = fill(NaN, length(kappa_grid))

                return_mats = RESIDUAL_MODE == :matrix
                return_ops = RESIDUAL_MODE == :operator

                for (i, kappa) in enumerate(kappa_grid)
                    lambda1[i], diag = safe_leading_odd_eigenvalue(
                        p;
                        kappa=kappa,
                        L=L,
                        M=M,
                        return_diag=true,
                        return_mats=return_mats,
                        return_ops=return_ops
                    )

                    if diag === nothing
                        converged[i] = false
                        niter[i] = 0
                        solver_info[i] = "error"
                        solver_resid[i] = NaN
                        residual_check[i] = NaN
                    else
                        converged[i] = diag.converged
                        niter[i] = diag.niter
                        solver_info[i] = string(diag.info)
                        solver_resid[i] = diag.solver_resid
                        try
                            residual_check[i] = residual_check_from_diag(diag)
                        catch err
                            @warn "residual_check failed" kappa L M err
                            residual_check[i] = NaN
                        end
                    end
                end

                dlambda = central_diff(kappa_grid, lambda1)
                crossings = find_zero_crossings(kappa_grid, lambda1)

                tag = @sprintf("h%s_L%s_M%d", tag_float(nu0), tag_float(L), M)

                curve_csv = joinpath(OUT_ROOT, "lambda_curve_" * tag * ".csv")
                write_curve_csv(curve_csv, kappa_grid, lambda1, dlambda,
                                converged, niter, solver_info, solver_resid, residual_check)

                resid_valid = filter(isfinite, residual_check)
                residual_max = isempty(resid_valid) ? NaN : maximum(resid_valid)

                plateau_stats = compute_plateau_stats(
                    kappa_grid, lambda1, converged, residual_check;
                    tail_frac=TAIL_FRAC,
                    flat_tol=TAIL_FLAT_TOL,
                    plateau_tol=PLATEAU_TOL,
                    plateau_run=PLATEAU_RUN,
                    nonconv_frac=NONCONV_TAIL_FRAC,
                    resid_thresh=RESIDUAL_THRESH
                )

                plateau_level = plateau_stats.plateau_level
                plateau_onset = plateau_stats.plateau_onset_kappa

                valid_idx = findall(isfinite, lambda1)
                kappa_peak = NaN
                lambda_peak = NaN
                if !isempty(valid_idx)
                    imax = valid_idx[argmax(lambda1[valid_idx])]
                    kappa_peak = kappa_grid[imax]
                    lambda_peak = lambda1[imax]
                end

                bad_conv_idx = findall(i -> !converged[i] && isfinite(lambda1[i]), eachindex(converged))
                bad_resid_idx = findall(
                    i -> isfinite(residual_check[i]) && residual_check[i] > RESIDUAL_THRESH && isfinite(lambda1[i]),
                    eachindex(residual_check)
                )

                if !isempty(valid_idx)
                    lambda_plot = joinpath(OUT_ROOT, "lambda_curve_" * tag * ".pdf")
                    save_lambda_plot(
                        kappa_grid, lambda1, dlambda, crossings;
                        L=L, M=M, nu0=nu0, outpath=lambda_plot,
                        plateau_onset=plateau_onset,
                        plateau_level=plateau_level,
                        kappa_peak=kappa_peak,
                        lambda_peak=lambda_peak,
                        bad_conv_idx=bad_conv_idx,
                        bad_resid_idx=bad_resid_idx
                    )

                    if RESIDUAL_MODE != :none
                        resid_plot = joinpath(OUT_ROOT, "residual_curve_" * tag * ".pdf")
                        save_residual_plot(kappa_grid, residual_check; L=L, M=M, nu0=nu0, outpath=resid_plot)
                    end
                else
                    @warn "All lambda1 values are NaN" nu0 L M
                end

                plateau_level_target = -nu0
                plateau_onset_target = p.λ + nu0
                plateau_level_match = isfinite(plateau_level) && abs(plateau_level - plateau_level_target) <= PLATEAU_TOL
                plateau_onset_match = isfinite(plateau_onset) && abs(plateau_onset - plateau_onset_target) <= onset_tol
                plateau_equals_boundary = isfinite(plateau_level) && abs(plateau_level - BOUNDARY_EIGENVALUE) < 1e-9

                notes = ""
                if RESIDUAL_MODE == :none
                    notes *= "resid_none;"
                end
                if plateau_stats.tail_flattening
                    notes *= "tail_flattening;"
                end
                if plateau_stats.exact_const_tail
                    notes *= "exact_const_tail;"
                end
                if plateau_stats.hard_clamp
                    notes *= "hard_clamp;"
                end
                if plateau_stats.nonconvergence_tail
                    notes *= "nonconv_tail;"
                end
                if plateau_stats.residual_spike_tail
                    notes *= "resid_spike_tail;"
                end
                if plateau_equals_boundary
                    notes *= "plateau_equals_boundary;"
                end

                push!(summary_rows, Dict(
                    "timestamp" => string(Dates.now()),
                    "nu0" => nu0,
                    "L" => L,
                    "M" => M,
                    "lambda_start" => lambda1[1],
                    "lambda_end" => lambda1[end],
                    "n_crossings" => length(crossings),
                    "kappa_peak" => kappa_peak,
                    "lambda_peak" => lambda_peak,
                    "plateau_level" => plateau_level,
                    "plateau_onset" => plateau_onset,
                    "plateau_run" => PLATEAU_RUN,
                    "plateau_level_target" => plateau_level_target,
                    "plateau_onset_target" => plateau_onset_target,
                    "plateau_level_match" => plateau_level_match ? 1 : 0,
                    "plateau_onset_match" => plateau_onset_match ? 1 : 0,
                    "plateau_equals_boundary" => plateau_equals_boundary ? 1 : 0,
                    "tail_flattening" => plateau_stats.tail_flattening ? 1 : 0,
                    "exact_const_tail" => plateau_stats.exact_const_tail ? 1 : 0,
                    "hard_clamp" => plateau_stats.hard_clamp ? 1 : 0,
                    "plateau_is_integer" => plateau_stats.plateau_is_integer ? 1 : 0,
                    "nonconv_tail_frac" => plateau_stats.nonconv_tail_frac,
                    "residual_tail_max" => plateau_stats.resid_tail_max,
                    "residual_tail_median" => plateau_stats.resid_tail_median,
                    "residual_max" => residual_max,
                    "residual_spike_tail" => plateau_stats.residual_spike_tail ? 1 : 0,
                    "nonconvergence_tail" => plateau_stats.nonconvergence_tail ? 1 : 0,
                    "plateau_flag" => plateau_stats.plateau_flag ? 1 : 0,
                    "notes" => notes
                ))

                if DO_EIGENVECTOR_TESTS
                    for kappa_ev in EIGENVECTOR_KAPPAS
                        lambda_ev, diag_ev = safe_leading_odd_eigenvalue(
                            p;
                            kappa=kappa_ev,
                            L=L,
                            M=M,
                            return_diag=true,
                            return_mats=false,
                            return_ops=false
                        )
                        if diag_ev === nothing || diag_ev.x === nothing || diag_ev.xgrid === nothing
                            push!(eigenvector_rows, Dict(
                                "timestamp" => string(Dates.now()),
                                "nu0" => nu0,
                                "L" => L,
                                "M" => M,
                                "kappa" => kappa_ev,
                                "lambda1" => lambda_ev,
                                "mass_outside" => NaN,
                                "boundary_mass" => NaN,
                                "mean_abs_u" => NaN,
                                "boundary_flag" => 0
                            ))
                            continue
                        end

                        metrics = compute_eigenvector_metrics(diag_ev.xgrid, diag_ev.x, p.Θ)
                        boundary_flag = isfinite(metrics.boundary_mass) && metrics.boundary_mass > BOUNDARY_MASS_THRESH

                        push!(eigenvector_rows, Dict(
                            "timestamp" => string(Dates.now()),
                            "nu0" => nu0,
                            "L" => L,
                            "M" => M,
                            "kappa" => kappa_ev,
                            "lambda1" => lambda_ev,
                            "mass_outside" => metrics.mass_outside,
                            "boundary_mass" => metrics.boundary_mass,
                            "mean_abs_u" => metrics.mean_abs_u,
                            "boundary_flag" => boundary_flag ? 1 : 0
                        ))

                        tag_ev = @sprintf("kappa%s_h%s_L%s_M%d", tag_float(kappa_ev), tag_float(nu0), tag_float(L), M)
                        eig_plot = joinpath(OUT_ROOT, "eigenvector_" * tag_ev * ".pdf")
                        save_eigenvector_plot(diag_ev.xgrid, diag_ev.x, kappa_ev, nu0;
                                              L=L, M=M, outpath=eig_plot, Theta=p.Θ)
                    end
                end

                if DO_SPECTRUM_TESTS && (SPECTRUM_ALL_LM || (L == SPECTRUM_LM[1] && M == SPECTRUM_LM[2]))
                    for kappa_spec in SPECTRUM_KAPPAS
                        lambda_spec, diag_spec = safe_leading_odd_eigenvalue(
                            p;
                            kappa=kappa_spec,
                            L=L,
                            M=M,
                            return_diag=true,
                            return_mats=true,
                            return_ops=false
                        )
                        if diag_spec === nothing || diag_spec.A === nothing || diag_spec.xgrid === nothing
                            @warn "spectrum: missing matrix" kappa_spec nu0 L M
                            continue
                        end

                        vals, vecs, weights = topk_odd_spectrum_from_matrix(
                            diag_spec.A,
                            diag_spec.xgrid;
                            corr_tol=1e-3,
                            k=TOPK
                        )

                        tag_spec = @sprintf("kappa%s_h%s_L%s_M%d", tag_float(kappa_spec), tag_float(nu0), tag_float(L), M)
                        spec_csv = joinpath(OUT_ROOT, "spec_" * tag_spec * ".csv")
                        open(spec_csv, "w") do io
                            writedlm(io, permutedims(["index", "eig_real", "eig_imag", "odd_weight"]), ',')
                            for i in 1:length(vals)
                                writedlm(io, permutedims(Any[i, real(vals[i]), imag(vals[i]), weights[i]]), ',')
                            end
                        end

                        spec_plot = joinpath(OUT_ROOT, "spectrum_" * tag_spec * ".pdf")
                        save_spectrum_plot(vals, kappa_spec, nu0, p; L=L, M=M, outpath=spec_plot)

                        branch_lambda = p.λ - kappa_spec
                        branch_nu = -nu0
                        match_lambda = any(abs.(real.(vals) .- branch_lambda) .< 1e-3)
                        match_nu = any(abs.(real.(vals) .- branch_nu) .< 1e-3)

                        push!(spectrum_rows, Dict(
                            "timestamp" => string(Dates.now()),
                            "nu0" => nu0,
                            "L" => L,
                            "M" => M,
                            "kappa" => kappa_spec,
                            "leading_lambda" => lambda_spec,
                            "branch_lambda" => branch_lambda,
                            "branch_nu" => branch_nu,
                            "match_lambda" => match_lambda ? 1 : 0,
                            "match_nu" => match_nu ? 1 : 0
                        ))
                    end
                end
            end
        end
    end

    # Clamp detection across hazard intensities (per L,M)
    clamp_map = Dict{Tuple{Float64, Int}, Bool}()
    level_map = Dict{Tuple{Float64, Int}, Vector{Float64}}()

    for row in summary_rows
        key = (row["L"], row["M"])
        if !haskey(level_map, key)
            level_map[key] = Float64[]
        end
        lvl = row["plateau_level"]
        if isfinite(lvl)
            push!(level_map[key], lvl)
        end
    end

    for (key, levels) in level_map
        if length(levels) < 2
            clamp_map[key] = false
            continue
        end
        lvl_min = minimum(levels)
        lvl_max = maximum(levels)
        lvl_range = lvl_max - lvl_min
        lvl_int = abs(lvl_min - round(lvl_min)) < 1e-12
        clamp_map[key] = (lvl_range < 1e-9) && lvl_int
    end

    # Write main summary
    summary_path = joinpath(OUT_ROOT, "plateau_hazard_sweep_summary.csv")
    summary_header = [
        "timestamp", "nu0", "L", "M",
        "lambda_start", "lambda_end",
        "n_crossings", "kappa_peak", "lambda_peak",
        "plateau_level", "plateau_onset", "plateau_run",
        "plateau_level_target", "plateau_onset_target",
        "plateau_level_match", "plateau_onset_match", "plateau_equals_boundary",
        "tail_flattening", "exact_const_tail", "hard_clamp", "plateau_is_integer",
        "nonconv_tail_frac", "residual_tail_max", "residual_tail_median", "residual_max",
        "residual_spike_tail", "nonconvergence_tail", "plateau_flag",
        "clamp_across_hazard", "notes"
    ]
    open(summary_path, "w") do io
        writedlm(io, permutedims(summary_header), ',')
        for row in summary_rows
            key = (row["L"], row["M"])
            clamp_flag = get(clamp_map, key, false)
            writedlm(io, permutedims(Any[
                row["timestamp"], row["nu0"], row["L"], row["M"],
                row["lambda_start"], row["lambda_end"],
                row["n_crossings"], row["kappa_peak"], row["lambda_peak"],
                row["plateau_level"], row["plateau_onset"], row["plateau_run"],
                row["plateau_level_target"], row["plateau_onset_target"],
                row["plateau_level_match"], row["plateau_onset_match"], row["plateau_equals_boundary"],
                row["tail_flattening"], row["exact_const_tail"], row["hard_clamp"], row["plateau_is_integer"],
                row["nonconv_tail_frac"], row["residual_tail_max"], row["residual_tail_median"], row["residual_max"],
                row["residual_spike_tail"], row["nonconvergence_tail"], row["plateau_flag"],
                clamp_flag ? 1 : 0,
                row["notes"]
            ]), ',')
        end
    end

    # Aggregated summary across L,M for each hazard intensity
    agg_path = joinpath(OUT_ROOT, "plateau_hazard_sweep_agg.csv")
    agg_header = [
        "nu0", "n_runs",
        "plateau_level_median", "plateau_level_std",
        "plateau_onset_median", "plateau_onset_std",
        "plateau_level_target", "plateau_onset_target"
    ]
    open(agg_path, "w") do io
        writedlm(io, permutedims(agg_header), ',')
        for nu0 in HAZARD_LIST
            rows = filter(r -> r["nu0"] == nu0, summary_rows)
            levels = filter(isfinite, [r["plateau_level"] for r in rows])
            onsets = filter(isfinite, [r["plateau_onset"] for r in rows])
            level_med = isempty(levels) ? NaN : median(levels)
            level_std = length(levels) >= 2 ? std(levels) : NaN
            onset_med = isempty(onsets) ? NaN : median(onsets)
            onset_std = length(onsets) >= 2 ? std(onsets) : NaN
            writedlm(io, permutedims(Any[
                nu0, length(rows),
                level_med, level_std,
                onset_med, onset_std,
                -nu0, 0.85 + nu0
            ]), ',')
        end
    end

    # Eigenvector metrics CSV
    if DO_EIGENVECTOR_TESTS
        ev_path = joinpath(OUT_ROOT, "eigenvector_metrics.csv")
        ev_header = [
            "timestamp", "nu0", "L", "M", "kappa", "lambda1",
            "mass_outside", "boundary_mass", "mean_abs_u", "boundary_flag"
        ]
        open(ev_path, "w") do io
            writedlm(io, permutedims(ev_header), ',')
            for row in eigenvector_rows
                writedlm(io, permutedims(Any[
                    row["timestamp"], row["nu0"], row["L"], row["M"], row["kappa"], row["lambda1"],
                    row["mass_outside"], row["boundary_mass"], row["mean_abs_u"], row["boundary_flag"]
                ]), ',')
            end
        end
    end

    # Spectrum summary CSV
    if DO_SPECTRUM_TESTS
        spec_path = joinpath(OUT_ROOT, "spectrum_summary.csv")
        spec_header = [
            "timestamp", "nu0", "L", "M", "kappa",
            "leading_lambda", "branch_lambda", "branch_nu",
            "match_lambda", "match_nu"
        ]
        open(spec_path, "w") do io
            writedlm(io, permutedims(spec_header), ',')
            for row in spectrum_rows
                writedlm(io, permutedims(Any[
                    row["timestamp"], row["nu0"], row["L"], row["M"], row["kappa"],
                    row["leading_lambda"], row["branch_lambda"], row["branch_nu"],
                    row["match_lambda"], row["match_nu"]
                ]), ',')
            end
        end
    end

    @info "Saved outputs to" OUT_ROOT
    println("\nDONE.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
