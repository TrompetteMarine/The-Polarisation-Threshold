#!/usr/bin/env julia
# ============================================================
# eigen_kappa_diagnostic.jl
#
# Diagnostics for λ₁(κ) robustness:
#   - loops over domain truncations L and grid resolutions M
#   - computes λ₁(κ) curves, all zero-crossings, peak, tail behavior
#   - finite-difference estimate of ∂κ λ₁(κ) and monotonicity checks
#   - (optional) spot-check of additional odd eigenvalues if supported
#   - saves per-(L,M) plots + a CSV summary table
#
# Usage:
#   julia --project=. scripts/eigen_kappa_diagnostic.jl
# ============================================================

using BeliefSim
using BeliefSim.OUResets
using BeliefSim.Types
using LinearAlgebra
using Plots
using Random
using Printf
using Dates
using DelimitedFiles
using Statistics
using Logging

# ----------------------------
# Configuration
# ----------------------------
function default_params()
    # The paper baseline : Params(λ=0.85, σ=0.80, Θ=2.00, c0=0.80, hazard=StepHazard(10.5)); hazard = \nu_0
    return Params(λ=0.85, σ=0.80, Θ=2.00, c0=0.80, hazard=StepHazard(10.5))
end

# Default sweep focuses on economically meaningful kappa >= 0; adjust constants if needed.
const KAPPA_MIN   = 0.0
const KAPPA_MAX   = 12.2
const KAPPA_N     = 400

# If strict, throw when lambda + kappa <= 0 (invalid linearisation region).
const STRICT_KAPPA = false

# Domain truncations and grid resolutions to test
const L_LIST = [ 12.0, 40.0]
const M_LIST = [201, 801]
#const L_LIST = [10.0, 12.0, 15.0, 20.0, 25, 30, 35, 40]
#const M_LIST = [201, 401, 601, 801]

# Optional: for spot-checking additional odd eigenvalues, if your implementation supports it
const DO_SPECTRUM_SPOTCHECK = true
const NEV_SPOTCHECK         = 5  # number of odd eigenvalues requested if supported

# Residual diagnostics controls
const RESIDUAL_MODE   = :matrix   # :matrix, :operator, :none
const RESIDUAL_THRESH = 1e-6

# Plateau detection controls
const TAIL_N            = 12
const TAIL_FLAT_TOL     = 1e-3
const NONCONV_TAIL_FRAC = 0.2

# Iteration plots are uninformative with dense eigen solves; suppress by default.
const PLOT_ITER = false

# Where to save outputs
const OUT_ROOT = joinpath("figs", "eigen_diag")

# ----------------------------
# Utilities
# ----------------------------
mkpath(OUT_ROOT)

# Robust call wrapper: avoids a single failure killing the sweep.
function safe_leading_odd_eigenvalue(p; κ::Float64, L::Float64, M::Int,
                                     return_diag::Bool=false,
                                     return_mats::Bool=false,
                                     return_ops::Bool=false)
    try
        return leading_odd_eigenvalue(
            p;
            κ=κ,
            L=L,
            M=M,
            return_diag=return_diag,
            return_mats=return_mats,
            return_ops=return_ops
        )
    catch err
        @warn "leading_odd_eigenvalue failed" κ L M err
        return (NaN, nothing)
    end
end

# Find all zero crossings (sign changes) via linear interpolation
function find_zero_crossings(κgrid::Vector{Float64}, λ::Vector{Float64})
    xs = Float64[]
    n = length(κgrid)
    for i in 1:(n-1)
        a, b = λ[i], λ[i+1]
        if !isfinite(a) || !isfinite(b)
            continue
        end
        if a == 0.0
            push!(xs, κgrid[i])
        elseif (a < 0 && b > 0) || (a > 0 && b < 0)
            w = -a / (b - a)
            push!(xs, κgrid[i] + w*(κgrid[i+1] - κgrid[i]))
        end
    end
    return xs
end

# Central-difference derivative dλ/dκ on the same κgrid
function central_diff(κgrid::Vector{Float64}, λ::Vector{Float64})
    n = length(κgrid)
    d = fill(NaN, n)
    for i in 2:(n-1)
        if !isfinite(λ[i-1]) || !isfinite(λ[i+1])
            continue
        end
        h = κgrid[i+1] - κgrid[i-1]
        d[i] = (λ[i+1] - λ[i-1]) / h
    end
    # One-sided at boundaries
    if isfinite(λ[1]) && isfinite(λ[2])
        d[1] = (λ[2] - λ[1]) / (κgrid[2] - κgrid[1])
    end
    if isfinite(λ[end-1]) && isfinite(λ[end])
        d[end] = (λ[end] - λ[end-1]) / (κgrid[end] - κgrid[end-1])
    end
    return d
end

# Optional spectrum spot-check: try to get more eigenvalues if your function supports it.
# This is intentionally defensive because we do not assume an API.
function spectrum_spotcheck(p; κ::Float64, L::Float64, M::Int, nev::Int)
    if !DO_SPECTRUM_SPOTCHECK
        return nothing
    end
    # Try common patterns. If all fail, return nothing.
    for kwargs in (
        (; κ=κ, L=L, M=M, nev=nev),
        (; κ=κ, L=L, M=M, k=nev),
        (; κ=κ, L=L, M=M, n=nev),
    )
        try
            # If leading_odd_eigenvalue accepts nev/k/n and returns a vector, take it
            out = leading_odd_eigenvalue(p; kwargs...)
            return out
        catch
            # keep trying
        end
    end
    return nothing
end

# Independent residual check for the returned eigenpair.
function residual_check_from_diag(diag)
    if RESIDUAL_MODE == :none || diag === nothing
        return NaN
    end
    x = diag.x
    if x === nothing
        return NaN
    end
    λ = diag.eigval
    if RESIDUAL_MODE == :matrix
        A = diag.A
        if A === nothing
            return NaN
        end
        Mmat = diag.Mmat
        Mx = Mmat === nothing ? x : Mmat * x
        return norm(A * x - λ * Mx) / max(norm(Mx), eps())
    elseif RESIDUAL_MODE == :operator
        apply_A = diag.apply_A
        if apply_A === nothing
            return NaN
        end
        apply_M = diag.apply_M
        Mx = apply_M === nothing ? x : apply_M(x)
        return norm(apply_A(x) - λ * Mx) / max(norm(Mx), eps())
    end
    return NaN
end

function tail_indices(n::Int, tail_n::Int)
    if n <= 0
        return 1:0
    elseif n <= tail_n
        return 1:n
    end
    return (n - tail_n + 1):n
end

function compute_tail_flags(λ1, converged, residuals; tail_n, flat_tol, resid_thresh, nonconv_frac)
    idx = tail_indices(length(λ1), tail_n)

    tail_vals = λ1[idx]
    tail_valid = filter(isfinite, tail_vals)
    tail_flattening = length(tail_valid) >= 2 &&
        (maximum(tail_valid) - minimum(tail_valid) < flat_tol)
    exact_const_tail = length(tail_valid) >= 2 &&
        all(x -> x == tail_valid[1], tail_valid)

    tail_conv = converged[idx]
    tail_count = length(tail_conv)
    frac_nonconv = tail_count == 0 ? 0.0 : count(x -> !x, tail_conv) / tail_count
    nonconv_flag = frac_nonconv > nonconv_frac

    tail_res = residuals[idx]
    tail_res_valid = filter(isfinite, tail_res)
    resid_tail_max = isempty(tail_res_valid) ? NaN : maximum(tail_res_valid)
    resid_tail_median = isempty(tail_res_valid) ? NaN : median(tail_res_valid)
    residual_spike = !isempty(tail_res_valid) && resid_tail_max > resid_thresh

    return (
        tail_flattening = tail_flattening,
        exact_const_tail = exact_const_tail,
        nonconvergence_tail = nonconv_flag,
        residual_spike_tail = residual_spike,
        nonconv_tail_frac = frac_nonconv,
        resid_tail_max = resid_tail_max,
        resid_tail_median = resid_tail_median
    )
end

function write_data_csv(outpath, κgrid, λ1, dλ, lambda_eff, status,
                        converged, niter, solver_info, solver_resid, residual_check)
    header = [
        "kappa", "lambda_eff", "lambda1", "dlambda", "status",
        "converged", "niter", "solver_info",
        "solver_resid", "residual_check"
    ]
    open(outpath, "w") do io
        writedlm(io, permutedims(header), ',')
        for i in eachindex(κgrid)
            writedlm(io, permutedims(Any[
                κgrid[i], lambda_eff[i], λ1[i], dλ[i], status[i],
                converged[i] ? 1 : 0,
                niter[i],
                solver_info[i],
                solver_resid[i],
                residual_check[i]
            ]), ',')
        end
    end
end

# Plot helper
function save_curve_plot(κgrid, λ, dλ, crossings; L, M, outpath,
                         κ_peak=NaN, λ_peak=NaN,
                         bad_conv_idx=Int[], bad_resid_idx=Int[],
                         resid_thresh=RESIDUAL_THRESH)
    valid_idx = findall(isfinite, λ)
    plt = plot(
        κgrid[valid_idx], λ[valid_idx];
        xlabel="κ",
        ylabel="λ₁(κ)",
        title=@sprintf("Leading odd eigenvalue (L=%.1f, M=%d)", L, M),
        linewidth=3,
        color=:black,
        legend=:topright,
        size=(850, 480),
        dpi=300,
        label="λ₁"
    )
    hline!(plt, [0.0]; color=:gray, linestyle=:dash, label="0")

    # Mark all crossings
    for (j, κx) in enumerate(crossings)
        vline!(plt, [κx]; color=:red, linestyle=:dash, label=(j==1 ? "crossings" : nothing))
        scatter!(plt, [κx], [0.0]; color=:red, label=nothing)
    end

    if isfinite(κ_peak) && isfinite(λ_peak)
        scatter!(plt, [κ_peak], [λ_peak]; color=:green, marker=:star5, label="peak")
    end

    if !isempty(bad_conv_idx)
        scatter!(plt, κgrid[bad_conv_idx], λ[bad_conv_idx];
                 color=:orange, marker=:circle, label="nonconverged")
    end

    if !isempty(bad_resid_idx)
        scatter!(plt, κgrid[bad_resid_idx], λ[bad_resid_idx];
                 color=:purple, marker=:diamond,
                 label=@sprintf("resid>%.1e", resid_thresh))
    end

    # Add derivative on a secondary axis if desired (simple overlay, dashed)
    # (kept as overlay to avoid backend differences in dual-axis support)
    valid_d_idx = findall(isfinite, dλ)
    if !isempty(valid_d_idx)
        plot!(plt, κgrid[valid_d_idx], dλ[valid_d_idx];
              linewidth=2, linestyle=:dot, color=:blue, label="∂κ λ₁ (FD)")
    end

    savefig(plt, outpath)
end

function save_residual_plot(κgrid, residuals; L, M, outpath, resid_thresh=RESIDUAL_THRESH)
    resid_plot = [ (isfinite(r) && r > 0) ? r : NaN for r in residuals ]
    valid_idx = findall(isfinite, resid_plot)
    plt = isempty(valid_idx) ? plot(;
        xlabel="κ",
        ylabel="||A*x - λ*M*x|| / ||M*x||",
        title=@sprintf("Residual check (L=%.1f, M=%d)", L, M),
        legend=:topright,
        yscale=:log10,
        size=(850, 480),
        dpi=300
    ) : plot(
        κgrid[valid_idx], resid_plot[valid_idx];
        xlabel="κ",
        ylabel="||A*x - λ*M*x|| / ||M*x||",
        title=@sprintf("Residual check (L=%.1f, M=%d)", L, M),
        linewidth=2,
        color=:black,
        legend=:topright,
        yscale=:log10,
        size=(850, 480),
        dpi=300,
        label="residual"
    )
    if isfinite(resid_thresh) && resid_thresh > 0
        hline!(plt, [resid_thresh]; color=:red, linestyle=:dash,
               label=@sprintf("thresh %.1e", resid_thresh))
    end
    savefig(plt, outpath)
end

function save_iter_plot(κgrid, niter; L, M, outpath)
    plt = plot(
        κgrid, niter;
        xlabel="κ",
        ylabel="iterations",
        title=@sprintf("Solver iterations (L=%.1f, M=%d)", L, M),
        linewidth=2,
        color=:black,
        legend=:topright,
        size=(850, 480),
        dpi=300,
        label="niter"
    )
    savefig(plt, outpath)
end

# ----------------------------
# Main
# ----------------------------
function main()
    Random.seed!(25)

    p = default_params()
    κgrid = collect(range(KAPPA_MIN, KAPPA_MAX, length=KAPPA_N))

    # Summary rows: write as CSV at the end
    header = [
        "timestamp", "L", "M",
        "du",
        "lambda_start", "lambda_end",
        "n_invalid_kappa", "n_fail", "min_lambda_eff",
        "n_crossings", "crossings",
        "kappa_peak", "lambda_peak",
        "dlambda_min", "dlambda_max",
        "monotone_increasing_flag",
        "tail_flattening_flag", "exact_const_tail_flag",
        "nonconvergence_tail_flag", "residual_spike_tail_flag",
        "plateau_flag",
        "residual_max", "residual_tail_max", "residual_tail_median",
        "nonconv_tail_frac",
        "notes"
    ]
    rows = Vector{Vector{Any}}()

    for L in L_LIST
        for M in M_LIST
            @info "Computing λ₁(κ) curve" L M

            tag = @sprintf("L%0.1f_M%d", L, M)
            tag_clean = replace(tag, "." => "p")
            du = 2 * L / (M - 1)

            nκ = length(κgrid)
            λ1 = fill(NaN, nκ)
            lambda_eff = fill(NaN, nκ)
            status = fill("ok", nκ)
            converged = fill(false, nκ)
            niter = fill(0, nκ)
            solver_info = fill("", nκ)
            solver_resid = fill(NaN, nκ)
            residual_check = fill(NaN, nκ)

            return_mats = RESIDUAL_MODE == :matrix
            return_ops = RESIDUAL_MODE == :operator

            for (i, κ) in enumerate(κgrid)
                # Domain guard: linearisation is invalid when lambda + kappa <= 0 (no mean reversion).
                lambda_eff[i] = p.λ + κ
                if lambda_eff[i] <= 0
                    if STRICT_KAPPA
                        throw(ArgumentError("Invalid kappa: lambda + kappa <= 0 (kappa=$(κ), lambda=$(p.λ))"))
                    end
                    status[i] = "invalid_kappa"
                    λ1[i] = NaN
                    converged[i] = false
                    solver_info[i] = "invalid_kappa"
                    solver_resid[i] = NaN
                    residual_check[i] = NaN
                    continue
                end

                λ1[i], diag = safe_leading_odd_eigenvalue(
                    p;
                    κ=κ,
                    L=L,
                    M=M,
                    return_diag=true,
                    return_mats=return_mats,
                    return_ops=return_ops
                )

                if diag === nothing
                    status[i] = "failure"
                    converged[i] = false
                    niter[i] = 0
                    solver_info[i] = "error"
                    solver_resid[i] = NaN
                    residual_check[i] = NaN
                else
                    status[i] = "ok"
                    converged[i] = diag.converged
                    niter[i] = diag.niter
                    solver_info[i] = string(diag.info)
                    solver_resid[i] = diag.solver_resid
                    try
                        residual_check[i] = residual_check_from_diag(diag)
                    catch err
                        @warn "residual_check failed" κ L M err
                        residual_check[i] = NaN
                    end
                end
            end

            crossings = find_zero_crossings(κgrid, λ1)
            dλ = central_diff(κgrid, λ1)

            data_path = joinpath(OUT_ROOT, "data_" * tag_clean * ".csv")
            write_data_csv(
                data_path,
                κgrid,
                λ1,
                dλ,
                lambda_eff,
                status,
                converged,
                niter,
                solver_info,
                solver_resid,
                residual_check
            )

            n_invalid_kappa = count(s -> s == "invalid_kappa", status)
            n_fail = count(s -> s == "failure", status)
            lambda_eff_valid = filter(x -> x > 0, lambda_eff)
            min_lambda_eff = isempty(lambda_eff_valid) ? NaN : minimum(lambda_eff_valid)

            tail_stats = compute_tail_flags(
                λ1, converged, residual_check;
                tail_n=TAIL_N,
                flat_tol=TAIL_FLAT_TOL,
                resid_thresh=RESIDUAL_THRESH,
                nonconv_frac=NONCONV_TAIL_FRAC
            )
            resid_valid = filter(isfinite, residual_check)
            resid_max = isempty(resid_valid) ? NaN : maximum(resid_valid)
            plateau_flag = tail_stats.tail_flattening ||
                tail_stats.exact_const_tail ||
                tail_stats.nonconvergence_tail ||
                tail_stats.residual_spike_tail

            # Derivative stats (ignore NaNs)
            valid_d = filter(isfinite, dλ)
            dl_min = isempty(valid_d) ? NaN : minimum(valid_d)
            dl_max = isempty(valid_d) ? NaN : maximum(valid_d)
            monotone_flag = isfinite(dl_min) && (dl_min > 0)

            # Peak (ignore NaNs)
            valid_idx = findall(isfinite, λ1)
            if isempty(valid_idx)
                @warn "All λ₁ are NaN for this (L,M)" L M
                notes = "all_nan;"
                if RESIDUAL_MODE == :none
                    notes *= "resid_none;"
                end
                if tail_stats.tail_flattening
                    notes *= "tail_flattening;"
                end
                if tail_stats.exact_const_tail
                    notes *= "exact_const_tail;"
                end
                if tail_stats.nonconvergence_tail
                    notes *= "nonconv_tail;"
                end
                if tail_stats.residual_spike_tail
                    notes *= "resid_spike_tail;"
                end
                push!(rows, Any[
                    string(Dates.now()),
                    L, M,
                    du,
                    λ1[1], λ1[end],
                    n_invalid_kappa, n_fail, min_lambda_eff,
                    length(crossings), join(string.(crossings), "|"),
                    NaN, NaN,
                    dl_min, dl_max,
                    monotone_flag ? 1 : 0,
                    tail_stats.tail_flattening ? 1 : 0,
                    tail_stats.exact_const_tail ? 1 : 0,
                    tail_stats.nonconvergence_tail ? 1 : 0,
                    tail_stats.residual_spike_tail ? 1 : 0,
                    plateau_flag ? 1 : 0,
                    resid_max,
                    tail_stats.resid_tail_max,
                    tail_stats.resid_tail_median,
                    tail_stats.nonconv_tail_frac,
                    notes
                ])
                continue
            end
            imax = valid_idx[argmax(λ1[valid_idx])]
            κ_peak, λ_peak = κgrid[imax], λ1[imax]

            notes = ""
            if RESIDUAL_MODE == :none
                notes *= "resid_none;"
            end
            if tail_stats.tail_flattening
                notes *= "tail_flattening;"
            end
            if tail_stats.exact_const_tail
                notes *= "exact_const_tail;"
            end
            if tail_stats.nonconvergence_tail
                notes *= "nonconv_tail;"
            end
            if tail_stats.residual_spike_tail
                notes *= "resid_spike_tail;"
            end

            # Optional spectrum spot-check at crossings and peak
            if DO_SPECTRUM_SPOTCHECK
                κ_check = unique(vcat(crossings, [κ_peak]))
                for κc in κ_check
                    try
                        out = spectrum_spotcheck(p; κ=κc, L=L, M=M, nev=NEV_SPOTCHECK)
                        if out !== nothing
                            # Just record that something was returned; user can extend to parse it.
                            notes *= @sprintf("spec@%.3f;", κc)
                        end
                    catch err
                        @warn "spectrum_spotcheck failed" κc L M err
                    end
                end
            end

            bad_conv_idx = findall(i -> !converged[i] && isfinite(λ1[i]), eachindex(converged))
            bad_resid_idx = findall(
                i -> isfinite(residual_check[i]) && residual_check[i] > RESIDUAL_THRESH && isfinite(λ1[i]),
                eachindex(residual_check)
            )

            # Save plots
            outplot = joinpath(OUT_ROOT, "eigen_curve_" * tag_clean * ".pdf")
            save_curve_plot(
                κgrid, λ1, dλ, crossings;
                L=L, M=M, outpath=outplot,
                κ_peak=κ_peak, λ_peak=λ_peak,
                bad_conv_idx=bad_conv_idx, bad_resid_idx=bad_resid_idx,
                resid_thresh=RESIDUAL_THRESH
            )

            if RESIDUAL_MODE != :none
                outresid = joinpath(OUT_ROOT, "residual_curve_" * tag_clean * ".pdf")
                save_residual_plot(κgrid, residual_check; L=L, M=M, outpath=outresid, resid_thresh=RESIDUAL_THRESH)
            end
            if PLOT_ITER && any(niter .> 0)
                outiter = joinpath(OUT_ROOT, "iter_curve_" * tag_clean * ".pdf")
                save_iter_plot(κgrid, niter; L=L, M=M, outpath=outiter)
            end

            push!(rows, Any[
                string(Dates.now()),
                L, M,
                du,
                λ1[1], λ1[end],
                n_invalid_kappa, n_fail, min_lambda_eff,
                length(crossings), join(string.(crossings), "|"),
                κ_peak, λ_peak,
                dl_min, dl_max,
                monotone_flag ? 1 : 0,
                tail_stats.tail_flattening ? 1 : 0,
                tail_stats.exact_const_tail ? 1 : 0,
                tail_stats.nonconvergence_tail ? 1 : 0,
                tail_stats.residual_spike_tail ? 1 : 0,
                plateau_flag ? 1 : 0,
                resid_max,
                tail_stats.resid_tail_max,
                tail_stats.resid_tail_median,
                tail_stats.nonconv_tail_frac,
                notes
            ])
        end
    end

    # Write summary CSV
    summary_path = joinpath(OUT_ROOT, "summary.csv")
    open(summary_path, "w") do io
        writedlm(io, permutedims(header), ',')
        for r in rows
            writedlm(io, permutedims(r), ',')
        end
    end
    @info "Saved summary" summary_path
    @info "Saved plots to" OUT_ROOT

    println("\nDONE.")
    println("Next: inspect summary.csv to see whether (i) number of crossings is stable across (L,M),")
    println("      and (ii) whether dlambda_min stays >0 (monotone) or goes negative (re-entrance).")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
