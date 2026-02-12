#!/usr/bin/env julia
# ============================================================
# eigen_kappa_validity_sweep.jl
#
# kappa-sweep diagnostics with domain validity guard and hazard-intensity sweep.
#
# Usage:
#   julia --project=. scripts/eigen_kappa_validity_sweep.jl
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

# Prefer library hazard evaluation when available.
const _HAS_HAZARD_NU = Ref(false)
try
    import BeliefSim.Hazard: ν
    _HAS_HAZARD_NU[] = true
catch
    _HAS_HAZARD_NU[] = false
end

# ----------------------------
# Configuration
# ----------------------------
# Default sweep focuses on economically meaningful kappa >= 0; adjust constants if needed.
const KAPPA_MIN = 0.0
const KAPPA_MAX = 20.0
const KAPPA_N   = 400

# If strict, throw when lambda + kappa <= 0 (invalid linearisation region).
const STRICT_KAPPA = false

const L_LIST = [10.0, 12.0, 15.0, 20.0]
const L0 = 10.0
const M0 = 801
const DX_TARGET = 2 * L0 / (M0 - 1)

const NU0_LIST = [7.0, 12.5, 50.0]

# Branch selection options for the spectral routine.
const USE_ODDSPACE = true
const USE_TRACKING = false
const ODD_SELECT = :parity
const TOL_ODD = 1e-6

const RESIDUAL_MODE   = :matrix   # :matrix, :operator, :none
const RESIDUAL_THRESH = 1e-6

const TAIL_FRAC    = 0.2
const PLATEAU_TOL  = 1e-6
const PLATEAU_RUN  = 8

# If strict, throw on hazard intensity mismatch; otherwise warn + record flag.
const STRICT_HAZARD_ASSERT = false

# Optional top-K odd eigenvalues check (dense eigen). Keep off by default.
const DO_TOPK = false
const TOPK = 10
const TOPK_KAPPAS = [8.0, 11.0, 20.0]  # ensure within [KAPPA_MIN, KAPPA_MAX] if DO_TOPK
const TOPK_ALL_LM = false
const TOPK_LM = (L0, M0)

const OUT_ROOT = joinpath("figs", "eigen_diag", "validity")
mkpath(OUT_ROOT)

# ----------------------------
# Utilities
# ----------------------------
function tag_float(x::Real)
    return replace(string(x), "." => "p")
end

function nearest_odd(n::Int)
    return isodd(n) ? n : n + 1
end

function m_for_L(L::Float64)
    m = Int(round(2 * L / DX_TARGET)) + 1
    return max(3, nearest_odd(m))
end

function default_params(nu0::Float64)
    return Params(λ=0.85, σ=0.80, Θ=2.00, c0=0.80, hazard=StepHazard(nu0))
end

function _get_prop(h, sym_a::Symbol, sym_b::Symbol)
    if hasproperty(h, sym_a)
        return getproperty(h, sym_a)
    elseif hasproperty(h, sym_b)
        return getproperty(h, sym_b)
    end
    return NaN
end

if _HAS_HAZARD_NU[]
    hazard_rate(h, u::Real, Θ::Real) = ν(h, float(u), float(Θ))
else
    function hazard_rate(h::StepHazard, u::Real, Θ::Real)
        ν0 = _get_prop(h, :ν0, :nu0)
        if !isfinite(ν0)
            @warn "StepHazard missing intensity field" typeof(h)
            return NaN
        end
        return abs(u) >= Θ ? float(ν0) : 0.0
    end

    function hazard_rate(h::LogisticHazard, u::Real, Θ::Real)
        νmax = _get_prop(h, :νmax, :numax)
        β = _get_prop(h, :β, :beta)
        if !(isfinite(νmax) && isfinite(β))
            @warn "LogisticHazard missing field(s)" typeof(h)
            return NaN
        end
        return float(νmax) / (1 + exp(-float(β) * (abs(u) - Θ)))
    end

    hazard_rate(h, u::Real, Θ::Real) = ( @warn "No hazard_rate method for hazard type" typeof(h); NaN )
end

function hazard_intensity_probe(p::Params; expected_nu0::Float64)
    h = p.hazard
    println("hazard type: ", typeof(h))
    println("hazard value: ", h)
    nu0_at_0 = hazard_rate(h, 0.0, p.Θ)
    nu0_at_100 = hazard_rate(h, 100.0, p.Θ)
    println(@sprintf("nu(0) = %.6f, nu(100) = %.6f", nu0_at_0, nu0_at_100))
    if !isfinite(nu0_at_0) || abs(nu0_at_0) > 1e-12
        @warn "nu(0) expected to be 0 for StepHazard" nu0_at_0
    end
    mismatch = !isfinite(nu0_at_100) || abs(nu0_at_100 - expected_nu0) > 1e-9
    if mismatch
        msg = @sprintf("hazard intensity mismatch: expected %.6f got %.6f", expected_nu0, nu0_at_100)
        if STRICT_HAZARD_ASSERT
            throw(ArgumentError(msg))
        else
            @warn msg
        end
    end
    return (nu0_at_100=nu0_at_100, mismatch=mismatch)
end

function safe_leading_odd_eigenvalue(p; kappa::Float64, L::Float64, M::Int,
                                     odd_select::Symbol=:parity,
                                     tol_odd::Float64=1e-6,
                                     track::Bool=false,
                                     prev_vec=nothing,
                                     oddspace::Bool=false,
                                     return_diag::Bool=false,
                                     return_mats::Bool=false,
                                     return_ops::Bool=false)
    try
        return leading_odd_eigenvalue(
            p;
            κ=kappa,
            L=L,
            M=M,
            odd_select=odd_select,
            tol_odd=tol_odd,
            track=track,
            prev_vec=prev_vec,
            oddspace=oddspace,
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

function compute_plateau_stats(kappa_grid, lambda1; tail_frac, plateau_tol, plateau_run)
    idx = tail_indices(length(lambda1), tail_frac)
    tail_vals = lambda1[idx]
    tail_valid = filter(isfinite, tail_vals)

    plateau_level = isempty(tail_valid) ? NaN : median(tail_valid)
    onset_idx = plateau_onset_index(lambda1, plateau_level; tol=plateau_tol, consecutive=plateau_run)
    onset_kappa = onset_idx > 0 ? kappa_grid[onset_idx] : NaN

    return (
        plateau_level = plateau_level,
        plateau_onset_idx = onset_idx,
        plateau_onset_kappa = onset_kappa
    )
end

function write_curve_csv(outpath, kappa_grid, lambda_eff, lambda1, dlambda,
                         status, converged, niter, solver_info, solver_resid, residual_check,
                         parity_defect_odd, parity_defect_even,
                         boundary_mass_1pct, boundary_mass_5pct,
                         hazard_mass, overlap_prev, generator_mass_err,
                         nu0_expected, nu0_used)
    header = [
        "kappa", "lambda_eff", "lambda1", "dlambda", "status",
        "converged", "niter", "solver_info", "solver_resid", "residual_check",
        "parity_defect_odd", "parity_defect_even",
        "boundary_mass_1pct", "boundary_mass_5pct",
        "hazard_mass", "overlap_prev", "generator_mass_err",
        "nu0_expected", "nu0_used"
    ]
    open(outpath, "w") do io
        writedlm(io, permutedims(header), ',')
        for i in eachindex(kappa_grid)
            writedlm(io, permutedims(Any[
                kappa_grid[i], lambda_eff[i], lambda1[i], dlambda[i], status[i],
                converged[i] ? 1 : 0,
                niter[i],
                solver_info[i],
                solver_resid[i],
                residual_check[i],
                parity_defect_odd[i],
                parity_defect_even[i],
                boundary_mass_1pct[i],
                boundary_mass_5pct[i],
                hazard_mass[i],
                overlap_prev[i],
                generator_mass_err[i],
                nu0_expected[i],
                nu0_used[i]
            ]), ',')
        end
    end
end

function save_lambda_plot(kappa_grid, lambda1, dlambda, crossings;
                          L, M, nu0, outpath,
                          plateau_onset=NaN, plateau_level=NaN,
                          kappa_peak=NaN, lambda_peak=NaN,
                          bad_conv_idx=Int[], bad_resid_idx=Int[])
    valid_idx = findall(isfinite, lambda1)
    plt = plot(
        kappa_grid[valid_idx], lambda1[valid_idx];
        xlabel="kappa",
        ylabel="lambda1(kappa)",
        title=@sprintf("lambda1 (nu0=%g, L=%.1f, M=%d)", nu0, L, M),
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

    valid_d_idx = findall(isfinite, dlambda)
    if !isempty(valid_d_idx)
        plot!(plt, kappa_grid[valid_d_idx], dlambda[valid_d_idx];
              linewidth=2, linestyle=:dot, color=:blue, label="d lambda1 / d kappa")
    end

    savefig(plt, outpath)
end

function save_residual_plot(kappa_grid, residuals; L, M, nu0, outpath)
    resid_plot = [ (isfinite(r) && r > 0) ? r : NaN for r in residuals ]
    valid_idx = findall(isfinite, resid_plot)
    plt = isempty(valid_idx) ? plot(;
        xlabel="kappa",
        ylabel="||A*x - lambda*M*x|| / ||M*x||",
        title=@sprintf("residual (nu0=%g, L=%.1f, M=%d)", nu0, L, M),
        legend=:topright,
        yscale=:log10,
        size=(850, 480),
        dpi=300
    ) : plot(
        kappa_grid[valid_idx], resid_plot[valid_idx];
        xlabel="kappa",
        ylabel="||A*x - lambda*M*x|| / ||M*x||",
        title=@sprintf("residual (nu0=%g, L=%.1f, M=%d)", nu0, L, M),
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

function save_spectrum_plot(vals, kappa::Float64, nu0::Float64, p::Params;
                            L, M, outpath)
    idx = 1:length(vals)
    plt = plot(
        idx, real.(vals);
        xlabel="odd eigenvalue index",
        ylabel="Re(lambda)",
        title=@sprintf("odd spectrum (kappa=%.2f, nu0=%g, L=%.1f, M=%d)", kappa, nu0, L, M),
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

# ----------------------------
# Main
# ----------------------------
function main()
    Random.seed!(25)

    kappa_grid = collect(range(KAPPA_MIN, KAPPA_MAX, length=KAPPA_N))
    dk = length(kappa_grid) >= 2 ? kappa_grid[2] - kappa_grid[1] : 1.0
    onset_tol = max(PLATEAU_TOL, 2.0 * dk)

    summary_rows = Vector{Dict{String, Any}}()

    for nu0 in NU0_LIST
        p = default_params(nu0)
        @info "Hazard intensity probe" nu0
        probe = hazard_intensity_probe(p; expected_nu0=nu0)
        observed_nu_at_100 = probe.nu0_at_100
        hazard_mismatch_flag = probe.mismatch ? 1 : 0

        for L in L_LIST
            M = m_for_L(L)
            du = 2 * L / (M - 1)
            @info "kappa sweep" nu0 L M du

            n = length(kappa_grid)
            lambda1 = fill(NaN, n)
            lambda_eff = fill(NaN, n)
            status = fill("ok", n)
            converged = fill(false, n)
            niter = fill(0, n)
            solver_info = fill("", n)
            solver_resid = fill(NaN, n)
            residual_check = fill(NaN, n)
            parity_defect_odd = fill(NaN, n)
            parity_defect_even = fill(NaN, n)
            boundary_mass_1pct = fill(NaN, n)
            boundary_mass_5pct = fill(NaN, n)
            hazard_mass = fill(NaN, n)
            overlap_prev = fill(NaN, n)
            generator_mass_err = fill(NaN, n)
            nu0_expected = fill(NaN, n)
            nu0_used = fill(NaN, n)

            return_mats = RESIDUAL_MODE == :matrix
            return_ops = RESIDUAL_MODE == :operator
            track_flag = USE_TRACKING && !USE_ODDSPACE
            prev_vec = nothing

            for (i, kappa) in enumerate(kappa_grid)
                # Domain guard: linearisation is invalid when lambda + kappa <= 0.
                lambda_eff[i] = p.λ + kappa
                if lambda_eff[i] <= 0
                    if STRICT_KAPPA
                        throw(ArgumentError("Invalid kappa: lambda + kappa <= 0 (kappa=$(kappa), lambda=$(p.λ))"))
                    end
                    status[i] = "invalid_kappa"
                    lambda1[i] = NaN
                    converged[i] = false
                    solver_info[i] = "invalid_kappa"
                    solver_resid[i] = NaN
                    residual_check[i] = NaN
                    continue
                end

                lambda1[i], diag = safe_leading_odd_eigenvalue(
                    p;
                    kappa=kappa,
                    L=L,
                    M=M,
                    odd_select=ODD_SELECT,
                    tol_odd=TOL_ODD,
                    track=track_flag,
                    prev_vec=prev_vec,
                    oddspace=USE_ODDSPACE,
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
                    parity_defect_odd[i] = diag.parity_defect_odd
                    parity_defect_even[i] = diag.parity_defect_even
                    boundary_mass_1pct[i] = diag.boundary_mass_1pct
                    boundary_mass_5pct[i] = diag.boundary_mass_5pct
                    hazard_mass[i] = diag.hazard_mass
                    overlap_prev[i] = diag.overlap_prev
                    generator_mass_err[i] = diag.generator_mass_err
                    nu0_expected[i] = diag.nu0_expected
                    nu0_used[i] = diag.nu0_used
                    try
                        residual_check[i] = residual_check_from_diag(diag)
                    catch err
                        @warn "residual_check failed" kappa L M err
                        residual_check[i] = NaN
                    end
                    if track_flag
                        prev_vec = hasproperty(diag, :v) ? diag.v : diag.x
                    end
                end
            end

            dlambda = central_diff(kappa_grid, lambda1)
            crossings = find_zero_crossings(kappa_grid, lambda1)
            kappa_cross_1 = length(crossings) >= 1 ? crossings[1] : NaN
            kappa_cross_2 = length(crossings) >= 2 ? crossings[2] : NaN

            plateau = compute_plateau_stats(
                kappa_grid, lambda1;
                tail_frac=TAIL_FRAC,
                plateau_tol=PLATEAU_TOL,
                plateau_run=PLATEAU_RUN
            )

            valid_idx = findall(isfinite, lambda1)
            kappa_peak = NaN
            lambda_peak = NaN
            if !isempty(valid_idx)
                imax = valid_idx[argmax(lambda1[valid_idx])]
                kappa_peak = kappa_grid[imax]
                lambda_peak = lambda1[imax]
            end

            n_invalid_kappa = count(s -> s == "invalid_kappa", status)
            n_fail = count(s -> s == "failure", status)
            lambda_eff_valid = filter(x -> x > 0, lambda_eff)
            min_lambda_eff = isempty(lambda_eff_valid) ? NaN : minimum(lambda_eff_valid)
            resid_valid = filter(isfinite, residual_check)
            residual_max = isempty(resid_valid) ? NaN : maximum(resid_valid)
            dlambda_valid = filter(isfinite, dlambda)
            dlambda_min = isempty(dlambda_valid) ? NaN : minimum(dlambda_valid)
            dlambda_max = isempty(dlambda_valid) ? NaN : maximum(dlambda_valid)
            generr_valid = filter(isfinite, generator_mass_err)
            generator_mass_err_median = isempty(generr_valid) ? NaN : median(generr_valid)

            boundary_mass_cross_max = NaN
            if isfinite(kappa_cross_1)
                idx_cross = findall(i -> isfinite(boundary_mass_5pct[i]) &&
                                    abs(kappa_grid[i] - kappa_cross_1) <= 2 * dk, eachindex(kappa_grid))
                if !isempty(idx_cross)
                    boundary_mass_cross_max = maximum(boundary_mass_5pct[idx_cross])
                end
            end

            mode_notes = String[]
            if track_flag
                overlap_valid = filter(isfinite, overlap_prev)
                if !isempty(overlap_valid) && minimum(overlap_valid) < 0.2
                    push!(mode_notes, "overlap_drop")
                end
            end
            parity_valid = filter(isfinite, parity_defect_odd)
            if !isempty(parity_valid) &&
               maximum(parity_valid) > 10 * max(TOL_ODD, 1e-12) &&
               minimum(parity_valid) <= TOL_ODD
                push!(mode_notes, "parity_jump")
            end
            mode_switch_flag = isempty(mode_notes) ? "" : join(mode_notes, ";")

            tag = @sprintf("h%s_L%s_M%d", tag_float(nu0), tag_float(L), M)
            curve_csv = joinpath(OUT_ROOT, "lambda_curve_" * tag * ".csv")
            write_curve_csv(curve_csv, kappa_grid, lambda_eff, lambda1, dlambda,
                            status, converged, niter, solver_info, solver_resid, residual_check,
                            parity_defect_odd, parity_defect_even,
                            boundary_mass_1pct, boundary_mass_5pct,
                            hazard_mass, overlap_prev, generator_mass_err,
                            nu0_expected, nu0_used)

            if !isempty(valid_idx)
                bad_conv_idx = findall(i -> !converged[i] && isfinite(lambda1[i]), eachindex(converged))
                bad_resid_idx = findall(
                    i -> isfinite(residual_check[i]) && residual_check[i] > RESIDUAL_THRESH && isfinite(lambda1[i]),
                    eachindex(residual_check)
                )

                lambda_plot = joinpath(OUT_ROOT, "lambda_curve_" * tag * ".pdf")
                save_lambda_plot(
                    kappa_grid, lambda1, dlambda, crossings;
                    L=L, M=M, nu0=nu0, outpath=lambda_plot,
                    plateau_onset=plateau.plateau_onset_kappa,
                    plateau_level=plateau.plateau_level,
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
            plateau_level_match = isfinite(plateau.plateau_level) && abs(plateau.plateau_level - plateau_level_target) <= PLATEAU_TOL
            plateau_onset_match = isfinite(plateau.plateau_onset_kappa) && abs(plateau.plateau_onset_kappa - plateau_onset_target) <= onset_tol

            push!(summary_rows, Dict(
                "timestamp" => string(Dates.now()),
                "nu0" => nu0,
                "L" => L,
                "M" => M,
                "du" => du,
                "observed_nu_at_100" => observed_nu_at_100,
                "hazard_mismatch_flag" => hazard_mismatch_flag,
                "n_invalid_kappa" => n_invalid_kappa,
                "n_fail" => n_fail,
                "min_lambda_eff" => min_lambda_eff,
                "n_crossings" => length(crossings),
                "kappa_cross_1" => kappa_cross_1,
                "kappa_cross_2" => kappa_cross_2,
                "kappa_peak" => kappa_peak,
                "lambda_peak" => lambda_peak,
                "dlambda_min" => dlambda_min,
                "dlambda_max" => dlambda_max,
                "boundary_mass_cross_max" => boundary_mass_cross_max,
                "generator_mass_err_median" => generator_mass_err_median,
                "mode_switch_flag" => mode_switch_flag,
                "plateau_level" => plateau.plateau_level,
                "plateau_onset" => plateau.plateau_onset_kappa,
                "plateau_level_target" => plateau_level_target,
                "plateau_onset_target" => plateau_onset_target,
                "plateau_level_match" => plateau_level_match ? 1 : 0,
                "plateau_onset_match" => plateau_onset_match ? 1 : 0,
                "residual_max" => residual_max
            ))

            if DO_TOPK && (TOPK_ALL_LM || (L == TOPK_LM[1] && M == TOPK_LM[2]))
                for kappa_spec in TOPK_KAPPAS
                    if kappa_spec < KAPPA_MIN || kappa_spec > KAPPA_MAX
                        continue
                    end
                    if p.λ + kappa_spec <= 0
                        continue
                    end

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

                    vals, _, weights = topk_odd_spectrum_from_matrix(
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
                end
            end
        end
    end

    # Write hazard sweep summary
    summary_path = joinpath(OUT_ROOT, "plateau_hazard_summary.csv")
    summary_header = [
        "timestamp", "nu0", "L", "M",
        "du",
        "observed_nu_at_100", "hazard_mismatch_flag",
        "n_invalid_kappa", "n_fail", "min_lambda_eff",
        "n_crossings", "kappa_cross_1", "kappa_cross_2",
        "kappa_peak", "lambda_peak",
        "dlambda_min", "dlambda_max",
        "boundary_mass_cross_max", "generator_mass_err_median", "mode_switch_flag",
        "plateau_level", "plateau_onset",
        "plateau_level_target", "plateau_onset_target",
        "plateau_level_match", "plateau_onset_match",
        "residual_max"
    ]
    open(summary_path, "w") do io
        writedlm(io, permutedims(summary_header), ',')
        for row in summary_rows
            writedlm(io, permutedims(Any[
                row["timestamp"], row["nu0"], row["L"], row["M"],
                row["du"],
                row["observed_nu_at_100"], row["hazard_mismatch_flag"],
                row["n_invalid_kappa"], row["n_fail"], row["min_lambda_eff"],
                row["n_crossings"], row["kappa_cross_1"], row["kappa_cross_2"],
                row["kappa_peak"], row["lambda_peak"],
                row["dlambda_min"], row["dlambda_max"],
                row["boundary_mass_cross_max"], row["generator_mass_err_median"], row["mode_switch_flag"],
                row["plateau_level"], row["plateau_onset"],
                row["plateau_level_target"], row["plateau_onset_target"],
                row["plateau_level_match"], row["plateau_onset_match"],
                row["residual_max"]
            ]), ',')
        end
    end

    @info "Saved outputs to" OUT_ROOT
    println("\nDONE.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
