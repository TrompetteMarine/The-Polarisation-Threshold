#!/usr/bin/env julia
# ============================================================
# eigen_hazard_wiring_sanity.jl
#
# Sanity checks for hazard wiring in the spectral operator.
#
# Usage:
#   julia --project=. scripts/eigen_hazard_wiring_sanity.jl
# ============================================================

using BeliefSim
using BeliefSim.OUResets
using BeliefSim.Types
using LinearAlgebra
using Printf
using Dates
using DelimitedFiles
using Logging

# ----------------------------
# Configuration
# ----------------------------
const L = 12.0
const M = 401
const NU0_LIST = [0.0, 1.0, 7.0, 10.0, 12.5, 50.0]
const KAPPA_LIST = [5.0, 12.0, 20.0]
const STRICT_ASSERT = true
const TOL = 1e-9

const OUT_ROOT = joinpath("figs", "eigen_diag", "validity")
mkpath(OUT_ROOT)

# ----------------------------
# Utilities
# ----------------------------
function default_params(nu0::Float64)
    return Params(λ=0.85, σ=0.80, Θ=2.00, c0=0.80, hazard=StepHazard(nu0))
end

function safe_leading_odd_eigenvalue(p; kappa::Float64, L::Float64, M::Int)
    try
        return leading_odd_eigenvalue(p; κ=kappa, L=L, M=M, return_diag=true, return_mats=true)
    catch err
        @warn "leading_odd_eigenvalue failed" kappa L M err
        return (NaN, nothing)
    end
end

function residual_check_from_diag(diag)
    if diag === nothing || diag.A === nothing || diag.x === nothing
        return NaN
    end
    Mmat = diag.Mmat
    x = diag.x
    lambda = diag.eigval
    Mx = Mmat === nothing ? x : Mmat * x
    return norm(diag.A * x - lambda * Mx) / max(norm(Mx), eps())
end

function unpack_jump_samples(samples, n::Int)
    uvals = fill(NaN, n)
    jvals = fill(NaN, n)
    for i in 1:min(n, length(samples))
        uvals[i] = samples[i][1]
        jvals[i] = samples[i][2]
    end
    return uvals, jvals
end

# ----------------------------
# Main
# ----------------------------
function main()
    rows = Vector{Vector{Any}}()

    for nu0 in NU0_LIST
        p = default_params(nu0)
        for kappa in KAPPA_LIST
            lambda1, diag = safe_leading_odd_eigenvalue(p; kappa=kappa, L=L, M=M)

            nu0_expected = diag === nothing ? NaN : diag.nu0_expected
            nu0_used = diag === nothing ? NaN : diag.nu0_used
            residual_check = residual_check_from_diag(diag)

            samples = diag === nothing ? Tuple{Float64, Float64}[] : diag.A_jump_diag_sample
            uvals, jvals = unpack_jump_samples(samples, 3)

            println(@sprintf("nu0=%.3f kappa=%.2f lambda1=%+.6f nu0_expected=%.3f nu0_used=%.3f residual=%.2e",
                            nu0, kappa, lambda1, nu0_expected, nu0_used, residual_check))

            if diag !== nothing && isfinite(nu0_expected) && isfinite(nu0_used)
                if STRICT_ASSERT
                    @assert abs(nu0_used - nu0_expected) <= TOL
                elseif abs(nu0_used - nu0_expected) > TOL
                    @warn "nu0_used mismatch" nu0_expected nu0_used
                end
            end

            if diag !== nothing
                for jv in jvals
                    if isfinite(jv) && isfinite(nu0_expected)
                        if STRICT_ASSERT
                            @assert abs(jv + nu0_expected) <= TOL
                        elseif abs(jv + nu0_expected) > TOL
                            @warn "jump-loss scaling mismatch" nu0_expected jv
                        end
                    end
                end
            end

            push!(rows, Any[
                string(Dates.now()),
                nu0,
                kappa,
                lambda1,
                nu0_expected,
                nu0_used,
                residual_check,
                uvals[1], jvals[1],
                uvals[2], jvals[2],
                uvals[3], jvals[3]
            ])
        end
    end

    outpath = joinpath(OUT_ROOT, "hazard_wiring_sanity.csv")
    header = [
        "timestamp", "nu0", "kappa", "lambda1",
        "nu0_expected", "nu0_used", "residual_check",
        "u_sample_1", "jump_diag_1",
        "u_sample_2", "jump_diag_2",
        "u_sample_3", "jump_diag_3"
    ]
    open(outpath, "w") do io
        writedlm(io, permutedims(header), ',')
        for r in rows
            writedlm(io, permutedims(r), ',')
        end
    end

    @info "Saved" outpath
    println("\nDONE.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
