#!/usr/bin/env julia
# ============================================================
# eigen_branch_tracking_demo.jl
#
# Compare odd-eigenvalue selection methods and branch tracking.
#
# Usage:
#   julia --project=. scripts/eigen_branch_tracking_demo.jl
# ============================================================

using BeliefSim
using BeliefSim.OUResets
using BeliefSim.Types
using LinearAlgebra
using Printf
using Dates
using DelimitedFiles
using Plots

# ----------------------------
# Configuration
# ----------------------------
const L = 12.0
const M = 401
const KAPPA = 5.0
const NU0_LIST = [0.0, 1.0, 7.0, 10.0, 12.5, 50.0]
const ODD_TOL = 1e-6
const CORR_TOL = 1e-3

const OUT_ROOT = joinpath("figs", "eigen_diag", "branch_tracking")
mkpath(OUT_ROOT)

# ----------------------------
# Utilities
# ----------------------------
function default_params(nu0::Float64)
    return Params(λ=0.85, σ=0.80, Θ=2.00, c0=0.80, hazard=StepHazard(nu0))
end

function run_method(p; method::Symbol, prev_vec=nothing)
    if method == :corr
        return leading_odd_eigenvalue(
            p;
            κ=KAPPA,
            L=L,
            M=M,
            corr_tol=CORR_TOL,
            odd_tol=ODD_TOL,
            odd_select=:corr,
            track=false,
            prev_vec=nothing,
            oddspace=false,
            return_diag=true
        )
    elseif method == :parity
        return leading_odd_eigenvalue(
            p;
            κ=KAPPA,
            L=L,
            M=M,
            corr_tol=CORR_TOL,
            odd_tol=ODD_TOL,
            odd_select=:parity,
            track=false,
            prev_vec=nothing,
            oddspace=false,
            return_diag=true
        )
    elseif method == :track
        return leading_odd_eigenvalue(
            p;
            κ=KAPPA,
            L=L,
            M=M,
            corr_tol=CORR_TOL,
            odd_tol=ODD_TOL,
            odd_select=:parity,
            track=true,
            prev_vec=prev_vec,
            oddspace=false,
            return_diag=true
        )
    elseif method == :oddspace
        return leading_odd_eigenvalue(
            p;
            κ=KAPPA,
            L=L,
            M=M,
            corr_tol=CORR_TOL,
            odd_tol=ODD_TOL,
            odd_select=:parity,
            track=false,
            prev_vec=nothing,
            oddspace=true,
            return_diag=true
        )
    else
        error("Unknown method: $(method)")
    end
end

function diag_field(diag, name::Symbol, default=NaN)
    if diag === nothing
        return default
    end
    return getproperty(diag, name)
end

# ----------------------------
# Main
# ----------------------------
function main()
    methods = [:corr, :parity, :track, :oddspace]
    rows = Vector{Vector{Any}}()

    prev_track_vec = nothing

    for nu0 in NU0_LIST
        p = default_params(nu0)
        for method in methods
            prev = method == :track ? prev_track_vec : nothing
            lambda1, diag = run_method(p; method=method, prev_vec=prev)
            if method == :track && diag !== nothing
                prev_track_vec = diag.x
            end

            parity_odd = diag_field(diag, :parity_defect_odd)
            parity_even = diag_field(diag, :parity_defect_even)
            boundary_mass_5pct = diag_field(diag, :boundary_mass_5pct)
            boundary_mass_1pct = diag_field(diag, :boundary_mass_1pct)
            hazard_mass = diag_field(diag, :hazard_mass)
            overlap_prev = diag_field(diag, :overlap_prev)
            residual = diag_field(diag, :residual)
            odd_fallback = diag_field(diag, :odd_selection_fallback, 0)

            println(@sprintf(
                "method=%-8s nu0=%6.2f lambda1=%+10.6f parity_odd=%.2e boundary5=%.2e hazard=%.2e overlap=%.2e resid=%.2e",
                string(method), nu0, lambda1, parity_odd, boundary_mass_5pct,
                hazard_mass, overlap_prev, residual
            ))

            push!(rows, Any[
                string(Dates.now()),
                string(method),
                nu0,
                KAPPA,
                lambda1,
                parity_odd,
                parity_even,
                boundary_mass_5pct,
                boundary_mass_1pct,
                hazard_mass,
                overlap_prev,
                residual,
                odd_fallback
            ])
        end
    end

    out_csv = joinpath(OUT_ROOT, "branch_tracking_demo.csv")
    header = [
        "timestamp", "method", "nu0", "kappa", "lambda1",
        "parity_defect_odd", "parity_defect_even",
        "boundary_mass_5pct", "boundary_mass_1pct",
        "hazard_mass", "overlap_prev", "residual",
        "odd_selection_fallback"
    ]
    open(out_csv, "w") do io
        writedlm(io, permutedims(header), ',')
        for r in rows
            writedlm(io, permutedims(r), ',')
        end
    end

    plt = plot(
        xlabel="nu0",
        ylabel="lambda1(kappa=5)",
        title="Odd eigenvalue branch selection comparison",
        linewidth=2,
        marker=:circle,
        legend=:topright,
        size=(850, 480),
        dpi=300
    )
    for method in methods
        xs = Float64[]
        ys = Float64[]
        for r in rows
            if r[2] == string(method)
                nu0 = r[3]
                lambda1 = r[5]
                if isfinite(lambda1)
                    push!(xs, nu0)
                    push!(ys, lambda1)
                end
            end
        end
        if !isempty(xs)
            plot!(plt, xs, ys; label=string(method))
        end
    end
    out_plot = joinpath(OUT_ROOT, "branch_tracking_lambda_vs_nu0.pdf")
    savefig(plt, out_plot)

    @info "Saved" out_csv out_plot
    println("\nDONE.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
