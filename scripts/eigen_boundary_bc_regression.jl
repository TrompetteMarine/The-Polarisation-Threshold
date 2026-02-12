#!/usr/bin/env julia
# ============================================================
# eigen_boundary_bc_regression.jl
#
# Boundary condition regression for leading odd eigenvalue.
#
# Usage:
#   julia --project=. scripts/eigen_boundary_bc_regression.jl
# ============================================================

using BeliefSim
using BeliefSim.OUResets
using BeliefSim.Types
using Statistics
using Printf
using Dates
using DelimitedFiles
using Logging
using Plots

# ----------------------------
# Configuration
# ----------------------------
const KAPPA_MIN = 0.0
const KAPPA_MAX = 20.0
const KAPPA_N = 200

const L = 12.0
const M = 401

const TAIL_FRAC = 0.2
const PLATEAU_TOL = 1e-6
const PLATEAU_RUN = 8

const OUT_ROOT = joinpath("figs", "eigen_diag", "boundary")
mkpath(OUT_ROOT)

# ----------------------------
# Utilities
# ----------------------------
function default_params()
    return Params(λ=0.85, σ=0.80, Θ=2.00, c0=0.80, hazard=StepHazard(10.5))
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
        plateau_onset_kappa = onset_kappa
    )
end

# ----------------------------
# Main
# ----------------------------
function main()
    kappa_grid = collect(range(KAPPA_MIN, KAPPA_MAX, length=KAPPA_N))
    p = default_params()

    modes = [
        ("reflecting", (:reflecting, nothing)),
        ("dirichlet", (:dirichlet, nothing)),
        ("legacy_damped", (:dirichlet, 10.0)),
    ]

    summary_rows = Vector{Vector{Any}}()

    plt = plot(; xlabel="kappa", ylabel="lambda1(kappa)",
               title=@sprintf("boundary comparison (L=%.1f, M=%d)", L, M),
               legend=:bottomleft, size=(900, 500), dpi=300)

    for (label, (boundary_mode, boundary_damp)) in modes
        lambda1 = fill(NaN, length(kappa_grid))

        for (i, kappa) in enumerate(kappa_grid)
            lambda1[i], _ = leading_odd_eigenvalue(
                p;
                κ=kappa,
                L=L,
                M=M,
                boundary=boundary_mode,
                boundary_damp=boundary_damp,
                return_diag=false
            )
        end

        crossings = find_zero_crossings(kappa_grid, lambda1)
        kappa_cross_1 = length(crossings) >= 1 ? crossings[1] : NaN
        kappa_cross_2 = length(crossings) >= 2 ? crossings[2] : NaN

        plateau = compute_plateau_stats(
            kappa_grid, lambda1;
            tail_frac=TAIL_FRAC,
            plateau_tol=PLATEAU_TOL,
            plateau_run=PLATEAU_RUN
        )

        diag_k20 = nothing
        kappa_target = 20.0
        if kappa_target >= KAPPA_MIN && kappa_target <= KAPPA_MAX
            _, diag_k20 = leading_odd_eigenvalue(
                p;
                κ=kappa_target,
                L=L,
                M=M,
                boundary=boundary_mode,
                boundary_damp=boundary_damp,
                return_diag=true
            )
        end

        boundary_mass_5pct = diag_k20 === nothing ? NaN : diag_k20.boundary_mass_5pct
        boundary_mass_1pct = diag_k20 === nothing ? NaN : diag_k20.boundary_mass_1pct
        parity_defect_odd = diag_k20 === nothing ? NaN : diag_k20.parity_defect_odd
        parity_defect_even = diag_k20 === nothing ? NaN : diag_k20.parity_defect_even

        @info "mode summary" label plateau.plateau_level plateau.plateau_onset_kappa kappa_cross_1 kappa_cross_2 boundary_mass_5pct parity_defect_odd

        push!(summary_rows, Any[
            string(Dates.now()),
            label,
            boundary_mode,
            boundary_damp === nothing ? "" : string(boundary_damp),
            plateau.plateau_level,
            plateau.plateau_onset_kappa,
            kappa_cross_1,
            kappa_cross_2,
            boundary_mass_5pct,
            boundary_mass_1pct,
            parity_defect_odd,
            parity_defect_even
        ])

        valid_idx = findall(isfinite, lambda1)
        plot!(plt, kappa_grid[valid_idx], lambda1[valid_idx]; label=label, linewidth=2)
    end

    outplot = joinpath(OUT_ROOT, "lambda_boundary_compare.pdf")
    savefig(plt, outplot)

    outcsv = joinpath(OUT_ROOT, "boundary_bc_summary.csv")
    header = [
        "timestamp", "mode", "boundary", "boundary_damp",
        "plateau_level", "plateau_onset",
        "kappa_cross_1", "kappa_cross_2",
        "boundary_mass_5pct", "boundary_mass_1pct",
        "parity_defect_odd", "parity_defect_even"
    ]
    open(outcsv, "w") do io
        writedlm(io, permutedims(header), ',')
        for row in summary_rows
            writedlm(io, permutedims(row), ',')
        end
    end

    @info "Saved" outplot outcsv
    println("\nDONE.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
