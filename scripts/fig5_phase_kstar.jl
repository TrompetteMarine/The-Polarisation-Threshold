#!/usr/bin/env julia
# =============================================================================
# Figure 5: Phase diagram of kappa*(c0, sigma)
# - Fixed baseline: lambda, theta, step hazard nu0 (unless CLI overrides)
# - Outputs: figs/fig5_phase_kstar.pdf and outputs/fig5_phase_kstar/kstar_grid.csv
# =============================================================================

using Pkg; Pkg.activate(".")

using Random
using Statistics
using Printf
using Dates
using CSV
using DataFrames

using BeliefSim
using BeliefSim.Types
using BeliefSim.Stats

# Optional plotting backends
CAIRO_AVAILABLE = false
try
    using CairoMakie
    CAIRO_AVAILABLE = true
catch
    @warn "CairoMakie not available, falling back to Plots.jl"
    using Plots
end

function parse_args()
    using ArgParse
    s = ArgParseSettings()
    @add_arg_table s begin
        "--lambda"; arg_type=Float64; default=0.85; help="Mean reversion λ"
        "--theta"; arg_type=Float64; default=2.0; help="Threshold Θ"
        "--nu0"; arg_type=Float64; default=10.6; help="Step hazard height ν0"
        "--c0-min"; arg_type=Float64; default=0.1; help="Minimum c0"
        "--c0-max"; arg_type=Float64; default=0.9; help="Maximum c0"
        "--sigma-min"; arg_type=Float64; default=0.2; help="Minimum σ"
        "--sigma-max"; arg_type=Float64; default=1.0; help="Maximum σ"
        "--nc0"; arg_type=Int; default=17; help="Grid size for c0"
        "--nsigma"; arg_type=Int; default=20; help="Grid size for σ"
        "--outdir"; arg_type=String; default="outputs/fig5_phase_kstar"; help="Output directory"
        "--fast"; action=:store_true; help="Enable fast mode (smaller grid/precision)"
        "--with-vstar"; action=:store_true; help="Compute V* per grid point (expensive)"
    end
    return parse_args(s)
end

function main()
    args = parse_args()

    fast_env = get(ENV, "FAST_MODE", "0") in ("1", "true", "yes", "on")
    fast = args["fast"] || fast_env
    with_vstar = args["with-vstar"]

    λ = args["lambda"]
    Θ = args["theta"]
    nu0 = args["nu0"]

    n_c0 = fast ? min(args["nc0"], 9) : args["nc0"]
    n_sigma = fast ? min(args["nsigma"], 10) : args["nsigma"]

    c0_min = args["c0-min"]
    c0_max = args["c0-max"]
    σ_min = args["sigma-min"]
    σ_max = args["sigma-max"]

    outdir = args["outdir"]
    figdir = "figs"
    mkpath(outdir)
    mkpath(figdir)

    # Critical kappa spectral grid parameters
    κmax = fast ? 2.5 : 3.0
    grid_points = fast ? 40 : 60
    L = fast ? 4.0 : 5.0
    M = fast ? 301 : 401

    # Optional V* budget (expensive for full grid)
    vstar_N = fast ? 1500 : 3000
    vstar_T = fast ? 120.0 : 200.0
    vstar_dt = fast ? 0.02 : 0.02
    vstar_burn = fast ? 30.0 : 50.0

    println("=" ^ 72)
    println("FIGURE 5: PHASE DIAGRAM κ*(c0, σ)")
    println("=" ^ 72)
    @printf("λ=%.3f, Θ=%.3f, ν0=%.3f\n", λ, Θ, nu0)
    @printf("c0 ∈ [%.2f, %.2f] (%d), σ ∈ [%.2f, %.2f] (%d)\n",
            c0_min, c0_max, n_c0, σ_min, σ_max, n_sigma)
    println("FAST_MODE = $(fast)")
    println("Compute V* per grid point = $(with_vstar)")
    println("Output directory: $outdir")

    c0_grid = collect(range(c0_min, c0_max; length=n_c0))
    σ_grid = collect(range(σ_min, σ_max; length=n_sigma))

    results = Vector{NamedTuple}()
    kstar_mat = fill(NaN, n_sigma, n_c0)
    vstar_mat = fill(NaN, n_sigma, n_c0)

    t_start = time()
    for (i_c0, c0) in enumerate(c0_grid)
        row_start = time()
        for (i_s, σ) in enumerate(σ_grid)
            p = Params(λ=λ, σ=σ, Θ=Θ, c0=c0, hazard=StepHazard(nu0))
            status = "ok"
            notes = ""
            kstar = NaN
            vstar = NaN
            try
                if with_vstar
                    vstar = estimate_Vstar(p; N=vstar_N, T=vstar_T, dt=vstar_dt,
                                           burn_in=vstar_burn, seed=101)
                end
                kstar = critical_kappa(p; Vstar=vstar, κmax=κmax,
                                       grid_points=grid_points, L=L, M=M)
            catch err
                status = "fail"
                notes = sprint(showerror, err)
                kstar = NaN
            end

            kstar_mat[i_s, i_c0] = kstar
            vstar_mat[i_s, i_c0] = vstar
            push!(results, (c0=c0, sigma=σ, kstar=kstar, vstar=vstar,
                            status=status, notes=notes))
        end
        row_time = time() - row_start
        @printf("Row %d/%d (c0=%.3f) done in %.2fs\n", i_c0, n_c0, c0, row_time)
    end
    total_time = time() - t_start

    # Save CSV
    df = DataFrame(results)
    csv_path = joinpath(outdir, "kstar_grid.csv")
    CSV.write(csv_path, df)

    # Write summary
    summary_path = joinpath(outdir, "summary.txt")
    open(summary_path, "w") do io
        println(io, "Figure 5: κ*(c0, σ) phase diagram")
        println(io, "Timestamp: $(Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))")
        @printf(io, "λ=%.4f, Θ=%.4f, ν0=%.4f\n", λ, Θ, nu0)
        @printf(io, "c0 range [%.3f, %.3f] with %d points\n", c0_min, c0_max, n_c0)
        @printf(io, "σ range  [%.3f, %.3f] with %d points\n", σ_min, σ_max, n_sigma)
        @printf(io, "κ grid: κmax=%.2f, grid_points=%d, L=%.2f, M=%d\n", κmax, grid_points, L, M)
        println(io, "FAST_MODE = $(fast)")
        println(io, "Compute V* per grid point = $(with_vstar)")
        println(io, "V* values recorded in kstar_grid.csv column vstar (NaN if disabled)")
        if with_vstar
            @printf(io, "V* budget: N=%d, T=%.1f, dt=%.3f, burn_in=%.1f\n",
                    vstar_N, vstar_T, vstar_dt, vstar_burn)
        end
        @printf(io, "Runtime: %.2f seconds\n", total_time)
        println(io, "Output CSV: $csv_path")
    end

    # Plot
    if CAIRO_AVAILABLE
        fig = Figure(size=(900, 650), fontsize=14, backgroundcolor=:white)
        ax = Axis(fig[1, 1], xlabel="c0", ylabel="σ", title="Critical coupling κ*(c0, σ)")
        hm = heatmap!(ax, c0_grid, σ_grid, kstar_mat; colormap=:viridis)
        contour!(ax, c0_grid, σ_grid, kstar_mat; levels=8, color=:black, linewidth=1.0)
        Colorbar(fig[1, 2], hm, label="κ*")
        fig_path = joinpath(figdir, "fig5_phase_kstar.pdf")
        save(fig_path, fig)
    else
        default(fontfamily="Computer Modern")
        plt = heatmap(c0_grid, σ_grid, kstar_mat; xlabel="c0", ylabel="σ",
                      title="Critical coupling κ*(c0, σ)", c=:viridis,
                      colorbar_title="κ*", size=(900, 650))
        contour!(plt, c0_grid, σ_grid, kstar_mat; levels=8, color=:black, linewidth=1.0)
        fig_path = joinpath(figdir, "fig5_phase_kstar.pdf")
        savefig(plt, fig_path)
    end

    println("Saved: $(joinpath(figdir, "fig5_phase_kstar.pdf"))")
    println("Saved: $csv_path")
    println("Saved: $summary_path")
    @printf("Total runtime: %.2fs\n", total_time)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
