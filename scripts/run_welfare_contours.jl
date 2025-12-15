#!/usr/bin/env julia
"""
Bifurcation-Based Welfare Analysis - Main Script

This script implements the complete welfare analysis pipeline described in the
"welfare_simulation_update" specification:

1. Computes stationary variance V*(θ, c₀) and welfare loss L(V) over a 2D grid
2. Simulates decentralised vs social planner optimization
3. Generates publication-grade contour plots with isolines
4. Saves results to CSV for further analysis

Theory: Sections 2-6, especially Theorem 4.8 for the welfare loss formula.
"""

using BeliefSim
using BeliefSim.OUResets
using BeliefSim.Types
using Plots
using LaTeXStrings
using Printf
using CSV
using DataFrames
using Statistics
using Optim

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

const DEBUG = true

# Baseline parameters (from paper calibration)
# Calibration note (Section 6 MFC):
# - αV and K_reset chosen so J_ind has an interior minimum around θ ≈ 0.9 for c₀ ≈ 0.45
# - c_pol set high enough so Φ shifts the planner leftward (θ_soc < θ_dec, V_soc < V_dec)
const PARAMS_BASE = Params(
    λ = 0.65,
    σ = 1.15,
    Θ = 0.87,  # Will be varied in grid
    c0 = 0.50,  # Will be varied in grid
    hazard = StepHazard(0.5)
)

const WELFARE_PARAMS = (
    αV = 1.0,
    K_reset = 0.35,
    c_pol = 2.0,
    φA = 0.1,
    κ_ratio_max = 2.0,
    use_fallback = true,
)
# Qualitative target (Prop. 6.3): J_ind attains an interior minimum near θ ≈ 0.9
# for c₀ ≈ 0.4–0.5, and Φ then pulls the planner to a lower θ (θ_soc < θ_dec,
# V_soc < V_dec). Adjust αV/K_reset/c_pol here if diagnostics show corr(J_ind, Φ)
# ≳ 0.95 or optima coincide.

# Grid resolution for (θ, c₀) space
const N_THETA = 25  # Tolerance threshold grid points
const N_C0 = 25     # Reset contraction grid points

# Parameter ranges
const THETA_MIN = 0.5
const THETA_MAX = 1.2
const C0_MIN = 0.2
const C0_MAX = 0.8

# Simulation settings
const N_AGENTS = 10_000
const T_SIM = 200.0
const DT = 0.01
const BURN_IN = 50.0

# Spectral solver settings
const L_SPECTRAL = 5.0
const M_SPECTRAL = 251  # Grid points for FP solver

# ----------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------

"""
    finite_clims(data; quantiles=(0.02, 0.98))

Compute robust color limits from finite values using quantiles.
"""
function finite_clims(data; quantiles=(0.02, 0.98))
    finite_vals = filter(isfinite, vec(data))
    if isempty(finite_vals)
        return (-1.0, 1.0)
    end
    return quantile(finite_vals, [quantiles[1], quantiles[2]])
end

"""
    save_both(plt, filename)

Save plot as both PDF and PNG at high DPI.
"""
function save_both(plt, filename)
    base = splitext(filename)[1]
    savefig(plt, "$(base).pdf")
    savefig(plt, "$(base).png")
    println("  ✓ Saved: $(basename(base)).pdf and .png")
end

surface_stats(arr) = begin
    vals = filter(isfinite, vec(arr))
    isempty(vals) && return (NaN, NaN, NaN)
    return (minimum(vals), maximum(vals), std(vals))
end

# ----------------------------------------------------------------------
# Step 1: Build 2D Welfare Surfaces
# ----------------------------------------------------------------------

"""
    compute_welfare_surface(params_base; kwargs...)

Compute 2D welfare surfaces L(θ, c₀) for both decentralised and social planner.

Returns a named tuple with:
- theta_grid, c0_grid: Parameter grids
- V_surface: Stationary variance V*(θ, c₀)
- L_dec_surface, L_soc_surface: Welfare losses
- W_dec_surface, W_soc_surface: Welfare levels (negative of the losses)
- Phi_surface: Externality Φ(V*)
- gap_surface: Externality wedge L_soc − L_dec (equals Φ when finite)
- lambda10_surface, lambda1_dot_surface: Spectral quantities
- kappa_star_surface: Polarisation thresholds κ*(V)

# Diagnostics: `main()` prints corr(L_dec, Φ) and fallback share so we can
# detect if Φ collapses to an affine rescaling of J_ind (corr ≳ 0.95).
"""
function compute_welfare_surface(
    params_base::Params;
    n_theta::Int = N_THETA,
    n_c0::Int = N_C0,
    theta_range = (THETA_MIN, THETA_MAX),
    c0_range = (C0_MIN, C0_MAX),
    verbose::Bool = true,
    # ---- hyper-parameters matching Section 6 ---------------------------
    αV::Float64 = WELFARE_PARAMS.αV,          # weight on dispersion in J_ind
    K_reset::Float64 = WELFARE_PARAMS.K_reset,     # reset-cost scale in J_ind
    c_pol::Float64 = WELFARE_PARAMS.c_pol,       # bifurcation loss weight c_pol
    φA::Float64 = WELFARE_PARAMS.φA,          # tail-risk weight in \Phi
    κ_ratio_max::Float64 = 2.0, # κ_max/κ* for \Phi
    δκ_spec::Float64 = 0.01,    # FD step for λ̇₁₀
    b_default::Float64 = 0.5,   # cubic coefficient b(V)
    use_fallback::Bool = WELFARE_PARAMS.use_fallback,  # allow Φ fallback to c_pol * V if spectral fails
)
    verbose && println("\n" * "="^70)
    verbose && println("Computing 2D Welfare Surfaces")
    verbose && println("="^70)

    # Diagnostics: reset Φ path counters so we can report spectral vs fallback usage
    reset_bifurcation_stats!()

    theta_grid = collect(range(theta_range[1], theta_range[2]; length = n_theta))
    c0_grid    = collect(range(c0_range[1],   c0_range[2];   length = n_c0))

    verbose && @printf("Grid: %d × %d = %d points\n", n_theta, n_c0, n_theta * n_c0)
    verbose && println("θ ∈ [$(theta_range[1]), $(theta_range[2])]")
    verbose && println("c₀ ∈ [$(c0_range[1]), $(c0_range[2])]")

    # Preallocate (θ = rows, c₀ = cols)
    V_surface           = Matrix{Float64}(undef, n_theta, n_c0)
    L_dec_surface       = Matrix{Float64}(undef, n_theta, n_c0)
    L_soc_surface       = Matrix{Float64}(undef, n_theta, n_c0)
    Phi_surface         = Matrix{Float64}(undef, n_theta, n_c0)
    lambda10_surface    = Matrix{Float64}(undef, n_theta, n_c0)
    lambda1_dot_surface = Matrix{Float64}(undef, n_theta, n_c0)
    kappa_star_surface  = Matrix{Float64}(undef, n_theta, n_c0)

    # NEW: welfare (negative of loss) and externality wedge
    W_dec_surface       = Matrix{Float64}(undef, n_theta, n_c0)
    W_soc_surface       = Matrix{Float64}(undef, n_theta, n_c0)
    gap_surface         = Matrix{Float64}(undef, n_theta, n_c0)  # L_soc − L_dec

    total_points = n_theta * n_c0
    completed    = 0

    verbose && println("\nComputing grid points...")

    for (i, theta) in enumerate(theta_grid)
        for (j, c0) in enumerate(c0_grid)

            # 1. Stationary dispersion V*(θ,c₀)
            V_ij = compute_stationary_variance(
                theta, c0, params_base;
                N       = N_AGENTS,
                T       = T_SIM,
                dt      = DT,
                burn_in = BURN_IN,
                seed    = 100 * i + j,
            )
            V_surface[i, j] = V_ij

            if isfinite(V_ij) && V_ij > 0.0
                # Params at this policy (θ,c₀)
                p_ij = Params(
                    λ      = params_base.λ,
                    σ      = params_base.σ,
                    Θ      = theta,
                    c0     = c0,
                    hazard = params_base.hazard,
                )

                # 2a. Decentralised welfare: J_ind only
                L_dec = welfare_loss(
                    V_ij, theta, p_ij;
                    regime   = :dec,
                    αV       = αV,
                    K_reset  = K_reset,
                )
                L_dec_surface[i, j] = L_dec

                # 2b. Spectral data and planner welfare
                λ10, λ1_dot = compute_lambda1_and_derivative(
                    V_ij, p_ij;
                    δκ = δκ_spec,
                    L  = L_SPECTRAL,
                    M  = M_SPECTRAL,
                )

                lambda10_surface[i, j]    = λ10
                lambda1_dot_surface[i, j] = λ1_dot

                kappa_star_surface[i, j] =
                    (isfinite(λ10) && isfinite(λ1_dot) && abs(λ1_dot) > 1e-12) ?
                    -λ10 / λ1_dot : NaN

                Phi_ij = bifurcation_loss(
                    V_ij, p_ij;
                    c_pol       = c_pol,
                    φA          = φA,
                    κ_ratio_max = κ_ratio_max,
                    δκ          = δκ_spec,
                    L           = L_SPECTRAL,
                    M           = M_SPECTRAL,
                    b_default   = b_default,
                    lambda10    = λ10,
                    lambda1_dot = λ1_dot,
                    use_fallback = use_fallback,
                )
                Phi_surface[i, j] = Phi_ij

                L_soc = welfare_loss(
                    V_ij, theta, p_ij;
                    regime      = :soc,
                    αV          = αV,
                    K_reset     = K_reset,
                    c_pol       = c_pol,
                    φA          = φA,
                    κ_ratio_max = κ_ratio_max,
                    δκ          = δκ_spec,
                    L           = L_SPECTRAL,
                    M           = M_SPECTRAL,
                    b_default   = b_default,
                    Phi         = Phi_ij,
                    lambda10    = λ10,
                    lambda1_dot = λ1_dot,
                    use_fallback = use_fallback,
                )
                L_soc_surface[i, j] = L_soc
                W_dec_surface[i, j] = -L_dec
                W_soc_surface[i, j] = -L_soc
                gap_surface[i, j]   = L_soc - L_dec  # should match Phi_ij
            else
                L_dec_surface[i, j]       = NaN
                L_soc_surface[i, j]       = NaN
                Phi_surface[i, j]         = NaN
                lambda10_surface[i, j]    = NaN
                lambda1_dot_surface[i, j] = NaN
                kappa_star_surface[i, j]  = NaN
                W_dec_surface[i, j]       = NaN
                W_soc_surface[i, j]       = NaN
                gap_surface[i, j]         = NaN
            end

            completed += 1
            if verbose && (completed % 50 == 0 || completed == total_points)
                progress = 100.0 * completed / total_points
                @printf("  Progress: %d/%d (%.1f%%)\n", completed, total_points, progress)
            end
        end
    end

    verbose && println("✓ Surface computation complete")

    return (
        theta_grid          = theta_grid,
        c0_grid             = c0_grid,
        V_surface           = V_surface,
        L_dec_surface       = L_dec_surface,
        L_soc_surface       = L_soc_surface,
        Phi_surface         = Phi_surface,
        lambda10_surface    = lambda10_surface,
        lambda1_dot_surface = lambda1_dot_surface,
        kappa_star_surface  = kappa_star_surface,
        W_dec_surface       = W_dec_surface,
        W_soc_surface       = W_soc_surface,
        gap_surface         = gap_surface,
    )
end

# Simple diagnostic helper to see where J_ind minimises along θ for fixed c₀ slices
function diagnose_private_optima(L_dec_surface, theta_grid, c0_grid; c0_targets = (0.4, 0.5))
    for c0_val in c0_targets
        j = findmin(abs.(c0_grid .- c0_val))[2]
        slice = L_dec_surface[:, j]
        if any(isfinite, slice)
            Lmin, idx = findmin(slice)
            θ_opt = theta_grid[idx]
            boundary = idx == 1 || idx == length(theta_grid)
            println("  Slice c₀≈$(round(c0_grid[j], digits=3)): argmin θ=$(round(θ_opt, digits=3)) (boundary? $(boundary)), L=$(Lmin)")
        else
            println("  Slice c₀≈$(round(c0_val, digits=3)): no finite entries")
        end
    end
end


# ----------------------------------------------------------------------
# Step 2: Find Optimal Policies
# ----------------------------------------------------------------------

"""
    find_optimal_policy(L_surface, theta_grid, c0_grid; mode=:min)

Find optimal (θ, c₀) by minimizing or maximizing welfare surface.

Returns (theta_opt, c0_opt, L_opt, idx_theta, idx_c0).
"""
function find_optimal_policy(L_surface, theta_grid, c0_grid; mode=:min)
    # Filter finite values
    finite_mask = isfinite.(L_surface)

    if !any(finite_mask)
        return (NaN, NaN, NaN, 0, 0)
    end

    if mode == :min
        L_opt, idx = findmin(L_surface[finite_mask] |> vec)
    else
        L_opt, idx = findmax(L_surface[finite_mask] |> vec)
    end

    # Find 2D index
    finite_indices = findall(finite_mask)
    idx_2d = finite_indices[idx]
    i, j = Tuple(idx_2d)

    theta_opt = theta_grid[i]
    c0_opt = c0_grid[j]

    return (theta_opt, c0_opt, L_opt, i, j)
end

# ----------------------------------------------------------------------
# Step 3: Publication-Grade Contour Plots
# ----------------------------------------------------------------------

"""
    plot_welfare_contours(data; nlevels=20, mode=:welfare)

Two-panel contour plot comparing decentralised vs social planner welfare.
"""
function plot_welfare_contours(data; nlevels::Int = 20, mode::Symbol = :welfare)
    theta_grid = data.theta_grid
    c0_grid    = data.c0_grid

    L_dec = data.L_dec_surface'
    L_soc = data.L_soc_surface'
    invalid = .!isfinite.(L_dec) .| .!isfinite.(L_soc)
    L_dec[invalid] .= NaN
    L_soc[invalid] .= NaN

    # Optimal policies (using original orientation)
    theta_dec, c0_dec, L_dec_opt, _, _ =
        find_optimal_policy(data.L_dec_surface, theta_grid, c0_grid; mode=:min)
    theta_soc, c0_soc, L_soc_opt, _, _ =
        find_optimal_policy(data.L_soc_surface, theta_grid, c0_grid; mode=:min)

    println("\nOptimal Policies:")
    @printf("  Decentralised: (θ=%.3f, c₀=%.3f) → L=%.5f\n", theta_dec, c0_dec, L_dec_opt)
    @printf("  Social planner: (θ=%.3f, c₀=%.3f) → L=%.5f\n", theta_soc, c0_soc, L_soc_opt)

    clim_dec = finite_clims(L_dec)
    clim_soc = finite_clims(L_soc)
    clim_global = (min(clim_dec[1], clim_soc[1]), max(clim_dec[2], clim_soc[2]))
    levels_shared = collect(range(clim_global[1], clim_global[2]; length=nlevels))
    iso_level = mean(clim_global)

    plt = plot(layout = (1, 2), size = (1600, 600), dpi = 300, margin = 8Plots.mm)

    # Panel 1: Decentralised welfare
    contourf!(plt[1], theta_grid, c0_grid, L_dec;
        levels = levels_shared, c = :viridis,
        clims = clim_global,
        xlabel = "Tolerance threshold θ",
        ylabel = "Reset contraction c₀",
        title  = "Decentralised Welfare L_dec(θ,c₀) = J_ind(θ,c₀)",
        fillalpha = 1.0, linewidth = 0, nan_color = :white,
        colorbar = true, colorbar_title = "L_dec")

    contour!(plt[1], theta_grid, c0_grid, L_dec;
        levels = levels_shared, color = :black, linewidth = 0.5,
        linealpha = 0.3, label = false, colorbar = false)
    contour!(plt[1], theta_grid, c0_grid, L_dec;
        levels = [iso_level], color = :orange, linewidth = 1.2,
        linestyle = :dash, label = "shared iso-level", colorbar = false)

    scatter!(plt[1], [theta_dec], [c0_dec];
        color = :red, markersize = 8, markershape = :star5,
        label = "Dec optimum (θ=$(round(theta_dec, digits=2)), c₀=$(round(c0_dec, digits=2)))")

    # Panel 2: Social planner welfare
    contourf!(plt[2], theta_grid, c0_grid, L_soc;
        levels = levels_shared, c = :plasma,
        clims = clim_global,
        xlabel = "Tolerance threshold θ",
        ylabel = "Reset contraction c₀",
        title  = "Social Planner Welfare L_soc(θ,c₀) = J_ind + Φ(V*)",
        fillalpha = 1.0, linewidth = 0, nan_color = :white,
        colorbar = true, colorbar_title = "L_soc")

    contour!(plt[2], theta_grid, c0_grid, L_soc;
        levels = levels_shared, color = :black, linewidth = 0.5,
        linealpha = 0.3, label = false, colorbar = false)
    contour!(plt[2], theta_grid, c0_grid, L_soc;
        levels = [iso_level], color = :orange, linewidth = 1.2,
        linestyle = :dash, label = false, colorbar = false)

    scatter!(plt[2], [theta_soc], [c0_soc];
        color = :cyan, markersize = 8, markershape = :star5,
        label = "Soc optimum (θ=$(round(theta_soc, digits=2)), c₀=$(round(c0_soc, digits=2)))")

    mkpath("figs")
    save_both(plt, "figs/fig_welfare_contours_corrected")

    return plt
end

"""
    plot_externality_surface(data; nlevels=20)

Single-panel contour plot of the bifurcation externality

    Φ(θ,c₀) = L_soc − L_dec,

interpreted as the planner's additional loss from polarisation risk.
"""
function plot_externality_surface(data; nlevels::Int = 20)
    theta_grid = data.theta_grid
    c0_grid    = data.c0_grid

    # surfaces for plotting
    Phi = data.gap_surface'
    invalid = .!isfinite.(Phi)
    Phi[invalid] .= NaN

    # Re-use optimal points (from original orientation)
    theta_dec, c0_dec, _, _, _ =
        find_optimal_policy(data.L_dec_surface, theta_grid, c0_grid; mode=:min)
    theta_soc, c0_soc, _, _, _ =
        find_optimal_policy(data.L_soc_surface, theta_grid, c0_grid; mode=:min)

    clim_ext   = finite_clims(Phi)
    levels_ext = collect(range(clim_ext[1], clim_ext[2]; length=nlevels))

    plt = plot(size=(800, 600), dpi=300, margin=8Plots.mm)

    contourf!(plt, theta_grid, c0_grid, Phi;
        levels = levels_ext, c = :balance,
        clims = (clim_ext[1], clim_ext[2]),
        xlabel = "Tolerance threshold θ",
        ylabel = "Reset contraction c₀",
        title  = "Bifurcation Externality Φ(θ,c₀) = L_soc − L_dec",
        fillalpha = 1.0, linewidth = 0, nan_color = :white,
        colorbar = true, colorbar_title = "Φ")

    contour!(plt, theta_grid, c0_grid, Phi;
        levels = levels_ext, color = :black, linewidth = 0.5,
        linealpha = 0.3, label = false, colorbar = false)

    # In principle Φ ≥ 0; still, emphasise zero contour if present
    contour!(plt, theta_grid, c0_grid, Phi;
        levels = [0.0], color = :black, linewidth = 2.0,
        linestyle = :solid, label = "Φ = 0", colorbar = false)

    # Mark both optima for visual wedge
    scatter!(plt, [theta_dec, theta_soc], [c0_dec, c0_soc];
        color = [:red, :cyan], markersize = 8, markershape = :star5,
        label = ["Dec optimum" "Soc optimum"])

    mkpath("figs")
    save_both(plt, "figs/fig_externality_surface")

    return plt
end

# ----------------------------------------------------------------------
# Step 4: Save Results to CSV
# ----------------------------------------------------------------------

"""
    save_results_csv(data, filename)

Save welfare surface data to CSV for further analysis.
"""
function save_results_csv(data, filename)
    theta_grid = data.theta_grid
    c0_grid = data.c0_grid

    # Flatten 2D arrays to 1D for DataFrame
    rows = []

    for (i, theta) in enumerate(theta_grid)
        for (j, c0) in enumerate(c0_grid)
            push!(rows, (
                theta = theta,
                c0 = c0,
                V = data.V_surface[i, j],
                L_dec = data.L_dec_surface[i, j],
                L_soc = data.L_soc_surface[i, j],
                Phi = data.Phi_surface[i, j],
                lambda10 = data.lambda10_surface[i, j],
                lambda1_dot = data.lambda1_dot_surface[i, j],
                kappa_star = data.kappa_star_surface[i, j],
                W_dec = data.W_dec_surface[i, j],
                W_soc = data.W_soc_surface[i, j],
                gap = data.gap_surface[i, j],
            ))
        end
    end

    df = DataFrame(rows)
    mkpath("outputs")
    CSV.write(filename, df)
    println("  ✓ Saved: $(basename(filename))")
end

# ----------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------

function main()
    println("\n" * "="^70)
    println("Bifurcation-Based Welfare Analysis")
    println("="^70)

    println("\nParameters:")
    println("  λ = $(PARAMS_BASE.λ), σ = $(PARAMS_BASE.σ)")
    println("  Hazard: ", PARAMS_BASE.hazard)

    # Step 1: Compute welfare surfaces
    data = compute_welfare_surface(
        PARAMS_BASE;
        verbose = true,
        αV = WELFARE_PARAMS.αV,
        K_reset = WELFARE_PARAMS.K_reset,
        c_pol = WELFARE_PARAMS.c_pol,
        φA = WELFARE_PARAMS.φA,
        κ_ratio_max = WELFARE_PARAMS.κ_ratio_max,
        use_fallback = WELFARE_PARAMS.use_fallback,
    )

    # Diagnostics: see whether Φ is collinear with J_ind
    Jmin, Jmax, Jstd = surface_stats(data.L_dec_surface)
    Pmin, Pmax, Pstd = surface_stats(data.Phi_surface)
    mask = isfinite.(data.L_dec_surface) .& isfinite.(data.Phi_surface)
    mask_vec = vec(mask)
    corr_J_Phi = any(mask_vec) ? cor(vec(data.L_dec_surface)[mask_vec], vec(data.Phi_surface)[mask_vec]) : NaN
    stats = get_bifurcation_stats()
    total_calls = stats.spectral + stats.fallback
    fallback_share = total_calls == 0 ? NaN : stats.fallback / total_calls

    println("\nDiagnostics (welfare components):")
    @printf("  J_ind stats: min=%.4f, max=%.4f, std=%.4f\n", Jmin, Jmax, Jstd)
    @printf("  Phi stats  : min=%.4f, max=%.4f, std=%.4f\n", Pmin, Pmax, Pstd)
    @printf("  corr(J_ind, Phi) over finite grid = %.4f\n", corr_J_Phi)
    @printf("  Φ fallback usage: %d spectral, %d fallback (share=%.2f)\n",
        stats.spectral, stats.fallback, fallback_share)

    println("\nPrivate-cost slices (θ argmin by c₀):")
    diagnose_private_optima(data.L_dec_surface, data.theta_grid, data.c0_grid; c0_targets = (0.4, 0.5))

    # Step 2: Generate contour plots
    println("\n" * "─"^70)
    println("Generating Welfare Surface Plots")
    println("─"^70)
    plot_welfare_contours(data)

    println("\n" * "─"^70)
    println("Generating Externality Surface Plot")
    println("─"^70)
    plot_externality_surface(data)


    # Step 3: Save CSV
    println("\n" * "─"^70)
    println("Saving Results to CSV")
    println("─"^70)
    save_results_csv(data, "outputs/welfare_corrected.csv")

    println("\n" * "="^70)
    println("Welfare Analysis Complete!")
    println("="^70)
    println("\nGenerated files:")
    println("  - figs/fig_welfare_contours_corrected.pdf (and .png)")
    println("  - figs/fig_externality_surface.pdf (and .png)")
    println("  - outputs/welfare_corrected.csv")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
