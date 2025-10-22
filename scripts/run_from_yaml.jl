#!/usr/bin/env julia
using YAML, CSV, DataFrames, Random, Printf, Statistics
using BeliefSim
using BeliefSim.Types: Params, KappaSweepResult
using BeliefSim.Stats: estimate_Vstar, critical_kappa, sweep_kappa, pitchfork_fit
using BeliefSim.Model: euler_maruyama_step!, reset_step!
using BeliefSim.Plotting: plot_bifurcation, plot_vector_field, plot_orbit
using BeliefSim.Utils: safe_array_slice, validate_simulation_params, safe_time_range
using Plots: savefig

function mkparams(pdict)
    hazard = pdict["hazard"]
    h = begin
        kind = String(lowercase(hazard["kind"]))
        if kind == "step"
            BeliefSim.Types.StepHazard(hazard["nu0"])
        elseif kind == "logistic"
            BeliefSim.Types.LogisticHazard(hazard["numax"], hazard["beta"])
        else
            error("Unknown hazard kind: $(hazard["kind"])")
        end
    end
    return Params(λ=pdict["lambda"], σ=pdict["sigma"], 
                 Θ=pdict["theta"], c0=pdict["c0"], 
                 hazard=h)
end

function main(cfgpath::String)
    cfg = YAML.load_file(cfgpath)
    name = get(cfg, "name", "run")
    seed = get(cfg, "seed", 0)
    N = cfg["N"]
    T = cfg["T"]
    dt = cfg["dt"]
    burn_in = cfg["burn_in"]
    p = mkparams(cfg["params"])

    # Validate parameters
    sim_params = validate_simulation_params(T=T, dt=dt, burn_in=burn_in, N=N)
    
    # First estimate V* and κ*
    @info "Estimating V* and κ*..."
    Vstar = estimate_Vstar(p; N=N, T=T, dt=dt, burn_in=burn_in, seed=seed)
    κstar = critical_kappa(p; Vstar=Vstar)

    sweep_cfg = cfg["sweep"]
    κfrom = sweep_cfg["kappa_from"]
    κto = κstar * sweep_cfg["kappa_to_factor_of_kstar"]
    points = sweep_cfg["points"]

    κgrid = collect(range(κfrom, κto; length=points))
    
    @info "Running κ sweep..."
    res = sweep_kappa(p, κgrid; N=N, T=T, dt=dt, burn_in=burn_in, seed=seed)

    # Simulate time series for additional analysis
    @info "Simulating time series..."
    seed != 0 && Random.seed!(seed)
    u = zeros(N)
    for step in 1:sim_params.total_steps
        g = mean(u)
        euler_maruyama_step!(u, κstar, g, p, dt)
        reset_step!(u, p, dt)
    end

    # Validate results
    if any(isnan, u) || any(isinf, u)
        @warn "Simulation produced invalid values. Filtering them out."
        u = filter(x -> !isnan(x) && !isinf(x), u)
    end

    if isempty(u)
        error("No valid simulation data available after filtering.")
    end

    # Extract valid data
    u_valid = safe_array_slice(u, sim_params.burn_in_idx, length(u))
    t_valid = safe_time_range(burn_in, T, length(u_valid))
    
    # Filter results for stability
    max_threshold = 1e6
    stable_mask = res.amp .< max_threshold
    res_filtered = KappaSweepResult(
        res.κ[stable_mask],
        res.amp[stable_mask],
        res.V[stable_mask]
    )

    # Fit pitchfork
    @info "Fitting pitchfork bifurcation..."
    fit = pitchfork_fit(res_filtered; ε=1e-4, κmin=κstar*0.9, κmax=κstar*1.5)

    # Create output directory
    outdir = cfg["output_dir"]
    isdir(outdir) || mkpath(outdir)

    # Save data
    @info "Saving results..."
    df = DataFrame(kappa=res.κ, amp=res.amp, variance=res.V)
    CSV.write(joinpath(outdir, "bifurcation.csv"), df)

    # Write summary
    open(joinpath(outdir, "summary.txt"), "w") do io
        @printf(io, "V* = %.6f\n", Vstar)
        @printf(io, "kappa* (canonical) ≈ %.6f\n", κstar)
        @printf(io, "Pitchfork fit: kappa* ≈ %.6f, b ≈ %.6f, R2 ≈ %.4f, used=%d\n",
                fit.κstar, fit.b, fit.R2, fit.used)
    end

    # Create plots
    @info "Creating plots..."
    
    # 1. Bifurcation diagram
    plt = plot_bifurcation(res_filtered; κstar=κstar, 
                          title="Pitchfork Bifurcation")
    savefig(plt, joinpath(outdir, "bifurcation.png"))
    
    # 2. Vector field plots at different κ values
    for κ_test in [0.8*κstar, κstar, 1.2*κstar]
        vf_plt = plot_vector_field(p, κ_test)
        savefig(vf_plt, joinpath(outdir, "vector_field_k$(round(κ_test, digits=3)).png"))
    end
    
    # 3. Orbit plot with envelope
    t_orbit = collect(t_valid)
    u_orbit = collect(u_valid)
    orbit_plt = plot_orbit(t_orbit, u_orbit, show_envelope=true)
    savefig(orbit_plt, joinpath(outdir, "orbit.png"))

    @info "Complete! Results saved to $outdir"
    println("\nSummary:")
    println("  V* = $Vstar")
    println("  κ* ≈ $κstar")
    println("  Pitchfork fit: κ*≈$(fit.κstar), b≈$(fit.b), R²≈$(fit.R2)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 1
        println("Usage: run_from_yaml.jl CONFIG.yaml")
        exit(1)
    end
    main(ARGS[1])
end