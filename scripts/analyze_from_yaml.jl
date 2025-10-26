"""
analyze_from_yaml.jl

Complete bifurcation analysis from YAML configuration.
Self-contained with all dependencies from the project.

Usage:
    julia --project=. scripts/analyze_from_yaml.jl configs/example_sweep.yaml
    julia --project=. scripts/analyze_from_yaml.jl --all  # Process all configs
"""

#═══════════════════════════════════════════════════════════════════════════
# IMPORTS AND ENVIRONMENT PREPARATION
#═══════════════════════════════════════════════════════════════════════════

using Logging
using Pkg

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
const OUTPUT_ROOT  = normpath(joinpath(PROJECT_ROOT, "outputs"))

Pkg.activate(PROJECT_ROOT)

function ensure_project_dependencies()
    try
        Pkg.instantiate()
    catch err
        bt = catch_backtrace()
        @error "Failed to instantiate project dependencies" exception=(err, bt)
        exit(1)
    end
end

ensure_project_dependencies()

function safe_import(modsym::Symbol; pkgname::AbstractString=String(modsym))
    try
        @eval using $(modsym)
    catch err
        bt = catch_backtrace()
        @error "Failed to load dependency $pkgname" exception=(err, bt)
        exit(1)
    end
end

"""
Attempt to load an optional dependency. Returns `true` if successful and
`false` otherwise while keeping the script running.
"""
function try_import(modsym::Symbol; pkgname::AbstractString=String(modsym),
                    install_hint::Union{Nothing,AbstractString}=nothing,
                    min_julia::Union{Nothing,VersionNumber}=nothing)
    try
        @eval import $(modsym)
        return true
    catch err
        @info "Optional dependency $pkgname unavailable – falling back";
        @debug "Failed to load optional dependency" exception=(err, catch_backtrace())
        pkgpath = Base.find_package(pkgname)
        if pkgpath === nothing
            suggestion = isnothing(install_hint) ?
                "julia --project=. -e 'using Pkg; Pkg.add(\"$pkgname\")'" : install_hint
            @info "  → Install with: $suggestion"
        else
            @info "  → Package detected at $pkgpath but failed to load"
        end
        if min_julia !== nothing && VERSION < min_julia
            @info "  → Requires Julia $(min_julia) or newer (current $(VERSION))"
        end
        return false
    end
end

using LinearAlgebra
using Statistics
using DifferentialEquations
using FFTW
using StatsBase
using Printf
using Dates
using Base.Threads

# Import only the simulation submodules we need (without triggering BifurcationKit)
module BeliefAnalysisCore
    include(joinpath(@__DIR__, "..", "src", "Utils.jl"))
    include(joinpath(@__DIR__, "..", "src", "Types.jl"))
    include(joinpath(@__DIR__, "..", "src", "Hazard.jl"))
    include(joinpath(@__DIR__, "..", "src", "Model.jl"))
    include(joinpath(@__DIR__, "..", "src", "Stats.jl"))
end

"""
Map requested κ/κ* ratios to κ values inside [κ_min, κ_max].
Falls back to an evenly spaced grid if κ* <= 0 or if the requested
ratios do not intersect the interval.
"""
function kappa_from_ratios(κ_star::Real, κ_min::Real, κ_max::Real,
                           ratios::AbstractVector{<:Real}; fallback_count::Int=length(ratios))
    if !isfinite(κ_star) || κ_star <= 0
        return collect(range(κ_min, κ_max; length=max(fallback_count, 2)))
    end

    raw = κ_star .* collect(ratios)
    span = maximum(abs.([κ_star, κ_min, κ_max]))
    tol = max(1e-9, 1e-6 * span)
    mask = (raw .>= κ_min - tol) .& (raw .<= κ_max + tol)
    selected = sort(unique(raw[mask]))

    if isempty(selected)
        return collect(range(κ_min, κ_max; length=max(fallback_count, 2)))
    end

    return selected
end

function ratio_ticks(κ_min::Real, κ_max::Real, κ_star::Real)
    if !isfinite(κ_star) || κ_star <= 0
        return nothing
    end

    r_min = max(0.0, floor(κ_min / κ_star * 2) / 2)
    r_max = ceil(max(κ_max / κ_star, 0.0) * 2) / 2
    ratios = collect(r_min:0.5:r_max)
    ticks = κ_star .* ratios
    labels = [string(round(r, digits=2), " κ*") for r in ratios]
    return (ticks, labels)
end

using .BeliefAnalysisCore.Types: Params, StepHazard, LogisticHazard
using .BeliefAnalysisCore.Stats: estimate_Vstar, estimate_g, critical_kappa

# Load YAML and plotting backend dependencies
safe_import(:YAML; pkgname="YAML")
safe_import(:CairoMakie; pkgname="CairoMakie")
safe_import(:ColorSchemes; pkgname="ColorSchemes")

# Diagnose optional BifurcationKit status early so users see actionable hints
const BIFURCATIONKIT_AVAILABLE = try_import(:BifurcationKit; pkgname="BifurcationKit",
                                            install_hint="julia --project=. -e 'using Pkg; Pkg.add(url=\"https://github.com/bifurcationkit/BifurcationKit.jl.git\")'",
                                            min_julia=v"1.9")
if BIFURCATIONKIT_AVAILABLE
    @info "BifurcationKit detected – advanced continuation scripts remain enabled"
else
    @info "Proceeding without BifurcationKit-dependent features (builtin normal-form tools will be used)"
end

# Optional Attractors.jl integration for basin computations
const ATTRACTORS_AVAILABLE = try_import(:Attractors; pkgname="Attractors",
                                        install_hint="julia --project=. -e 'using Pkg; Pkg.add(\"Attractors\"); Pkg.add(\"DynamicalSystems\")'") &&
                             try_import(:DynamicalSystems; pkgname="DynamicalSystems",
                                        install_hint="julia --project=. -e 'using Pkg; Pkg.add(\"DynamicalSystems\")'")
if ATTRACTORS_AVAILABLE
    @info "Attractors.jl detected – enabling advanced basin computation"
else
    @info "Proceeding with internal basin computation (Attractors.jl not active)"
end

# Load bifurcation modules
include(joinpath(@__DIR__, "..", "src", "bifurcation", "model_interface.jl"))
include(joinpath(@__DIR__, "..", "src", "bifurcation", "plotting_cairo.jl"))

using .ModelInterface
using .PlottingCairo

#═══════════════════════════════════════════════════════════════════════════
# YAML CONFIGURATION LOADER
#═══════════════════════════════════════════════════════════════════════════

"""
Ensure a YAML mapping has string keys for consistent access.
"""
function ensure_string_dict(value, context::AbstractString)
    value isa AbstractDict || throw(ArgumentError("$context must be a mapping, got $(typeof(value))"))
    return Dict(String(k) => v for (k, v) in value)
end

function coerce_string(value, context::AbstractString; allow_empty::Bool=false)
    str = String(value)
    str = strip(str)
    if !allow_empty && isempty(str)
        throw(ArgumentError("$context cannot be empty"))
    end
    return str
end

function expect_string!(dict::AbstractDict, key::AbstractString; default::Union{Nothing,AbstractString}=nothing,
                         allow_empty::Bool=false, context::AbstractString=key)
    value = if haskey(dict, key)
        dict[key]
    elseif default === nothing
        throw(ArgumentError("Missing required field: $context"))
    else
        default
    end
    str = coerce_string(value, context; allow_empty=allow_empty)
    dict[key] = str
    return str
end

function coerce_real(value, context::AbstractString)
    num = if value isa Real
        Float64(value)
    elseif value isa AbstractString
        parsed = tryparse(Float64, value)
        parsed === nothing && throw(ArgumentError("$context must be numeric, got '$value'"))
        parsed
    else
        throw(ArgumentError("$context must be numeric, got $(typeof(value))"))
    end
    isfinite(num) || throw(ArgumentError("$context must be finite"))
    return num
end

function expect_real!(dict::AbstractDict, key::AbstractString; min::Real=-Inf, max::Real=Inf,
                      default::Union{Nothing,Real,AbstractString}=nothing, context::AbstractString=key)
    value = if haskey(dict, key)
        dict[key]
    elseif default === nothing
        throw(ArgumentError("Missing required field: $context"))
    else
        default
    end
    num = coerce_real(value, context)
    if num < min || num > max
        throw(ArgumentError("$context must be between $min and $max, got $num"))
    end
    dict[key] = num
    return num
end

function expect_int!(dict::AbstractDict, key::AbstractString; min::Integer=typemin(Int),
                     max::Integer=typemax(Int), default=nothing, context::AbstractString=key)
    value = if haskey(dict, key)
        dict[key]
    elseif default === nothing
        throw(ArgumentError("Missing required field: $context"))
    else
        default
    end

    num = if value isa Integer
        Int(value)
    elseif value isa Real
        rounded = round(Int, value)
        abs(value - rounded) <= eps(Base.max(abs(value), 1.0)) || throw(ArgumentError("$context must be an integer, got $value"))
        rounded
    elseif value isa AbstractString
        parsed = tryparse(Int, value)
        parsed === nothing && throw(ArgumentError("$context must be an integer, got '$value'"))
        parsed
    else
        throw(ArgumentError("$context must be an integer, got $(typeof(value))"))
    end

    if num < min || num > max
        throw(ArgumentError("$context must be between $min and $max, got $num"))
    end

    dict[key] = num
    return num
end

function sanitize_output_dir(raw::AbstractString)
    mkpath(OUTPUT_ROOT)
    candidate = normpath(isabspath(raw) ? raw : joinpath(PROJECT_ROOT, raw))
    rel_project = relpath(candidate, PROJECT_ROOT)
    if startswith(rel_project, "..")
        throw(ArgumentError("output_dir must reside inside the project directory: $raw"))
    end
    rel_output = relpath(candidate, OUTPUT_ROOT)
    if startswith(rel_output, "..")
        throw(ArgumentError("output_dir must be within $(OUTPUT_ROOT). Received: $raw"))
    end
    return candidate
end

function sanitize_run_name(name_in)
    name = coerce_string(name_in, "name"; allow_empty=false)
    sanitized = replace(name, r"[^A-Za-z0-9_.-]" => "_")
    return isempty(sanitized) ? "config" : sanitized
end

"""
Load and validate YAML configuration
"""
function load_yaml_config(filepath::String)
    if !isfile(filepath)
        throw(ArgumentError("Config file not found: $filepath"))
    end
    
    cfg_raw = YAML.load_file(filepath)
    cfg = ensure_string_dict(cfg_raw, "root")

    # Validate required fields
    required_fields = ["params", "N", "T", "dt", "output_dir"]
    for field in required_fields
        haskey(cfg, field) || throw(ArgumentError("Missing required field: $field"))
    end

    cfg["params"] = ensure_string_dict(cfg["params"], "params")
    params = cfg["params"]

    for param in ["lambda", "sigma", "theta", "c0", "hazard"]
        haskey(params, param) || throw(ArgumentError("Missing required parameter: $param"))
    end

    expect_real!(params, "lambda"; context="params.lambda")
    expect_real!(params, "sigma"; min=0.0, context="params.sigma")
    expect_real!(params, "theta"; context="params.theta")
    expect_real!(params, "c0"; context="params.c0")

    hazard_cfg = ensure_string_dict(params["hazard"], "params.hazard")
    kind = lowercase(expect_string!(hazard_cfg, "kind"; context="params.hazard.kind"))
    if kind == "step"
        expect_real!(hazard_cfg, "nu0"; context="params.hazard.nu0")
    elseif kind == "logistic"
        expect_real!(hazard_cfg, "numax"; context="params.hazard.numax")
        expect_real!(hazard_cfg, "beta"; context="params.hazard.beta")
    else
        throw(ArgumentError("Unsupported hazard kind: $(hazard_cfg["kind"]) — use 'step' or 'logistic'."))
    end
    hazard_cfg["kind"] = kind
    params["hazard"] = hazard_cfg

    cfg["N"] = expect_int!(cfg, "N"; min=2, context="N")
    cfg["T"] = expect_real!(cfg, "T"; min=0.0, context="T")
    cfg["dt"] = expect_real!(cfg, "dt"; min=1e-6, context="dt")
    cfg["burn_in"] = expect_real!(cfg, "burn_in"; min=0, default=0, context="burn_in")
    cfg["seed"] = expect_int!(cfg, "seed"; default=0, context="seed")

    outdir_raw = expect_string!(cfg, "output_dir"; context="output_dir")
    cfg["output_dir"] = sanitize_output_dir(outdir_raw)

    if haskey(cfg, "sweep")
        sweep_cfg = ensure_string_dict(cfg["sweep"], "sweep")
        if haskey(sweep_cfg, "points")
            sweep_cfg["points"] = expect_int!(sweep_cfg, "points"; min=2, context="sweep.points")
        end
        for key in ["kappa_from", "kappa_to", "kappa_from_factor_of_kstar", "kappa_to_factor_of_kstar"]
            if haskey(sweep_cfg, key)
                expect_real!(sweep_cfg, key; context="sweep.$key")
            end
        end
        cfg["sweep"] = sweep_cfg
    end

    if haskey(cfg, "name")
        cfg["name"] = sanitize_run_name(cfg["name"])
    end

    return cfg
end

"""
Convert YAML parameters to ModelInterface format
"""
function yaml_to_model_params(cfg::Dict)
    params = cfg["params"]

    # Extract core parameters
    λ = Float64(params["lambda"])
    σ = Float64(params["sigma"])
    Θ = Float64(params["theta"])
    c0 = Float64(params["c0"])

    hazard_cfg = params["hazard"]
    hazard = begin
        kind = lowercase(String(hazard_cfg["kind"]))
        if kind == "step"
            StepHazard(Float64(hazard_cfg["nu0"]))
        elseif kind == "logistic"
            LogisticHazard(Float64(hazard_cfg["numax"]), Float64(hazard_cfg["beta"]))
        else
            error("Unsupported hazard kind: $(hazard_cfg["kind"]) — use 'step' or 'logistic'.")
        end
    end

    seed = get(cfg, "seed", 0)
    N = cfg["N"]
    T = cfg["T"]
    dt = cfg["dt"]
    burn_in = cfg["burn_in"]

    p_sim = Params(; λ=λ, σ=σ, Θ=Θ, c0=c0, hazard=hazard)

    @info "  Estimating stationary dispersion V* via Monte Carlo" N=N T=T dt=dt burn_in=burn_in seed=seed
    Vstar = estimate_Vstar(p_sim; N=N, T=T, dt=dt, burn_in=burn_in, seed=seed)

    @info "  Estimating odd-mode decay g at κ=0"
    g_est = estimate_g(p_sim;
                       N=max(N, 20_000),
                       T=min(max(T, 40.0), 120.0),
                       dt=min(dt, 0.01),
                       seed=seed)

    κ_star = critical_kappa(p_sim; Vstar=Vstar, g=g_est, N=N, T=T, dt=dt, burn_in=burn_in, seed=seed)

    # Calibrate cubic nonlinearity so that |u₁-u₂|/2 ≈ √((κ-κ*)/κ*) for κ > κ*
    σ_sq = max(σ^2, eps())
    prefactor = (2 * λ * Vstar) / σ_sq
    β = prefactor * max(κ_star, eps())
    p_model = (
        λ=λ,
        σ=σ,
        Θ=Θ,
        c0=c0,
        Vstar=Vstar,
        g=g_est,
        beta=β,
        kappa=0.0,
        kstar=κ_star,
        seed=seed,
    )

    meta = (
        params=p_sim,
        hazard=hazard,
        Vstar=Vstar,
        g=g_est,
        κ_star=κ_star,
    )

    return p_model, meta
end

#═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
#═══════════════════════════════════════════════════════════════════════════

"""
Compute autocorrelation function
"""
function autocorrelation(x::Vector{Float64}, max_lag::Int)
    n = length(x)
    x_centered = x .- mean(x)
    
    acf = zeros(max_lag + 1)
    c0 = sum(x_centered.^2) / n
    
    if c0 < 1e-12
        return acf
    end
    
    for lag in 0:max_lag
        if lag == 0
            acf[1] = 1.0
        else
            c_lag = sum(x_centered[1:n-lag] .* x_centered[lag+1:n]) / n
            acf[lag+1] = c_lag / c0
        end
    end
    
    return acf
end

"""
Compute power spectral density
"""
function compute_psd(x::Vector{Float64}, dt::Float64)
    n = length(x)
    x_centered = x .- mean(x)
    
    # Apply Hanning window
    window = 0.5 .- 0.5 .* cos.(2π .* (0:n-1) ./ (n-1))
    x_windowed = x_centered .* window
    
    # Compute FFT
    X = fft(x_windowed)
    psd = abs2.(X) ./ n
    freqs = fftfreq(n, 1/dt)
    
    # Keep only positive frequencies
    pos_idx = freqs .>= 0
    
    return freqs[pos_idx], psd[pos_idx]
end

"""
Order parameter for polarization: half the difference between the two agents.
Falls back to mean if dimensionality ≠ 2.
"""
polarization_coordinate(u::AbstractVector) = length(u) == 2 ? 0.5 * (u[1] - u[2]) : mean(u)

polarization_amplitude(u::AbstractVector) = abs(polarization_coordinate(u))

"""
Classify attractor by running trajectory
"""
function classify_attractor(u0::Vector{Float64}, p; tmax=500.0)
    function ode!(du, u, p, t)
        F = ModelInterface.f(u, p)
        du .= F
    end

    prob = ODEProblem(ode!, u0, (0.0, tmax), p)

    try
        sol = solve(prob, Tsit5(); reltol=1e-6, abstol=1e-6)
        u_final = sol.u[end]
        pol_final = polarization_coordinate(u_final)

        amp_ref = ModelInterface.polarization_amplitude(p)
        tol = amp_ref > 0 ? max(0.02, 0.1 * amp_ref) : 0.02

        if abs(pol_final) < tol
            return 0  # Consensus
        elseif pol_final > tol
            return 1  # Positive polarization
        elseif pol_final < -tol
            return -1  # Negative polarization
        else
            return 0
        end
    catch
        return NaN
    end
end

"""
Integrate from u0 until it lands ε-close to an attractor or tmax is reached.
Returns (label, t_hit) with label ∈ {-1,0,+1} or NaN if integration failed.
label = 0 → consensus, ±1 → polarized branches (sign = sign of odd mode).
"""
function classify_flow(u0_in::AbstractVector{<:Real}, p;
                       tmax::Float64=300.0, dt::Float64=0.05, ε::Float64=1e-3)
    # ensure a concrete Vector{Float64}
    u0 = Vector{Float64}(u0_in)

    # precompute target equilibria
    pol_eqs_raw = ModelInterface.polarized_equilibria(p)    # may return vectors/tuples
    pol_eqs = [Vector{Float64}(e) for e in pol_eqs_raw]
    consensus = zeros(2)

    function ode!(du, u, p, t)
        du .= ModelInterface.f(u, p)
    end

    prob = ODEProblem(ode!, u0, (0.0, tmax), p)
    integ = init(prob, Tsit5(); dt=dt, save_everystep=false)  # fast stepping

    t = 0.0
    try
        while t < tmax
            step!(integ)
            u = integ.u
            t = integ.t

            # consensus check
            if norm(u .- consensus) < ε
                return (0, t)
            end
            # polarized check
            if !isempty(pol_eqs)
                dmin, arg = findmin(map(e -> norm(u .- e), pol_eqs))
                if dmin < ε
                    s = sign(0.5 * (pol_eqs[arg][1] - pol_eqs[arg][2]))
                    return (s ≥ 0 ? 1 : -1, t)
                end
            end
        end
    catch
        return (NaN, t)   # integration failure → mark as NaN
    end

    # fallback classification: sign of odd mode at final time
    odd = 0.5 * (integ.u[1] - integ.u[2])
    lab = abs(odd) < 5ε ? 0 : (odd > 0 ? 1 : -1)
    return (lab, tmax)
end

"""
Compute a supersampled basin grid and convergence times.
Supersampling (ss≥1) anti-aliases boundaries (majority vote on labels).
Returns (xs, ys, labels::Matrix{Int8}, times::Matrix{Float32}).
This is the built-in fallback implementation used when Attractors.jl is
unavailable or fails.
"""
function compute_basin_grid_fallback(κ::Real, p_base; xmin=-3.0, xmax=3.0, ymin=-3.0, ymax=3.0,
                                     res::Int=250, ss::Int=2, tmax::Float64=300.0,
                                     dt::Float64=0.05, ε::Float64=1e-3)
    p   = ModelInterface.kappa_set(p_base, κ)
    xs  = range(xmin, xmax; length=res)
    ys  = range(ymin, ymax; length=res)
    dx  = step(xs);  dy = step(ys)

    labels = Matrix{Int8}(undef, res, res)
    times  = Matrix{Float32}(undef, res, res)

    Threads.@threads for i in 1:res
        for j in 1:res
            labs = Int8[]
            ts   = Float64[]
            # supersample at subcell centers
            for a in 0:ss-1, b in 0:ss-1
                x = xs[i] + (a + 0.5)/ss * dx - 0.5*dx
                y = ys[j] + (b + 0.5)/ss * dy - 0.5*dy
                lab, thit = classify_flow([x, y], p; tmax=tmax, dt=dt, ε=ε)
                push!(labs, Int8(isnan(lab) ? 0 : lab))
                push!(ts,  thit)
            end
            # majority vote for label; mean for time
            labels[i, j] = StatsBase.mode(labs)
            times[i, j]  = Float32(mean(ts))
        end
    end
    return xs, ys, labels, times
end

"""
Attempt to compute the basin grid using Attractors.jl. Falls back to the
internal implementation if Attractors.jl is not available or raises an
exception.
"""
function compute_basin_grid_attractors(κ::Real, p_base; xmin=-3.0, xmax=3.0,
                                       ymin=-3.0, ymax=3.0, res::Int=250,
                                       ss::Int=2, tmax::Float64=300.0,
                                       dt::Float64=0.05, ε::Float64=1e-3)
    try
        p   = ModelInterface.kappa_set(p_base, κ)
        flow! = (du, u, p, t) -> (du .= ModelInterface.f(u, p))
        ds = DynamicalSystems.ContinuousDynamicalSystem(flow!, zeros(2), p)

        ss_eff = max(ss, 1)
        res_eff = max(res * ss_eff, 2)
        xs_fine = range(xmin, xmax; length=res_eff)
        ys_fine = range(ymin, ymax; length=res_eff)

        grid = let constructed = false, g = nothing
            if isdefined(Attractors, :RectangleGrid)
                try
                    g = Attractors.RectangleGrid(xs_fine, ys_fine)
                    constructed = true
                catch err
                    @debug "RectangleGrid(xs, ys) failed" exception=(err, catch_backtrace())
                end
                if !constructed
                    try
                        g = Attractors.RectangleGrid((xmin, xmax), (ymin, ymax); dims=(res_eff, res_eff))
                        constructed = true
                    catch err
                        @debug "RectangleGrid span constructor failed" exception=(err, catch_backtrace())
                    end
                end
            end
            if !constructed && isdefined(Attractors, :Grid)
                try
                    g = Attractors.Grid((xs_fine, ys_fine))
                    constructed = true
                catch err
                    @debug "Grid constructor failed" exception=(err, catch_backtrace())
                end
            end
            if !constructed
                @warn "Failed to construct Attractors grid; falling back"
                return compute_basin_grid_fallback(κ, p_base;
                                                   xmin=xmin, xmax=xmax,
                                                   ymin=ymin, ymax=ymax,
                                                   res=res, ss=ss,
                                                   tmax=tmax, dt=dt, ε=ε)
            end
            g
        end

        method = nothing
        if isdefined(Attractors, :AttractorsViaIntegrators)
            base_kwargs = (; tfinal=tmax, Δt=dt, abstol=1e-9, reltol=1e-9)
            stop_kwargs = if isdefined(Attractors, :norm_stop)
                (; stopping_condition=Attractors.norm_stop(ε))
            else
                (;)
            end
            method = Attractors.AttractorsViaIntegrators(; merge(base_kwargs, stop_kwargs)...)
        end

        kwargs = (; show_progress=false)
        if method !== nothing
            kwargs = merge(kwargs, (; method))
        end

        basin_result = Attractors.basins_of_attraction(ds, grid; kwargs...)

        basins_raw = if hasproperty(basin_result, :basins)
            getproperty(basin_result, :basins)
        elseif basin_result isa Tuple
            basin_result[1]
        else
            basin_result
        end

        times_raw = if hasproperty(basin_result, :times)
            getproperty(basin_result, :times)
        elseif hasproperty(basin_result, :mean_convergence_time)
            getproperty(basin_result, :mean_convergence_time)
        else
            nothing
        end

        basins_mat = Array{Int8}(undef, res_eff, res_eff)
        for j in 1:res_eff, i in 1:res_eff
            basins_mat[i, j] = Int8(basins_raw[i, j])
        end

        times_mat = if times_raw === nothing
            nothing
        else
            Float32.(times_raw)
        end

        if ss_eff > 1
            labels = Matrix{Int8}(undef, res, res)
            times = Matrix{Float32}(undef, res, res)
            for i in 1:res, j in 1:res
                i1 = (i - 1) * ss_eff + 1
                i2 = i1 + ss_eff - 1
                j1 = (j - 1) * ss_eff + 1
                j2 = j1 + ss_eff - 1
                block = @view basins_mat[i1:i2, j1:j2]
                labels[i, j] = StatsBase.mode(vec(block))
                if times_raw === nothing
                    times[i, j] = Float32(NaN)
                else
                    block_t = @view times_mat[i1:i2, j1:j2]
                    times[i, j] = Float32(mean(block_t))
                end
            end
        else
            labels = basins_mat
            times = times_mat === nothing ? fill(Float32(NaN), size(labels)) : times_mat
        end

        xs = range(xmin, xmax; length=res)
        ys = range(ymin, ymax; length=res)

        if any(t -> !isfinite(t), times)
            @info "Attractors.jl did not provide convergence times – recomputing with fallback"
            _, _, _, times = compute_basin_grid_fallback(κ, p_base;
                                                         xmin=xmin, xmax=xmax,
                                                         ymin=ymin, ymax=ymax,
                                                         res=res, ss=ss,
                                                         tmax=tmax, dt=dt, ε=ε)
        end

        return xs, ys, labels, times
    catch err
        @warn "Attractors.jl basin computation failed; using fallback" exception=(err, catch_backtrace())
        return compute_basin_grid_fallback(κ, p_base;
                                           xmin=xmin, xmax=xmax,
                                           ymin=ymin, ymax=ymax,
                                           res=res, ss=ss,
                                           tmax=tmax, dt=dt, ε=ε)
    end
end

function compute_basin_grid(κ::Real, p_base; kwargs...)
    if ATTRACTORS_AVAILABLE
        return compute_basin_grid_attractors(κ, p_base; kwargs...)
    else
        return compute_basin_grid_fallback(κ, p_base; kwargs...)
    end
end


# Lightweight vector-field overlay that doesn't depend on PlottingCairo internals
function overlay_vectorfield!(ax, f::Function, p;
                              lims::NTuple{4,Float64}=(-3.0, 3.0, -3.0, 3.0),
                              density::Int=28, scale::Float64=0.12)
    xmin, xmax, ymin, ymax = lims
    xs = range(xmin, xmax; length=density)
    ys = range(ymin, ymax; length=density)

    # build grids
    X = repeat(collect(xs)', length(ys), 1)
    Y = repeat(collect(ys),  1, length(xs))
    U = similar(X); V = similar(Y)

    # normalize vectors and scale to figure size
    L = scale * min(xmax - xmin, ymax - ymin)
    for j in axes(X, 1), i in axes(X, 2)
        x = X[j, i]; y = Y[j, i]
        F = f([x, y], p)
        fx, fy = F[1], F[2]
        m = hypot(fx, fy)
        if m > 0
            U[j, i] = (fx / m) * L
            V[j, i] = (fy / m) * L
        else
            U[j, i] = 0.0; V[j, i] = 0.0
        end
    end

    # Use arrows2d! (no deprecation), with explicit positions/directions
    # Point2f / Vec2f are available via CairoMakie / Makie
    positions  = Point2f.(X, Y)
    directions = Vec2f.(U, V)
    arrows2d!(ax, positions, directions; shaftwidth=1.0, tipwidth=6, tiplength=10, color=:gray55)
    return nothing
end


"""
Compute Lyapunov exponent
"""
function compute_lyapunov(f::Function, jac::Function, u0::Vector{Float64}, p;
                         tmax=1000.0, dt=0.1)
    n = length(u0)
    u = copy(u0)
    w = randn(n)
    w ./= norm(w)
    
    λ_sum = 0.0
    n_steps = 0
    
    t = 0.0
    while t < tmax
        J = jac(u, p)
        F = f(u, p)
        
        u .+= F .* dt
        w .+= (J * w) .* dt
        
        w_norm = norm(w)
        if w_norm > 1e-12
            λ_sum += log(w_norm)
            w ./= w_norm
        end
        
        t += dt
        n_steps += 1
    end
    
    return n_steps > 0 ? λ_sum / (n_steps * dt) : 0.0
end

"""
Pre-compute equilibrium diagnostics across a κ-grid.
Returns consensus stability and polarized branch information for plotting.
"""
function equilibrium_diagnostics(κ_values::AbstractVector{<:Real}, p_base)
    κ_vec = collect(float.(κ_values))
    n = length(κ_vec)

    consensus_max = Vector{Float64}(undef, n)
    consensus_imag = Vector{Float64}(undef, n)
    polarized_amp = zeros(Float64, n)
    polarized_exists = falses(n)
    polarized_max = fill(NaN, n)
    polarized_imag = fill(NaN, n)

    for (i, κ) in enumerate(κ_vec)
        p = ModelInterface.kappa_set(p_base, κ)

        J_cons = ModelInterface.jacobian(zeros(2), p)
        eig_cons = eigvals(J_cons)
        consensus_max[i] = maximum(real.(eig_cons))
        consensus_imag[i] = maximum(abs.(imag.(eig_cons)))

        amp = ModelInterface.polarization_amplitude(p)
        if amp > 0
            polarized_exists[i] = true
            polarized_amp[i] = amp
            u_pol = [amp, -amp]
            eig_pol = eigvals(ModelInterface.jacobian(u_pol, p))
            polarized_max[i] = maximum(real.(eig_pol))
            polarized_imag[i] = maximum(abs.(imag.(eig_pol)))
        end
    end

    return (; κ=κ_vec,
            consensus_max=consensus_max,
            consensus_imag=consensus_imag,
            polarized_amp=polarized_amp,
            polarized_exists=polarized_exists,
            polarized_max=polarized_max,
            polarized_imag=polarized_imag)
end

function format_ratio_label(κ::Real, κ_star::Real; digits_ratio::Int=3, digits_kappa::Int=3)
    if isfinite(κ_star) && κ_star > 0
        ratio = κ / κ_star
        fmt = Printf.Format("κ = %." * string(digits_kappa) * "f (κ/κ* = %." * string(digits_ratio) * "f)")
        return Printf.format(fmt, κ, ratio)
    else
        fmt = Printf.Format("κ = %." * string(digits_kappa) * "f")
        return Printf.format(fmt, κ)
    end
end

"""
Batch-means standard error for a (possibly autocorrelated) series.
Splits x into nb batches (default 10) and returns the SE of batch means.
"""
function batch_means_se(x::AbstractVector{<:Real}; nb::Int=10)
    n = length(x)
    nb = clamp(nb, 2, min(50, n))               # keep sane
    L  = ceil(Int, n / nb)
    bmeans = Float64[]
    for i in 1:nb
        lo = 1 + (i-1)*L
        hi = min(n, i*L)
        lo > hi && break
        push!(bmeans, mean(@view x[lo:hi]))
    end
    nb_eff = length(bmeans)
    nb_eff ≥ 2 || return 0.0
    return std(bmeans) / sqrt(nb_eff)
end


#═══════════════════════════════════════════════════════════════════════════
# PLOTTING FUNCTIONS
#═══════════════════════════════════════════════════════════════════════════

"""
Plot 1: Extended bifurcation diagram
"""
function plot_bifurcation(κ_range, p_base)
    @info "  Computing bifurcation diagram..."

    κ_vals = collect(κ_range)
    diag = equilibrium_diagnostics(κ_vals, p_base)
    κ_star = getproperty(p_base, :kstar)

    fig = Figure(size=(1400, 1000))

    consensus_mean = zeros(length(κ_vals))
    stable_consensus = diag.consensus_max .< 0

    polarized_mask = diag.polarized_exists .& (.!isnan.(diag.polarized_max))
    stable_polarized = polarized_mask .& (diag.polarized_max .< 0)

    # Panel 1: Mean equilibrium
    ax1 = Axis(fig[1, 1:2];
              xlabel="Coupling strength κ",
              ylabel="Mean belief ⟨u⟩",
              title="Bifurcation Diagram: Consensus → Polarization")

    if any(stable_consensus)
        lines!(ax1, κ_vals[stable_consensus], consensus_mean[stable_consensus];
              linewidth=3, color=:blue, label="Consensus (stable)")
    end
    if any(.!stable_consensus)
        lines!(ax1, κ_vals[.!stable_consensus], consensus_mean[.!stable_consensus];
              linewidth=3, color=:red, linestyle=:dash, label="Consensus (unstable)")
    end

    vlines!(ax1, [κ_star]; color=:green, linestyle=:dash, linewidth=2,
            label="κ* = $(round(κ_star, digits=4))")
    ticks_main = ratio_ticks(minimum(κ_vals), maximum(κ_vals), κ_star)
    ticks_main === nothing || (ax1.xticks = ticks_main)
    axislegend(ax1, position=:lt)

    # Panel 2: Stability indicator
    ax2 = Axis(fig[2, 1];
              xlabel="κ",
              ylabel="max Re(λ)",
              title="Stability (negative = stable)")

    lines!(ax2, κ_vals, diag.consensus_max; linewidth=2, color=:blue, label="Consensus")
    if any(polarized_mask)
        lines!(ax2, κ_vals[polarized_mask], diag.polarized_max[polarized_mask];
              linewidth=2, color=:orange, label="Polarized")
    end
    hlines!(ax2, [0]; color=:black, linestyle=:dash, linewidth=1.5)
    axislegend(ax2, position=:lt)

    # Panel 3: Oscillation frequency
    ax3 = Axis(fig[2, 2];
              xlabel="κ",
              ylabel="|Im(λ)|",
              title="Oscillation Frequency")

    lines!(ax3, κ_vals, diag.consensus_imag; linewidth=2, color=:blue)
    if any(polarized_mask)
        lines!(ax3, κ_vals[polarized_mask], diag.polarized_imag[polarized_mask];
              linewidth=2, color=:orange)
    end

    # Panel 4: Polarization amplitude
    ax4 = Axis(fig[3, 1:2];
              xlabel="κ",
              ylabel="Polarization |u₁ - u₂|/2",
              title="Opinion Divergence")

    lines!(ax4, κ_vals, zeros(length(κ_vals)); color=:gray70, linewidth=1, linestyle=:dash)
    if any(stable_polarized)
        lines!(ax4, κ_vals[stable_polarized], diag.polarized_amp[stable_polarized];
              linewidth=3, color=:orange, label="Polarized (stable)")
        lines!(ax4, κ_vals[stable_polarized], -diag.polarized_amp[stable_polarized];
              linewidth=3, color=:orange)
    end
    if any(polarized_mask .& (.!stable_polarized))
        mask = polarized_mask .& (.!stable_polarized)
        lines!(ax4, κ_vals[mask], diag.polarized_amp[mask];
              linewidth=3, color=:red, linestyle=:dot, label="Polarized (unstable)")
        lines!(ax4, κ_vals[mask], -diag.polarized_amp[mask];
              linewidth=3, color=:red, linestyle=:dot)
    end

    vlines!(ax4, [κ_star]; color=:green, linestyle=:dash, linewidth=2)
    ticks_div = ratio_ticks(minimum(κ_vals), maximum(κ_vals), κ_star)
    ticks_div === nothing || (ax4.xticks = ticks_div)
    axislegend(ax4, position=:lt)

    return fig
end

"""
Plot 2: Phase portraits
"""
function plot_phase_portraits(κ_values, p_base)
    @info "  Computing phase portraits..."
    
    n = length(κ_values)
    fig = Figure(size=(1200, 600 * ceil(Int, n/2)))
    
    for (idx, κ) in enumerate(κ_values)
        p = ModelInterface.kappa_set(p_base, κ)
        
        row = div(idx - 1, 2) + 1
        col = mod(idx - 1, 2) + 1
        
        κ_star = getproperty(p, :kstar)
        title_str = "Phase Space: " * format_ratio_label(κ, κ_star)
        ax = Axis(fig[row, col];
                 xlabel="u₁ (agent 1)",
                 ylabel="u₂ (agent 2)",
                 title=title_str,
                 aspect=DataAspect())
        
        # Vector field
        PlottingCairo.phase_portrait!(ax, ModelInterface.f, p;
                                     lims=(-3.0, 3.0, -3.0, 3.0),
                                     density=35, alpha=0.25)
        
        # Consensus line
        lines!(ax, [-3, 3], [-3, 3]; 
              color=:gray, linestyle=:dash, linewidth=2)
        
        # Consensus equilibrium
        u_cons = zeros(2)
        eig_cons = eigvals(ModelInterface.jacobian(u_cons, p))
        stable_cons = all(real.(eig_cons) .< 0)
        scatter!(ax, [u_cons[1]], [u_cons[2]];
                color=stable_cons ? :green : :orange,
                markersize=14,
                marker=stable_cons ? :circle : :xcross,
                label=stable_cons ? (idx == 1 ? "Consensus" : nothing)
                                  : (idx == 1 ? "Consensus (unstable)" : nothing))

        # Polarized equilibria
        pol_idx = 0
        for u_pol in ModelInterface.polarized_equilibria(p)
            pol_idx += 1
            eig_pol = eigvals(ModelInterface.jacobian(u_pol, p))
            stable_pol = all(real.(eig_pol) .< 0)
            scatter!(ax, [u_pol[1]], [u_pol[2]];
                    color=stable_pol ? :red : :purple,
                    markersize=14,
                    marker=stable_pol ? :rect : :utriangle,
                    label=stable_pol ? (pol_idx == 1 ? "Polarized" : nothing)
                                     : (pol_idx == 1 ? "Polarized (unstable)" : nothing))
        end
    end

    return fig
end

"""
Plot 3: Publication-grade basins of attraction.
Left column: categorical basins with crisp separatrices (contours at -0.5,0,0.5),
vector field overlay, and equilibria markers. Right column: log10 convergence time.
"""
function plot_basins(κ_values, p_base; resolution::Int=300,
                     domain=(-3.0, 3.0, -3.0, 3.0),
                     supersample::Int=2, tmax::Float64=300.0, dt::Float64=0.05,
                     ε::Float64=1e-3)
    @info "  Computing publication-grade basins (res=$resolution, ss=$supersample)..."
    nκ = length(κ_values)
    rows = nκ
    fig = Figure(size=(1600, 420 * rows))

    for (row, κ) in enumerate(κ_values)
        κ_star = getproperty(p_base, :kstar)
        title_str = "Basin: " * format_ratio_label(κ, κ_star; digits_ratio=3)
        xmin, xmax, ymin, ymax = domain
        xs, ys, lab, tmat = compute_basin_grid(κ, p_base;
                                               xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                                               res=resolution, ss=supersample,
                                               tmax=tmax, dt=dt, ε=ε)
        labT   = permutedims(lab, (2, 1))         # labels, Int8 but transposed for plotting
        labTf  = Float64.(labT)                   # Float64 copy for line-contours
        tlog   = log10.(max.(1e-3, tmat))
        tlogT  = permutedims(tlog, (2, 1))


        # ---- Left: categorical basin map with separatrices ----
        axL = Axis(fig[row, 1];
                   title=title_str, xlabel="u₁", ylabel="u₂", aspect=DataAspect(),
                   xgridvisible=false, ygridvisible=false)

        # Left: categorical basins
        heatmap!(axL, xs, ys, labT; colorrange=(-1.0, 1.0), colormap=Reverse(:RdBu))
        # Draw line contours (poly recipe ⇒ strokewidth)
        contour!(axL, xs, ys, labTf; levels=[-0.5, 0.0, 0.5],
                color=:black, linewidth=1.4)

        # light vector field overlay (from your plotting util)
        # light vector field overlay (self-contained)
        # overlay (self-contained)
        overlay_vectorfield!(axL, ModelInterface.f,
                            ModelInterface.kappa_set(p_base, κ);
                            lims=(xmin, xmax, ymin, ymax), density=28, scale=0.12)

        # diagonal consensus line
        lines!(axL, [xmin, xmax], [xmin, xmax]; color=:gray65, linestyle=:dash, linewidth=1.0)

        # mark equilibria
        pκ = ModelInterface.kappa_set(p_base, κ)
        scatter!(axL, [0.0], [0.0]; color=:forestgreen, marker=:circle, markersize=13, strokewidth=0.5, label="Consensus")
        for (k, ueq) in enumerate(ModelInterface.polarized_equilibria(pκ))
            scatter!(axL, [ueq[1]], [ueq[2]];
                     color=:firebrick, marker=:rect, markersize=12, strokewidth=0.5,
                     label=k==1 ? "Polarized" : nothing)
        end
        axislegend(axL, position=:lb)

        # ---- Right: convergence time map (log scale) ----
        axR = Axis(fig[row, 2];
                   title="Convergence time (log₁₀ seconds)",
                   xlabel="u₁", ylabel="u₂", aspect=DataAspect(),
                   xgridvisible=false, ygridvisible=false)
        hm = heatmap!(axR, xs, ys, tlogT; colormap=:viridis)
        # reuse separatrix to show slow regions near boundaries (poly recipe ⇒ linewidth)
        contour!(axR, xs, ys, labTf; levels=[-0.5, 0.0, 0.5],
                 color=:black, linewidth=1.2)
        Colorbar(fig[row, 3], hm)

        # clean limits
        xlims!(axL, xmin, xmax); ylims!(axL, ymin, ymax)
        xlims!(axR, xmin, xmax); ylims!(axR, ymin, ymax)
    end

    fig
end


"""
Plot 4: Time series analysis (annotated).
Adds near-threshold diagnostics: steady polarization p̄ (± CI),
prediction p̂ = sqrt((κ-κ*)/κ*), ratio r = p̄/p̂, and relaxation time τ_c.
Also overlays ±p̂ on the polarization panel when κ>κ*.
"""
function plot_timeseries(κ, p_base; u0=[0.5, -0.5], tmax=500.0, dt=0.1)
    @info "  Analyzing time series for κ=$κ..."
    p = ModelInterface.kappa_set(p_base, κ)
    κ_star = getproperty(p, :kstar)

    # ODE
    function ode!(du, u, p, t)
        du .= ModelInterface.f(u, p)
    end
    sol = solve(ODEProblem(ode!, u0, (0.0, tmax), p), Tsit5(); saveat=dt)

    # Extract series
    t   = sol.t
    U   = reduce(hcat, sol.u)
    u1  = vec(U[1, :]);  u2 = vec(U[2, :])
    m   = (u1 .+ u2) ./ 2
    pol = 0.5 .* (u1 .- u2)           # signed odd mode
    pabs = abs.(pol)                   # amplitude

    # --- Diagnostics ---
    T  = length(pabs)
    B  = fld(6T, 10)                   # 60% burn-in
    post = pabs[B+1:end]
    pbar = mean(post)
    se   = batch_means_se(post; nb=10)

    # ACF(1) on the transient (or whole series if too short)
    acf1_slice = (B ≥ 10) ? pol[1:B] : pol
    ρ1  = cor(acf1_slice[1:end-1], acf1_slice[2:end])
    ρ1  = clamp(isfinite(ρ1) ? ρ1 : 0.0, 1e-6, 0.999)
    τc  = -1 / log(ρ1)                 # relaxation time proxy

    # Near-threshold prediction (normal-form calibration used in yaml_to_model_params)
    p̂ = (isfinite(κ_star) && κ > κ_star) ? sqrt((κ - κ_star)/κ_star) : 0.0
    r  = (p̂ > 0) ? pbar / p̂ : NaN

    # --- Figure ---
    fig = Figure(size=(1600, 1200))
    label_ratio = format_ratio_label(κ, κ_star; digits_ratio=3)

    # Panel 1: beliefs
    ax1 = Axis(fig[1, 1:2]; xlabel="Time", ylabel="Belief",
               title="Evolution: " * label_ratio)
    lines!(ax1, t, u1;  color=:blue,  linewidth=1.8, label="Agent 1")
    lines!(ax1, t, u2;  color=:red,   linewidth=1.8, label="Agent 2")
    lines!(ax1, t, m;   color=:green, linewidth=2.0, label="Mean")
    hlines!(ax1, [0]; color=:black, linestyle=:dash, linewidth=1)
    axislegend(ax1, position=:rt)

    # Annotation block on the top panel
    ratio = isfinite(κ_star) && κ_star>0 ? round(κ/κ_star, digits=2) : NaN
    info_lines = String[
        "κ/κ* = $(isnan(ratio) ? "—" : string(ratio))",
        "p̄ = $(round(pbar,digits=3)) ± $(round(1.96*se,digits=3))",
        "τ_c ≈ $(round(τc,digits=1))  (ρ₁=$(round(ρ1,digits=3)))"
    ]
    if κ > κ_star
        push!(info_lines, "p̂ = $(round(p̂,digits=3));  r = $(round(r,digits=2))")
    end
    txt = join(info_lines, "   |   ")
    xmax = maximum(t)
    ymax = maximum(vcat(u1, u2))
    text!(ax1, xmax*0.60, ymax - 0.05*(abs(ymax)+1e-6); text=txt, align=(:left,:top), fontsize=12)

    # Panel 2: polarization amplitude
    ax2 = Axis(fig[2, 1]; xlabel="Time", ylabel="|u₁ - u₂|/2", title="Polarization")
    lines!(ax2, t, pabs; color=:purple, linewidth=1.8, label="|u₁-u₂|/2")
    if κ > κ_star && p̂ > 0
        hlines!(ax2, [p̂, -p̂]; color=:red, linestyle=:dash, linewidth=1.5)
        text!(ax2, t[end]*0.80, p̂*(1 + 0.05); text="p̂", fontsize=11, color=:red)
    end

    # Panel 3: state magnitude
    ax3 = Axis(fig[2, 2]; xlabel="Time", ylabel="‖u‖", title="State Magnitude")
    lines!(ax3, t, sqrt.(u1.^2 .+ u2.^2); color=:green, linewidth=1.8)

    # Panel 4: ACF (lags in time units)
    ax4 = Axis(fig[3, 1]; xlabel="Lag", ylabel="ACF", title="Autocorrelation")
    max_lag = min(500, length(pol) ÷ 4)
    acf = autocorrelation(pol, max_lag)
    lines!(ax4, (0:max_lag) .* dt, acf; color=:blue, linewidth=2)
    hlines!(ax4, [0.0]; color=:black, linestyle=:dash, linewidth=1)

    # Panel 5: spectrum (log–log)
    ax5 = Axis(fig[3, 2]; xlabel="Frequency", ylabel="Power", title="Spectrum",
               xscale=log10, yscale=log10)
    freqs, psd = compute_psd(pol, dt)
    valid = (freqs .> 0) .& (psd .> 1e-12)
    lines!(ax5, freqs[valid], psd[valid]; color=:blue, linewidth=2)

    # Panel 6: phase trajectory
    ax6 = Axis(fig[4, 1:2]; xlabel="u₁", ylabel="u₂", title="Phase Trajectory", aspect=DataAspect())
    stride = max(1, length(u1) ÷ 2000)
    lines!(ax6, u1[1:stride:end], u2[1:stride:end]; color=:dodgerblue, linewidth=1.5)
    lines!(ax6, [-3, 3], [-3, 3]; color=:gray, linestyle=:dash, linewidth=1)
    scatter!(ax6, [u1[1]],  [u2[1]];  color=:blue, markersize=13, label="Start")
    scatter!(ax6, [u1[end]], [u2[end]]; color=:red,  markersize=13, label="End")
    axislegend(ax6, position=:lt)

    return fig
end


"""
Plot 5: Return maps
"""
function plot_return_maps(κ_values, p_base)
    @info "  Computing return maps..."
    
    n_κ = length(κ_values)
    fig = Figure(size=(1600, 1200))
    
    scenarios = [
        (u0=[1.0, -1.0], tmax=800.0, name="Polarized"),
        (u0=[0.1, 0.1], tmax=800.0, name="Consensus"),
        (u0=[2.0, -2.0], tmax=800.0, name="Extreme")
    ]
    
    dt = 0.1
    delay = 20
    
    for (κ_idx, κ) in enumerate(κ_values)
        p = ModelInterface.kappa_set(p_base, κ)
        
        for (s_idx, scen) in enumerate(scenarios)
            function ode!(du, u, p, t)
                F = ModelInterface.f(u, p)
                du .= F
            end
            
            prob = ODEProblem(ode!, scen.u0, (0.0, scen.tmax), p)
            sol = solve(prob, Tsit5(); saveat=dt)
            
            traj = reduce(hcat, sol.u)
            u1 = traj[1, :]
            
            label_ratio = format_ratio_label(κ, getproperty(p, :kstar); digits_ratio=3, digits_kappa=2)
            ax = Axis(fig[κ_idx, s_idx];
                     xlabel="u₁(t)", ylabel="u₁(t+τ)",
                     title=label_ratio * " — $(scen.name)",
                     aspect=DataAspect())
            
            if var(u1) < 1e-6
                eq = mean(u1)
                scatter!(ax, [eq], [eq]; markersize=20, color=:red, marker=:xcross)
                padding = 0.5
                lims = (eq - padding, eq + padding)
                xlims!(ax, lims)
                ylims!(ax, lims)
            else
                x_vals = u1[1:end-delay]
                y_vals = u1[delay+1:end]
                
                scatter!(ax, x_vals, y_vals; markersize=2, alpha=0.3, color=:blue)
                
                all_vals = [x_vals; y_vals]
                lims = extrema(all_vals)
                padding = 0.1 * (lims[2] - lims[1])
                plot_lims = (lims[1] - padding, lims[2] + padding)
                
                lines!(ax, [plot_lims[1], plot_lims[2]], [plot_lims[1], plot_lims[2]];
                      color=:red, linestyle=:dash, linewidth=2)
                
                xlims!(ax, plot_lims)
                ylims!(ax, plot_lims)
                
                corr_val = cor(x_vals, y_vals)
                text!(ax, plot_lims[1] + 0.05*(plot_lims[2]-plot_lims[1]),
                     plot_lims[2] - 0.05*(plot_lims[2]-plot_lims[1]);
                     text="ρ=$(round(corr_val, digits=2))",
                     fontsize=12, align=(:left, :top))
            end
        end
    end
    
    return fig
end

"""
Plot 6: Parameter scan
"""
function plot_parameter_scan(κ_range, p_base; n_points=150)
    @info "  Computing parameter scan..."
    
    κ_vals = collect(range(κ_range[1], κ_range[2], length=n_points))
    κ_star = getproperty(p_base, :kstar)
    diag = equilibrium_diagnostics(κ_vals, p_base)

    consensus_stable_mask = diag.consensus_max .< 0
    consensus_unstable_mask = .!consensus_stable_mask

    polarized_mask = diag.polarized_exists .& (.!isnan.(diag.polarized_max))
    polarized_stable_mask = polarized_mask .& (diag.polarized_max .< 0)
    polarized_unstable_mask = polarized_mask .& (.!polarized_stable_mask)

    fig = Figure(size=(1200, 720))

    ax = Axis(fig[1, 1];
             xlabel="Coupling κ", ylabel="Polarization |u₁ - u₂|/2",
             title="Bifurcation Structure")

    if any(consensus_stable_mask)
        lines!(ax, κ_vals[consensus_stable_mask], zeros(sum(consensus_stable_mask));
               color=:steelblue, linewidth=3, label="Consensus (stable)")
    end
    if any(consensus_unstable_mask)
        lines!(ax, κ_vals[consensus_unstable_mask], zeros(sum(consensus_unstable_mask));
               color=:steelblue, linewidth=3, linestyle=:dash,
               label="Consensus (unstable)")
    end
    if any(polarized_stable_mask)
        lines!(ax, κ_vals[polarized_stable_mask], diag.polarized_amp[polarized_stable_mask];
               color=:firebrick, linewidth=3, label="Polarized (stable)")
        lines!(ax, κ_vals[polarized_stable_mask], -diag.polarized_amp[polarized_stable_mask];
               color=:firebrick, linewidth=3)
    end
    if any(polarized_unstable_mask)
        lines!(ax, κ_vals[polarized_unstable_mask], diag.polarized_amp[polarized_unstable_mask];
               color=:darkorange, linewidth=3, linestyle=:dot,
               label="Polarized (unstable)")
        lines!(ax, κ_vals[polarized_unstable_mask], -diag.polarized_amp[polarized_unstable_mask];
               color=:darkorange, linewidth=3, linestyle=:dot)
    end

    vlines!(ax, [κ_star]; color=:seagreen, linestyle=:dash, linewidth=3,
           label="κ* = $(round(κ_star, digits=4))")

    ticks = ratio_ticks(minimum(κ_vals), maximum(κ_vals), κ_star)
    ticks === nothing || (ax.xticks = ticks)

    axislegend(ax, position=:lt)

    return fig
end

"""
Plot 7: Lyapunov spectrum
"""
function plot_lyapunov(κ_range, p_base; tmax=2000.0)
    @info "  Computing Lyapunov exponents..."
    
    κ_vals = collect(κ_range)
    lyap_values = Float64[]
    
    u0 = [0.1, 0.1]
    
    for (i, κ) in enumerate(κ_vals)
        p = ModelInterface.kappa_set(p_base, κ)
        
        λ = compute_lyapunov(
            ModelInterface.f, ModelInterface.jacobian, u0, p;
            tmax=tmax, dt=0.1
        )
        
        push!(lyap_values, λ)
        
        if i % 5 == 0
            @info "    Progress: $(i)/$(length(κ_vals))"
        end
    end
    
    fig = Figure(size=(1200, 700))
    
    ax = Axis(fig[1, 1];
             xlabel="Coupling κ", ylabel="Lyapunov exponent λ",
             title="Estimated leading eigenvalue ")
    
    lines!(ax, κ_vals, lyap_values; linewidth=3, color=:blue)
    hlines!(ax, [0]; color=:red, linestyle=:dash, linewidth=2)
    
    band!(ax, κ_vals, fill(-1, length(lyap_values)),
          min.(lyap_values, 0); color=(:green, 0.2))
    band!(ax, κ_vals, zeros(length(lyap_values)),
          max.(lyap_values, 0); color=(:red, 0.2))

    κ_star = getproperty(p_base, :kstar)
    ticks = ratio_ticks(minimum(κ_vals), maximum(κ_vals), κ_star)
    ticks === nothing || (ax.xticks = ticks)
    
    return fig
end

#═══════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS PIPELINE
#═══════════════════════════════════════════════════════════════════════════

"""
Run complete analysis from YAML config
"""
function analyze_config(config_path::String)
    @info "═══════════════════════════════════════════════════════════════"
    @info "  Comprehensive Analysis from YAML"
    @info "═══════════════════════════════════════════════════════════════"
    @info "Config: $config_path"

    @info ""

    # Load configuration
    cfg = load_yaml_config(config_path)
    config_name = haskey(cfg, "name") ? cfg["name"] : sanitize_run_name(splitext(basename(config_path))[1])

    @info "Configuration: $config_name"
    @info "  λ = $(cfg["params"]["lambda"])"
    @info "  σ = $(cfg["params"]["sigma"])"
    @info "  Θ = $(cfg["params"]["theta"])"
    @info "  c₀ = $(cfg["params"]["c0"])"
    @info ""
    
    # Convert to model parameters
    p_base, meta = yaml_to_model_params(cfg)
    κ_star = getproperty(p_base, :kstar)

    results = Dict{String, String}()
    results["kappa_star"] = string(κ_star)
    results["Vstar"] = string(p_base.Vstar)
    results["g"] = string(p_base.g)

    @info "Model parameters:"
    @info "  λ = $(p_base.λ)"
    @info "  σ = $(p_base.σ)"
    @info "  V* ≈ $(round(p_base.Vstar, digits=5))"
    @info "  g ≈ $(round(p_base.g, digits=5))"
    @info "  β = $(p_base.beta)"
    @info "  κ* (theory) = $(round(κ_star, digits=5))"
    @info ""
    
    # Determine κ range
    if haskey(cfg, "sweep")
        sweep_cfg = cfg["sweep"]
        n_points = get(sweep_cfg, "points", 121)

        if haskey(sweep_cfg, "kappa_from_factor_of_kstar") && isfinite(κ_star) && κ_star > 0
            κ_min = κ_star * Float64(sweep_cfg["kappa_from_factor_of_kstar"])
        else
            κ_min = Float64(get(sweep_cfg, "kappa_from", max(0.0, 0.5 * κ_star)))
        end

        if haskey(sweep_cfg, "kappa_to")
            κ_max = Float64(sweep_cfg["kappa_to"])
        else
            factor = Float64(get(sweep_cfg, "kappa_to_factor_of_kstar", 3.0))
            κ_max = (isfinite(κ_star) && κ_star > 0) ? κ_star * factor : κ_min + max(abs(κ_min), 1.0)
        end
    else
        κ_min = max(0.0, 0.4 * κ_star)
        κ_max = (isfinite(κ_star) && κ_star > 0) ? 2.5 * κ_star : κ_min + 1.0
        n_points = 121
    end

    if κ_max <= κ_min
        bump = max(abs(κ_star) * 0.25, 1e-3)
        κ_max = κ_min + bump
    end

    @info "Analysis range: κ ∈ [$(round(κ_min, digits=3)), $(round(κ_max, digits=3))]"
    @info ""
    
    # Set up output
    outdir = joinpath(cfg["output_dir"], "comprehensive_analysis_$config_name")
    mkpath(outdir)
    @info "Output directory: $outdir"
    @info ""
    
    # Set theme
    PlottingCairo.set_theme_elegant!()
    
    # Generate plots
    κ_range_fine = range(κ_min, κ_max; length=n_points)
    κ_range_coarse = range(κ_min, κ_max; length=max(20, div(n_points, 5)))
    κ_selected = kappa_from_ratios(κ_star, κ_min, κ_max, [0.5, 0.9, 1.1, 1.6]; fallback_count=4)
    
    # Plot 1: Bifurcation
    try
        @info "Generating Plot 1: Bifurcation diagram"
        fig = plot_bifurcation(κ_range_fine, p_base)
        filename = "01_bifurcation.png"
        save(joinpath(outdir, filename), fig)
        results["bifurcation"] = filename
        @info "  ✓ Saved: $filename"
    catch e
        @error "  ✗ Failed" exception=e
    end
    
    # Plot 2: Phase portraits
    try
        @info "Generating Plot 2: Phase portraits"
        fig = plot_phase_portraits(κ_selected, p_base)
        filename = "02_phase_portraits.png"
        save(joinpath(outdir, filename), fig)
        results["phase_portraits"] = filename
        @info "  ✓ Saved: $filename"
    catch e
        @error "  ✗ Failed" exception=e
    end
    
    # Plot 3: Basins
    try
        @info "Generating Plot 3: Basins of attraction"
        κ_basin = kappa_from_ratios(κ_star, κ_min, κ_max, [0.6, 0.9, 1.1, 1.5]; fallback_count=4)
        fig = plot_basins(κ_basin, p_base; resolution=40)
        filename = "03_basins.png"
        save(joinpath(outdir, filename), fig)
        results["basins"] = filename
        @info "  ✓ Saved: $filename"
    catch e
        @error "  ✗ Failed" exception=e
    end
    
    # Plot 4: Time series
    try
        @info "Generating Plot 4: Time series analysis"
        dt = cfg["dt"]
        T = min(cfg["T"], 500.0)
        
        κ_times = kappa_from_ratios(κ_star, κ_min, κ_max, [0.75, 1.05, 1.4]; fallback_count=3)
        for κ in κ_times
            fig = plot_timeseries(κ, p_base; tmax=T, dt=dt)
            ratio_tag = isfinite(κ_star) && κ_star > 0 ? @sprintf("%0.2f", κ / κ_star) : @sprintf("%0.2f", κ)
            filename = "04_timeseries_ratio_$(replace(ratio_tag, "." => "p")).png"
            save(joinpath(outdir, filename), fig)
        end
        results["timeseries"] = "04_timeseries_*.png"
        @info "  ✓ Saved: 04_timeseries_*.png"
    catch e
        @error "  ✗ Failed" exception=e
    end
    
    # Plot 5: Return maps
    try
        @info "Generating Plot 5: Return maps"
        κ_return = kappa_from_ratios(κ_star, κ_min, κ_max, [0.8, 1.05, 1.4]; fallback_count=3)
        fig = plot_return_maps(κ_return, p_base)
        filename = "05_return_maps.png"
        save(joinpath(outdir, filename), fig)
        results["return_maps"] = filename
        @info "  ✓ Saved: $filename"
    catch e
        @error "  ✗ Failed" exception=e
    end
    
    # Plot 6: Parameter scan
    try
        @info "Generating Plot 6: Parameter scan"
        fig = plot_parameter_scan((κ_min, κ_max), p_base; n_points=n_points)
        filename = "06_parameter_scan.png"
        save(joinpath(outdir, filename), fig)
        results["parameter_scan"] = filename
        @info "  ✓ Saved: $filename"
    catch e
        @error "  ✗ Failed" exception=e
    end
    
    # Plot 7: Lyapunov
    try
        @info "Generating Plot 7: Lyapunov spectrum"
        fig = plot_lyapunov(κ_range_coarse, p_base; tmax=min(2000.0, cfg["T"]))
        filename = "07_lyapunov.png"
        save(joinpath(outdir, filename), fig)
        results["lyapunov"] = filename
        @info "  ✓ Saved: $filename"
    catch e
        @error "  ✗ Failed" exception=e
    end
    
    # Generate summary
    summary_path = joinpath(outdir, "summary.txt")
    open(summary_path, "w") do io
        println(io, "═══════════════════════════════════════════════════════════════")
        println(io, "Comprehensive Bifurcation Analysis")
        println(io, "═══════════════════════════════════════════════════════════════")
        println(io, "")
        println(io, "Configuration: $config_name")
        println(io, "Generated: $(now())")
        println(io, "")
        println(io, "Parameters:")
        println(io, "  λ = $(cfg["params"]["lambda"])")
        println(io, "  σ = $(cfg["params"]["sigma"])")
        println(io, "  Θ = $(cfg["params"]["theta"])")
        println(io, "  c₀ = $(cfg["params"]["c0"])")
        println(io, "")
        println(io, "Model (normal form calibration):")
        println(io, "  λ = $(p_base.λ)")
        println(io, "  σ = $(p_base.σ)")
        println(io, "  Θ = $(p_base.Θ)")
        println(io, "  c₀ = $(p_base.c0)")
        println(io, "  Hazard = $(meta.hazard)")
        println(io, "  V* ≈ $(round(p_base.Vstar, digits=6))")
        println(io, "  g ≈ $(round(p_base.g, digits=6))")
        println(io, "  β = $(p_base.beta)")
        println(io, "  κ* = $(round(κ_star, digits=6))")
        println(io, "  Theory: κ* = g σ² / (2 λ V*)")
        theory_val = p_base.g * p_base.σ^2 / (2 * p_base.λ * p_base.Vstar)
        println(io, "         = $(round(theory_val, digits=6)) (numerical check)")
        println(io, "")
        println(io, "Analysis Range:")
        println(io, "  κ ∈ [$(round(κ_min, digits=3)), $(round(κ_max, digits=3))]")
        println(io, "  Points: $n_points")
        if isfinite(κ_star) && κ_star > 0
            println(io, "  κ/κ* ∈ [$(round(κ_min/κ_star, digits=3)), $(round(κ_max/κ_star, digits=3))]")
        end
        println(io, "")
        println(io, "Generated Plots:")
        for (key, file) in sort(collect(results))
            println(io, "  $key: $file")
        end
        println(io, "")
        println(io, "═══════════════════════════════════════════════════════════════")
    end
    
    @info ""
    @info "═══════════════════════════════════════════════════════════════"
    @info "  ✅ Analysis Complete!"
    @info "═══════════════════════════════════════════════════════════════"
    @info "Output: $outdir"
    @info "Summary: $summary_path"
    @info "Plots generated: $(length(results))"
    @info "═══════════════════════════════════════════════════════════════"
    
    return results
end

"""
Process all YAML files in a directory
"""
function analyze_all(config_dir::String="configs")
    @info "Scanning directory: $config_dir"
    
    if !isdir(config_dir)
        @error "Directory not found: $config_dir"
        return Dict()
    end
    
    yaml_files = filter(f -> endswith(lowercase(f), ".yaml") || endswith(lowercase(f), ".yml"),
                       readdir(config_dir, join=true))
    
    if isempty(yaml_files)
        @error "No YAML files found in $config_dir"
        return Dict()
    end
    
    @info "Found $(length(yaml_files)) configuration(s)"
    
    all_results = Dict{String, Any}()
    
    for config_file in yaml_files
        config_name = sanitize_run_name(splitext(basename(config_file))[1])
        @info ""
        @info "Processing: $config_name"

        try
            results = analyze_config(config_file)
            all_results[config_name] = results
        catch e
            @error "Failed to process $config_name" exception=(e, catch_backtrace())
            all_results[config_name] = :failed
        end
    end
    
    @info ""
    @info "═══════════════════════════════════════════════════════════════"
    @info "  ✅ Batch Processing Complete"
    @info "═══════════════════════════════════════════════════════════════"
    @info "Processed: $(length(all_results)) configurations"
    @info "  Successful: $(sum(values(all_results) .!= :failed))"
    @info "  Failed: $(sum(values(all_results) .== :failed))"
    @info "═══════════════════════════════════════════════════════════════"
    
    return all_results
end

#═══════════════════════════════════════════════════════════════════════════
# COMMAND LINE INTERFACE
#═══════════════════════════════════════════════════════════════════════════

function main()
    if length(ARGS) == 0
        println("""
        Usage:
          julia --project=. scripts/analyze_from_yaml.jl <config.yaml>
          julia --project=. scripts/analyze_from_yaml.jl --all [config_dir]
        
        Examples:
          julia --project=. scripts/analyze_from_yaml.jl configs/example_sweep.yaml
          julia --project=. scripts/analyze_from_yaml.jl --all
          julia --project=. scripts/analyze_from_yaml.jl --all configs/
        """)
        return
    end
    
    if ARGS[1] == "--all"
        config_dir = length(ARGS) >= 2 ? ARGS[2] : "configs"
        analyze_all(config_dir)
    else
        config_path = ARGS[1]
        analyze_config(config_path)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
