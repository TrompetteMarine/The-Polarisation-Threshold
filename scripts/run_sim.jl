#!/usr/bin/env julia
using ArgParse
using BeliefSim
using BeliefSim.Types
using BeliefSim.Stats

function main()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--lambda"; arg_type=Float64; default=1.0; help="Mean reversion λ"
        "--sigma"; arg_type=Float64; default=1.0; help="Noise σ"
        "--theta"; arg_type=Float64; default=2.0; help="Threshold Θ"
        "--c0"; arg_type=Float64; default=0.5; help="Reset contraction c0 ∈ (0,1)"
        "--nu0"; arg_type=Float64; default=0.5; help="Step hazard height ν0"
        "--N"; arg_type=Int; default=20000; help="Population size"
        "--T"; arg_type=Float64; default=300.0; help="Horizon"
        "--dt"; arg_type=Float64; default=0.01; help="Time step"
        "--burnin"; arg_type=Float64; default=100.0; help="Burn-in window"
        "--seed"; arg_type=Int; default=0; help="Random seed (0=random)"
        "--kappa"; arg_type=Float64; default=0.0; help="Coupling κ"
        "--mode"; arg_type=String; default="vstar"; help="vstar | sweep"
    end
    args = parse_args(s)

    p = Params(λ=args["lambda"], σ=args["sigma"], Θ=args["theta"], 
               c0=args["c0"], hazard=StepHazard(args["nu0"]))

    try
        if args["mode"] == "vstar"
            Vstar = estimate_Vstar(p; N=args["N"], T=args["T"], dt=args["dt"], 
                                  burn_in=args["burnin"], seed=args["seed"])
            κcrit = critical_kappa(p; Vstar=Vstar)
            println("V* = $Vstar")
            println("κ* ≈ $κcrit")
        elseif args["mode"] == "sweep"
            Vstar = estimate_Vstar(p; N=args["N"], T=args["T"], dt=args["dt"], 
                                  burn_in=args["burnin"], seed=args["seed"])
            κcrit = critical_kappa(p; Vstar=Vstar)
            κgrid = collect(range(0.0, stop=2.0*κcrit, length=21))
            res = BeliefSim.Stats.sweep_kappa(p, κgrid; N=args["N"], T=args["T"], 
                                             dt=args["dt"], burn_in=args["burnin"], 
                                             seed=args["seed"])
            println("# kappa\tamp\tvariance")
            for (κ, a, V) in zip(res.κ, res.amp, res.V)
                println("$κ\t$a\t$V")
            end
        else
            error("Unknown mode: $(args["mode"]). Use 'vstar' or 'sweep'")
        end
    catch e
        @error "Simulation failed" exception=(e, catch_backtrace())
        exit(1)
    end
end

main()