#!/usr/bin/env julia
using CSV, DataFrames, Plots

if length(ARGS) < 1
    println("Usage: plot_bifurcation.jl bifurcation.csv [kappa_star] [out.png]")
    exit(1)
end

csvpath = ARGS[0]
κstar = length(ARGS) >= 2 ? parse(Float64, ARGS[1]) : NaN
out = length(ARGS) >= 3 ? ARGS[2] : "bifurcation.png"

df = DataFrame(CSV.File(csvpath))
p = plot(df.kappa, df.amp, marker=:circle, xlabel="κ", ylabel="|ḡ|", label="amplitude", title="Pitchfork")
if !isnan(κstar)
    vline!(p, [κstar], linestyle=:dash, label="κ*")
end
savefig(p, out)
println("Saved ", out)
