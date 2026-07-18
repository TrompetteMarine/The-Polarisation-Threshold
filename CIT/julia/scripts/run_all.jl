#!/usr/bin/env julia
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using CITReplication

publication = "--publication" in ARGS
output = "outputs"
for (i, argument) in enumerate(ARGS)
    if argument == "--output" && i < length(ARGS)
        global output = ARGS[i+1]
    end
end

report = run_pipeline(output; publication)
for key in sort(collect(keys(report)))
    println(key, ": ", report[key])
end
