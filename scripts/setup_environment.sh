#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v julia >/dev/null 2>&1; then
    echo "Error: Julia is not installed or not available on PATH." >&2
    echo "Please install Julia 1.8 or newer before running this script." >&2
    exit 1
fi

echo "Instantiating Julia project at $ROOT_DIR"
julia --project=. --color=yes -e '
using Pkg

function ensure_general_registry!()
    try
        Pkg.Registry.add("General")
    catch err
        msg = sprint(showerror, err)
        if occursin("already added", msg)
            @info "General registry already present"
        else
            rethrow(err)
        end
    end
end

ensure_general_registry!()

function ensure_git_dependency!(name, url; rev=nothing)
    spec_kwargs = rev === nothing ? (; name=name, url=url) : (; name=name, url=url, rev=rev)
    try
        Pkg.add(Pkg.PackageSpec(; spec_kwargs...))
    catch err
        msg = sprint(showerror, err)
        if occursin("already exists", msg)
            @info "Dependency $(name) already sourced from $(url)"
        else
            rethrow(err)
        end
    end
end

# Ensure all project dependencies are sourced directly from their Git repositories.
for dep in [
    (name = "ArgParse", url = "https://github.com/carlobaldassi/ArgParse.jl.git"),
    (name = "BifurcationKit", url = "https://github.com/bifurcationkit/BifurcationKit.jl.git"),
    (name = "CairoMakie", url = "https://github.com/MakieOrg/CairoMakie.jl.git"),
    (name = "Colors", url = "https://github.com/JuliaGraphics/Colors.jl.git"),
    (name = "DifferentialEquations", url = "https://github.com/SciML/DifferentialEquations.jl.git"),
    (name = "FFTW", url = "https://github.com/JuliaMath/FFTW.jl.git"),
    (name = "ForwardDiff", url = "https://github.com/JuliaDiff/ForwardDiff.jl.git"),
    (name = "GR", url = "https://github.com/jheinen/GR.jl.git"),
    (name = "Graphs", url = "https://github.com/JuliaGraphs/Graphs.jl.git"),
    (name = "JSON3", url = "https://github.com/quinnj/JSON3.jl.git"),
    (name = "Parameters", url = "https://github.com/mauro3/Parameters.jl.git"),
    (name = "Plots", url = "https://github.com/JuliaPlots/Plots.jl.git"),
    (name = "ProgressMeter", url = "https://github.com/timholy/ProgressMeter.jl.git"),
    (name = "RecipesBase", url = "https://github.com/JuliaPlots/RecipesBase.jl.git"),
    (name = "StatsBase", url = "https://github.com/JuliaStats/StatsBase.jl.git"),
    (name = "YAML", url = "https://github.com/JuliaData/YAML.jl.git"),
]
    ensure_git_dependency!(dep.name, dep.url)
end
try
    Pkg.instantiate()
catch err
    @warn "Pkg.instantiate() failed, attempting resolve before retry" exception = err
    Pkg.resolve()
    Pkg.instantiate()
end
Pkg.precompile()'

# ensure common output directories exist
mkdir -p figs outputs

echo "Environment ready. Use 'julia --project .' to run scripts within this repository."

