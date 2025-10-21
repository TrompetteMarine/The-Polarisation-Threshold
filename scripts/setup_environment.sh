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

function ensure_dependencies!()
    required = [
        PackageSpec(name = "ArgParse"),
        PackageSpec(name = "BifurcationKit"),
        PackageSpec(name = "CairoMakie"),
        PackageSpec(name = "Colors"),
        PackageSpec(name = "DifferentialEquations"),
        PackageSpec(name = "FFTW"),
        PackageSpec(name = "ForwardDiff"),
        PackageSpec(name = "GR"),
        PackageSpec(name = "Graphs"),
        PackageSpec(name = "JSON3"),
        PackageSpec(name = "Parameters"),
        PackageSpec(name = "Plots"),
        PackageSpec(name = "ProgressMeter"),
        PackageSpec(name = "RecipesBase"),
        PackageSpec(name = "StatsBase"),
        PackageSpec(name = "YAML"),
    ]
    for spec in required
        try
            Pkg.add(spec)
        catch err
            @error "Failed to add dependency" spec = spec.name exception = err
            rethrow(err)
        end
    end
end

ensure_general_registry!()
Pkg.Registry.update()
ensure_dependencies!()
Pkg.instantiate()
Pkg.precompile()'

# ensure common output directories exist
mkdir -p figs outputs

echo "Environment ready. Use 'julia --project .' to run scripts within this repository."

