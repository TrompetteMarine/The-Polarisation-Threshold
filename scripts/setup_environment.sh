#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v julia >/dev/null 2>&1; then
    echo "Error: Julia is not installed or not available on PATH." >&2
    echo "Please install Julia 1.10 or newer before running this script." >&2
    exit 1
fi

echo "Instantiating Julia project at $ROOT_DIR"
julia --project=. --color=yes -e '
using Pkg

function ensure_general_registry!()
    regs = try
        Pkg.Registry.reachable_registries()
    catch err
        @info "Unable to query registries, refreshing metadata" exception = err
        Pkg.Registry.update()
        Pkg.Registry.reachable_registries()
    end
    if all(reg.name != "General" for reg in regs)
        @info "Adding General registry"
        Pkg.Registry.add(Pkg.RegistrySpec(name="General", url="https://github.com/JuliaRegistries/General.git"))
    end
end

function ensure_manifest_consistency!()
    project_deps = collect(keys(Pkg.project().dependencies))
    manifest = try
        Pkg.manifest()
    catch err
        @info "Manifest read failed, triggering resolve" exception = err
        Pkg.resolve()
        return
    end
    manifest_deps = manifest === nothing ? String[] : collect(keys(manifest.dependencies))
    missing = setdiff(project_deps, manifest_deps)
    if !isempty(missing)
        @info "Resolving manifest to include new direct dependencies" missing
        Pkg.resolve()
    end
end

ensure_general_registry!()
ensure_manifest_consistency!()
try
    Pkg.instantiate()
catch err
    @warn "Pkg.instantiate() failed, attempting resolve before retry" exception = err
    Pkg.resolve()
    Pkg.instantiate()
end
Pkg.precompile()
'

# ensure common output directories exist
mkdir -p figs outputs

echo "Environment ready. Use 'julia --project .' to run scripts within this repository."

