#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "════════════════════════════════════════════════════════════"
echo "  BeliefSim.jl Environment Setup"
echo "════════════════════════════════════════════════════════════"
echo ""

if ! command -v julia >/dev/null 2>&1; then
    echo "❌ Error: Julia is not installed" >&2
    exit 1
fi

echo "✓ Found Julia $(julia --version)"
echo ""

echo "🧹 Cleaning previous installation..."
rm -f Manifest.toml
echo ""

echo "📦 Installing dependencies..."
julia --project=. --color=yes -e '
using Pkg

println("  → Updating registry...")
Pkg.Registry.update()

println("  → Installing core packages...")
Pkg.instantiate()

println("  → Installing optional packages...")
try
    Pkg.add("CairoMakie")
    println("  ✓ CairoMakie installed")
catch e
    @warn "CairoMakie installation failed (optional)" exception=e
end

try
    Pkg.add(url="https://github.com/bifurcationkit/BifurcationKit.jl.git")
    println("  ✓ BifurcationKit installed")
catch e
    @warn "BifurcationKit installation failed (optional)" exception=e
end

println("  → Precompiling...")
Pkg.precompile()
'

mkdir -p figs outputs

echo ""
echo "✅ Verifying installation..."
julia --project=. -e '
using BeliefSim
println("  ✓ BeliefSim loaded")
'

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ✅ Setup Complete!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Test with: julia --project=. examples/quickstart.jl"