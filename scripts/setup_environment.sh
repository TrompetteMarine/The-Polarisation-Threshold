#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "════════════════════════════════════════════════════════════"
echo "  BeliefSim.jl Environment Setup"
echo "════════════════════════════════════════════════════════════"
echo ""

# Function to install Julia using juliaup
install_julia() {
    echo "📥 Julia not found. Installing Julia via juliaup..."
    echo ""
    
    if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
        # Linux or macOS
        echo "  → Downloading and installing juliaup..."
        curl -fsSL https://install.julialang.org | sh -s -- --yes
        
        # Source the juliaup environment
        export PATH="$HOME/.juliaup/bin:$PATH"
        
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
        # Windows
        echo "❌ Error: Automatic installation on Windows requires manual setup" >&2
        echo "Please install Julia from: https://julialang.org/downloads/" >&2
        echo "Or install juliaup from: https://github.com/JuliaLang/juliaup" >&2
        exit 1
    else
        echo "❌ Error: Unsupported operating system: $OSTYPE" >&2
        exit 1
    fi
    
    # Verify installation
    if ! command -v julia >/dev/null 2>&1; then
        echo "❌ Error: Julia installation failed" >&2
        echo "Please restart your terminal and run this script again," >&2
        echo "or install Julia manually from: https://julialang.org/downloads/" >&2
        exit 1
    fi
    
    echo "  ✓ Julia installed successfully!"
    echo ""
}

# Check if Julia is installed
if ! command -v julia >/dev/null 2>&1; then
    install_julia
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