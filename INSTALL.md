# Installation Guide

## Method 1: Automatic (Recommended)
```bash
./scripts/setup_environment.sh
```

## Method 2: Manual Installation

### Core Installation
```bash
# 1. Clean slate
rm -f Manifest.toml

# 2. Install core dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# 3. Test core functionality
julia --project=. -e 'using BeliefSim; println("✓ Core installation successful")'
```

### Optional: Bifurcation Analysis
```bash
# Install Makie for visualization
julia --project=. -e 'using Pkg; Pkg.add("CairoMakie")'

# Install BifurcationKit for advanced analysis
julia --project=. -e 'using Pkg; Pkg.add(url="https://github.com/bifurcationkit/BifurcationKit.jl.git")'
```

## Troubleshooting

### Issue: "expected package CairoMakie to be registered"

**Solution**: Remove `CairoMakie` from `Project.toml` if present, then install separately:
```bash
# Edit Project.toml and remove CairoMakie line
# Then run:
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.add("CairoMakie")'
```

### Issue: Registry errors

**Solution**: Update or reset registry:
```bash
julia -e 'using Pkg; Pkg.Registry.update()'
# Or if that fails:
rm -rf ~/.julia/registries/General
julia -e 'using Pkg; Pkg.Registry.add("General")'
```

### Issue: Precompilation errors

**Solution**: Clean compiled cache:
```bash
rm -rf ~/.julia/compiled
julia --project=. -e 'using Pkg; Pkg.precompile()'
```

## Verification
```bash
# Test core functionality
julia --project=. examples/quickstart.jl

# Test optional features (if installed)
julia --project=. scripts/make_phase_portraits.jl --kappas "0.9,1.0,1.1"
```