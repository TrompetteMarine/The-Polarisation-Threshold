# Phylogenetic Bifurcation Analysis Guide

## Overview

The phylogenetic bifurcation analysis extends BeliefSim.jl with tools for generating "phylogenetic tree" bifurcation diagrams. These visualizations show the complete attractor structure as the social coupling parameter κ sweeps through the critical threshold κ*, analogous to the classic logistic map bifurcation diagrams.

## Quick Start

### Generate the main phylogenetic diagram

```bash
julia --project=. scripts/fig_phylogenetic_tree.jl
```

**Output:**
- `figs/phylogenetic_bifurcation.pdf` - Main tuning fork diagram
- `figs/phylogenetic_bifurcation.png` - Raster version

**Features:**
- Dense parameter sweep (500 κ points, 30 ICs per point)
- Stable vs unstable equilibrium classification
- Theoretical envelope ±√(μ/b) overlay
- Inset scaling verification (β ≈ 0.5)

### Generate attractor evolution gallery

```bash
julia --project=. scripts/fig_attractor_evolution.jl
```

**Output:**
- `figs/phase_portrait_gallery.pdf` - 2×3 grid of phase portraits
- `figs/basin_evolution.pdf` - Basin splitting dynamics
- `figs/trajectory_fates.pdf` - Convergence comparison

## Theoretical Background

### Reduced Normal Form

The center-manifold reduction yields 1D dynamics for the polarization amplitude a(t):

```
ȧ = μ(κ)·a - b·a³
```

where:
- **μ(κ) = μ_slope · (κ - κ*)** is the linear growth rate
- **b > 0** is the cubic coefficient (supercritical bifurcation)
- **μ_slope = 2λV*/σ²** from the micro-level parameters

### Equilibrium Structure

- **κ < κ*:** Single stable equilibrium at a = 0 (consensus)
- **κ = κ*:** Pitchfork bifurcation (μ = 0)
- **κ > κ*:** Three equilibria:
  - a = 0 (unstable consensus)
  - a = ±√(μ/b) (stable polarized states)

### Scaling Law

Near the bifurcation: |a*(κ)| ∝ (κ - κ*)^β with **β = 0.5** (supercritical pitchfork)

## Module API

### Core Module: `PhylogeneticDiagram.jl`

Located in `src/bifurcation/PhylogeneticDiagram.jl`

#### Data Structures

**`NormalFormParams`**
```julia
struct NormalFormParams
    κ_star::Float64    # Critical coupling
    b::Float64         # Cubic coefficient
    μ_slope::Float64   # Transversality slope
    Vstar::Float64     # Stationary dispersion
end
```

**`PhylogeneticResult`**
```julia
struct PhylogeneticResult
    κ_vals::Vector{Float64}      # Coupling values
    a_vals::Vector{Float64}      # Amplitude values
    stable_mask::Vector{Bool}    # Stability flags
    n_initial_conditions::Int
    params::NormalFormParams
end
```

#### Key Functions

**`calibrate_normal_form`**
```julia
calibrate_normal_form(κ_star, Vstar; λ=1.0, σ=0.8, b_default=0.5)
```

Calibrate normal form parameters from micro-level quantities.

**`phylogenetic_sweep`**
```julia
phylogenetic_sweep(nf::NormalFormParams;
                  κ_points=500,
                  κ_max_factor=2.0,
                  n_ics=30,
                  a_range=(-3.0, 3.0),
                  T_burn=500.0,
                  dt=0.01,
                  cluster_tol=0.05,
                  adaptive_burn=true,
                  seed=0)
```

Perform dense parameter sweep with attractor classification.

**Returns:** `PhylogeneticResult`

**`equilibria`**
```julia
equilibria(κ, nf::NormalFormParams)
```

Compute all equilibria at given κ. Returns vector of equilibrium values.

**`stability`**
```julia
stability(a_eq, κ, nf::NormalFormParams)
```

Check if equilibrium a_eq is stable at coupling κ.

**`scaling_exponent`**
```julia
scaling_exponent(result::PhylogeneticResult;
                κ_range_factor=(1.0, 1.5))
```

Verify scaling exponent β ≈ 0.5. Returns (β, C, r²).

**`theoretical_envelope`**
```julia
theoretical_envelope(κ_grid, nf::NormalFormParams)
```

Compute theoretical envelope ±√(μ/b) for κ > κ*.

## Usage Examples

### Basic REPL Usage

```julia
using BeliefSim
using BeliefSim.Types, BeliefSim.Stats

# Load phylogenetic module
include("src/bifurcation/PhylogeneticDiagram.jl")
using .PhylogeneticDiagram

# Calibrate from micro parameters
p = Params(λ=1.0, σ=0.8, Θ=2.0, c0=0.5, hazard=StepHazard(0.6))
Vstar = estimate_Vstar(p; N=20_000, T=300.0, dt=0.01, burn_in=100.0, seed=2025)
κstar = critical_kappa(p; N=20_000, T=350.0, dt=0.01, burn_in=120.0, seed=2026)

# Setup normal form
nf = calibrate_normal_form(κstar, Vstar; λ=p.λ, σ=p.σ)

# Run phylogenetic sweep
result = phylogenetic_sweep(nf; κ_points=500, n_ics=30, seed=2027)

# Check scaling
β, C, r² = scaling_exponent(result)
@show β, r²  # Should see β ≈ 0.5

# Plot with CairoMakie (if available)
using CairoMakie

fig = Figure(resolution=(900, 650))
ax = Axis(fig[1, 1], xlabel="κ", ylabel="a*")

stable_idx = findall(result.stable_mask)
scatter!(ax, result.κ_vals[stable_idx], result.a_vals[stable_idx],
    markersize=3, color=:steelblue)

save("my_phylogenetic.pdf", fig)
```

### Custom Parameter Exploration

```julia
# Explore different cubic coefficients
for b in [0.3, 0.5, 0.7, 1.0]
    nf = NormalFormParams(κstar, b, μ_slope, Vstar)
    result = phylogenetic_sweep(nf; κ_points=300, n_ics=20)

    β, C, r² = scaling_exponent(result)
    println("b = $b: β = $β, r² = $r²")
end
```

### Compute Specific Equilibria

```julia
# At a specific κ value
κ_test = 1.2 * κstar
eq_list = equilibria(κ_test, nf)

for a_eq in eq_list
    is_stable = stability(a_eq, κ_test, nf)
    stability_str = is_stable ? "stable" : "unstable"
    println("Equilibrium a = $(round(a_eq, digits=3)) is $stability_str")
end
```

## YAML Configuration

The file `configs/phylogenetic_analysis.yaml` provides comprehensive configuration options.

### Key Sections

**Micro-level parameters:**
```yaml
params:
  lambda: 1.0
  sigma: 0.8
  theta: 2.0
  c0: 0.5
  hazard:
    kind: step
    nu0: 0.6
```

**Phylogenetic sweep:**
```yaml
phylogenetic_sweep:
  kappa_points: 500
  kappa_max_factor: 2.0
  n_initial_conditions: 30
  a_range: [-3.0, 3.0]
  T_burn: 500.0
  dt: 0.01
  cluster_tolerance: 0.05
  adaptive_burn: true
  seed: 2027
```

**Phase portraits:**
```yaml
phase_portraits:
  kappa_ratios: [0.5, 0.9, 0.99, 1.01, 1.2, 1.5]
  n_trajectories: 10
  T_integrate: 50.0
```

**Visualization:**
```yaml
plotting:
  backend: "cairomakie"
  phylogenetic_diagram:
    resolution: [900, 650]
    stable_color: "steelblue"
    unstable_color: "gray"
```

## Script Architecture

### `fig_phylogenetic_tree.jl`

**Steps:**
1. Calibrate micro-level parameters (V*, κ*)
2. Setup reduced normal form
3. Perform phylogenetic sweep
4. Verify scaling exponent
5. Generate visualization with inset

**Customization:**
Edit the parameter definitions at the top of the script:
```julia
p = Params(λ=1.0, σ=0.8, Θ=2.0, c0=0.5, hazard=StepHazard(0.6))
```

### `fig_attractor_evolution.jl`

**Steps:**
1. Setup parameters
2. Generate phase portrait gallery (6 panels)
3. Compute basin evolution (3 panels)
4. Compare trajectory fates (2 panels)

**Customization:**
Adjust κ ratios for phase portraits:
```julia
κ_ratios = [0.5, 0.9, 0.99, 1.01, 1.2, 1.5]
```

## Performance Considerations

### Computational Cost

**Main phylogenetic sweep:**
- Default: 500 κ points × 30 ICs × ~50k time steps
- Runtime: ~30-60 seconds on modern hardware
- Memory: < 1 GB

**Attractor evolution:**
- Phase portraits: 6 panels × 8 trajectories × ~5k time steps
- Basins: 3 panels × 300 ICs × ~10k time steps
- Runtime: ~20-40 seconds
- Memory: < 500 MB

### Optimization Tips

**Reduce resolution for quick prototyping:**
```julia
result = phylogenetic_sweep(nf;
    κ_points=200,      # Instead of 500
    n_ics=10,          # Instead of 30
    T_burn=200.0)      # Instead of 500.0
```

**Disable adaptive burn-in for speed:**
```julia
result = phylogenetic_sweep(nf; adaptive_burn=false)
```

**Use coarser time step:**
```julia
result = phylogenetic_sweep(nf; dt=0.02)  # Instead of 0.01
```

## Validation and Quality Checks

### Expected Results

**Scaling exponent:**
- β should be 0.48-0.52 (theoretical: 0.5)
- R² should be > 0.98 for good fit

**Symmetry:**
- Positive and negative branches should be symmetric
- Check: `maximum(abs.(a_vals_pos + a_vals_neg)) < 1e-3`

**Equilibrium accuracy:**
- At κ > κ*, stable branches should match √(μ/b)
- Tolerance: < 0.01

### Diagnostic Outputs

Both scripts print detailed summaries:
```
Summary statistics:
   ────────────────────────────────────────────────────────────
   Critical coupling:          κ* = 1.0234
   Stationary dispersion:      V* = 0.8756
   Normal form μ-slope:        2.1875
   Cubic coefficient:          b  = 0.5000
   Scaling exponent:           β  = 0.498 ± 0.05
   Regression R²:              0.9956
   Total attractor points:     1247
   Stable points:              831 (66.6%)
   ────────────────────────────────────────────────────────────
```

## Dependencies

### Required
- Julia ≥ 1.8
- BeliefSim.jl (this package)
- Statistics, Random, LinearAlgebra (stdlib)
- StatsBase

### Optional
- **CairoMakie:** Professional vector graphics (strongly recommended)
  - Install: `using Pkg; Pkg.add("CairoMakie")`
- **Plots.jl:** Fallback plotting backend (always available)

### Automatic Fallbacks

Scripts detect missing dependencies and adapt:
```julia
try
    using CairoMakie
    CAIRO_AVAILABLE = true
catch
    @warn "CairoMakie not available, falling back to Plots.jl"
    using Plots
    CAIRO_AVAILABLE = false
end
```

## Troubleshooting

### "CairoMakie not found"

**Solution:** Install CairoMakie or use Plots.jl fallback
```bash
julia --project=. -e 'using Pkg; Pkg.add("CairoMakie"); Pkg.precompile()'
```

### "Scaling exponent β far from 0.5"

**Causes:**
1. Insufficient burn-in time → increase `T_burn`
2. Coarse κ grid near threshold → increase `κ_points`
3. Wrong fitting range → adjust `κ_range_factors` in `scaling_exponent()`

**Solution:**
```julia
# Increase burn-in
result = phylogenetic_sweep(nf; T_burn=1000.0)

# Or use adaptive burn
result = phylogenetic_sweep(nf; adaptive_burn=true)
```

### "Asymmetric branches"

**Causes:**
1. Insufficient ICs → increase `n_ics`
2. Clustering tolerance too large → decrease `cluster_tol`

**Solution:**
```julia
result = phylogenetic_sweep(nf; n_ics=50, cluster_tol=0.02)
```

### Memory issues

**Solution:** Reduce resolution
```julia
result = phylogenetic_sweep(nf; κ_points=200, n_ics=15)
```

## Advanced Topics

### Custom Attractor Classification

Modify the clustering algorithm:
```julia
function cluster_attractors_kmeans(values, n_clusters)
    # Use KMeans.jl or custom algorithm
    # ...
end
```

### Particle System Validation

Compare reduced model against full N-particle simulation:
```julia
# Run at selected κ values
for κ in [0.8, 1.0, 1.2] .* κstar
    # Full particle system
    res = sweep_kappa(p, [κ]; N=20_000, T=300.0, dt=0.01, burn_in=100.0)
    a_particle = res.amp[1]

    # Reduced model prediction
    eq_list = equilibria(κ, nf)
    stable_eq = filter(a -> stability(a, κ, nf), eq_list)

    println("κ = $κ:")
    println("  Particle: a = $a_particle")
    println("  Theory:   a = $(maximum(abs.(stable_eq)))")
end
```

### Hysteresis Detection

Check for bistability (supercritical bifurcations shouldn't have hysteresis):
```julia
# Forward sweep
result_fwd = phylogenetic_sweep(nf; seed=100)

# Backward sweep (reverse κ grid)
result_bwd = phylogenetic_sweep(nf; seed=200)
# Manually reverse κ grid or modify sweep function

# Compare
max_difference = maximum(abs.(result_fwd.a_vals - result_bwd.a_vals))
@assert max_difference < 1e-3 "Hysteresis detected!"
```

## References

- **Task specification:** See the comprehensive task document provided
- **Normal form theory:** Center manifold reduction for pitchfork bifurcations
- **Logistic map analogy:** Feigenbaum diagrams (period-doubling cascade)
- **BeliefSim paper:** Main manuscript on OU-with-resets dynamics

## Support

For questions or issues:
1. Check this guide and the main README
2. Examine example scripts for usage patterns
3. Open a GitHub issue with reproducible example

---

**Last updated:** 2025-12-04
