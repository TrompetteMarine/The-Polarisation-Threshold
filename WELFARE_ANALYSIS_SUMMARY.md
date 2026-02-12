# Bifurcation-Based Welfare Analysis - Implementation Summary

## Overview

This document summarizes the implementation of a rigorous bifurcation-theoretic framework for welfare analysis in the Polarisation Threshold model, as specified in the `welfare_simulation_update` requirements.

## What Was Implemented

### 1. Core Spectral Helper Functions (in `src/OUResets.jl`)

Four new exported functions added to the `OUResets` module:

#### `compute_stationary_variance(theta, c0, params; kwargs...)`
- **Purpose**: Solves variance balance equation 2λV* = σ² + Λ_reset(V*; θ, c₀)
- **Implementation**: Simulation-based via modified params at κ=0
- **Returns**: Stationary dispersion V*(θ, c₀) or NaN if fails
- **Theory**: Section 2, equation (2.9)

#### `compute_lambda1_and_derivative(V, params; δκ=0.01, kwargs...)`
- **Purpose**: Computes spectral quantities λ₁₀(V) and λ̇₁₀(V)
- **Implementation**: Finite differences using existing `leading_odd_eigenvalue`
- **Returns**: `(lambda10, lambda1_dot)` tuple
- **Theory**: Section 4, local expansion λ₁(κ; V) ≈ λ₁₀(V) + λ̇₁₀(V)·κ

#### `compute_b_cubic(V, params; b_default=0.5)`
- **Purpose**: Extracts cubic coefficient b(V) from normal form
- **Implementation**: Currently uses calibrated default (extensible to Dirichlet-form)
- **Returns**: Cubic coefficient b(V) > 0
- **Theory**: Section 5, centre-manifold reduction

#### `welfare_loss(V, params; nquad=50, kwargs...)`
- **Purpose**: Computes welfare loss L(V) from Theorem 4.8
- **Implementation**: Numerical integration (trapezoidal rule)
- **Formula**: L(V) = ∫[(κ-κ*)/κ*]·√[α(κ-κ*)/b]·f(κ) dκ over [κ*, 2κ*]
- **Returns**: Welfare loss L(V) or NaN
- **Theory**: Theorem 4.8, assumes κ ~ Uniform[0, 2κ*] conditional on V

### 2. Main Welfare Contours Script (`scripts/run_welfare_contours.jl`)

Complete pipeline implementing the specification:

**Features:**
- Computes 2D welfare surfaces L(θ, c₀) on configurable grid (default 25×25)
- Calculates V*, λ₁₀, λ̇₁₀, κ*, and L(V) at each grid point
- Finds optimal policies for decentralised (β=0) and social planner (β>0)
- Generates publication-grade 3-panel contour plot
- Saves complete CSV with all spectral quantities
- High-resolution output (300 DPI) in both PDF and PNG

**Configuration constants:**
```julia
N_THETA = 25              # θ grid points
N_C0 = 25                 # c₀ grid points
THETA_MIN/MAX = 0.5/1.2   # Tolerance range
C0_MIN/MAX = 0.2/0.8      # Contraction range
BETA_DEC = 0.0            # Decentralised externality
BETA_SOC = 0.3            # Social planner externality
N_AGENTS = 10_000         # Simulation size
L_SPECTRAL = 5.0          # Spectral domain
M_SPECTRAL = 251          # FP grid points
```

**Outputs:**
- `figs/fig_welfare_contours.pdf` (and .png): 3-panel contour plot
- `outputs/welfare_contours_data.csv`: Complete grid data

### 3. Test Scripts

#### `scripts/test_welfare_functions.jl`
Comprehensive diagnostics testing all four core functions:
- Tests single point computation
- Sweeps L(V) over range of dispersion levels
- Verifies all spectral quantities are finite and sensible

**Results:**
```
✓ compute_stationary_variance: V* = 0.838
✓ compute_lambda1_and_derivative: λ₁₀ = -1.30, λ̇₁₀ = 2.00 → κ* ≈ 0.65
✓ compute_b_cubic: b = 0.5
✓ welfare_loss: L(V) = 0.323
```

#### `scripts/test_welfare_quick.jl`
Quick validation on single grid point (< 1 minute runtime)

### 4. Documentation (`README.md`)

Added comprehensive section "Bifurcation-based welfare contours" including:
- Theory overview and connection to paper sections
- Function signatures with mathematical formulas
- Usage examples
- Configuration guide
- Comparison with simulation-based welfare approach

## Technical Details

### Mathematical Framework

The welfare loss L(V) captures the cost of polarisation arising from social coupling κ exceeding the bifurcation threshold κ*(V):

1. **Threshold**: κ*(V) = -λ₁₀(V) / λ̇₁₀(V) from spectral analysis
2. **Growth rate**: α(V) = λ̇₁₀(V) (spectral slope)
3. **Cubic coefficient**: b(V) from normal form
4. **Amplitude**: a(κ, V) ≈ √[α(V)(κ - κ*)/b(V)] for κ > κ*
5. **Distribution**: f(κ | V) = 1/(2κ*) for κ ∈ [0, 2κ*]

The welfare integral combines:
- **(κ - κ*)/κ***: relative excess coupling
- **√[α(κ - κ*)/b]**: polarisation amplitude
- **f(κ)**: coupling distribution weight

### Key Design Decisions

1. **Simulation-based V* computation**: More robust than root-finding for complex reset dynamics
2. **Finite differences for λ̇₁₀**: Simpler than analytical derivatives, sufficient accuracy with δκ=0.01
3. **Trapezoidal integration**: Simple, stable quadrature for smooth integrand
4. **NaN masking**: Failed computations return NaN for clean contour plotting
5. **Collected levels**: `collect(range(...))` for Plots.jl compatibility

### Bug Fixes Applied

**Issue**: Plots.jl error "objects of type Vector{Float64} are not callable"

**Root cause**: Incompatibility between how `levels` and `clims` were passed

**Solution**:
1. Collect ranges to explicit arrays: `levels = collect(range(...))`
2. Make clims explicit tuples: `clims=(clim[1], clim[2])`
3. Add `colorbar=false` to overlay contours to prevent conflicts

## Usage Examples

### Run full welfare analysis (25×25 grid, ~10-15 minutes):
```bash
julia --project=. scripts/run_welfare_contours.jl
```

### Test implementation (< 2 minutes):
```bash
julia --project=. scripts/test_welfare_functions.jl
```

### Quick validation (< 1 minute):
```bash
julia --project=. scripts/test_welfare_quick.jl
```

### Compute welfare at specific V:
```julia
using BeliefSim.OUResets, BeliefSim.Types

p = Params(λ=0.65, σ=1.15, Θ=0.87, c0=0.50, hazard=StepHazard(0.5))
V = 0.8
L = welfare_loss(V, p)
println("L($V) = $L")
```

## Differences from Simulation-Based Welfare

The existing `compute_welfare_curves` in `src/OUResets.jl` uses a simpler approach:
- **Simulation-based**: W_dec = benefit - αV - amp²
- **Phenomenological**: Ad-hoc cost terms (dispersion + polarisation²)
- **No threshold awareness**: Doesn't explicitly model κ*(V)

The new bifurcation-based approach:
- **Theory-driven**: Derived from spectral structure and normal form
- **Threshold-aware**: Explicitly incorporates κ*(V) and distribution
- **Micro-founded**: Links individual dynamics to aggregate welfare through eigenvalues

Both approaches are valid for different purposes:
- Use **simulation-based** for quick exploratory analysis and intuition
- Use **bifurcation-based** for rigorous theoretical results and publication

## Future Extensions

### Already Implemented
- ✓ Core spectral helpers
- ✓ Welfare loss L(V)
- ✓ 2D surface computation
- ✓ Publication-grade contour plots
- ✓ CSV data export
- ✓ Comprehensive testing
- ✓ Documentation

### Possible Refinements
- [ ] Analytical b(V) via Dirichlet-form (currently uses calibrated default)
- [ ] Parallel grid computation for faster surface generation
- [ ] Interpolation-based surface smoothing
- [ ] Optimal policy optimization via Optim.jl (currently uses grid search)
- [ ] Extended sensitivity analysis over (α, β, λ, σ) parameter space
- [ ] Interactive visualization with Makie.jl

## Files Modified/Created

### Modified:
- `src/OUResets.jl`: Added 4 new functions (280 lines)
- `README.md`: Added bifurcation-based welfare section

### Created:
- `scripts/run_welfare_contours.jl`: Main analysis script (430 lines)
- `scripts/test_welfare_functions.jl`: Test suite (100 lines)
- `scripts/test_welfare_quick.jl`: Quick validation (60 lines)
- `scripts/test_plotting_fix.jl`: Plotting diagnostic (50 lines)
- `WELFARE_ANALYSIS_SUMMARY.md`: This document

## Verification

All tests pass successfully:
```bash
✓ Module compiles without errors
✓ All four functions return finite values
✓ Spectral quantities match expected ranges
✓ Welfare loss is positive and sensible
✓ Plotting works with fixed syntax
```

## Support

For questions or issues:
- Check `scripts/test_welfare_functions.jl` for diagnostic output
- Review theory in paper Sections 2-6 and Theorem 4.8
- Consult README.md section "Bifurcation-based welfare contours"
- Inspect CSV output for detailed grid data

## Citation

When using this welfare analysis framework, cite:
> Bontemps Gabriel, *The Polarisation Threshold*, 2025.
> Bifurcation-based welfare analysis implementation available at
> https://github.com/TrompetteMarine/The-Polarisation-Threshold
