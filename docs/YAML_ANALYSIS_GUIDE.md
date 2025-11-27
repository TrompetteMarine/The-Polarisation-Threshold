# YAML Analysis Guide

## Quick Start
```bash
# Single config
julia --project=. scripts/analyze_from_yaml.jl configs/example_sweep.yaml

# All configs
julia --project=. scripts/analyze_from_yaml.jl --all
```

## Output Structure
```
outputs/
└── comprehensive_analysis_[config_name]/
    ├── 01_bifurcation.png          # Main bifurcation diagram
    ├── 02_phase_portraits.png      # Vector fields
    ├── 03_basins.png               # Initial condition outcomes
    ├── 04_timeseries_ratio_*.png   # Time evolution at selected κ/κ*
    ├── 05_return_maps.png          # Recurrence analysis
    ├── 06_parameter_scan.png       # Equilibrium branches
    ├── 07_lyapunov.png             # Chaos indicator
    └── summary.txt                 # Analysis summary
```

## Dependencies

All dependencies are in `Project.toml`:
- `YAML.jl` - Config parsing
- `CairoMakie.jl` - Plotting
- `DifferentialEquations.jl` - ODE solving
- `FFTW.jl` - Spectral analysis
- No `BifurcationKit.jl` dependency – the YAML analysis runs on the in-repo normal-form tools.

## Script Features

✅ **Self-contained** - No external file dependencies  
✅ **Robust error handling** - Continues on plot failures  
✅ **Progress logging** - Clear status messages  
✅ **Auto YAML installation** - Installs if missing  
✅ **Parameter validation** - Checks config integrity  
✅ **Batch processing** - Handle multiple configs  
✅ **Summary generation** - Complete analysis report

## Parameters Used

The script extracts from your YAML:
- `λ` (lambda) - Mean reversion rate
- `σ` (sigma) - Noise strength
- `Θ` (theta) - Reset threshold
- `c₀` - Reset contraction
- `hazard` - State-dependent Poisson reset specification (step or logistic)
- `T` - Simulation horizon (used for plots 4 & 7)
- `dt` - Time step (used for time-series analysis)
- `sweep` - κ range and resolution. Supports either absolute bounds (`kappa_from`, `kappa_to`) or
  ratio-based bounds (`kappa_from_factor_of_kstar`, `kappa_to_factor_of_kstar`). If omitted the script
  spans roughly `[0.4, 2.5] × κ*` to highlight the pitchfork transition.

And calibrates the reduced normal form by Monte Carlo simulation of the full Poisson-reset model:
- `V*` — stationary dispersion at κ = 0 (estimated via Euler–Maruyama with state-dependent Poisson resets).
- `g` — odd-mode decay rate at κ = 0 (estimated by injecting a small antisymmetric perturbation).
- `β = (2λV*/σ²)·κ*` — cubic saturation chosen so the polarized branch obeys `|u₁-u₂|/2 ≈ √((κ-κ*)/κ*)`.
- `κ*` — spectral threshold defined by λ₁(κ*) = 0 where λ₁ is the leading odd eigenvalue of the linearised generator. A shortcut `κ*_{OU} ≈ g σ² / (2 λ V*)` serves only as a coarse OU benchmark and should not be used as general theory.

## Troubleshooting

**"Config file not found"**
→ Check path is correct relative to project root

**"CairoMakie not available"**
→ Run: `using Pkg; Pkg.add("CairoMakie")`

**Plot generation fails**
→ Check logs, script continues to next plot; calibration values are still written to `summary.txt`

**No YAML files found**
→ Ensure files end in `.yaml` or `.yml`