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
    ├── 04_timeseries_k*.png        # Time evolution (multiple)
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
- `T` - Simulation time (for plots 4, 7)
- `dt` - Time step (for plot 4)
- `sweep` - κ range and resolution

And computes:
- `β = σ/λ` - Normalized noise
- `κ* = λ` - Critical coupling (approximation)

## Troubleshooting

**"Config file not found"**
→ Check path is correct relative to project root

**"CairoMakie not available"**
→ Run: `using Pkg; Pkg.add("CairoMakie")`

**Plot generation fails**
→ Check logs, script continues to next plot

**No YAML files found**
→ Ensure files end in `.yaml` or `.yml`