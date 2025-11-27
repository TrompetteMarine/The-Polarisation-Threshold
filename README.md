# The Polarisation Threshold — Replication Package for BeliefSim.jl

This repository accompanies the manuscript on the polarisation threshold in Ornstein–Uhlenbeck (OU) belief dynamics with stochastic resets. It provides source code, scripts, and documentation to reproduce every quantitative result and figure reported in the paper, along with diagnostic tools for extended exploration. The codebase is centred on the `BeliefSim.jl` Julia package, which implements Monte Carlo simulators, statistical estimators, and reduced-form bifurcation analyses for the OU-with-resets model.

> **Core theoretical identities.** The stationary dispersion \(V^*\) solves \(2\lambda V^*=\sigma^2+\Lambda_{\text{reset}}(V^*)\). The leading eigen-slope at the origin is \(\lambda'_1(0)=2\lambda V^*/\sigma^2\). The critical coupling predicted by the mean-field theory is \(\kappa^* = g\,\sigma^2/(2\lambda V^*)\); the short-cut \(g \approx \lambda\) yields the practical estimate \(\kappa^* \approx \sigma^2/(2V^*)\).

---

## Table of contents

1. [How to cite](#how-to-cite)
2. [Repository layout](#repository-layout)
3. [Software requirements](#software-requirements)
4. [Environment setup](#environment-setup)
5. [Verifying the installation](#verifying-the-installation)
6. [Reproducing the main results](#reproducing-the-main-results)
7. [YAML configuration reference](#yaml-configuration-reference)
8. [Output structure](#output-structure)
9. [Extended analyses and optional tools](#extended-analyses-and-optional-tools)
10. [Testing and quality assurance](#testing-and-quality-assurance)
11. [Troubleshooting](#troubleshooting)
12. [Support, contributions, and licence](#support-contributions-and-licence)

---

## How to cite

> Bontemps Gabriel, *The Polarisation Threshold*, 2025. Replication materials available at `https://github.com/TrompetteMarine/The-Polarisation-Threshold`.

---

## Repository layout

| Path | Description |
| ---- | ----------- |
| `src/` | `BeliefSim.jl` source code: stochastic simulator (`Model.jl`), hazard kernels (`Hazard.jl`), parameter types (`Types.jl`), summary statistics (`Stats.jl`), plotting utilities (`Plotting.jl`), and reduced bifurcation modules (`bifurcation/`). |
| `scripts/` | Command-line entry points for simulations, YAML-driven analyses, comprehensive diagnostic figure generation, and environment setup. |
| `configs/` | Ready-to-use YAML configuration files (e.g. `example_sweep.yaml`) for scripted replications. |
| `examples/` | Minimal Julia scripts demonstrating how to estimate \(V^*\), sweep \(\kappa\), and explore network coupling. |
| `docs/` | Supplemental documentation: getting started, YAML analysis guide, troubleshooting notes for optional dependencies. |
| `outputs/` | Default location for generated tables, figures, and logs (created automatically by the scripts). |
| `env-nods/` | Alternative environment specification for systems without DynamicalSystems dependencies. |
| `test/` | Unit tests for `BeliefSim.jl` and optional bifurcation checks. |
| `Makefile` | Convenience targets for instantiation (`make instantiate`), testing (`make test`), and linting hooks. |

---

## Software requirements

- **Julia:** 1.8, 1.9, 1.10, or 1.12 (LTS recommended: 1.10).
- **Operating system:** Linux, macOS, or Windows with a working Julia installation.
- **Git:** for cloning this repository.
- **Optional packages:** CairoMakie (vector graphics), BifurcationKit (advanced continuation), Attractors/DynamicalSystems (basin sampling). The setup script installs these when available, and all core replications run without them.

---

## Environment setup

### Automated bootstrap (recommended)

```bash
# Clone the repository
git clone https://github.com/TrompetteMarine/The-Polarisation-Threshold.git
cd The-Polarisation-Threshold

# Install and precompile all dependencies, optional tools, and output directories
./scripts/setup_environment.sh
```

The script cleans previous manifests, instantiates the environment, attempts to add CairoMakie and the latest BifurcationKit, precompiles, creates `figs/` and `outputs/`, and verifies that `BeliefSim` loads.

### Manual installation

If you prefer explicit control over package resolution:

```julia
julia --project=. -e '
using Pkg
Pkg.instantiate()
Pkg.precompile()
'
```

To include the optional dependencies used by the extended analysis scripts:

```julia
julia --project=. -e '
using Pkg
Pkg.add("CairoMakie")
Pkg.add("Attractors")
Pkg.add("DynamicalSystemsBase")
Pkg.add(url="https://github.com/bifurcationkit/BifurcationKit.jl.git")
Pkg.precompile()
'
```

---

## Verifying the installation

1. Start a Julia REPL with the project activated:
   ```julia
   julia --project=.
   ```
2. Load the package and run the bundled smoke test:
   ```julia
   using BeliefSim
   include("examples/quickstart.jl")
   ```
   This script draws \(V^*\) and \(\kappa^*\) using the default parameters and reports the values in the console. A successful run confirms that core dependencies are compiled and functional. For a CLI check, run:
   ```bash
   julia --project=. scripts/run_sim.jl --mode vstar --seed 2025
   ```

---

## Reproducing the main results

All commands below assume the repository root as the working directory and that the project environment is active (`julia --project=.`).

### 1. Minimal replication in the REPL

```julia
using BeliefSim
using BeliefSim.Types, BeliefSim.Stats, BeliefSim.Plotting

p = Params(λ=1.0, σ=0.8, Θ=2.0, c0=0.5, hazard=StepHazard(0.5))
Vstar = estimate_Vstar(p; N=20_000, T=300.0, dt=0.01, burn_in=100.0, seed=123)
κstar = critical_kappa(p; Vstar=Vstar)              # canonical shortcut g ≈ λ
κgrid = collect(range(0.0, 2.0*κstar, length=21))
res = sweep_kappa(p, κgrid; N=20_000, T=300.0, dt=0.01, burn_in=100.0, seed=42)
plt = plot_bifurcation(res; κstar=κstar, title="Pitchfork")
savefig(plt, "outputs/bifurcation.png")
```

This reproduces the near-threshold pitchfork diagram reported in the manuscript.

### 2. Command-line interface (no custom coding)

`scripts/run_sim.jl` estimates \(V^*\) and \(\kappa^*\) or sweeps a \(\kappa\) grid directly from the shell:

```bash
# Dispersion and threshold estimates
julia --project=. scripts/run_sim.jl --mode vstar --lambda 1.0 --sigma 0.8 --theta 2.0 --c0 0.5 --nu0 0.5 --seed 2025

# Sweep κ with tabulated output
julia --project=. scripts/run_sim.jl --mode sweep --lambda 1.0 --sigma 0.8 --theta 2.0 --c0 0.5 --nu0 0.5 --seed 2025
```

### 3. Batch replication from YAML configuration

`configs/example_sweep.yaml` stores the parameters used in the paper. Run:

```bash
julia --project=. scripts/run_from_yaml.jl configs/example_sweep.yaml
```

The script calibrates \(V^*\), computes \(\kappa^*\), executes the \(\kappa\) sweep, fits the cubic coefficient, and saves plots and tables under `outputs/example_pitchfork/`.

### 4. Full diagnostic replication (comprehensive figure set)

For the complete panel of figures (bifurcation diagrams, phase portraits, basins, time-series diagnostics, Lyapunov indicators), use the YAML-driven analysis pipeline:

```bash
julia --project=. scripts/analyze_from_yaml.jl configs/example_sweep.yaml
```

Pass `--all` to process every YAML file in `configs/`. The script manages optional dependencies automatically, patches known upstream issues when needed, and logs its progress. The resulting directory contains the numbered figures and a `summary.txt` detailing the calibrated quantities.

### 5. Deterministic reduced-model figures

The `scripts/comprehensive_analysis.jl` harnesses the reduced normal-form surrogate to generate detailed vector fields, nullclines, limit cycles, and continuation diagnostics:

```bash
julia --project=. scripts/comprehensive_analysis.jl
```

A CairoMakie backend is required; the script exits early with guidance if unavailable.

### 6. Reproducing specialised figures

- `scripts/make_phase_portraits.jl --kappas 0.8,1.0,1.2 --lims -3,3` generates publication-quality phase portraits.
- `scripts/scan_hopf_and_cycles.jl` and `scripts/scan_homoclinic.jl` scan for Hopf and homoclinic bifurcations, respectively. Adjust command-line flags in each script (see the header comments) to match the scenarios discussed in the manuscript.

---

## YAML configuration reference

YAML files provide reproducible parameter sets for the stochastic simulations and reduced analyses. The main fields in `configs/example_sweep.yaml` are:

```yaml
name: example_pitchfork          # Output subdirectory name
seed: 2025                       # RNG seed for reproducibility
N: 20000                         # Population size
T: 300.0                         # Simulation horizon
dt: 0.01                         # Euler–Maruyama time step
burn_in: 100.0                   # Warm-up horizon before statistics
params:
  lambda: 1.0                    # Mean reversion λ
  sigma: 0.8                     # Noise σ
  theta: 2.0                     # Reset threshold Θ
  c0: 0.5                        # Reset contraction c₀
  hazard:
    kind: step                   # Reset hazard (step or logistic)
    nu0: 0.5                     # Parameter for the chosen hazard
sweep:
  kappa_from: 0.0                # Lower bound of κ grid
  kappa_to_factor_of_kstar: 2.0  # Upper bound expressed as a multiple of κ*
  points: 21                     # Number of κ grid points
output_dir: outputs/example_pitchfork
```

Additional keys accepted by `scripts/analyze_from_yaml.jl` include explicit `kappa_to` bounds, plotting options, and toggles for optional analyses. See `docs/YAML_ANALYSIS_GUIDE.md` for a comprehensive field-by-field explanation.

---

## Output structure

Scripts write self-describing artefacts to subfolders of `outputs/`. For the YAML analysis pipeline, the directory tree is:

```
outputs/
└── comprehensive_analysis_<config_name>/
    ├── 01_bifurcation.png          # Main bifurcation diagram
    ├── 02_phase_portraits.png      # Vector fields across κ
    ├── 03_basins.png               # Basin-of-attraction tiling
    ├── 04_timeseries_ratio_*.png   # Time evolution at selected κ/κ*
    ├── 05_return_maps.png          # Recurrence diagnostics
    ├── 06_parameter_scan.png       # Equilibrium continuation
    ├── 07_lyapunov.png             # Chaos indicator
    └── summary.txt                 # Calibrated V*, κ*, cubic fit, notes
```

`summary.txt` records all numerical outputs (dispersion, threshold, cubic coefficient, fitting window) for archival purposes. CSV exports of sweep data (κ grid, amplitudes, variances) are provided alongside the figures when `scripts/run_from_yaml.jl` is used.

---

## Extended analyses and optional tools

- **Network coupling experiments:** `examples/network_demo.jl` shows how to couple agents on Graphs.jl networks via `src/Network.jl`.
- **Custom plotting:** `scripts/plot_bifurcation.jl` converts stored CSV results into publication-ready figures using CairoMakie.
- **BifurcationKit integration:** The project ships with lightweight continuation routines in `src/bifurcation/`. When BifurcationKit is available, additional continuation scripts (`scripts/scan_hopf_and_cycles_bifkit.jl`, `scripts/scan_homoclinic_BifKIt.jl`) can be run for cross-validation against the internal toolkit.

All optional paths degrade gracefully: the scripts detect missing packages, provide installation hints, and fall back to the built-in solvers or plotting backends.

---

## Testing and quality assurance

Run the automated test suite before submitting modified replication materials:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

The suite validates module loading, parameter construction, variance and threshold estimators, and κ sweeps (`test/runtests.jl`). When BifurcationKit is installed the bifurcation-specific regression tests in `test/test_bifurcation.jl` are executed automatically.

Continuous integration is not bundled with this repository; please report any test failures together with your Julia version and platform.

---

## Troubleshooting

Consult the documents in `docs/` for detailed remediation steps:

- `docs/getting_started.md` — minimal REPL instructions.
- `docs/YAML_ANALYSIS_GUIDE.md` — CLI usage, dependency list, and output descriptions for the YAML pipeline.
- `docs/troubleshooting.md` — resolutions for common optional dependency failures (BifurcationKit on Julia 1.8, Attractors/DynamicalSystems installation, Makie attribute changes, `subscript` constant errors). The YAML analysis script applies a safe patch for known upstream issues automatically and prints guidance when manual intervention is required.

---

## Support, contributions, and licence

- **Issues and questions:** Open a GitHub issue with a short description, your Julia version, operating system, and relevant log excerpts. For reproducibility, attach the YAML file or script used to generate the failing result.
- **Contributions:** Please fork the repository, add tests for new features, and ensure `Pkg.test()` passes before opening a pull request. Contributions that improve documentation, reproducibility, or numerical robustness are especially welcome.
- **Licence:** MIT. All code, scripts, and documentation in this repository are released under the MIT Licence.

---

Happy replicating!

## Reproducing cherry picked-requested figures from the paper

Each script saves a PDF under `figs/` (created automatically). Run them from the
repository root with the project activated:

```bash
julia --project=. scripts/fig1_ou_resets.jl
julia --project=. scripts/fig2_eigen_kappa.jl
julia --project=. scripts/fig3_bifurcation.jl
julia --project=. scripts/fig4_welfare.jl
```

- **Figure 1:** OU sample path with reset markers and stationary density with/without resets.
- **Figure 2:** Leading odd eigenvalue \(\lambda_1(\kappa)\) highlighting the zero crossing \(\kappa^*\).
- **Figure 3:** Pitchfork bifurcation of the order parameter with symmetric branches from biased simulations.
- **Figure 4:** Welfare comparison between decentralised and planner solutions, marking \(\kappa^{\text{dec}}\) and \(\kappa^{\text{soc}}\).


