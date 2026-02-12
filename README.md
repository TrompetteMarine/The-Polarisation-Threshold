# The Polarisation Threshold — Replication Package for BeliefSim.jl

This repository accompanies the manuscript on the polarisation threshold in Ornstein–Uhlenbeck (OU) belief dynamics with stochastic resets. It provides source code, scripts, and documentation to reproduce every quantitative result and figure reported in the paper, along with diagnostic tools for extended exploration. The codebase is centred on the `BeliefSim.jl` Julia package, which implements Monte Carlo simulators, statistical estimators, and reduced-form bifurcation analyses for the OU-with-resets model.

> **Core theoretical anchors.** The stationary dispersion \(V^*\) solves \(2\lambda V^*=\sigma^2+\Lambda_{\text{reset}}(V^*)\). The critical coupling \(\kappa^*\) is defined spectrally as the unique value where the leading odd eigenvalue of the linearised generator crosses zero. A crude OU-style shortcut \(\kappa^*_{OU} \approx g\,\sigma^2/(2\lambda V^*)\) can be used for back-of-the-envelope intuition, but it is not valid in general once resets and state-dependent hazards matter.

---

## Table of contents

1. [How to cite](#how-to-cite)
2. [Repository layout](#repository-layout)
3. [Software requirements](#software-requirements)
4. [Environment setup](#environment-setup)
5. [Verifying the installation](#verifying-the-installation)
6. [Reproducing the main results](#reproducing-the-main-results)
7. [Model simulated in numerics](#model-simulated-in-numerics)
8. [Interpreting symmetry](#interpreting-symmetry)
9. [Reproducing figures](#reproducing-figures)
10. [Outputs glossary](#outputs-glossary)
11. [Sanity checks](#sanity-checks)
12. [YAML configuration reference](#yaml-configuration-reference)
13. [Output structure](#output-structure)
14. [Extended analyses and optional tools](#extended-analyses-and-optional-tools)
15. [Testing and quality assurance](#testing-and-quality-assurance)
16. [Troubleshooting](#troubleshooting)
17. [Support, contributions, and licence](#support-contributions-and-licence)

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
| `scripts/generate_bifurcation_figures.jl` | Publication-quality figure generator driven by the ensemble sweep CSVs. |

---

## Software requirements

- **Julia:** 1.8, 1.9, 1.10, or 1.12 (LTS recommended: 1.10).
- **Operating system:** Linux, macOS, or Windows with a working Julia installation.
- **Git:** for cloning this repository.
- **Optional packages:** CairoMakie (vector graphics), LaTeXStrings (math labels), BifurcationKit (advanced continuation), Attractors/DynamicalSystems (basin sampling). The setup script installs these when available, and all core replications run without them.

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

p = Params(λ=0.85, σ=0.8, Θ=2.0, c0=0.85, hazard=StepHazard(10.5))
Vstar = estimate_Vstar(p; N=20_000, T=600.0, dt=0.01, burn_in=100.0, seed=123)
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

- `scripts/make_phase_portraits.jl --kappas 0.8,1.0,1.2 --lims -3,3` generates phase portraits.
- `scripts/scan_hopf_and_cycles.jl` and `scripts/scan_homoclinic.jl` scan for Hopf and homoclinic bifurcations, respectively. Adjust command-line flags in each script (see the header comments) to match the scenarios discussed in the manuscript.

### 7. Reproducing the core Julia figures (Fig 1–6 + Fig A)

The paper integrates self-contained figures generated by the scripts in `scripts/` plus a YAML-driven Fig A. Run each script with the project activated to save the PDFs into `figs/`:

```bash
julia --project=. scripts/fig1_ou_resets.jl
julia --project=. scripts/fig2_eigen_kappa.jl
julia --project=. scripts/fig3_bifurcation.jl
julia --project=. scripts/fig4_welfare.jl
julia --project=. scripts/fig5_phase_kstar.jl
julia --project=. scripts/fig6_density_snapshots.jl
```

Optional (longer runtime): ensemble-validated Figure 6 with confidence intervals, growth-rate tests, and phase diagram.
This run writes `outputs/parameter_sweep/*.csv`, which are consumed by the publication-quality figure generator below.

```bash
julia --project=. scripts/fig6_ensemble_enhanced.jl
```

Publication-quality bifurcation figures (post-processing of the ensemble sweep CSVs):

```bash
julia --project=. scripts/generate_bifurcation_figures.jl
```

Fig A (zoomed pitchfork around κ*) is generated via the YAML analysis pipeline:

```bash
julia --project=. scripts/analyze_from_yaml.jl configs/figA_zoom.yaml
```

Each script sets its own seed for reproducibility and writes the corresponding `figs/fig*.pdf` file consumed by the paper LaTeX includes. The Fig A YAML pipeline writes to `outputs/comprehensive_analysis_figA_zoom_pitchfork/`.

Convenience Makefile targets:
- `make fig5`
- `make fig6` (baseline single-run density snapshots)
- `make figs_abc`

### 8. Extended welfare analysis (Figure 4 supplements)

The welfare analysis includes a comprehensive suite of visualizations beyond the main Figure 4. Three specialized scripts generate various perspectives on the welfare landscape:

#### Main welfare figures (`fig4_welfare.jl`)

```bash
julia --project=. scripts/fig4_welfare.jl
```

Generates the core Figure 4 materials:
- **Main paper figure:** 1D welfare comparison showing κ^dec vs κ^soc (`fig4_welfare.pdf`)
- **Enhanced 3-panel contour:** Decentralised, Planner, and Externality surfaces (`fig4_welfare_enhanced.pdf`)
- **Individual panels:** Separate files for each contour panel (`fig4_panel_decentralised.pdf`, `fig4_panel_planner.pdf`, `fig4_panel_difference.pdf`)
- **Cross-sections:** Welfare slices at different V* levels (`fig4_welfare_crosssections.pdf`, plus individual `fig4_slice_low/mid/high_dispersion.pdf`)
- **Optimal comparison:** Ridge lines κ^dec(V*) vs κ^soc(V*) with shaded externality wedge (`fig4_welfare_optimal_comparison.pdf`)

**Key features:**
- Augmented axis ranges (6-8% padding) for better readability
- Consistent color limits using robust quantile-based clipping
- Power-law level spacing (x^0.6) to emphasize low-welfare regions
- Both combined multi-panel and individual panel outputs for journal flexibility

#### Supplementary welfare plots (`fig4_extra_plots.jl`)

```bash
julia --project=. scripts/fig4_extra_plots.jl
```

Generates supplementary figures for the online appendix and presentations:
- **Externality ridge:** Δκ(V*) wedge showing the optimal policy gap (`fig4_externality_ridge.pdf`)
- **Welfare difference surfaces:** 2-panel ΔW visualization with and without ridge overlays (`fig4_welfare_difference_surface.pdf`, plus individual panels)
- **Complementary slices:** W(V* | κ) cross-sections at fixed coupling (`fig4_welfare_slices_kappa.pdf`, plus individual `fig4_slice_low/mid/high_kappa.pdf`)
- **Pedagogical contours:** Annotated surfaces with "Safe plateau", "Welfare crater", and "Danger zone" labels for teaching (`fig4_pedagogical_contour.pdf`, plus individual panels)
- **Labeled contours:** Inspired by CairoMakie's labeled contour example, showing exact welfare values with nonlinear level spacing (`fig4_labeled_contours.pdf`, plus individual `fig4_panel_labeled_*.pdf`)

**Advanced features:**
- Symmetric logarithmic spacing for difference plots (emphasizes near-zero structure)
- Median-filtered ridge lines to remove argmax discontinuities
- Debug mode (`DEBUG = true`) for surface diagnostics and slice validation

#### Volcano-style surface explorer (`fig4_surface_explorer.jl`)

```bash
julia --project=. scripts/fig4_surface_explorer.jl
```

Generates clean topographic-style visualizations inspired by the volcano contour example:
- **Multiple colorschemes:** terrain (topographic), thermal (heat map), deep (ocean), dense (high-contrast)
- **Clean aesthetics:** Pure filled contours with integrated colorbars (no overlay lines)
- **Individual surfaces:** Decentralised and Planner welfare with various color palettes
- **Combined comparison:** 2×2 grid showing different colorscheme views (`combined_colorschemes.pdf`)

All volcano-style outputs saved to: `figs/volcano/`

**Design philosophy:**
- Emphasizes "landscape" metaphor (welfare as elevation/temperature)
- Higher resolution (80×20 grid) for ultra-smooth contours
- Explicit colorbar legends showing exact welfare-to-color mapping
- Simpler, more intuitive for non-specialist audiences

#### Output organization

All welfare scripts save both PDF and PNG formats for maximum compatibility:

```
figs/
├── fig4_welfare.pdf                           # Main paper (1D)
├── fig4_welfare_enhanced.pdf                  # 3-panel contour (main)
├── fig4_panel_*.pdf                           # Individual panels (all types)
├── fig4_slice_*.pdf                           # Cross-section slices
├── fig4_*_ridge.pdf                           # Ridge/wedge comparisons
├── fig4_labeled_contours.pdf                  # Nonlinear level spacing
├── fig4_pedagogical_contour.pdf               # Annotated for teaching
├── fig5_phase_kstar.pdf                       # κ*(c0, σ) phase diagram
├── fig6_density_snapshots.pdf                 # Density snapshots below/above κ*
├── fig6_ensemble_enhanced.pdf                 # Ensemble-averaged density snapshots + CI
├── fig6_phase_diagram.pdf                     # λ₁ vs κ/κ* phase diagram (ensemble)
├── fig6_observables_comparison.pdf            # Variance/kurtosis/bimodality/overlap
├── fig_bifurcation_diagram.pdf                # Publication-quality |a*| vs κ/κ*
├── fig_loglog_scaling.pdf                     # Log-log scaling test (β)
├── fig_hysteresis_test.pdf                    # Forward/backward sweep comparison
├── fig_variance_ushape.pdf                    # Variance U-shape with V_min
├── fig_density_evolution.pdf                  # Composite density evolution (if data saved)
├── fig_validation_summary.pdf                 # 4-panel summary (talks/posters)
└── volcano/
    ├── decentralised_terrain.pdf              # Volcano-style surfaces
    ├── planner_*.pdf                          # Various colorschemes
    ├── difference_balance.pdf                 # Symmetric difference
    └── combined_colorschemes.pdf              # 2×2 comparison grid
```

Ensemble validation outputs (from `scripts/fig6_ensemble_enhanced.jl`) are written to:
- `outputs/ensemble_results/` (ensemble growth rates, trajectories, observables)
- `outputs/parameter_sweep/` (κ/κ* sweep CSVs such as `equilibrium_sweep.csv`, `parameter_sweep.csv`, and optional `backward_sweep.csv`)
- `docs/` (ensemble validation report + methodology notes)
- `manuscript_snippets/` (LaTeX table + caption snippets)

**Configuration:**
- Set `DEBUG = true` at the top of each script for detailed diagnostics
- Modify grid resolution by editing `build_grids()` function
- Adjust color limits via `finite_clims()` quantile parameters
- Customize power-law exponents in `compute_powerlaw_levels()`

#### Bifurcation-based welfare contours (`run_welfare_contours.jl`)

```bash
julia --project=. scripts/run_welfare_contours.jl
# or, equivalently:
julia --project=. scripts/run_welfare_corrected.jl
```

This script now follows the Section 6 mean-field-control split:
- **Private loss** \(J_{\text{ind}}(θ,c₀) = α_V·V^* + K/θ^2\) (no polarisation term)
- **Planner loss** \(J_{\text{soc}} = J_{\text{ind}} + Φ(V^*)\) where \(Φ\) is the bifurcation externality built from \(\kappa^*(V^*)\), \(λ̇₁₀(V^*)\), and the cubic normal form.

**Core functions (in `src/OUResets.jl`):**

1. **`compute_stationary_variance(theta, c0, params)`** — Monte Carlo estimate of \(V^*(θ,c₀)\).
2. **`compute_lambda1_and_derivative(V, params)`** — leading odd eigenvalue \(λ₁₀\) and slope \(λ̇₁₀\); gives κ*(V).
3. **`compute_b_cubic(V, params)`** — calibrated cubic coefficient \(b(V)\) (placeholder).
4. **`private_cost(V, theta; αV, K_reset)`** — decentralised objective \(J_{\text{ind}}\).
5. **`bifurcation_loss(V, params; ...)`** — externality \(Φ(V) = c_{\text{pol}}E[a^2] + φ_A E[a^4]\) with optional cached eigenvalues and a tracked fallback proxy when the spectral routine fails.
6. **`welfare_loss(V, theta, params; regime=:dec|:soc, ...)`** — thin wrapper that returns \(J_{\text{ind}}\) or \(J_{\text{ind}} + Φ\).

**Generated outputs:**

- **`figs/fig_welfare_contours_corrected.pdf`**: 3-panel contour plot
  - Decentralised welfare L_dec(θ, c₀)
  - Social planner welfare L_soc(θ, c₀) = L_dec + Φ(V*)
  - Externality surface Φ(V*)
  - Overlaid optimal policies marked with stars

- **`outputs/welfare_corrected.csv`**: Complete grid data
  - Columns: theta, c0, V, L_dec, L_soc, Phi, lambda10, lambda1_dot, kappa_star
  - One row per grid point for further analysis in R/Python/Matlab

**Theory:**

The welfare loss captures the cost of polarisation arising from social coupling κ exceeding the bifurcation threshold κ*(V). Unlike the simple simulation-based welfare in `fig4_welfare.jl` (which uses amplitude² and dispersion costs), this approach:

- Derives welfare from the **spectral gap** of the linearised generator
- Incorporates the **normal form bifurcation structure** (pitchfork with cubic coefficient b)
- Explicitly models the **distribution of coupling intensities** conditional on V
- Provides a **rigorous connection** between micro-level dynamics and aggregate welfare

**Testing:**

Verify the implementation with:
```bash
julia --project=. scripts/test_welfare_functions.jl
```

This runs diagnostics on all four core functions and sweeps L(V) over a range of dispersion levels.

**Configuration:**

Edit the following constants at the top of `run_welfare_contours.jl`:
- `αV`, `K_reset`: Private-cost weights
- `c_pol`, `φA`, `κ_ratio_max`: Externality weights and κ-range for Φ
- `N_THETA`, `N_C0`: Grid resolution (default 25×25)
- `THETA_MIN/MAX`, `C0_MIN/MAX`: Parameter ranges
- `N_AGENTS`, `T_SIM`: Simulation budget for variance estimation
- `L_SPECTRAL`, `M_SPECTRAL`: Spectral solver domain and grid points

### 9. Phylogenetic bifurcation diagrams

The repository includes advanced tools for generating "phylogenetic tree" bifurcation diagrams that show the complete attractor structure as κ sweeps through the critical threshold. These diagrams are analogous to the classic logistic map bifurcation diagrams but with the clean supercritical pitchfork structure of this model.

**Main phylogenetic diagram:**
```bash
julia --project=. scripts/fig_phylogenetic_tree.jl
```

This script:
- Calibrates the reduced normal form from micro-level parameters
- Performs a dense parameter sweep (500 κ points, 30 initial conditions each)
- Classifies equilibria as stable or unstable
- Generates a "tuning fork" diagram with theoretical envelope
- Includes an inset verifying the β = 0.5 scaling exponent

**Attractor evolution gallery:**
```bash
julia --project=. scripts/fig_attractor_evolution.jl
```

This generates:
- Phase portrait gallery (2×3 grid) showing flow evolution across κ
- Basin of attraction evolution showing how basins split at the bifurcation
- Trajectory fate comparison (κ < κ* vs κ > κ*)

**Configuration:**

For batch processing or custom parameter sets, use the YAML configuration:
```bash
# Edit configs/phylogenetic_analysis.yaml to customize parameters
julia --project=. scripts/run_phylogenetic_analysis_from_yaml.jl configs/phylogenetic_analysis.yaml
```

The YAML configuration controls:
- Sweep resolution and initial condition sampling
- Adaptive burn-in near the critical point
- Particle system validation (optional)
- Visualization settings and output formats

**Module documentation:**

The core implementation is in `src/bifurcation/PhylogeneticDiagram.jl`, which provides:
- `NormalFormParams`: Structure for the reduced cubic normal form ȧ = μ(κ)·a - b·a³
- `phylogenetic_sweep`: Dense parameter sweep with attractor classification
- `calibrate_normal_form`: Automatic calibration from V* and κ*
- `scaling_exponent`: Verification of the β = 0.5 supercritical scaling

Load the module explicitly when needed:
```julia
include("src/bifurcation/PhylogeneticDiagram.jl")
using .PhylogeneticDiagram
```

**Requirements:**

- CairoMakie is recommended for vector graphics output
- Falls back to Plots.jl if CairoMakie is unavailable
- All core functionality works with the base BeliefSim installation

---

## Model simulated in numerics

The ensemble pipelines (`fig6_ensemble_enhanced.jl` and the downstream figure generator) simulate the **OU-with-Poisson-resets (OU-PR)** model using `StepHazard(nu0)` and partial resets of depth `c0`. This is the numerical counterpart of the OU-with-resets generator used in the theory. Boundary-hitting OU (OU-BR) is not simulated in the default scripts unless explicitly stated.

## Interpreting symmetry

Because the dynamics are Z2-symmetric, above kappa* each run selects a + or - branch. Consequently, the **signed mean** E[m(t)] can remain near zero even when symmetry breaking occurs. We therefore report **E|m(t)|**, **RMS(m(t))**, and the **aligned mean** (sign-flipped per run) as primary order parameters, and we export terminal mean diagnostics to show bimodality. Density plots distinguish the **mixture density** (symmetric by construction) from the **aligned density** (branch-aligned).

## Reproducing figures

Run the ensemble pipeline (produces metadata + CSVs), then the publication figure generator:

```bash
julia --project=. scripts/fig6_ensemble_enhanced.jl
julia --project=. scripts/generate_bifurcation_figures.jl
```

For distributed runs (use all CPUs minus one), start Julia with workers:

```bash
julia --project=. -p $(($(sysctl -n hw.ncpu)-1)) scripts/fig6_ensemble_enhanced.jl
```

For threaded runs instead, use:

```bash
JULIA_NUM_THREADS=$((`sysctl -n hw.ncpu`-1)) julia --project=. scripts/fig6_ensemble_enhanced.jl
```

Outputs land in:
- `outputs/ensemble_results/` (metadata, trajectories, terminal means, densities)
- `outputs/parameter_sweep/` (sweep CSVs; used for phase‑diagram plots)
- `figs/` (publication‑ready PDFs)

## Outputs glossary

Key files written by the ensemble pipeline:

- `outputs/ensemble_results/metadata.json`  
  Parameters, simulation settings, and computed quantities (V*, kappa*, V_baseline, kappa_eff, beta_hat, C_hat).
- `outputs/ensemble_results/ensemble_trajectories.csv`  
  Columns: `mean_signed`, `mean_abs`, `mean_rms`, `mean_aligned` (decided runs only), their CI bounds, plus variance and CI. Also includes branch shares (`decided_share`, `plus_share`, `minus_share`, `undecided_share`) per scenario.
- `outputs/ensemble_results/terminal_means.csv`  
  Columns: `scenario`, `run_id`, `mean_late`, `abs_mean_late`, `branch_sign`, `decided_flag`.
- `outputs/ensemble_results/density_snapshots.csv`  
  Columns: mixture/aligned/plus/minus densities with CI bounds on a common grid.
- `outputs/ensemble_results/density_moments.csv`  
  Columns: `integral_mixture`, `integral_aligned`, `mu_mix`, `mu_aligned`, `var_mix` per scenario and snapshot time (sanity checks).

## Sanity checks

- **Below kappa\***: mean_abs(T) ≈ O(N^{-1/2}) and terminal means unimodal near 0.
- **Above kappa\***: mean_abs(T) > 0 and terminal means bimodal around +/- m*.
- **Density alignment**: mixture density remains symmetric, aligned density reveals the branch.
- **Density moments**: integrals are ~1; for below/critical, mu_mix ≈ 0; above kappa*, mu_aligned > 0 at late times.

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
- **Custom plotting:** `scripts/plot_bifurcation.jl` converts stored CSV results into figures using CairoMakie.
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

## Reproducing the other figures of the paper

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
