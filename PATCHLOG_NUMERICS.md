# Patch Log — Numerics/Theory Docking (NUMERICS_THEORY_DOCKING_R1)

## What was wrong
- Symmetry cancellation: ensemble-signed means and mixture densities can look mean-zero above kappa*, hiding the odd-mode symmetry-breaking instability.
- Plotting scripts used hard-coded kappa*/V* constants, which can diverge from the actual run and break reproducibility.
- Density snapshots were exported only as mixture densities (no aligned/branch-conditioned view).
- No terminal-mean diagnostics were exported to show bimodality across runs.

## What changed
- **Symmetry-aware order parameters**: `ensemble_trajectories.csv` now includes mean_abs and mean_rms (with CI), alongside signed mean.
- **Branch diagnostics**: new `terminal_means.csv` with per-run late-time mean and branch sign.
- **Density exports**: `density_snapshots.csv` now includes mixture, aligned, plus, and minus densities (with CI where possible).
- **Metadata**: `metadata.json` is written with parameters, simulation settings, and computed quantities (V*, kappa*, V_baseline, kappa*_eff, beta_fit, C_fit).
- **Plots**: density panels distinguish mixture vs aligned; mean panel plots E|m(t)| with signed mean overlay; terminal mean histogram added.
- **Pipeline fixes**: plotting scripts load kappa*/V* from metadata.json and fall back only with explicit warnings; sweep CSV loader handles both equilibrium_sweep.csv and parameter_sweep.csv.
- **Docs**: README and statistical methodology updated to explain symmetry and outputs; figure captions updated for aligned density.

## Files touched
- `scripts/fig6_ensemble_enhanced.jl`
- `scripts/statistical_tests.jl`
- `scripts/visualization.jl`
- `scripts/generate_bifurcation_figures.jl`
- `docs/statistical_methodology.md`
- `manuscript_snippets/figure_captions.tex`
- `README.md`

## QA checklist (manual)
- [ ] Run `julia --project=. scripts/fig6_ensemble_enhanced.jl`
- [ ] Verify outputs:
  - `outputs/ensemble_results/metadata.json`
  - `outputs/ensemble_results/ensemble_trajectories.csv` (mean_abs/mean_rms columns present)
  - `outputs/ensemble_results/terminal_means.csv`
  - `outputs/ensemble_results/density_snapshots.csv`
- [ ] Run `julia --project=. scripts/generate_bifurcation_figures.jl`
- [ ] Inspect figures:
  - `figs/fig6_ensemble_enhanced.pdf` shows E|m(t)| and mixture vs aligned densities
  - `figs/fig_density_evolution.pdf` shows mixture vs aligned, plus/minus above kappa*, and terminal mean histogram
- [ ] Sanity checks:
  - Below kappa*: mean_abs(T) ~ 0 and terminal means unimodal
  - Above kappa*: mean_abs(T) > 0 and terminal means bimodal
  - Mixture density symmetric; aligned density reveals branch
