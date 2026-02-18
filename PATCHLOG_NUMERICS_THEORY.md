# PATCHLOG_NUMERICS_THEORY.md

## Summary
This patch aligns the ensemble numerics and plotting pipeline with the Z2-symmetric pitchfork theory. The main changes:
- Introduced **Route A (empirical κ\*_A)** based on growth of the symmetry-aware order parameter.
- Introduced **Route B (theoretical κ\*_B)** using a rank‑one susceptibility calculation for the OU‑PR model.
- Switched primary order parameters to **M\* = E|m(t)|** and **aligned mean** (branch‑conditioned), avoiding symmetry cancellation.
- Exported symmetry‑aware densities (mixture/plus/minus/aligned) and moment diagnostics.
- Removed hard‑coded κ\* from plotting scripts; figures now read `outputs/threshold/metadata.json`.

## Truth Map (Model Contract)
**Simulation model (OU‑PR, sign‑preserving contraction)**
- SDE: `dU = (-λ U + κ m) dt + σ dW`, where `m = E[U]` (empirical mean in finite N).
- Hazard: `ν(u)` depends on `|u|` (e.g., `StepHazard(ν0)`), symmetric in sign.
- Reset map: `U -> c0 * U` (sign‑preserving contraction).
- Z2 symmetry: invariance under `u -> -u`, so the mixture mean can remain ~0 above κ\*.

**Legacy theoretical code (mismatch)**
- `src/OUResets.jl` contains a spectral operator using drift `-(λ-κ)x` and a reset mapping that does **not** match the OU‑PR mean‑field model. This is retained as **legacy** and is no longer used by the ensemble pipeline.

## What Changed
### Route A (empirical κ\*_A)
- Implemented growth‑rate scan using `M_abs(t) = E|m_i(t)|`.
- Bootstrapped slope estimates and CI; zero‑crossing defines κ\*_A.
- Outputs: `outputs/threshold/growth_scan.csv`, `kappa_star_A.json`.

### Route B (theoretical κ\*_B)
- Constructed the linear operator A0 for OU‑PR with correct reset gain term `(1/c0) ν(x/c0) ρ(x/c0)`.
- Computed odd‑mode susceptibility `Φ(0)` on an odd subspace; κ\*_B = 1/Φ(0).
- Outputs: `outputs/threshold/kappa_star_B.json`, `susceptibility.csv`, `metadata.json`.

### Symmetry‑Aware Observables
- Exported **mean_signed**, **mean_abs**, **mean_rms**, **mean_aligned**, plus CI bands.
- Branch signs computed from late‑time mean; undecided runs excluded from aligned statistics.
- New exports:
  - `outputs/ensemble_results/ensemble_trajectories.csv`
  - `outputs/ensemble_results/terminal_means.csv`
  - `outputs/ensemble_results/density_snapshots.csv` (mixture/plus/minus/aligned)
  - `outputs/ensemble_results/density_moments.csv`

### Figures and Metadata
- `scripts/generate_bifurcation_figures.jl` loads κ\*_A/κ\*_B from `outputs/threshold/metadata.json`.
- Bifurcation and scaling plots now use **M\*** rather than variance‑based amplitude.
- Added A vs B agreement panel in summary figure.

## Files Touched
- `scripts/fig6_ensemble_enhanced.jl`
- `scripts/ensemble_utils.jl`
- `scripts/generate_bifurcation_figures.jl`
- `README.md`
- `manuscript_snippets/figure_captions.tex`
- `manuscript_snippets/results_text.tex`
- `manuscript_snippets/bifurcation_validation_table.tex`

## Reproduction Checklist
1. Run the ensemble pipeline (generates metadata and CSVs):
   ```bash
   julia --project=. scripts/fig6_ensemble_enhanced.jl
   ```
2. Generate publication figures (reads outputs/threshold + sweep CSVs):
   ```bash
   julia --project=. scripts/generate_bifurcation_figures.jl
   ```

## QA Checks
- **Symmetry cancellation visible**: E[m(t)] ~ 0 above κ\*, while E|m(t)| > 0.
- **Density alignment**: mixture density symmetric; aligned density reveals branch.
- **No hard‑coded κ\***: figures use `outputs/threshold/metadata.json`.
- **Outputs present**:
  - `outputs/threshold/metadata.json`
  - `outputs/ensemble_results/density_snapshots.csv`
  - `outputs/ensemble_results/ensemble_trajectories.csv`

