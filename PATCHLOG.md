# PATCHLOG.md

## OU Reset Numerics Consistency Patch

### Summary
This patch aligns the numerical outputs and figures with the Z2-symmetric pitchfork theory. It removes misleading mean plotting, adds symmetry-aware order parameters and branch diagnostics, exports density moments with sanity checks, and updates documentation for reproducibility and interpretation.

### Key changes
1. **Symmetry-aware branch classification**
   - Added late-time branch classification with an adaptive threshold based on the median standard error.
   - Exports `decided_flag` for each run and overall branch shares per scenario.

2. **Order parameter exports**
   - Added `mean_aligned(t)` (decided runs only) with CI bounds to `ensemble_trajectories.csv`.
   - Retained `mean_signed`, `mean_abs`, and `mean_rms` with CI bounds.

3. **Density moments + sanity checks**
   - New `density_moments.csv` export with mixture/aligned integrals and moments.
   - Fail-fast checks:
     - Density integrals close to 1.
     - Mixture mean near 0 below/critical.
     - Aligned mean positive at late times above kappa* (warns if not).

4. **Figure semantics fixed**
   - `plot_density_evolution()` now separates `E|m(t)|` (nonnegative axis) from mixture mean `E[m(t)]`.
   - Adds annotation of `mu_mix` and `mu_aligned` at late time for the above scenario.

5. **Documentation**
   - README updated with new outputs and sanity checks.

### Files changed
- `scripts/fig6_ensemble_enhanced.jl`
- `scripts/generate_bifurcation_figures.jl`
- `README.md`
- `PATCHLOG.md`

### QA checklist
- [ ] Run: `julia --project=. scripts/fig6_ensemble_enhanced.jl`
- [ ] Confirm outputs:
  - `outputs/ensemble_results/ensemble_trajectories.csv`
  - `outputs/ensemble_results/terminal_means.csv`
  - `outputs/ensemble_results/density_snapshots.csv`
  - `outputs/ensemble_results/density_moments.csv`
  - `outputs/ensemble_results/metadata.json`
- [ ] Run: `julia --project=. scripts/generate_bifurcation_figures.jl`
- [ ] Confirm figures:
  - `figs/fig_density_evolution.pdf` (E|m(t)| nonnegative; E[m(t)] separate)
  - `figs/fig6_ensemble_enhanced.pdf`

