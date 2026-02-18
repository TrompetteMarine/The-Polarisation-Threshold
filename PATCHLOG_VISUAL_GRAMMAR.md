# Patch Log — Unified Plot Grammar

## Overview
This patch introduces a shared plotting grammar for CairoMakie figures and refactors
figure-generating code to follow consistent styling, legend, and colorbar rules.
No simulation logic or numerical outputs were modified.

## New Shared Grammar
- Added `scripts/plot_grammar.jl` with:
  - `apply_plot_grammar!(mk)` for Theme defaults (fonts, grids, frame, linewidths).
  - `style_axis!`, `add_time_colorbar!`, `add_style_legend!`, `time_color`.
- Grammar constants live in `PlotGrammar.GRAMMAR`.

## Refactors
### `scripts/visualization.jl`
- Added `plot_ensemble_dashboard(...)` with:
  - 2×2 density block (κ<κ*, κ≈κ*, κ>κ* aligned, κ>κ* mixture).
  - Shared time colorbar + single style legend (aligned vs mixture).
  - CI bands only for final snapshot.
  - Plus/minus curves only at final snapshot; no legend entries.
  - Trajectories (E|m|, Var, Kurtosis) below density block.
- `plot_ensemble_figure(...)` now delegates to the dashboard for CairoMakie.
- `plot_phase_diagram` and `plot_observables` use the shared grammar.

### `scripts/generate_bifurcation_figures.jl`
- Uses `plot_grammar.jl` helpers and theme.
- `plot_density_evolution`:
  - No per-time legends; time encoded via color + shared colorbar.
  - Mixture vs aligned encoded by line style.
  - CI bands only at final snapshot.
  - Scenario colors labelled in-plot (no extra legend).
- Single-axis figures (bifurcation, hysteresis, scaling, U-shape) now use consistent line widths and theme.

## Output Compatibility
All PDF filenames remain unchanged:
`fig6_ensemble_enhanced.pdf`, `fig_density_evolution.pdf`, `fig_bifurcation_diagram.pdf`,
`fig_loglog_scaling.pdf`, `fig_hysteresis_test.pdf`, `fig_variance_ushape.pdf`,
`fig_validation_summary.pdf`, `fig6_observables_comparison.pdf`, `fig6_phase_diagram.pdf`.

## Notes
- Plots.jl fallback paths were kept minimal; no per-time legends.
- `src/plotting.jl` (Plots backend) retains legacy defaults (not refactored here).
