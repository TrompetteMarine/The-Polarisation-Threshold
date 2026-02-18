# Style Debt (Plot Grammar Audit)

- scripts/visualization.jl :: plot_ensemble_figure / plot_ensemble_dashboard  
  Issue: per-time legends, inconsistent line styles, no shared time colorbar, CI shown for every snapshot.  
  Fix: time encoded by color + shared colorbar; one style legend; CI only at final snapshot; line styles encode type.

- scripts/visualization.jl :: plot_observables  
  Issue: multiple per-axis legends and inconsistent line/CI styling.  
  Fix: single shared legend; consistent linewidths; lighter CI bands.

- scripts/generate_bifurcation_figures.jl :: plot_density_evolution  
  Issue: repeated legends per panel and time labels; mixed legend semantics; inconsistent line styles.  
  Fix: shared time colorbar + one style legend; line styles encode mixture vs aligned; CI only final snapshot.

- scripts/generate_bifurcation_figures.jl :: plot_validation_summary  
  Issue: inconsistent linewidths and mixed legend placement.  
  Fix: unified linewidths + grammar theme; no extra legends.

Notes:
- src/plotting.jl (Plots backend) uses its own defaults for legacy scripts; not refactored in this pass.
