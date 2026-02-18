# Patch Log — Bifurcation/Scaling + Visuals (R2)

## What changed
- **Bifurcation diagram** now uses the correct odd‑mode order parameter:
  - Primary: `m_aligned_star` (aligned mean amplitude).
  - Secondary: `m_abs_star` (hollow markers) if available.
  - Removed variance‑based amplitude as primary (variance remains a diagnostic elsewhere).
- **Log–log scaling** now fits:
  - `x = log(κ − κ*_B)` with κ*_B from `metadata.json`.
  - `y = log(M*_aligned)` within a near‑critical window.
  - Bootstrapped slope CI and explicit reference slope 1/2.
- **Kurtosis removed** from observables panels; replaced with overlap or decided‑share when available.
- **Unified plot grammar** applied consistently:
  - Time encoded by color + shared colorbar.
  - Mixture vs aligned encoded by linestyle + alpha.
  - CI bands only on final snapshot where applicable.
  - One compact legend per figure; no per‑time legends.

## Files patched
- `scripts/generate_bifurcation_figures.jl`
- `scripts/visualization.jl`
- `scripts/fig6_ensemble_enhanced.jl` (pass observables into plotting)
- `scripts/plot_grammar.jl` (shared style helpers)

## How to reproduce
1) Ensemble outputs:
```
julia --project=. scripts/fig6_ensemble_enhanced.jl
```
2) Figure generation:
```
julia --project=. scripts/generate_bifurcation_figures.jl
```

## Expected results
- `fig_bifurcation_diagram.pdf` shows **M*_aligned** vs κ/κ*_B with κ*_B line and κ*_A band (if present).
- `fig_loglog_scaling.pdf` uses **M*_aligned**, near‑critical window, and shows slope ± 95% CI.
- No kurtosis panel in the observables/dashboard figures.
- Legends are compact; time is shown via a shared colorbar where relevant.
