# BeliefSim.jl — OU-with-Resets Mean-Field Simulator

This package simulates the **OU-with-resets** belief dynamics model with social coupling and provides
a minimal, reproducible pipeline to recover the **polarization threshold** \(\kappa^\*\) and the
**near-threshold** supercritical pitchfork profile, as developed in the accompanying manuscript.

> Core formulae (refer to the PDF): stationary dispersion \(V^*\) solves \(2\lambda V^*=\sigma^2+\Lambda_{\rm reset}(V^*)\);
> the leading eigen-slope is \(\lambda_1'(0)=2\lambda V^*/\sigma^2\); the critical coupling is
> \(\kappa^* = g\,\sigma^2/(2\lambda V^*)\), with the canonical short-cut \(g\approx\lambda\Rightarrow\kappa^*\approx\sigma^2/(2V^*)\).

---

## 0) Requirements

- Julia **1.10** or later.
- On first use, instantiate dependencies:

```julia
] activate .
] instantiate
```

---

## 1) Quick replication — minimal run

```julia
using BeliefSim
using BeliefSim.Types, BeliefSim.Stats, BeliefSim.Plotting

# Parameters (paper-like defaults; tweak as needed)
p = Params(λ=1.0, σ=0.8, Θ=2.0, c0=0.5, hazard=StepHazard(0.5))

# Estimate V* and κ*
Vstar = estimate_Vstar(p; N=20_000, T=300.0, dt=0.01, burn_in=100.0, seed=123)
κstar = critical_kappa(p; Vstar=Vstar)  # canonical g≈λ

# Sweep and plot
κgrid = collect(range(0.0, 2.0*κstar, length=21))
res = sweep_kappa(p, κgrid; N=20_000, T=300.0, dt=0.01, burn_in=100.0, seed=42)

plt = plot_bifurcation(res; κstar=κstar, title="Pitchfork")
savefig(plt, "bifurcation.png")
```

This will produce a `bifurcation.png` figure and console output with estimates.

---

## 2) Reproduce with a YAML config (batch & clean outputs)

Use the example configuration at `configs/example_sweep.yaml`. Run:

```bash
julia --project scripts/run_from_yaml.jl configs/example_sweep.yaml
```

Outputs (in `outputs/example_pitchfork/` by default):
- `bifurcation.csv` — κ grid, |ḡ| amplitude, variance
- `summary.txt` — V*, κ* (canonical), and cubic-fit estimates (κ*, b, R²)
- `bifurcation.png` — Pitchfork diagram with κ* line

### YAML fields

```yaml
name: example_pitchfork
seed: 2025            # RNG seed for reproducibility
N: 20000              # population size
T: 300.0              # horizon
dt: 0.01              # time step
burn_in: 100.0        # burn-in for steady-state summaries
params:
  lambda: 1.0
  sigma: 0.8
  theta: 2.0
  c0: 0.5
  hazard:
    kind: step        # or 'logistic'
    nu0: 0.5          # for step; for logistic use 'numax' and 'beta'
sweep:
  kappa_from: 0.0
  kappa_to_factor_of_kstar: 2.0
  points: 21
output_dir: outputs/example_pitchfork
```

---

## 3) Estimating the cubic coefficient (center-manifold fit)

Near \(\kappa^*\), the supercritical pitchfork has steady amplitude \(A\approx\sqrt{(\kappa-\kappa^*)/b}\).
We linearize as \(\kappa \approx \kappa^* + b A^2\) and fit \(\kappa\) on \(A^2\) (OLS) just above
threshold.

```julia
using BeliefSim.Stats
fit = pitchfork_fit(res; ε=1e-4, κmin=0.9*κstar, κmax=1.5*κstar)
@show fit.κstar fit.b fit.R2 fit.used
```

- `κstar` — estimated threshold from the fit (should be near canonical estimate).
- `b` — cubic coefficient; **b > 0** is consistent with supercriticality.
- `R2` — goodness of fit on the selected window.
- `used` — number of points used in the regression.

---

## 4) Command-line interface (no code)

Two ready-made scripts are provided:

```bash
# Minimal: estimate V* and κ* or sweep κ
julia --project scripts/run_sim.jl --mode vstar --lambda 1.0 --sigma 0.8 --theta 2.0 --c0 0.5 --nu0 0.5
julia --project scripts/run_sim.jl --mode sweep --lambda 1.0 --sigma 0.8 --theta 2.0 --c0 0.5 --nu0 0.5

# Config-driven batch run with plots
julia --project scripts/run_from_yaml.jl configs/example_sweep.yaml
```

---

## 5) Design & numerical notes

- **Discrete time step \(dt\)**: reduce until stable (e.g., 0.005–0.02). Smaller `dt` improves Poisson thinning accuracy.
- **Population size**: larger \(N\) stabilizes moment estimates (suggested: \(\ge 2\times 10^4\)).
- **Burn-in & horizon**: long enough for steady-state moments to stabilize (e.g., burn-in of 100 on the default scale).
- **Hazard boundedness**: both step and logistic hazards are bounded to keep reset thinning numerically sound.
- **Reproducibility**: all top-level runners accept a `seed`; fix it for exact replication.

---

## 6) Source map

- `src/Types.jl` — `Params` and hazard types
- `src/Hazard.jl` — `ν(h, u, Θ)` dispatch
- `src/Model.jl` — Euler–Maruyama updates and reset operator
- `src/Simulate.jl` — population simulation & summaries
- `src/Stats.jl` — `estimate_Vstar`, `estimate_g`, `critical_kappa`, `sweep_kappa`, `pitchfork_fit`
- `src/Network.jl` — neighbor-mean coupling (Graphs.jl)
- `src/Plotting.jl` — `plot_bifurcation(res; κstar=...)`
- `scripts/` — CLI runners
- `configs/` — example YAML configuration
- `examples/` — quick scripts to try the model

---

## 7) Minimal theory-to-code checklist

- OU core and reset rule implement Eqs. (9), (11) in the manuscript.
- Mean-field coupling implements Eq. (13).
- The dispersion estimator targets \(V^*\) in the balance (Eqs. (16)–(17)).
- The threshold calculation uses Theorem 5.4 (canonical shortcut included).
- The pitchfork fit operationalizes the near-threshold normal form to estimate the cubic \(b\).

---

## 8) Troubleshooting

- **Flat pitchfork**: increase `T` and `burn_in`, or refine `dt`.
- **Noisy \(V^*\)**: increase `N`, and ensure hazard parameters are not vanishingly small or huge.
- **Fit unstable**: restrict the window (`κmin`, `κmax`) closer to the transition and increase the number of \(\kappa\)-grid points.

MIT License.
