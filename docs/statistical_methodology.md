# Statistical Methodology

## Symmetry-aware order parameters

Because the model is Z₂-symmetric, above κ* each realization selects a + or − branch. As a result, the
ensemble-signed mean E[m(t)] can remain near zero even when symmetry breaking occurs. We therefore report
sign-invariant summaries:

- **E|m(t)|** = mean across runs of |m_r(t)|
- **RMS(m(t))** = sqrt(mean across runs of m_r(t)^2)

These are the primary order parameters in the figures and CSV exports. The signed mean is still exported
for completeness as a diagnostic of symmetry cancellation.

## Growth-rate estimation

We estimate growth rates using log-linear regression on |m(t)|. Confidence intervals use a t-distribution
with df = n_ensemble − 1. One-sample t-tests are used to test H₀: λ = 0.

## Threshold estimation (Route A / Route B)

- **Route A (empirical)**: scan κ and fit the growth rate of E|m(t)|; κ*_A is the zero-crossing of λ̂(κ), with bootstrap CI.
- **Route B (theory)**: compute κ*_B from the odd-mode susceptibility of the linearised OU‑PR generator with sign‑preserving resets.
Both values are stored in `outputs/threshold/metadata.json` and used by the plotting scripts.

## Branch diagnostics

For each run we define a branch sign using the late-time mean:

```
s_r = sign(mean_t m_r(t) over t ∈ [T − Δ, T])
```

with Δ ≈ 0.15T (or 50 time units). If s_r = 0 we set s_r = +1. These branch signs are used to compute
aligned densities and to export terminal_means.csv (showing bimodality above κ*).
