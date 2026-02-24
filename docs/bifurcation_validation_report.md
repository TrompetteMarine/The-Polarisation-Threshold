# Bifurcation Validation Report (Figure 6)

Generated: 2026-02-19 02:02:19

## Model Parameters
| Parameter | Value |
|-----------|-------|
| lambda | 0.85 |
| sigma | 1.15 |
| theta | 2.0 |
| c0 | 0.8 |
| nu0 | 10.6 |

## Critical Values
- V* = 0.682233
- kappa*_B (theory) = 0.988614
- kappa*_A (empirical) = 0.794052
- kappa*_A CI = [0.700514, 0.797279]
- kappa_varmin (variance minimum, diagnostic) = 1.191079
- kappa*_ref (used for scenarios) = 0.794052
- V_baseline (minimum variance across sweep, near kappa*) = 0.390309
- kappa/kappa* at V_baseline = 1.5

## Simulation Settings
- N = 20000
- T = 400.0
- dt = 0.01
- Ensemble size = 50 (scenarios), 30 (sweep)
- Sweep cap: kappa/kappa* <= 1.5
- Adaptive T: increases near kappa* to mitigate critical slowing down (cap T_max = 1200.0)
- Sweep mode: equilibrium (t_measure = 50.0)

---

## Order Parameter
Primary order parameter: **M*(t) = E|m_i(t)|** across runs, with optional **aligned mean**
M_aligned(t) = E[s_i m_i(t)] over decided runs (branch signs from late-time averages).
Because of symmetry, the signed mixture mean E[m(t)] can remain near zero above kappa*.
Variance-based amplitude sqrt(V - V_baseline) is exported as a secondary diagnostic.
Scaling uses the baseline-corrected amplitude M_corr = sqrt(max(M_abs^2 − M0^2, 0)),
with M0 the median M_abs below kappa*_B.

## Test 1: Scaling Exponent (H0: beta = 0.5)
| Statistic | Value |
|-----------|-------|
| beta_hat | 0.1857 |
| Standard Error | 0.0052 |
| 95% CI | [0.1733, 0.1981] |
| t-statistic | -60.116 |
| p-value | 0.0 |
| Amplitude C | 1.4875 |
| Points used | 9 |
| Delta window | [0.01, 0.05] |
| Verdict | FAIL (neither estimator agrees) |

## Test 2: Hysteresis (Supercriticality)
| Statistic | Value |
|-----------|-------|
| Mean |Delta M*| | 0.00127 |
| Max |Delta M*| | 0.023939 |
| t-statistic | 2.079 |
| p-value | 0.042 |
| Verdict | FAIL: unexpected — both sweeps start disordered yet differ (p=0.042) |

## Test 3: Critical Point Localization
| Statistic | Value |
|-----------|-------|
| kappa*_B (theory) | 0.9886 |
| Bootstrap 95% CI | [0.397, 0.5932] |
| Verdict | WIDE CI: kappa*=0.446 +/- 0.098 |

---

## Overall Verdict
**FAIL**: validation did not pass all criteria
