# Bifurcation Validation Report (Figure 6)

Generated: 2026-02-18 17:38:55

## Model Parameters
| Parameter | Value |
|-----------|-------|
| lambda | 0.85 |
| sigma | 0.8 |
| theta | 2.0 |
| c0 | 0.8 |
| nu0 | 10.6 |

## Critical Values
- V* = 0.371425
- kappa*_B (theory) = 0.864766
- kappa*_A (empirical) = 0.792807
- kappa*_A CI = [0.692841, 0.820375]
- kappa_varmin (variance minimum, diagnostic) = 1.18921
- kappa*_ref (used for scenarios) = 0.792807
- V_baseline (minimum variance across sweep, near kappa*) = 0.141082
- kappa/kappa* at V_baseline = 1.5

## Simulation Settings
- N = 20000
- T = 200.0
- dt = 0.01
- Ensemble size = 20 (scenarios), 10 (sweep)
- Sweep cap: kappa/kappa* <= 1.5
- Adaptive T: increases near kappa* to mitigate critical slowing down (cap T_max = 1200.0)
- Sweep mode: equilibrium (t_measure = 30.0)

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
| beta_hat | 2.1093 |
| Standard Error | 0.1088 |
| 95% CI | [1.8817, 2.3369] |
| t-statistic | 14.797 |
| p-value | 0.0 |
| Amplitude C | 186.4162 |
| Points used | 21 |
| Delta window | [0.01, 0.1] |
| Verdict | FAIL (neither estimator agrees) |

## Test 2: Hysteresis (Supercriticality)
| Statistic | Value |
|-----------|-------|
| Mean |Delta M*| | 0.016555 |
| Max |Delta M*| | 0.139247 |
| t-statistic | 1.889 |
| p-value | 0.0641 |
| Verdict | PASS: no significant hysteresis (max diff=0.1392) |

## Test 3: Critical Point Localization
| Statistic | Value |
|-----------|-------|
| kappa*_B (theory) | 0.8648 |
| Bootstrap 95% CI | [0.6053, 0.7351] |
| Verdict | WIDE CI: kappa*=0.626 +/- 0.065 |

---

## Overall Verdict
**FAIL**: validation did not pass all criteria
