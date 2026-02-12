# Bifurcation Validation Report (Figure 6)

Generated: 2026-02-11 11:55:42

## Model Parameters
| Parameter | Value |
|-----------|-------|
| lambda | 0.85 |
| sigma | 0.8 |
| theta | 2.0 |
| c0 | 0.8 |
| nu0 | 10.6 |

## Critical Values
- V* = 0.370362
- kappa* = 1.830508
- V_baseline (minimum variance across sweep, near kappa*) = 0.099795
- kappa/kappa* at V_baseline = 0.85

## Simulation Settings
- N = 5000
- T = 200.0
- dt = 0.01
- Ensemble size = 5 (scenarios), 5 (sweep)
- Sweep cap: kappa/kappa* <= 1.5
- Adaptive T: enabled for kappa/kappa* > 1.2 to avoid numerical explosions
- Sweep mode: equilibrium (t_measure = 30.0)

---

## Order Parameter
We measure polarization with |a*| = sqrt(V - V_baseline), where V is the population variance at equilibrium.
The baseline variance is taken as the minimum variance across the sweep (near kappa*), reflecting maximal consensus.
This captures the symmetric split into ±a* when the mean remains near zero.

## Test 1: Scaling Exponent (H0: beta = 0.5)
| Statistic | Value |
|-----------|-------|
| beta_hat | 3.8319 |
| Standard Error | 0.3719 |
| 95% CI | [3.0217, 4.6421] |
| t-statistic | 8.96 |
| p-value | 0.0 |
| Amplitude C | 1.99 |
| Points used | 14 |
| Verdict | FAIL: beta=3.832 differs from 0.5 (p=0.0) |

## Test 2: Hysteresis (Supercriticality)
| Statistic | Value |
|-----------|-------|
| Mean |Delta a*| | 0.00527 |
| Max |Delta a*| | 0.01771 |
| t-statistic | -2.662 |
| p-value | 0.0186 |
| Verdict | FAIL: hysteresis detected (mean diff=0.0053, p=0.0186) |

## Test 3: Critical Point Localization
| Statistic | Value |
|-----------|-------|
| kappa* (analytic) | 1.8305 |
| Bootstrap 95% CI | [1.4644, 1.739] |
| Verdict | WIDE CI: kappa*=1.551 +/- 0.137 |

---

## Overall Verdict
**FAIL**: validation did not pass all criteria
