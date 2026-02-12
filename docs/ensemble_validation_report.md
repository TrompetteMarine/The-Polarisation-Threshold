# Ensemble Validation Report (Figure 6)

Generated: 2026-02-07 22:38:26

## Model Parameters
- lambda = 0.85
- sigma = 0.8
- theta = 2.0
- c0 = 0.8
- nu0 = 10.6

## Critical Values
- V* = 0.371425
- kappa* = 1.830508

## Growth Rate Summary
```
3×12 DataFrame
 Row │ scenario  kappa_ratio  kappa    lambda_mean  lambda_std  lambda_ci_lower  lambda_ci_upper  r_squared  p_value   status  window_start  window_end
     │ String    Float64      Float64  Float64      Float64     Float64          Float64          Float64    Float64   String  Float64       Float64
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ below             0.8  1.46441   0.00594304  0.00758034       0.00052039       0.0113657   0.0858191  0.982483  FAIL            10.0        50.0
   2 │ critical          1.0  1.83051   1.37188e-5  5.20048e-5      -2.34832e-5       5.09208e-5  0.0461207  0.425752  PASS            10.0        50.0
   3 │ above             1.5  2.74576  -1.08503e-7  4.00021e-5      -2.87243e-5       2.85073e-5  0.0349416  0.503328  FAIL            10.0        50.0
```
