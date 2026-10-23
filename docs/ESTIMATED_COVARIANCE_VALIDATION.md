# Finite-sample covariance calibration: a falsifiable gate

The existing graph-closure and weighted-GLS identity is exact for a specified positive-definite edge covariance, but a naive chi-square tail probability assumes that covariance is **known**. Estimating it from a finite sample changes the statistic's null distribution.

## Probability model and corrected reference

For a fixed connected graph, A is its incidence matrix and Z is an independent cycle matrix satisfying ZA=0. Let q=rank(Z)=m-n+1 and take a held-out y=Ax+ε with ε~N(0,Σ). Its cycle closure c=Zy=Zε has **theoretical mean zero**.

Take K **independent** calibration observations from the same stationary Gaussian edge-noise law, separate from the held-out y. Their node offsets may vary and cancel under Z. Form the centered unbiased sample covariance S_hat of their K cycle vectors (Wishart degrees of freedom K-1). For Q=cᵀ S_hat⁻¹ c:

    ((K-q)/(q(K-1))) Q ~ F(q, K-q).

The numpy-only API `cmbcst.estimated_covariance.calibrated_cycle_statistic` returns Q, the naive chi-square p-value, the scaled F statistic and degrees of freedom. Evaluate the corrected p-value with `scipy.stats.f.sf(result['f_statistic'], result['f_numerator_dof'], result['f_denominator_dof'])`.

## Reproduce

```sh
python scripts/audit_finite_covariance.py --trials 2400 --seed 2026
python -m pytest tests/test_estimated_covariance.py -q
python scripts/run_sandbox_suites.py estimated_covariance
```

The deterministic audit compares nominal 5% rejection to observed rejection over independent simulated trials for K=8, 16, and 64 with three independent cycles. Tests check the F rescaling, agreement with SciPy and projected sample covariance, tree/no-redundancy behavior, rank guards, and deterministic replay.

## Limits

The F reference assumes the **test measurement is independent** of calibration observations and every observation has the same stationary multivariate Gaussian noise covariance. It relies on a **known-zero cycle mean**. Subtracting an estimated mean from the held-out vector changes the reference law. Shared calibration/test samples, temporal correlation, changing topology, drift, heavy tails, censoring, adaptive threshold choices, or non-Gaussian errors can invalidate the correction.

The K>=q+3 guard excludes particularly low denominator-degrees-of-freedom cases for numerical stability; the F distribution itself does not require this stronger guard. A tree has no cycle statistic. These checks are synthetic and do not establish field sensor performance, fault attribution, real-time behavior, or security certification. The F identity is established statistical theory rather than a new theorem.