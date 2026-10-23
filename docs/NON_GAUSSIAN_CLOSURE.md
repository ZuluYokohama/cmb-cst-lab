# Non-Gaussian negative control for finite-sample closure calibration

The finite-sample scaled-F reference in `estimated_covariance.py` is **exact only under its specified independent stationary Gaussian calibration model**. Even matching the population noise covariance does not extend that result to arbitrary noise distributions.

This reproducible synthetic negative control takes a connected five-sensor, seven-edge graph with three independent cycles. The Gaussian arm has independent standard-normal full-edge vectors. The Student-t arm multiplies each full-edge vector by an independently drawn common scale `sqrt((df-2)/chi2_df)`. For `df > 2`, the resulting elliptical Student-t distribution has the **same population covariance matrix** as the Gaussian vectors. Both use `K` independent calibration vectors and one independent held-out vector per trial.

The audit forms sample covariance in the cycle subspace, calculates the scaled-F score, verifies the vectorized score against the shipped scalar-graph function, and measures rejection frequency at a nominal 5% threshold.

```sh
python scripts/audit_non_gaussian_closure.py --trials 2400 --seed 2026
python -m pytest tests/test_non_gaussian_closure.py -q
```

The point is to **exhibit an assumption failure**, not to claim a new robust estimator. In the heavy-tailed scenario, the Gaussian-based F reference can exhibit inflated rejection frequency. The result depends on the specific Student-t model, `df`, calibration count, seed, and network topology. It is not a measured real-sensor false-alarm rate.

Possible research next steps: independent time-series calibration, heavy-tail-robust covariance estimators, calibrated empirical conformal/rank procedures with valid exchangeability assumptions, and external sensor datasets. Those possibilities are not implemented or validated here.