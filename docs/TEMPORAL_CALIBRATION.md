# Temporal-dependence negative control for finite-calibration closure

The finite-covariance F test is conditional on **independent** Gaussian calibration vectors. Gaussian marginal distributions and correct population covariance do not imply independent observations. A correlated sample covariance does not have the independent-Wishart law needed for the F reference.

## Fixed synthetic design

- A connected five-node seven-edge scalar graph has three independent fundamental cycles.
- A paired control uses 16 iid standard-Gaussian edge calibration vectors per synthetic trial.
- The experimental arm uses an AR(1) edge sequence with lag-one correlation rho=0.8: x_0=e_0; x_t=rho*x_(t-1)+sqrt(1-rho^2)*e_t.
- Both arms share the exact same Gaussian innovations and held-out measurement per trial, so each time point has the **same N(0,I) marginal covariance**. Only the dependence among calibration observations differs.
- The held-out measurement is independent of calibration in both arms, explicitly isolating calibration-row dependence.
- Each arm constructs its centered sample covariance in cycle space and applies the existing finite-F reference anyway, measuring how often nominal 5% tests reject under the known null.

## Reproduction

```sh
python scripts/audit_temporal_calibration.py --trials 2400 --seed 2026
python -m pytest tests/test_temporal_calibration.py -q
```

Tests cover nominal iid rejection, exaggerated false-alarm frequency under rho=0.8, rho=0 identity, reproducibility, input limits and agreement with the shipped calibrated-cycle reference for selected draws.

## Interpretation and limits

This is an **assumption-failure demonstration**, not a new estimator. F-based p-values from temporally correlated calibration observations generally have no claimed nominal calibration. The direction and size of failure depend on serial correlation, calibration sample count, network, seed, and chosen noise model.

A heuristic effective-sample-size substitution does **not** automatically restore the exact Wishart/F law. No such unverified adjustment is applied here. The test says nothing about particular real sensors, field deployments, unique fault attribution, operational safety, or cryptographic sandbox assurance.

The next useful tests are correlation-robust calibration with external held-out sessions, controlled drift and nonstationarity, and comparison with independently implemented time-series inference procedures under explicit assumptions.