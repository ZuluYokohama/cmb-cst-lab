# Sparse-network closure validation (experimental)

The new sparse-graph solver fits scalar additive sensor differences on any connected undirected graph with one oriented observation per edge. It is a reference demonstration, not operational sensor software.

With incidence matrix A, data y = A x + noise and known symmetric positive-definite edge covariance Sigma, gauge-fixed generalized least squares minimizes (y-Ax)^T Sigma^(-1) (y-Ax).

A spanning-tree fundamental-cycle matrix Z satisfies Z A = 0 and has m-n+1 independent rows. The GLS residual score equals (Zy)^T (Z Sigma Z^T)^(-1) (Zy), up to numerical error.

A tree has m=n-1 and no cycles. It has no closure-based detection power: even corrupted observations can be explained as consistent relative sensor offsets. Redundant closure detects inconsistencies but does not uniquely attribute a defective or adversarial sensor.

Commands:
- python scripts/demo_sparse_network.py
- python -m pytest tests/test_sparse_network.py -q
- python scripts/run_sandbox_suites.py sparse_network

The tests cover exact cycle constraints, matching the existing full K5 implementation, a separate scipy.optimize.least_squares solution, correlated covariance, gauge behavior, invalid inputs, and a limited nominal chi-square simulation.

Limitations: graph topology is sparse, **numerical covariance/GLS operations remain dense**; no matrix-free memory or throughput claim is justified. No support for angular wrapping, SO(3)/SE(3), untrusted covariance, robust adversarial statistics, or field sensors. Nominal p-values require correctly specified Gaussian noise and covariance. The AST-bundled CI suite is not a new JoVin isolation attestation.

Next evidence gates: assess conditioning and numerical scaling; independent graph estimators; corrupted/misspecified covariance; heavy-tailed and correlated outliers; controlled real-sensor measurements.

## Covariance misspecification stress test

See NOISE_MODEL_CALIBRATION.md for the reproducible comparison of correct covariance with underreported variance and the resulting false-alarm inflation. The chi-square p-value is conditional on noise-model validity.
