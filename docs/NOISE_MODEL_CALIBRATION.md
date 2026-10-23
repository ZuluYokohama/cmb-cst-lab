# Statistical calibration gate for sensor graph closure

The sparse scalar cochain framework gives a rigorous GLS/closure identity, but **its nominal chi-square tail probability requires a correct noise model**.

This check constructs a fixed, redundant 5-sensor graph with 7 pair edges and 3 independent cycles, injects synthetic Gaussian measurement noise, and calculates the same statistic two ways:

1. Using the true edge covariance: rejection at a nominal 5% threshold should be approximately 5% over repeated trials.
2. Underreporting every edge variance by a factor of four: the fitted offsets remain mathematically defined, but the nominal false-alarm rate rises dramatically because the reported covariance is wrong.

Run: python scripts/audit_sparse_noise.py

The script is deterministic for a supplied seed, verifies its batched cycle statistics against the shipped sparse-graph GLS implementation, and supports configurable trial counts for further checks. CI tests the statistical direction of the effect with broad bounds rather than asserting an exact Monte Carlo fraction.

This demonstration is **not evidence of a novel statistical result**. It operationalizes a known assumption check and prevents the repository's cochain language from being interpreted as automatic fault detection, identification, or security certification.

Next gates: covariance estimated from a finite number of observations, spatial/time correlation, rare outliers, changing sensor bias, incomplete edges, and independent field datasets. None has been resolved by this experiment.


## Finite independent calibration samples

See [estimated-covariance validation](ESTIMATED_COVARIANCE_VALIDATION.md). With finite independent Gaussian calibration, the closure score has a scaled F—not chi-square—reference; empirical covariance from correlated or reused observations still needs a separate treatment.


## Non-Gaussian contamination

The [heavy-tailed negative control](NON_GAUSSIAN_CLOSURE.md) tests the conditional F correction on an equal-covariance Student-t noise model; a correct second moment does not imply a correct Gaussian false-alarm reference.
