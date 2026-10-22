# CMB calibration manuscript: scope, corrections, and pre-submission gates

**Status:** working research draft, 2026-10-22; not submitted to arXiv or a journal.
**Primary references:** [Lonappan, Keating & Arnold 2026](https://arxiv.org/abs/2609.09149), [their code](https://github.com/antolonappan/cosmic_birefringence), [Cole & Shiu 2018](https://arxiv.org/abs/1712.08159).

## Core findings and non-findings

- Differential pair-angle calibration (28 edges of an 8-node complete graph) is the published estimator, not a novel discovery of this lab.
- Adding 56 oriented triangles gives 21 independent closure constraints, and the filled two-complex has H^1=0. This is established combinatorial topology, not a cosmic measurement.
- The covariance-weighted triangle-closure quadratic form is **identical** to the GLS residual chi-square using the same edge observations and covariance. Closure and GLS do **not** provide independent confirmations.
- With a general covariance, the GLS residual is orthogonal in the **inverse-covariance metric**, not necessarily in the Euclidean metric. Earlier descriptions calling it an ordinary im(d1.T) curl were incomplete.
- A coherent **detector-node** offset is absorbed by fitted angles and cannot be detected by triangle closure. A fault on a **single pairwise measurement** generally can.
- The new single-edge GLR scan implements well-established statistics with an empirical maximum-statistic threshold for familywise error. It is a **baseline diagnostic**, not a new cosmology estimator.
- A deterministic **synthetic** experiment (4,000 independent calibration nulls; 4,000 null holdouts; 4,000 injected draws) has 5% held-out false-alarm fraction, 84% detection, and 81.5% simultaneous detection plus correct edge localization for one selected 4-sigma edge injection. These figures do **not** describe the Planck data, uncertainty across other covariances, or superiority over existing comparators.
- Real-sky persistence remains exploratory. The current main-branch output uses 1,000 Gaussian nulls, +1-corrected two-sided tails and Holm control across 13 tests. The two smallest adjusted component-count p-values are about 0.078; this is **not** a corrected discovery. Noise anisotropy, foregrounds, masks and null-model adequacy remain incompletely validated.
- The PD1 return value is a dual-based lower-bound proxy, not an independently established exact cubical-complex loop-persistence result. In the Planck patch analysis, only PD0 is used.
- Sheaf cohomology, non-Gaussian primordial phase extraction, and new physical E/B curl invariants are **not implemented/validated scientific outputs of this codebase**.

## Reproduction checks already present in public repository

The original README reports real-map Planck PR4 reproduction; this revision does not rerun the approximately 17 GB download and CPU/GPU processing pipeline. The JoVin hash-chained receipts cover selected numpy kernels and are not proof that the entire astronomical procedure was independently executed or reviewed.

## Required follow-on verification (not yet completed)

1. Record map file hashes, exact processing masks, output commit, numerical environment, theory transfer and covariance estimation.
2. Rerun the real-sky pipeline from source maps and compare the reproducible outputs to the original LKA paper without tuning masks on the result.
3. Independently verify the current corrected persistence output and test Monte Carlo uncertainty under realistic anisotropic noise, foreground, mask and multiple-testing scenarios; the present 1,000 Gaussian simulations are not a sufficient physical-null validation.
4. Compare the edge GLR with **equivalent** covariance-weighted GLS residual diagnostics, leave-one-edge-out fits, and robust/outlier methods. The GLR may be algebraically equivalent to well-known studentized residuals, so do not label it superior without a difference established by a blinded test.
5. Preregister contamination classes, calibration metrics, confidence interval/reporting rules and independent held-out seeds before new performance claims.
6. Obtain independent mathematical and experimental cosmology review, including numerical provenance and authorship attribution.

## Commands

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/test_edge_diagnostics.py tests/test_persistence_pvalues.py
python scripts/run_edge_fault_pilot.py
# Full-data reconstruction (requires the original Planck data products):
python -m cmbcst.data
python scripts/run_reproduction.py
```

The edge script writes `results/edge_fault_pilot.json`; full-data reproduction changes other caches and must be inspected before publication.

## Candidate draft title

**Gauge-Invariant Cycle Diagnostics for CMB Polarization Calibration: A Reproducibility Audit and Covariance-Aware Edge-Fault Pilot**

Proposed arXiv subject areas: **astro-ph.IM**, **astro-ph.CO**, subject to moderation. The experiment as currently implemented is more appropriately viewed as a technical note/methodological pilot than a proved new estimator.
