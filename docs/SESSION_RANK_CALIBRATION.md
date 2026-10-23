# Whole-session empirical rank calibration — assumptions and limits

The finite-sample F null reference requires independent calibration **rows**. In the previous temporal experiment, AR(1) dependence violated that assumption even though all individual rows were Gaussian with equal marginal covariance.

This experiment uses a separate source of exchangeability: whole calibration **sessions**. Every independent session contains correlated calibration rows and an independent held-out test observation; an identical, fixed algorithm reduces each session to a single scalar closure score. Independent null-reference sessions and a new held-out session are exchangeable **as scores** if their entire data-generating distributions match.

The conservative upper-tail empirical p-value is:

    p = (1 + number_of_reference_scores >= test_score)/(N_reference + 1).

Under exchangeability of the test score with all independent null reference scores, and fixed score construction, this is superuniform (with conservative ties). Dependence **inside** each session is allowed, but shared observation windows, reference leakage, nonstationarity across sessions, adaptive selection, or changing scoring procedures can break validity. At least 19 references are needed even to resolve a 5% tail (and at 19 the strict p<0.05 rule cannot reject). More reference sessions improve resolution and precision; they do not cure shift.

## Synthetic experiment

On a connected 5-node, 7-edge graph with 3 independent cycles, we generate independent *episodes*, each with K=16 stationary Gaussian AR(1) calibration rows at rho=0.8 and an independent held-out edge measurement. A reference bank of 1,800 episode scores is fixed before comparing 1,700 other independent null episodes.

We compare nominal 5% rejection using the invalid F reference versus empirical rank calibration. We then deliberately violate between-session exchangeability by switching tested episodes from rho=0.8 to rho=0.95 while retaining the same reference bank. The rank method no longer has a 5% guarantee under that shift.

```sh
python scripts/audit_session_rank.py --reference-count 1800 --test-count 1700 --seed 2026
python -m pytest tests/test_session_rank.py -q
python scripts/run_sandbox_suites.py empirical_calibration
```

Tests check ties, exact rank resolution, deterministic replay, controlled null rejection, within-session correlation, a deliberately shifted distribution, error handling, and AST-bundled kernel execution. This is a reference method based on existing conformal/rank-testing principles, not a new theorem or deployment-ready detector.

## Limits and next gate

Identically distributed *sessions* and independent score generation are crucial. The reference bank is assumed to consist of genuine null sessions, not contaminated or adversarial ones. The particular simulation does not evaluate unknown session drift, multiple correlated incoming alerts, adaptive repeated testing, or field sensor data. Reusing a bank for many decisions introduces multiplicity and dependence that require separate analysis.

No claim of real-time sensor fault identification, operational safety, novel mathematical topology, or security certification follows from this demonstration.