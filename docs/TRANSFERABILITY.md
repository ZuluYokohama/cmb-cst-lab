# Transferability, verification limits, and commercial research

This note distinguishes demonstrated CMB research from a small experimental transfer of its cochain mathematics. It is not a production claim, security certification, license grant, or valuation.

## Demonstrated in the repository

- Planck PR4 reproduction of pair-angle calibration and a covariance-aware graph/Hodge decomposition.
- For the **filled** complete graph on eight maps, H^1=0; after covariance weighting, GLS residual energy equals the quadratic form of triangle-closure discrepancies. The nonzero residual is **coexact in the covariance metric**, not automatically Euclidean coexact.
- Region-resolved diagnostics and identifiable-parameter checks are scientific extensions with published source code and recorded numerical results.
- Tests, pinned source hashes, CI runs, and JoVin receipt metadata support **auditability of specific calculations**. They do not independently verify all external data, scientific assumptions, or the security of the runtime.

## Experimental transfer: scalar multi-sensor differences

Run: `python scripts/demo_sensor_network.py`.

`src/cmbcst/sensor_network.py` reuses the **same shipped** `complete_complex`, `solve_network`, and `closure_chi2` functions for synthetic scalar measurements y_ij = a_i - a_j + noise on a complete network.

The demo shows that (i) a global additive offset remains unobservable without an external anchor, (ii) consistent measurements close, and (iii) a single corrupted pair measurement creates detectable closure energy under the assumed noise covariance. The implementation accepts fully correlated, symmetric positive-definite edge covariance.

**Scope limitations:** This is scalar additive-offset calibration for a complete graph. It does not yet support sparse/disconnected arrays, angular wrapping, noncommutative SO(3)/SE(3) transport, nonlinear clock models, robust outlier contamination, real-time latency guarantees, automatic fault localization, or proven spoofing detection. Correlated or incorrectly estimated noise can invalidate nominal p-values. Real physical sensor experiments and independent baselines are required before operational use.

## AST extraction and receipts: precise assurance boundary

The `scripts/jovin_verify.py` runner extracts existing Python source using `ast` into bundled modules; its configured external JoVin harness executes suites and records outcome metadata and source/test hashes. CI executes those bundled suites with pytest, without issuing JoVin receipts. The lab's existing receipts report the state of **those specific historical runs**; a green CI run on a new branch is not a JoVin attestation.

AST parsing is a code-extraction step, **not** a security sandbox. A hash chain is tamper-evident only under appropriate assumptions about hash integrity, ledger anchoring, signing/trust, and verifier independence. Neither makes the tested program correct or proves the execution host was isolated. For regulated deployments, independently evaluate the harness threat model, permissions, reproducible builds, immutable artifact storage, signed provenance, and isolation.

## Scientific and commercial claims

- The astrophysics pipeline reproduces and extends published methods; it does not claim new cosmological physics.
- Fisher scoring is not independently established as the cause of the larger beta uncertainty: masks, pseudo-C_l analysis, dust assumptions, and other methodological differences also changed. MCMC is not inherently a guarantee of correct error bars.
- Enterprise sensor calibration and AI-generated-code verification are **candidate pilot applications**, not commercially validated products. Detectable triangle-closure inconsistencies do not uniquely attribute a defective or adversarial sensor.
- No customer contracts, product-market fit, enterprise pricing, security certification, or deployable safety features are demonstrated by this repository.

## IP and licensing boundary

The repository is public, but a root-level LICENSE was not present at the time this note was prepared. Public visibility does not by itself establish permission for unrestricted reuse. The repository separately preserves an upstream MIT notice for the port of `mk_likelihood.py`; obligations tied to that imported material remain relevant. The original estimator, physics, and mathematical identities cannot be claimed as new proprietary inventions merely through this implementation. No new license or transfer of rights is made by this document.

Before choosing AGPL, dual licensing, or a private commercial module, audit copyright ownership, every third-party component, and applicable contribution rights with a qualified adviser. A future commercial pilot should keep genuinely new deliverables and provenance separate and use written client agreements.

## Proposed evidence gates

1. **Math:** test scalar gauge invariance, closure/residual equality, noise calibration, and correlated covariance against an independent solver.
2. **Transfer:** compare detection power and false-positive rates with robust least squares and conventional graph-cycle baselines on real multi-sensor datasets.
3. **Efficiency:** measure wall time, RAM, and scaling versus sensor count; this initial full-triangle implementation is not optimized for large arrays.
4. **Security:** threat-model AST extraction, the JoVin runtime, receipt forging, replay, and ledger anchoring before making assurances.
5. **IP:** review provenance and licensing before public release of any commercial components.
