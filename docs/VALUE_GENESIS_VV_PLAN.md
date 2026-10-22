# CMB–CST evidence-to-value qualification

**Status:** research and release **gates**, not certification or a finding of exclusive ownership. Updated 2026-10-22.

## Purpose

Retain the value of the *human-directed systems-engineering process* without overstating the novelty or exclusive ownership of published CMB physics, public input data, upstream MIT code or non-copyrightable AI output. Scientific validity, provenance/authorship, commercial enforceability and product viability must be evaluated **independently**.

The end-to-end path is:

`human objective → prior-art boundary → source/data lineage → formal model → implementation → test → observed output → independent baseline → attempted falsification → reproducible claim → applicable rights → deliberate release`

Each transition has a measurable failure condition, not just a narrative explanation.

## Validation ledger

| Layer | Present evidence or scope | Pass condition | Current state |
| --- | --- | --- | --- |
| 1. Source/antecedents | Published Lonappan–Keating–Arnold estimator, Planck data, MIT upstream source | Upstream module-level comparison, proper permissions and source-version citations | **PARTIAL** |
| 2. Math/topology | Eight vertices, 28 oriented edges, rank-seven incidence, 21 residual dimensions | Exact linear-algebra checks, covariance-weighted residual/closure identity, nullspace controls | **KERNEL EVIDENCE; independent review pending** |
| 3. Likelihood | Port of published likelihood with binned / Fisher-scoring differences | Numerical comparison to cited method under matched masks, spectra, covariance, priors and precision | **UNVERIFIED at matched conditions** |
| 4. Instrument/datasets | Public Planck PR4/NPIPE products | Product hashes, masks, beam/spectra lineage, reproducible fetch, version lock | **INCOMPLETE** |
| 5. Computational V&V | Unit tests, local JoVin receipts, CI | Clean reproducible runs on independent setup; adversarial numerical tests and independent reviewer | **LOCAL ONLY; not independent certification** |
| 6. Edge fault diagnostics | Draft PR #3 synthetic GLR | Fixed baselines, holdout calibration, power/FWER intervals across covariances and faults; raw sky analysis separate | **EXPLORATORY SYNTHETIC** |
| 7. Sky persistence | Cached Planck patch summary | Realistic anisotropic-noise/foreground nulls, multiplicity correction, preregistered tests, corrected rerun | **NO ANOMALY CLAIM** |
| 8. Novelty | Hodge language/diagnostic packaging and verification workflow | Show independent measurable result *beyond* standard GLS, prior topology and existing software practice | **NOT ESTABLISHED** |
| 9. Human authorship | Prompt/commit/choice/test record in progress | Trace sufficiently original human expression, independent sources, work-for-hire agreements and AI terms | **UNRESOLVED** |
| 10. Release/commercialization | Draft rights notice; no blanket license | Patent/public-disclosure triage; legal title review; file-by-file permissions; privacy/secret audit | **HOLD** |

## Technical falsification program

1. **Analytical equivalence:** On random connected graphs and positive-definite *correlated* edge covariance, verify GLS residual \(r\), edge closure \(d_1 y\), \(d_1 d_0=0\), and \(r^T C^{-1}r=(d_1 y)^T(d_1 C d_1^T)^+(d_1 y)\). The closure statistic is mathematically **the same evidence** as the GLS residual, not independent confirmation.
2. **Gauge invariance and identifiability:** Compare reference-gauge choices and common/node offsets; triangle closure cannot detect coherent node calibration offsets. Deliberately construct a disconnected or rank-deficient case to confirm rejection.
3. **Numerical stress:** Test singular/near-singular covariance, anisotropy, finite precision, pseudoinverse tolerance, orientation permutations, and data ordering. Record the asserted tolerance before execution.
4. **Estimator reproducibility:** Compare upstream model and port on identical synthetic observables; then on identical Planck inputs and preprocessing. Separate implementation discrepancy from scientific-model discrepancy.
5. **Fault detection:** Benchmark GLR against GLS residuals, leave-one-out, robust regression, likelihood ratios and fixed/holdout nulls. Report confidence intervals, false-discovery controls, power, bias and selected-case limitations. The current pilot cannot establish general superiority.
6. **Real sky:** Rerun the corrected Monte Carlo p-value estimator; model masks, noise anisotropy and foreground uncertainty; predefine patch/test family and multiplicity treatment. Do not infer new physics from uncalibrated tails.
7. **Security & provenance:** Review all branches/history for accidental secrets; produce a dependency SBOM, source lineage diffs, third-party notices, private session hashes and a genuine chronological human-decision ledger. Reject incomplete origin assertions.

## Rights and publication controls

- Do **not** equate public source with public domain, AI use with loss of all human copyrights, or a code repository with ownership of the published scientific method.
- Before any further enabling public disclosure, inventory dated prior public disclosures and obtain patent counsel's decision on whether anything technically patentable remains; U.S. grace-period rules do not replicate in most European cases.
- Store novel undisclosed research and enterprise integrations in a **separate private repository**, not a branch of this public repository.
- Do not place private session transcripts or unpublished invention details in CI, Issues, GitHub Pages or public source files.
- Preserve known upstream MIT notice and isolate any future original source licensing to specifically audited files. AGPL + commercial licensing, if later selected, requires chain-of-title and contributor-rights review.
- Identify the actual lawful rights holder(s), human contributions and any employer/contractual claims before registering, licensing or assigning rights.
- Treat preprints and software as **separate licensing surfaces**. Disclose significant AI involvement and cite original sources in manuscripts.

## Required decision record for each claim

```yaml
claim_id: CMB-CST-###
claim_text: ""
scope: [reproduction, synthetic, observational, theoretical, software]
antecedent_citations: []
implementation_ref: "<commit:path:function>"
human_decision_evidence_private_sha256: ""
inputs_and_checksums: []
run_environment: ""
baseline_comparators: []
negative_controls: []
measured_result: ""
uncertainty_and_limits: ""
independent_review: "PENDING"
third_party_rights: "UNRESOLVED"
human_copyright_chain: "UNRESOLVED"
patent_disclosure_gate: "HOLD"
decision: "HOLD"
```

**Release rule:** no scientific priority, commercialization-readiness, third-party verification, or exclusive-ownership claims from a single passing unit test, a synthetic result, a locally generated receipt, or a rights notice. The relevant independently supported gates must pass before the claim is published.
