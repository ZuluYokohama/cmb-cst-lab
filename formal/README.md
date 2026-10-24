# Exact rational support for the finite-scale features

`ScaleInvariance.lean` contains nine kernel-checked algebraic theorems with
Lean 4.35.0, using only its standard library. They prove cancellation of an
affine brightness offset in increments, homogeneous scaling of second/fourth/
sixth sample moment sums, and invariance of normalized fourth/sixth moment
ratios and a directional contrast when the denominators and scale are nonzero.

The proof uses exact rationals and explicit assumptions. It does not prove
the log-slope summaries, the Python/CUDA floating-point implementation on every
input, any sky anomaly, multifractality, cosmology, or cryptographic security.
It uses Lean's standard foundations: propext, Classical.choice and Quot.sound.
No admitted theorem or extra axiom declaration is present. No Mathlib download
is needed. The supporting algebra is elementary established mathematics, not
a new theorem or a physical discovery.

Pinned official Linux archive:
`https://github.com/leanprover/lean4/releases/download/v4.35.0/lean-4.35.0-linux.tar.zst`

SHA256: `7289597a44543246c9227b8cae501767e095ce17f6d448b95fc4a8dc97268f6d`.

After checking and extracting that archive, run from the repository:

```sh
python scripts/verify_lean.py --lean /path/to/lean/bin/lean --archive /path/to/lean-4.35.0-linux.tar.zst
```

The verifier records the compiler version/commit, source and numerical module
hashes, nine theorem names, compiler output, assumptions and limits in
`results/multiscale_v1/lean_verification.json`. CI repeats the proof with the
same pinned compiler archive. SHA256 identifies bytes; these receipts are not
digital signatures and do not establish authorship or scientific truth.

## Acquisition geometry

[PolarizationGeometry.lean](PolarizationGeometry.lean) adds two exact rational
identities: rotating Q/U preserves their squared norm, and rotating both a
projection vector and Q/U preserves the measured projection. The assumption
is c²+s²=1. Both use the same pinned compiler and standard foundations.

These identities explain an amplitude-only diagnostic's blindness to a uniform
polarization-basis rotation. They do not prove floating-point reconstruction,
valid beam optics or a physical discovery. The separate
[compiler record](../data/evidence/acquisition_lean.json) and
[acquisition report](../docs/ACQUISITION_RUN.md) preserve their scope; CI checks
both proof files. The original nine theorem sources and evidence remain intact.

## Operational projection under smoothing

[OperationalClosure.lean](OperationalClosure.lean) adds seven theorems using the
same pinned Lean 4.35.0 compiler and standard library. Two conditional typed-
function results show that equal recorded inputs with unequal target outputs
exclude a deterministic predictor or factor map on a domain containing that
pair. Five exact rational checks instantiate the argument with four-node
adjacent averaging and second/fourth moments. The conditional results need no
axioms; the toy checks use the standard foundations listed above.

The [compiler evidence](../data/evidence/closure_lean.json) records all seven
theorems and their axioms. There are 18 supporting theorems across the three
files, all checked in CI. The [bounded Planck counterfactual study](../docs/CLOSURE_FIBERS_RUN.md)
found no qualifying witness and left the repair comparison untested. The proofs
do not certify numerical sky calculations, successful optimization, universal
repair, physical detections or mathematical novelty. Earlier proof sources and
their evidence remain unchanged.

## Recorded v2 witness

[ClosureFibersV2Witness.lean](ClosureFibersV2Witness.lean) adds six axiom-free
theorems for the first verified v2 pair, tile 24 / start 56. They check the four
44-entry integer-bin tables, equal current records and unequal future records,
including primary future bins 232 and 1007. The final theorem imports and
specializes the existing no-factor theorem with an explicit assumption that
the retained global records agree. It does not copy the generic proof.

The [compiler receipt](../data/evidence/closure_v2_first_witness_lean.json) binds
the exact tables, numerical evidence, coefficients and source by SHA-256 and
records the pinned Lean 4.35.0 output. The compiled dependency stays in a
temporary cache. CI repeats the check. There are now 24 supporting theorems in
four files. These six prove recorded-table mathematics and a conditional
conclusion; identifying those tables with floating-point sky measurements
remains an empirical assumption, not a formal pipeline or physical proof.
