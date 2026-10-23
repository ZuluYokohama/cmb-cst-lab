# Exact representation/orientation control V1

The user's phrase "represent and rotate for impact based on merit" admits
several readings. Pending clarification, the bounded implementation is an
exact map-grid symmetry audit plus an evidence-based research priority brief.
It is not authorization to optimize a rotation for significance or rewrite
previous experimental outcomes. Earlier publication/JoVin requirements remain.

Before evaluation, freeze the audit code, numerical sources, authentic JoVin
receipt, observed patch-cache hash and this specification. Preserve all prior
source plans, draw records, proofs and conclusions.

Use three existing spherical_topology_v1 patches, eight fixed-seed Gaussian
128x128 arrays (seed710395+i), a row+2column ramp, and one Gaussian array with
a +12 spike at its center (seed710410). Thirteen cases in total. Retain three
representations: original pixels, per-patch mean/std standardization, and
empirical normal scores. These representations ask different amplitude/marginal
questions; they are not independent observations or opportunities to select
the most attractive sky result.

Test all eight D4 square-grid symmetries R^k F^f: quarter turns k=0,1,2,3,
with and without a horizontal reflection. These are exact pixel permutations
and preserve open-grid adjacency. They do not interpolate data, sample a new
sky location, test arbitrary SO(3) rotations, or establish rotational invariance
of the whole projection/instrument pipeline.

Expected behavior within each representation:

- First five multiscale features and first five curve columns are invariant.
- The signed horizontal/vertical contrast changes sign for odd quarter turns;
  reflection alone preserves it. This is covariance under relabeling, not a
  spurious physical anomaly. No signed statistic is silently replaced by its
  absolute value in an earlier experiment.
- Finite PD0 count and total lifetime are invariant to these graph symmetries.
- Pixel histograms are exactly preserved and rank-normalization commutes with
  each permutation within floating-point tolerance.
- Every multiscale curve is computed independently on CPU and CUDA float64.
  Require max(abs(measured-expected)/(1+abs(expected)))<=1e-10 for invariant/
  covariant curves, features and lifetimes, and CPU/GPU agreement. Require exact
  component counts and histograms, finite outputs, and preserve every failure.

Do not introduce a p-value or new discovery threshold: these are numerical
equivariance checks. Any change from raw to normal-score feature values is a
change of scientific representation and must not be reported as a robustness
pass/fail against raw values. Standardization should leave the six affine-
invariant feature values unchanged within tolerance.

Deliver every transformed-case diagnostic and CPU/GPU error, plus a report
separating implementation symmetry, statistical evidence and physical merit.
Retain the nine earlier Lean rational invariants; do not describe them as a
formal proof of D4 topology invariance or of any new physical discovery.
