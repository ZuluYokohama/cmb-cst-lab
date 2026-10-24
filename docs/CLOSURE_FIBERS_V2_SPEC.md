# CST smoothing fibers v2: batched search and independent replay

This is a separate prospective protocol, informed by the incomplete v1 search.
Preserve `docs/CLOSURE_FIBERS_SPEC.md`, every byte under
`results/closure_fibers_v1/`, and all earlier frozen numerical modules. Output
only to `results/closure_fibers_v2/`; temporary rebuildable operators belong in
`data/work/closure_fibers_v2/`. Never restart or extend a frozen run's budget.

## Question and limits

Apply the supplied CST working paper section 8.1 to recorded finite-precision
representations: can matching current summaries yield different summaries after
identical smoothing? Compare baseline, ordinary-power response control and
connected-fourth-order response repair. A verified pair rules out a deterministic
factor map on a domain containing that pair. No finite search without a witness
proves closure. Separating an old witness does not establish universal repair;
lower maxima from a fresh adversarial search do not establish detector superiority.
No cosmological detection, physical ceiling, fractal dimension or priority claim
is licensed. The supplied paper is not redistributed.

V1 used batches of four and already scheduled four searches per tile. Its largest
saved feasible endpoint had raw K4 gap 0.017660102450468743, tile perturbation RMS
0.036371500409516505 and global perturbation RMS 0.010991346931965748. Those values
do not show saturation of the 0.25 RMS limits. Similar discovered gaps do not
establish a physical maximum. A raw K4 gap and a robust-scale bin width are
different units; report their values separately. The fixed K4 scale at 2 degrees
in v1 was 0.07443557055963251. All RMS arrays use explicit order **tile, global**.

## Unchanged counterfactual domain and representations

Use the existing filtered 143 GHz A/B harmonics, masks and transported sampling
geometry; eight CPU transform threads, single BLAS thread, float64/complex128.
State is coadd plus fixed A-minus-B difference. Additional smoothing multiplies
each harmonic by exp[-u*l*(l+1)*sigma_60arcmin^2/2], for u=0,1,2,3. Primary target
is raw dimensionless normalized connected fourth-order K4 at 2 degrees and u=1.
Global E/B auto/cross spectra remain recorded. Construct 128 orthogonal rotation
planes, four per each of 32 selected multipoles spanning l30--216, in the
orthogonal complement of the split difference. Store actual numerical bases.
Angles remain within +/-pi/4. Perturbation RMS remains <=0.25 of reference coadd
RMS, both globally and at the tile anchors, throughout accepted trajectories.

Use tiles 24,6,30,34,21,11,152,170,189,174,186,166, the previously declared
geometry-only selection. Retain all 52 feature equations and four lags. Baseline
records the first 32 increment features plus 12 band powers (A, B and real A/B
cross power in l30--64,65--128,129--256,257--512). Capacity adds four derivatives
of mean increment power; repair adds four derivatives of connected K4 instead.

Fix scales from all 79 original reference tiles before any search, using
max(1.4826*MAD,0.001*max(1,abs(median))). Reference-centered local quantization is
nearest-even with width 0.001 robust-scale units. V2 input and changed-future-bin
safety margin is **1e-5 robust-scale units**. Spectral bin width remains 1e-6;
numerical gate remains scaled 1e-10; direct sampling epsilon remains 1e-12.

## Batched solver and matched starts

Keep seed 82024102027 and the stored v1 seed derivation. There are 128 starts
per case, 64 maximizing and 64 minimizing; source and fresh initializations are
distinct, and all three fresh representations share raw kicks. Save numerical
initializations before search; no replacement draws after a failed retraction.
For each tile run source baseline, fresh baseline, fresh capacity, fresh repair,
regardless of source success, before moving to the next tile. All representations
for a tile use the same card: tiles 34,170,166 on GTX 1660 Ti, remaining nine on
RTX 5070. The full manifest contains 48 cases and 6,144 starts.

Use native batches B=128 on RTX 5070 and B=64 on GTX 1660 Ti. Hold transported,
centered operators resident on device, with analytic batched feature Jacobians,
target gradients and squared-RMS gradients. No spherical transform is called
inside optimization. Python loops over iterations, representations and tiles
are permitted; no per-start Python optimization loop is used. Actual memory,
latency and activity are measured; no saturation, FLOPS or 20-second case-time
guarantee is assumed. Fixed-angle software benchmarks occur before freeze.

The measured exact CPU batched SVD, including transfers, was substantially faster
than exact CUDA Jacobi or QR SVD on this hardware (22--42 ms versus 0.49--2.11 s
in the fixed fixtures). Production therefore uses a native batched CPU SVD with
eight threads for tangent/restoration factorization while feature/Jacobian and
objective batches stay on their assigned GPU. Preserve the same rank cutoff
and verify rank/projected-direction agreement before freeze. A shared lock
serializes multithreaded SVD and direct spherical transforms; no padding work or
artificial VRAM allocation is used to manufacture GPU activity.

SVD tangent rank cutoff is 1e-10 relative to the largest singular value. Initial
step norm is 0.1; at most five Newton restorations and ten step halvings. Newton
restoration targets zero current-feature residual, stopping at infinity norm
<=1e-6 robust-scale units. An improving proposal may be accepted if all residuals
are within +/-0.00049, inside the quantization bins, and RMS/angle constraints hold; record whether
the tighter center target was reached. Append squared-RMS constraint gradients
when the corresponding ratio is >=0.24. Iteratively append outward active box
faces until the projected direction is tangent or inward; avoid padding a full
identity matrix when no box faces are active. These are algorithmic constraints,
not evidence that those boundaries caused v1 stalls. Stop a start at 1,000
iterations or 50 consecutive rejected proposals. Record reasons and active-set,
center-restoration, finite-value, rank and margin diagnostics.

When a trial crosses an RMS boundary, include its positive squared-RMS excess
and squared-RMS gradient in the same five-step Newton restoration. This allows
curved feasible motion near a hard boundary; acceptance still requires each
ratio <=0.25. Do not increase the number of corrections or relax the gate.

## Independent verification and tiers

Independently recompute every started start's best feasible endpoint in NumPy.
Reconstruct its actual harmonics and directly sample all four smoothing times;
do not trust optimization matrices alone. Check input bins, RMS and angle bounds,
spectra, unchanged split difference, direct/matrix feature and derivative
agreement, and stored actual search-GPU outputs against the fixed 1e-10 gate.
Save per-row evidence, failures and incomplete verification checkpoints. Keep
the fixed largest feasible endpoint per cell without substituting a runner-up
after precision failure. For this reported cell maximum additionally evaluate
all 52 features at all four times on both actual GPUs, require a safe changed
future representation bin, and bind reconstructed coefficient bytes by SHA-256.
Full coefficients or a lossless sparse coefficient archive may be retained;
numerical bases, angles and input hashes must permit reconstruction.

Report counts of verified **cell-maximum** witnesses at raw K4 gaps >=0.010,
>=0.015, >=0.025 and >=0.050. The last retains v1's primary threshold; the new
tiers were chosen after seeing v1 and are frozen prospectively only for v2.
They are not multiples of a robust-scale bin width. Separately label per-row
CPU/direct/search-GPU qualifications; do not imply every row received both-GPU
replay unless it actually did. Publish whether each augmentation distinguishes
source witnesses and all fresh matched outcomes, failures and missing cells.
Do not compare unequal or incomplete fresh groups as completed equal-budget
experiments.

## Budget, preflight and publication

The production cap is 14,400 seconds including preprocessing, optimization and
scientific verification. Pipeline one CPU verification worker after each saved
cell alongside GPU search, retaining eight transform threads. Stop new search
work at 13,500 seconds, leaving a final 900-second verification reserve. Stop
when either recorded monotonic elapsed or UTC elapsed reaches the cap; record
both clocks and any discrepancy. Never terminate shared WSL. Checkpoint durable
row/cell evidence before stopping; unfinished starts and verifications stay
incomplete. No scientific recomputation after the cap to promote candidates.

Before freezing: test exact closure/identity controls, an exact nonclosure toy,
active RMS and box faces, center restoration, native batch-vs-row calculations,
analytic derivatives against autograd and independent finite differences,
spectral/split/transport/semigroup identities and hostile controls. Extend JoVin
with compact tests and actual outputs from both GPUs under its unchanged input
limit. Authenticate current source/test receipts, run regression and check all
prior frozen artifacts. Reuse the pinned Lean conditional proof and rotation
identities with their precise assumptions; no floating-point proof is claimed.

Freeze source, tests, protocol, input/operator hashes, initialization manifest,
software versions, current JoVin receipts and measured preflight. Publish the
actual complete or incomplete outcome, report and inspected plots through
existing PR #17 without merging. Preserve every failed artifact and exclude
raw downloads, working caches, compiler binaries and the supplied paper.
