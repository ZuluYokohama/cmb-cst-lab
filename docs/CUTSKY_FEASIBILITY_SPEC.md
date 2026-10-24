# Cut-sky integration H1 pilot V1

This is a separate synthetic feasibility experiment and integration profile.
It preserves all frozen v1/v2 results, numerical modules, tests and protocols.
It does not test new cosmology, a physical noise law, universal closure,
inference performance, or the merit of new K4 response features.

## Question and fixed domain

Can finite counterfactual perturbations preserve the declared **realized,
coupled binned** NaMaster spectra as well as a canonical full-sky spectral Gram
and empirical A−B difference? A certified point demonstrates feasibility for
this specified construction; no point found is an inconclusive finite search.

Use NSIDE32, lmax64, COSMO Q/U, Galactic coordinates, uK_CMB and common declared
identity response. The fixed input generator seed is 82024102027: common
Gaussian E/B sky with amplitudes [8,1.6]/(ell+1), independent split noises with
amplitude .35/(ell+1), and a common .02*sin(pixel_index) residual. Retain that
pixel residual unchanged. This is simulated input, not Planck or a calibrated
foreground/instrument simulation. Dataset generation, map I/O, analysis,
operator construction, continuation and independent verification are within
the new run budget.

The new map-facing interface explicitly uses the area-weighted synthesis
adjoint, not an inverse. It keeps the original maps and their residual, adding
the same synthesized coadd increment to both. Global spectral claims concern
the canonical retained alms; no claim is made that reanalyzing the full residual
maps gives the same spectral Gram. Raw observed A−B is also checked directly.

Build 128 stored rotation planes: four per each ell30–61, in the complement of
both E/B split-difference vectors. Preserve the entire four-field Gram at every
retained ell. Angles remain within ±pi/4. Demand at least **1%** retained-harmonic
coadd RMS perturbation, with at most **25%** globally and in mask-squared-weighted
observed coadd RMS. This finite minimum prevents arbitrarily tiny motions from
being called useful degrees of freedom.

## Realized observing constraints

Use nominal cap fractions .10/.30/.60, each with deterministic four-degree holes
and cosine apodization widths 8/12 degrees. Report actual positive-pixel
fractions and weight moments rather than treating nominal caps as effective
survey coverage. Both splits have the same scalar mask and response.

Pin pymaster3.0.1, healpy1.20.1, h5py3.16.0 and fitsio1.4.2 in this first profile;
record all actual numerical dependencies and build provenance. Purify B, do
not purify E; field and mask lmax both64, n_iter=n_iter_mask=3. Different mask
and field band limits with purification are explicitly unsupported after a
preserved upstream3.0.1 broadcast failure. No templates, anisotropic Q/U
weights, beam deconvolution or implicit pixel-window correction are used.
The apodized mask must exclude every invalid input pixel.

Retain AA, BB and AB, each with EE/EB/BE/BB, in bands [2,18),[18,34),[34,50),
[50,65): **48 recorded values**. These are binned coupled pseudo-C_ell,
not all individual multipoles or decoupled bandpowers. NaMaster's ensemble
mode-coupling relation is not an individual-realization lock.

Cache the actual masked/purified linear response to each numerical cosine/sine
basis map. With z=[1,cos(theta)−1,sin(theta)], form symmetric real quadratics
q_k=z^T Q_k z. Include original observed references in the constant coordinate;
do not replace their residual with synthesized band-limited proxies.
Fix each feature scale before continuation as
max(abs(reference_feature),1e−3*max(abs(reference_features))).
The zero-state direct/cache discrepancy must satisfy the stricter 1e−12 gate
before either tolerance case of a mask starts. A failure stops the study as an
execution qualification failure, preserving incomplete cases; it does not
establish finite-family infeasibility.

## Continuation and prospective precision comparison

For each of six masks, run four fixed directions at each of two separately
declared equality tolerances: scaled **1e−10 and 1e−12**. Total: 12 cases,
48 starts. The same global direction IDs are used at both tolerances for a
mask. Store the actual numeric family bases. Family seed derivation is
SeedSequence([seed,ell]); direction derivation is
SeedSequence([seed,profile_index*4+start]), normalized standard normals.

Use SVD tangent projection, relative rank cutoff1e−10, initial step norm0.1,
up to five Newton corrections and ten halvings. Restore toward
min(1e−12,tolerance/10), but accept only within the declared equality tolerance.
Stop after100 iterations or25 consecutive rejected proposals. Seek finite RMS
amplitude rather than a future-statistic objective. Do not tune seeds,
thresholds, constraints, masks or parameter counts after observing outcomes.
Report rank, finite continuation, stalls and all failed/incomplete starts.

Independent verification reconstructs the candidate from stored alms/bases,
adds the common perturbation to the original maps, builds full NaMaster fields
and recomputes all48 spectra. It independently checks canonical full Gram,
empirical difference, global/observed RMS, angle bounds and actual forward
values on both GPUs. CPU continuation and direct NaMaster transforms are
explicitly separate from the native batched GPU evaluator. No saturation or
global-optimization claim is made.

Current spectral quantization uses width .001 in the frozen feature-scale
units with safety margin1e−5; the equality gates are much tighter. No future
features are evaluated in H1, so a finite feasible candidate must not be labeled
a closure counterexample or discovery. The tolerance comparison measures
whether finite amplitude survives stricter locks; failures at stricter precision
remain failures and do not trigger replacement candidates.

## Budget, evidence and release

The full new job has an **1,800-second cap**, using the maximum of monotonic and
UTC elapsed time. Stop new preprocessing/search starts at1,620 seconds, reserving
180 seconds for verification. A watchdog stops only this process at the cap;
never terminate shared WSL. No duplicate, restart or budget extension is allowed.
Started IDs and solver states are saved before verification; interrupted or
failed candidates remain auditable. A new version is required for another run.
Stage writes are serialized between the watchdog and worker. Scientific end
clocks are captured before disabling the watchdog; a late completion is marked
incomplete. Subsequent saved-evidence serialization has a separate metadata
timestamp and performs no new scientific calculations.

Before freeze, run actual FITS/HDF5, independent ducc/healpy, operator-adjoint,
hostile-input, quadratic/finite-difference, finite-continuation and certificate
tests. Record native RTX128/GTX64 actual quadratic outputs and independent
NaMaster reconstruction controls. Issue authentic JoVin receipts for compact
source-extracted CPU checks and explicitly recorded GPU fixtures under the
unchanged sandbox limits. Preserve development failures. Existing24 supporting
Lean declarations remain scoped mathematical evidence; they do not prove
NaMaster arithmetic or physical validity.

Freeze source, protocol, test, dependency, preflight and receipt hashes before
the run. Bind generated input maps, family bases, Q tensors, candidate arrays
and results by SHA-256. Post-run helpers perform saved-metadata reporting,
hashing and live receipt authentication only; they may not recompute/promote
scientific candidates after the cap. Report all planned cases and their actual
outcomes, tested scope and limitations. Publish through the existing unmerged
PR, excluding dependency binaries/caches/raw downloads and the supplied paper.
