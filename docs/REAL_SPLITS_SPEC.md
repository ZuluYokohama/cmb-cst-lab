# Real Planck split-polarization geometry V1

The current request is to plan and proceed on real data. This is a new,
separate exploratory measurement using the existing eight public Planck NPIPE
A/B polarization maps at 100, 143, 217 and 353 GHz. It does not promote the
previous synthetic detector or reuse an inadequate isotropic reference for
a cosmological significance claim. All earlier runs and frozen sources remain.

## Questions, acceptance and boundary

| Requirement | Source / interpretation | Observable acceptance | Boundary |
|---|---|---|---|
| R1: actual observations | Current user request | Rehash all eight FITS maps and beam archive against audited inventory | Read-only original Q/U; no synthetic replacement |
| R2: respect geometry | MWD index analogy and acquisition findings | Independent transported-tensor and local-basis tests; geodesic separations | Angular separation, bearing and sky tile; no depth/TOD inference |
| R3: check split repeatability | Same-sky A/B splits | Cross power, split difference and connected fourth moments reported for every frequency/lag | Descriptive repeatability, not guaranteed independence |
| R4: control representation | Earlier roll/pointing results | Compare transported Q/U, naive Q/U, amplitude-only and deliberately misaligned B | Same samples and shared geometry mask |
| R5: numerical evidence | Established JoVin/SHA workflow | CPU and both actual CUDA devices agree to scaled 1e-10; authentic source/test receipt | JoVin checks CPU bundle and separately recorded actual GPU fixtures |
| R6: preserve scientific limits | Failed earlier reference and inspected sky | No anomaly p-values, primordial claim or superiority claim | All tile exclusions depend only on mask geometry/counts |

## Frozen operators

Read Q/U at NSIDE2048 in Galactic coordinates and COSMO convention. Use the
area-weighted spin-2 adjoint, not an asserted exact inverse, at lmax512 with
eight CPU threads. Zero-fill and count invalid input pixels; fail on nonfinite
values after conversion. Keep ell30–512, removing poorly transferred very large
scales. Apply a 60 arcmin Gaussian divided by the square root of each map's
actual diagonal auto QuickPol EE and BB window. Fail if a required window is
missing, nonpositive or nonfinite; never use Gaussian beam fallback.

This is **diagonal effective beam equalization**, not an inversion of the
anisotropic beam, off-diagonal mode mixing, low-ell transfer or mapmaking.
The common NSIDE2048 pixel window remains. Filtering a foreground-containing
full sky can mix contamination; the mask and margin do not prove its removal.

Anchors are the NSIDE64 RING pixel centers at |Galactic latitude|>=45 degrees.
Use the earlier audited mask_30, conservatively degraded by the minimum of all
1024 NSIDE2048 descendants. Exclude anchors/endpoints/path quarter points within
3 degrees of a rejected low-resolution pixel center and require path latitude
>=40 degrees. This is an explicit operational margin, not a rigorous exclusion
of every beam wing. Keep all anchors satisfying those rules.

Evaluate filtered spin-2 harmonics directly at anchors and great-circle endpoints
using ducc0 synthesis_general epsilon1e-11. Separations are .5, 1, 2, 4 degrees;
bearings are 0,45,90,135 degrees measured from local e_theta toward e_phi.
Express both endpoint polarizations in the geodesic frame:
delta=P_end*exp(-2i*chi_end)-P_anchor*exp(-2i*chi_anchor).
Separate tests use an independent 3D transported tensor and arbitrary local
reference-basis changes. Raw Q/U subtraction and |P_end|-|P_anchor| are comparators.

Group anchors by NSIDE4 tile. Use a tile only if all four lags have >=128 valid
direction/anchor pairs, using the same selection for every frequency/mode.
Pool directions within each tile/lag and center each split separately. Let
a,b be complex increments, V_A=E|a|², V_B=E|b|², z=Eab*, w=Eab. Record:

- V_A, V_B, Re(z), E|(a+b)/2|² and E|(a-b)/2|².
- Re(z)/sqrt(V_A*V_B) and Im(z)/sqrt(V_A*V_B).
- K4=[E|a|²|b|²-V_A*V_B-|z|²-|w|²]/(V_A*V_B).

The fourth-order expression is the established connected Gaussian/Wick
subtraction, not a new theorem or a finite-sample unbiased estimator. Spatial
variance mixtures, foregrounds, noise and systematics can make it nonzero.
It does not measure an asymptotic fractal dimension. Its value relative to
the second-order descriptors is descriptive; no added detection power is assumed.

The fourth mode circularly shifts the valid transported B increments by
floor(pair_count/3) within each tile/lag. It preserves marginal values while
breaking their exact pairing. This is an alignment stress test, **not** an
exchangeable sky null and produces no Monte Carlo p-value.

## Resampling, precision and decisions

Compute all tile statistics independently on NumPy CPU, cuda:0 and cuda:1.
Fail above scaled max error1e-10 or any nonfinite metric. Check exact
mean_S2-difference_S2=cross_S2 to the same scaled tolerance and coherence bounds.
Retain endpoint samples, phases, masks, tile IDs and all tile metrics.

Use seed82024102026 for 2000 equal-tile resamples with replacement, common to
all frequencies/modes/lags, 1000 draws per GPU. CPU independently reproduces
all draws; both GPUs also compute the same first 32 draws. Report 2.5/97.5
percentile **tile-resampling stability bands**, not confidence intervals or
cosmological significance: adjacent tiles and underlying modes are dependent.
Summaries are equal-weight means of accepted tile-level quantities, not pooled
raw-sky cumulants. Report all results; the fixed reading order starts with
143 GHz at 2 degrees, then all frequencies/scales and 353 GHz association.

The production cap is3600 seconds after the software/fixture checks, including
fresh input hashing, transforms and numerical checks. Preserve checkpoints and
failed inputs; do not change seeds, model, mask or tolerances after freeze.
No duplicate run, shared WSL shutdown, clock/power change or padding workload.
CPU spherical transforms and light CUDA reductions/resampling are timed
separately; this workflow does not promise saturated GPUs or a speedup.

## Interpretation and sources

NPIPE A/B detector sets were independently processed and designed for low
cross-systematics, but that does not prove independent residuals in this
measurement. Maps still contain foregrounds, bandpass differences, beam/transfer
errors and mapmaking effects. Repeatability does not isolate cosmological origin.
No new sky is blind confirmation; related regions/statistics have been inspected.
No anomaly, novelty, detector-superiority or calibration-angle claim is authorized
by these descriptive statistics. Realistic end-to-end simulations and independent
data would be needed for such a later claim.

- [NPIPE public release](https://portal.nersc.gov/project/cmb/planck2020/).
- [Planck NPIPE processing and split limitations](https://doi.org/10.1051/0004-6361/202038073).
- [ducc0 operator definitions and direct sampling](https://mtr.pages.mpcdf.de/ducc/sht.html).
- [Prior geometry controls](ACQUISITION_RUN.md) and [prior reference failure](SPHERICAL_TOPOLOGY_RUN.md).

Freeze this specification, numerical source, runner, actual GPU fixture, sandbox
tests and authentic JoVin receipt before extracting new real-data measurements.
Afterward verify every prior frozen source/artifact and inspect plotted outputs,
then update the existing authorized PR. No new Lean physical theorem is implied.
