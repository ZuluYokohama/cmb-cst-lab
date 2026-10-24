# Controlled spherical response pilot V1

## Requirement and interpretation

The user accepted combining controlled feature responses with a test of added
information beyond conventional summaries. This is a bounded method-development
benchmark, not a Planck anomaly search. The previous empirical uncleaned
full-sphere spectrum gives a substantial local spread mismatch; changing its
amplitude alone would not validate its foreground/noise assumptions.

This pilot substitutes an explicitly specified synthetic forward model whose
inputs are known. That permits a measured method comparison. It does **not**
repair or validate the observational Planck reference. No real-sky p-value or
new cosmological/fractal claim is authorized by a synthetic benchmark pass.

| ID | Acceptance example | Boundary |
|---|---|---|
| R1 | Source, protocol, independent seed blocks and JoVin receipt freeze before scientific draws | Separate response_v1 outputs; previous results remain intact |
| R2 | All rotated grids evaluate the same band-limited sphere at their own spherical coordinates | Four rotations about one fixed center, not all SO(3) orientations |
| R3 | Every baseline/response feature and final score agrees between CPU and CUDA to scaled 1e-10, with finite values | Spherical synthesis, rank preprocessing and model fitting are CPU; feature kernels/aggregation/scoring use CUDA independently |
| R4 | Fitting sees only null training draws; calibration, null validation and injected alternatives are disjoint | No test-driven model or threshold selection |
| R5 | Compare baseline and baseline-plus-response at the same nominal .05 false-alarm rate | Rejection is model discrepancy, not physical attribution |
| R6 | Preserve all six planned power comparisons and an out-of-model stress test | No seed, feature or threshold retuning to force a gain |

## Fixed forward model

Use CAMB's total lensed TT spectrum in µK², raw C_ell, through ell=256 with the
repository's PLANCK18 parameters; zero ell=0,1. Draw independent real Gaussian
harmonic shell directions conditional on those fixed powers. This is not a
fully lensed non-Gaussian CMB simulation: the lensed two-point spectrum is used
with Gaussian conditional directions. It is an explicit benchmark distribution.

Use a HEALPix NSIDE128 intermediate grid and eight transform threads. Any map
to harmonic conversion is an area-weighted adjoint, not an asserted exact
inverse. Components are then band-limited; identical downstream operators apply
to every role. The celestial temperature signal is mixed with:

- A deterministic, broad Galactic-band foreground proxy with azimuthal
  modulation; independently drawn amplitude Uniform(0,60) µK in every sky.
- An independent band-limited Gaussian white-spectrum noise field, multiplied
  on the sphere by 1+.6 sin(theta) cos(2phi), reprojected to harmonics and
  normalized to unit global RMS; amplitude Uniform(5,25) µK in every sky.

The foreground proxy is normalized to unit RMS on the full intermediate sphere.
Neither component is a calibrated physical Planck foreground/noise simulation.
The exact template formulas, synthesis, normalization and amplitudes are frozen
in the runner. These nuisance amplitudes vary in training, calibration and
matched validation, rather than being fitted from the positive controls.

Evaluate a 20-degree, 64x64 gnomonic patch centered at colatitude pi/4,
longitude .7 radians, at grid rotations 0,22.5,45,67.5 degrees about that center.
Use ducc0.sht.synthesis_general (epsilon=1e-11) directly at spherical locations,
without rotating/interpolating a raster. Multiply all observed harmonic
components by Gaussian beams of FWHM30,60,120 arcmin before evaluation.
These are continuous spherical sampling orientations with changing square
footprints; they are not exact invariance tests or physical instrument rotations.

Each view has raw and empirical normal-score representations. Rank preprocessing
uses the existing tie-aware CPU function. It is shared preprocessing, not an
independent GPU rank algorithm. Reference and CUDA feature paths receive the
same representation arrays and compute their statistics separately.

## Features and fixed learning rule

For each representation, smoothing and angle, phi has nine entries: the six
existing multiscale features at pixel lags1,2,4,8,16 and upper-excursion Euler
characteristics at standardized levels -1,0,1. Euler is V-E+F for grid vertices,
edges with both endpoints active and filled faces with all four vertices active,
divided by4096; open boundaries. It is an established Minkowski/topological
summary, not persistent homology and not a newly invented statistic.

The 17-dimensional conventional baseline at raw/30arcmin/0degrees contains:
log(std), skewness, excess kurtosis; log(S2(lag)/variance) for all five lags;
the nine phi values. Thus the baseline already includes geometry and finite-
scale statistics; the experiment tests the responses' added value, not a claim
to beat every conventional method or a power-spectrum-only straw comparator.

The 90 response values contain, for both representations:

1. At each smoothing, the mean phi over the three nonzero angles minus phi at
   angle zero: 3x2x9=54 values.
2. At angle zero, phi at each of the two larger smoothings minus phi at the
   smallest smoothing: 2x2x9=36 values.

Standardize each feature using only training means and sample SDs; SD<=1e-12
uses scale1. Fit ridge regression of standardized responses on standardized
baselines, penalty .1*n_train*I. Residualization is approximate and does not
prove independence or immunity to nuisances. Construct Mahalanobis scores for
baseline alone and for the joint baseline+response residual vector. Covariances
use fixed shrinkage .9*C+.1*I. The 17 and107 dimensional models are fixed in
advance; no alternative labels are used to fit or tune them.

## Samples, calibration, decisions

Base seed765902132026. Independent blocks: training768 (offset0), calibration999
(offset10000), matched null512 (offset20000), then128 draws for each of six
alternative groups. Alternatives are local quadratic signal coupling, compact
positive sources and a windowed filament/edge; each has amplitudes .15 and .35
relative to the original local signal RMS. Blocks use offsets30000/31000,
40000/41000,50000/51000 respectively. Within each draw the modified signal's
harmonic shell powers are renormalized to the original TT C_ell before nuisance
addition. Record every relative shell-power error, real m=0 and zero shell check.

An additional256-draw stress block (offset60000) adds an unmodeled azimuthal
stripe component of local RMS40µK to otherwise matched observations. It tests
model discrepancy; its flags are not counted as matched-null false alarms or
identified as primordial signals. It is not used for fitting or selecting a
method. Total3303 draws, batch8.

For each of the two scores use upper-tail p=(1+number calibration scores >=
test score)/1000; flag p<=.05. Report matched-null false-alarm rates and Wilson
intervals for both methods, plus nuisance-amplitude subgroup rates. The gross
calibration gate fails if either method flags more than40/512 matched nulls.
It is a diagnostic cutoff, not proof of realistic astrophysical assumptions.

For each alternative group compare paired baseline/combined flags. Compute the
one-sided exact binomial tail for combined-only wins among discordant pairs;
apply Holm correction to all six comparisons. A provisional benchmark gain
requires adjusted p<=.05, at least .10 absolute detection-rate improvement,
the matched-null gate and all numerical checks. All remaining comparisons are
reported. Paired bootstrap95% intervals (2000 resamples, seed833600+group index)
are descriptive; no seeds, thresholds or assumptions are tuned after results.

Freeze protocol, source/test hashes, software versions, roles and source-bound
JoVin receipts before draws. Retain features, scores, draw parameters, all
decisions, failures and timing envelopes. Verify authentic receipt-chain
membership and prior frozen artifacts afterward; inspect the scientific plot.
Any numerical failure preserves its batch and ends the run. Scientific failure
or no added value is a valid outcome. Any observed gain remains specific to this
synthetic distribution and these baselines; it needs broader comparisons and
independent observation checks before a novelty or physics claim.

Prior art informing this protocol: [nuisance-hardened compression](https://arxiv.org/abs/1903.01473),
[spherical scattering](https://arxiv.org/abs/2407.07007),
[model-misspecification tests](https://arxiv.org/abs/2412.15100).
Our ridge residualization is not the efficient-score construction in the first
paper, and these response contrasts are not a scattering transform. The proposed
contribution is a benchmarked combination and response test, subject to further
literature review, not a priority claim over these established methods.
