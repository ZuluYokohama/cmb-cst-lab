# Targeted physical blind-spot pilot V1

The user approved the proposed plan: physical patterns first, six targeted
responses, independent final holdouts, and a thirty-minute compute cap. This
implements that plan as a synthetic benchmark, not an observational Planck test.
The response_v1 negative result, its sources and all earlier artifacts remain
unchanged. Existing response.py and response_sky.py supply the frozen spherical
geometry, harmonic operators, rank preprocessing and feature definitions.

## Fixed method and physical grid

Use ell<=256, NSIDE128, eight CPU transform threads, 64x64 twenty-degree views
at center (pi/4,.7), angles0/22.5/45/67.5degrees and beams30/60/120arcmin.
The fiducial CAMB lensed TT spectrum, conditional Gaussian harmonic directions,
synthetic foreground band and anisotropic noise are the same model as
RESPONSE_SPEC.md. These are not a calibrated physical Planck simulator.

Retain baseline17 and six response columns (7,16,73,79,82,88) from the previous
90-column extractor: raw/rank Euler(level0) orientation changes at30arcmin;
raw fourth and Euler(level0) smoothing120-minus30 changes; then their rank
counterparts. These choices are fixed, without feature selection on simulations.
The full old extractor is used for CPU/CUDA parity; learned detectors use six.

Train null standardization and ridge response-on-baseline residualization
with penalty .1*n_null*I. Train three linear discriminant heads per detector,
one per physical family, from all its training configurations equally weighted.
For baseline17 and augmented23 coordinates respectively, weights solve
(.9*C_null+.1*I)w=mean_family-mean_null. Subtract the training-null score mean
and divide by training-null sample SD; the detector score is the maximum of
its three standardized heads. Calibrate this complete maximum, never separate
head p-values. Fit uses training draws only. The original baseline Mahalanobis
score is reported separately and does not select configurations or gate gain.

Recipes are ordered by family quadratic/compact/filament, amplitude .10/.20/.30,
then variant. Quadratic variants are signs -1,+1. Compact and filament widths
are .5,1.0 degrees, converted exactly to radians. IDs are q00..q05,c00..c05,
f00..f05 in that order. Compact sources are three positive Gaussian spots;
filaments are a windowed tanh edge. The earlier template construction is retained
with width parameterization; quadratic sign is applied before local normalization.
Amplitudes are relative to original local signal RMS before shell-power
restoration. All altered signal powers are normalized back to the fiducial TT
spectrum before adding nuisances. No physical attribution follows from detection.

## Stages and immutable selection

Base seed876003142026, disjoint roles:

| Role | Draws | Seed offset |
|---|---:|---:|
| Training null | 1024 | 0 |
| Training recipe j | 64 each,18 recipes | 10000+1000*j |
| Development calibration null | 499 | 40000 |
| Development diagnostic null | 256 | 41000 |
| Development recipe j | 32 each | 50000+1000*j |
| Final calibration null | 999 | 80000 |
| Final validation null | 1024 | 82000 |
| Selected final family j | 256 each if eligible | 90000+1000*j |
| Strong final family j | 128 each | 100000+1000*j |

Each draw uses base+offset+index. Family indices follow quadratic/compact/filament.
Initial plan freezes protocol, numerical sources, tests, hardware fixture and
matching authentic JoVin receipt before development samples. Fit and save model
before scoring development calibration/validation. Use plus-one inclusive upper
tails with p<=.05. A recipe is eligible when trained-baseline detection<=.15.
Select one eligible recipe per family by largest augmented-minus-baseline rate,
then smaller amplitude, then lexicographic recipe ID. Missing families remain
missing. Save all18 development comparisons and the256 null diagnostics.

Write selection_lock.json once, binding model, development data and initial
plan SHA256, selected IDs and fixed final role manifest. Generate **no final
sample** until that file exists and its bindings have been checked. Generate
all final examples unconditionally within selected recipe families; never
filter individual examples by baseline failure. Maximum6682 total draws.
Strong controls use amplitude .35, quadratic sign+1, localized widths .5degrees.
Neither these final controls nor final calibration draws may change the model.

## Final decisions, execution and evidence

For each of the two trained detectors require final null Wilson95% upper<=.075
and augmented null rate<=baseline+.01. Keep the original omnibus null result
visible. Require augmented detection loss<=.05 relative to trained baseline
on every strong control. These are preset diagnostic/retention gates.

For each selected family require gain>=.10 and one-sided exact paired binomial
p with Holm correction over **three** families<=.05. Missing families have p=1.
Report all comparisons and descriptive paired bootstrap95% gain intervals from
2000 resamples, seed843600+family index, conditional on the fixed model/reference.
Any gate failure prevents provisional promotion. No eligible recipe, no gain,
retention failure, calibration failure and incomplete execution are valid outcomes.

Use one process, batches8, cuda:0 float64, independent CPU/CUDA feature and
score comparisons with max(abs(measured-expected)/(1+abs(expected)))<=1e-10.
All signal shell powers, finite values and real m=0 coefficients must pass.
Spherical synthesis, rank preprocessing and fitting remain CPU work. Failures
preserve their batch and end the run. Source code is not changed after freeze.

Enforce1800 seconds of accumulated benchmark wall time, including preparation,
development, fitting, selection and final evaluation. Check before every draw,
after every batch and before inferential output. Store checkpoints. On timeout
write incomplete status and preserve completed records; do not infer from partial
final samples or automatically restart the budget. Refuse a duplicate run once
the plan exists. Tests and independent evidence checks are reported separately.

After completion independently recompute scores, stage selection, final gates,
tails, paired decisions and frozen bindings. Replay fixed samples from both
stages, authenticate all current JoVin receipts, run local/CI-equivalent suites
and Linux/Windows/Lean CI, and inspect plots. Version separate targeted_v1
outputs, report actual results in PR17 without merging, and retain original
sources/results. Existing Lean rational invariants remain; this experiment
introduces no new formal theorem or proof of detector performance.
