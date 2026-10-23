# GPU acquisition-geometry experiment V1

The user approved planning and execution of a known-sky acquisition experiment,
asked for substantial GPU work, and selected a one-hour campaign cap after
software validation. Their MWD analogy motivates explicit angular position,
detector incidence, orientation, beam footprint and derived reconstruction
indices. None of these indices is inferred radial depth.

## Model and scope

This first acquisition test is a512x512 periodic local tangent plane spanning
20degrees, with a fixed Cartesian Stokes basis. It is not the spherical Planck
forward model, a full spin-2 isotropic CMB realization, or actual Planck TOD.
It introduces a new, separately versioned model rather than modifying old runs.

Eight synthetic detectors observe each raster position. Their incidence angles
are0,10,20,30,40,50,55,60degrees; beam ellipse angles are22.5*h degrees;
polarization angles20+22.5*h degrees. Gains1+.02*sin(2*pi*h/8), polarization
efficiency .95, pointing offsets .75*[cos(2*pi*h/8),sin(2*pi*h/8)] pixels.
Each Gaussian beam has minor sigma1.2pixels and major sigma1.2/cos(incidence).
This explicit projected-footprint formula is a toy instrument response, not a
measured Planck beam. Desired reconstruction has a circular sigma3pixel beam.

Independent Gaussian local I/Q/U fields have a red Fourier filter
(1+(frequency/.04)^4)^(-1/2), zero DC and radial band limit .2cycles/pixel;
their source RMS values are100,5,5µK before the common reconstruction beam.
Independent detector white noise has sigma .5µK. Work uses orthonormal real FFTs,
with Nyquist modes outside the retained band to preserve Hermitian reality.

At each frequency, d_h=g_h*beam_h*pointing_phase_h*
[I+rho_h*(Q*cos(2psi_h)+U*sin(2psi_h))]+noise_h.
Fit a3-component least-squares inverse using all eight detectors, then apply
the common beam. Reject nonpositive eigenvalues or Gram condition>1e8; never
substitute a pseudoinverse. Static geometry/conditioning are prepared on CPU;
inverse weights are independently solved on each GPU in chunks256 to bound
cuSOLVER workspace. Production sky/noise generation, transforms, forward mixing,
reconstruction and metrics remain GPU resident until small metric transfers.

Reconstruct identical observations using four assumptions:

1. Correct geometry.
2. Polarization roll offset +10degrees; all other metadata correct.
3. Ignored pointing offsets; all other metadata correct.
4. Circular sigma1.2pixel beams; all other metadata correct.

The global roll changes Q/U while preserving polarization amplitude. This is
a known basis degeneracy, not a novel cosmological discovery. Wrong pointing
or differential beam assumptions can convert intensity structure into spurious
polarization; this experiment measures that effect in the declared toy model.

## Workload, resource policy and indices

Use cuda:0 RTX5070 for196608 noisy polarized skies and cuda:1 GTX1660Ti for65536,
concurrently. Each GPU also runs256 independent noise-free polarized controls
and256 noisy intensity-only controls:263168 total skies, eight detector maps
per sky, four reconstructions. The split follows the approximately3:1 throughput
ratio measured before science. Use float64/complex128 throughout; no TF32 or
reduced-precision scientific path. Preflight chooses the largest tested batch
whose conservative allocation estimate and measured tensor peak stay below
75% of measured available memory. Freeze those batches and hardware names.

Each production batch has a device-local Torch generator with seed
986004152026+device_index*100000+batch_index. Noise-free controls use seed
986004152026+1000000+device_index*10000+batch_index; intensity-only controls
use +2000000+device_index*10000+batch_index. Freeze counts, seeds, batch shapes,
versions, source bytes, GPU fixture, protocol and authentic JoVin receipt before
science. GPU RNG reproducibility is scoped to the recorded backend/device;
full-resolution audit input arrays are published for independent CPU checks.

Store metrics per sky, reconstruction mode and I/Q/U/amplitude component in
compressed NPZ files. Store batch ID, seed and row ranges. Export32 labeled
timeline rows from the first actual observation, with sample index, synthetic
raster-time index, detector channel, angular x/y, pointing offsets, incidence,
beam angle, polarization angle and measured µK. Raster time spacing1ms is an
explicit artificial cadence, not spacecraft or drilling telemetry.

Collect nvidia-smi utilization, memory, temperature and power once per second,
and CUDA-event elapsed time, throughput and allocated/reserved memory peaks.
Measure actual use; do not promise100% utilization or maximum power and do not
pad the run with meaningless loops. Do not change clocks/power limits or stop
shared WSL. The complete production campaign, audits and final checks must stop
at3600 seconds; checkpoints and incomplete status survive interruption/timeout.
Refuse duplicates once plan.json exists. Development profiling/tests are outside
the campaign and separately recorded. An earlier development workspace OOM is
preserved locally; it changed batching of static solves, not model assumptions.

## Checks and decision rules

- Noise-free correct-geometry absolute I/Q/U RMSE<=1e-10µK for every control.
- Every production/control metric finite; polarization-amplitude difference
  between correct and roll-error reconstructions <=1e-10 scaled max error.
- On each GPU the mean correct-reconstruction I/Q/U noise MSE agrees within2%
  with sigma²/Npixels times the weighted sum of squared inverse coefficients.
  RFFT weights are2 for interior columns,1 for DC/Nyquist columns. This tests
  the exact declared noise model, not realistic Planck noise adequacy.
- One complete512-grid input per GPU is retained, independently computed in
  NumPy, and checked against production GPU metrics and maps to scaled1e-10.
  The primary input is also reconstructed on the secondary GPU at the same
  tolerance. Small real CUDA fixtures on both GPUs are checked by JoVin.
- Poor orientation coverage is an expected negative control: rank-deficient
  reconstruction must be rejected. Preserve unexpected failures and inputs.
- Report wrong-geometry MSE, polarization leakage and correct/noisy comparisons
  without introducing a data-dependent anomaly threshold. No physical claim
  follows from the expected response of this constructed instrument.

Freeze formal/PolarizationGeometry.lean with two exact rational identities:
polarization norm invariance and invariance of a co-rotated projection under
c²+s²=1. Compile with the existing SHA-pinned Lean4.35.0 and preserve compiler
output/axiom lists. These proofs do not establish floating-point correctness,
optics validity, reconstruction conditioning or physical novelty.

Afterward authenticate all current JoVin receipts and source/test hashes,
verify prior frozen artifacts, inspect scientific and performance plots, and
publish actual results in the existing PR without merging. Keep all original
Planck and failed analyses intact. A software/numerical/precision failure,
weak GPU utilization, or lack of new scientific information must remain visible.
