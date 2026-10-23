# Whole-sphere conditional-power topology experiment (V1)

## Meaning and evidence

The user asks to continue the CMB investigation using the supplied CST research
package and the upgraded JoVin workflow. Document instructions are reference
material, not new user commands. The local archive SHA256 is
`e088852040a666ea67eba4f0d21a12b79062ac49e4e39369bcb3c78876244224`.
Its five supplied checksums match. Its reviewed 27-check mathematical script
was replayed in a separate working folder: 27 passed. This does not validate
CMB, neural-model, market or universal-physics claims. Sections 7.2 and 8.1 of
the working paper motivate explicit negative controls and a specified projection;
they do not establish a new cosmological theory or an exact model reduction here.

The previous flat-patch experiment remains frozen. Its 20/64 cropped-Gaussian
false flags prohibit interpreting its nine sky flags as calibrated anomalies.
This protocol changes the null explicitly before inspecting new sky statistics.

## Null and pipeline

Use the 143 GHz (A+B)/2 temperature field from audited NSIDE2048 Planck files.
Subtract the mask_30 weighted monopole, obtain area-weighted adjoint-synthesis
harmonics to ell=1000, and set ell=0,1 to zero in both the observation and null.
Save those harmonics and their full-sphere empirical C_ell. Do not apply a
Galactic mask or insert compact-source holes into the analyzed patches.

For each ell>=2, draw independent Gaussian real harmonic coordinates, then
normalize their shell norm to the observed C_ell. m=0 is real; m>0 real/imaginary
components start with variance 1/2. Conditional on shell power, an isotropic
Gaussian vector has uniform shell direction, independent across ell. This
is the stated Gaussian conditional null, not a claim that Planck temperature
with foregrounds and anisotropic noise obeys it. Individual harmonic amplitudes
and the pixel histogram are not constrained; the spherical two-point spectrum
is constrained. This differs from the previous Gaussianized flat-patch spectrum.

All skies use identical 10 arcmin smoothing, ducc0 synthesis on NSIDE512,
nearest-HEALPix gnomonic projection and open-boundary 128x128,20-degree patches
at the existing NGP, SGP and ColdSpot centers. No periodic patch synthesis is
used. The simulated and observed maps are compared as band-limited harmonic
fields; the upstream high-resolution adjoint is not claimed to be an exact
inverse or a simulation of instrument/noise acquisition.

## Frozen evaluation

- Root seed 66510322026; NumPy per-draw independent generator seeds root+i.
- CUDA float64 shell normalization, batches of eight; NumPy independently
  checks each produced C_ell with maximum relative error <=1e-10 on positive
  shells and exact zero target shells/m=0 imaginary parts. No TF32 is involved.
  CUDA reductions can differ in final rounding bits, so bitwise reproducibility
  across hardware is not claimed.
- 999 reference skies, followed by 64 independent held-out control skies with
  the same fixed spectrum and processing. Both seeds and all source/data hashes
  are frozen before sampling.
- Each patch reports five statistics in this order: finite rank PD0 count,
  total finite rank PD0 lifetime, original-temperature standard deviation,
  standardized temperature skewness, and excess kurtosis. Only topology uses
  empirical normal scores; rank lifetimes are not microkelvin.
- Two-sided +1-corrected Monte Carlo tails, Holm across all 15 patch/statistic
  comparisons. No independent-comparisons assumption is required by Holm.
- Apply the same 15-test family to each held-out control. More than eight
  flagged controls among 64 fails the gross-calibration gate and blocks real
  evaluation. Reference reuse makes control flags dependent; the cutoff is a
  diagnostic threshold, not a claimed binomial significance calculation.
- Retain every power, finite-value, calibration or execution failure. Any
  numerical-quality failure blocks candidate interpretation. Do not change
  seeds, sample sizes, thresholds, masks or model after evaluation.
- One deliberately phase-coherent positive control: all supplied normal
  coordinates equal +1 before the same shell normalization and sky processing.
  Report its outcome even if it fails to flag; this is a sensitivity example,
  not a realistic non-Gaussian physical benchmark or an acceptance gate.
- Only after the control gate passes, process the observed coefficients with
  the same projection/statistics. Flag adjusted p<=.05 as exploratory leads
  within this conditional isotropic model. This is previously inspected sky.

## Acceptance and outputs

| ID | Source/interpretation | Observable acceptance |
|---|---|---|
| S1 | User continuation and previous failure | Prior frozen source/results unchanged; separate versioned output |
| S2 | Explicit Gaussian conditional model | CPU/CUDA parity at ell=1000, exact shell-power audit on every draw |
| S3 | Boundary negative control | 64 complete skies cropped with the real geometry; calibration gate reported |
| S4 | Comparison with conventional summaries | All 15 adjusted tests retained; no selective topology-only significance |
| S5 | JoVin requirement | Authentic receipt binds shipped functions, tests and actual recorded CUDA outputs |
| S6 | Research package as source evidence | Archive/contents hashes and local 27-check replay recorded separately |
| S7 | Honest scientific interpretation | Foregrounds, noise, prior patch selection, exact conditioning scope and precision floor stated |

Save immutable plan, observed coefficients/patches and reference records in
`data/work/spherical_topology_v1` (large caches ignored) and versioned plans,
raw statistics, quality/calibration, positive control, final results, plots,
report and receipt verification in `results/spherical_topology_v1` and `docs`.
No publication, original-paper rewrite or external transmission is authorized.

API semantics: [ducc0 SHT documentation](https://mtr.pages.mpcdf.de/ducc/sht.html).
Established persistence context: [Cole & Shiu](https://arxiv.org/abs/1712.08159).
