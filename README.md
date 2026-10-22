# CMB-CST Lab

A first-pass, independent reproduction of the differential polarization-calibration network of
**Lonappan, Keating & Arnold (2026)**, [arXiv:2609.09149](https://arxiv.org/abs/2609.09149)
(ApJL 1009, L12), on the public Planck PR4 (NPIPE) maps. It is paired with a cochain/Hodge
reading of the same network and a persistent-homology look at the real sky.

**Website:** https://zuluyokohama.github.io/cmb-cst-lab/ (GitHub Pages, served from `docs/`).
It is the same self-contained studio as `studio/index.html`: a page with the K₈ network, per-map angles,
pair spectra, triangle closures, the β comparison, persistence on the real sky, and the
verification receipts. Open it in a browser.

## Summary

1. **The estimator works on public data.** Eight NPIPE A/B maps (100–353 GHz) give 28
   birefringence-blind pair angles. A gauge-fixed network fit turns them into one angle per map,
   and a Minami–Komatsu (MK) fit of the same maps gives the same pattern.
2. **The network is a cochain complex.** The paper's incidence matrix is the coboundary d₀ on K₈
   (the paper notes A·1 = 0 is the gauge). With all 56 triangles filled, H¹(K₈) = 0, so there is
   no harmonic part, and what the fit does not explain is triangle-closure failure. Because the
   fit is weighted by the edge covariance C, that residual is a covariance-weighted coexact
   component (C⁻¹r ∈ im d₁ᵀ), not a Euclidean curl. The weighted closure χ² equals the
   network fit's residual χ², and the sandbox checks both statements. This edge-space curl is a
   graph-closure diagnostic; it is unrelated to CMB B-modes.
3. **β agrees with the paper, at lower precision.** The MK β is consistent with the published
   value but carries about twice its error; see *Differences from the paper*.
4. **Persistence on the real sky (exploratory).** No Galactic-cap asymmetry or lifetime anomaly
   appears. However, the PD₀ feature count is high in all three patches (NGP, SGP, Cold Spot)
   relative to the current Gaussian simulations. A shared offset like that points first to a
   simulation/data mismatch (noise anisotropy, foregrounds, pipeline), and it is unexplained
   until the validation track runs; see `results/persistence.json`.

## Results

Every number in the first column is computed in this repository. Paper values are quoted for
comparison.

<!-- RESULTS:START -->
| Quantity | This repository | Paper |
|---|---|---|
| Sky fraction (masks "0" / "30") | 0.920 / 0.600 | 0.92 / 0.62 |
| Network residual χ² (21 dof), f_sky 0.92 | 28.8, PTE 0.12 | not reported |
| Network residual χ² (21 dof), f_sky 0.62 | 30.7, PTE 0.08 | not reported |
| Network vs MK, χ²_diff (7 dof) | 3.32, PTE 0.85 | 8.18, PTE 0.32 |
| Largest per-map pull | 143B +0.70 | 143B +1.8 |
| Pearson r over 28 pair angles | 0.984 | 0.98 |
| MK β, f_sky 0.92 | 0.25° ± 0.23° | 0.37° ± 0.11° |
| Anchored β, f_sky 0.92 | 0.24° ± 0.23° | 0.37° ± 0.12° |
| Anchored β, f_sky 0.62 | 0.23° ± 0.23° | 0.40° ± 0.13° |
| H¹(K₈), filled complex | b₁ = 0 (rank d₀ = 7, rank d₁ = 21) | not discussed |
| NGP − SGP mean PD₀ lifetime, 143 GHz | -0.001 (sims +0.001 ± 0.030, p = 0.92) | not in paper |
| Spatial stationarity of the calibration pattern, 12 equal-area regions (primary) | χ² 70.9 / 77, PTE 0.67 | new (not in paper) |
| Same-frequency A/B edges across regions, 12 equal-area regions (primary) | χ² 45.8 / 44, PTE 0.397 | new (not in paper) |
| Spatial stationarity of the calibration pattern, 8 latitude bands (robustness) | χ² 67.9 / 49, PTE 0.04 | new (not in paper) |
| Same-frequency A/B edges across regions, 8 latitude bands (robustness) | χ² 49.0 / 28, PTE 0.008 | new (not in paper) |
| Dust EE/BB, ℓ < 300 (ΛCDM-subtracted) 143/217/353 | 1.41 / 1.40 / 1.40 | 1.41 / 1.37 / 1.42 |

Per-map angles (network, gauge α_143A = 0, f_sky 0.92):

| Map | 100A | 100B | 143A | 143B | 217A | 217B | 353A | 353B |
|---|---|---|---|---|---|---|---|---|
| α (deg) | -0.32 ± 0.10 | -0.50 ± 0.09 | 0 (gauge) | +0.13 ± 0.06 | -0.10 ± 0.05 | -0.12 ± 0.05 | -0.25 ± 0.05 | -0.27 ± 0.06 |
<!-- RESULTS:END -->

## Verification

The numerical kernels are numpy-only on purpose, so they can be checked in the **JoVin** sandbox:
a network-less container on WSL with Python 3.12 + numpy, 512 MB RAM and a 10 s limit. The
verifier extracts the exact shipped functions with `ast` (never hand-copied), runs each suite, and
appends a hash-chained receipt to `data/evidence/jovin_receipts.json`.

| Suite | What it pins down |
|---|---|
| `network` | d₁d₀ = 0, A·1 = 0, H¹ = 0 on the filled K₈, gauge-invariant GLS, closure χ² = residual χ², curl covariance, χ²₂₁ calibration, χ² survival function |
| `differential` | exact pair-angle recovery, blindness to β and sky EB, covariance calibrated over 40 Wishart skies |
| `mk_core` | the likelihood port matches the reference model rows to 1e-14, exact α/β recovery, agreement with the network |
| `persistence` | β₀ from PD₀ equals flood-fill counts; the PD₁ lower-bound behaviour is pinned |
| `accel` | the bootstrap's numpy kernels, and **Gram matrices recorded on this machine's GPU and NPU**, recomputed in float64 from the same normals (see *Hardware acceleration*) |

<!-- RECEIPTS:START -->
| Suite | Result | Receipt |
|---|---|---|
| `network` | pass 12/12 | `20261022T182616Z-55c98a3fc8f4` |
| `differential` | pass 7/7 | `20261022T171103Z-7469960ebbb7` |
| `mk_core` | pass 8/8 | `20261022T171106Z-676f9cad61b0` |
| `persistence` | pass 8/8 | `20261022T182619Z-703f2a4eadd6` |
| `accel` | pass 10/10 | `20261022T172121Z-5cd7a09a18e0` |
| `spatial` | pass 8/8 | `20261022T174604Z-cea3f175e138` |
<!-- RECEIPTS:END -->

The harness counts one more item than each file defines (its own collection check).

**Continuous integration.** `.github/workflows/verify.yml` runs on every push and pull request,
on Ubuntu and Windows. It runs the local tests, and all sandbox suites under plain pytest via
`scripts/run_sandbox_suites.py`, using the same `ast` bundles as JoVin. CI has no JoVin harness
and no GPU, so it issues no receipts and skips the XPU tests. The recorded GPU/NPU attestation
in the `accel` suite is still recomputed there.

Steps that need ducc0 or the full-resolution maps cannot run in the sandbox. They are covered by
local `pytest`, including an end-to-end injection on synthetic HEALPix maps: per-map rotations
plus a common β are applied in map space, and the pipeline recovers them. Over 12 injected skies
(N_side 256, |b| > 20° cut) the pair-angle pulls have standard deviation 0.90 and mean −0.02, and
the network χ² averages 19.4 for 21 dof. The Gaussian errors are unbiased and about 10%
conservative under a cut sky.

## Hardware acceleration

The bootstrap behind χ²_diff draws 2,000 Wishart samples of the bandpowers and refits both
estimators on each one. Two changes make that practical on a laptop:

- **GPU draws (`src/cmbcst/accel.py`).** On an Intel Arc GPU (PyTorch XPU; developed on an Arc
  140V) the normals Z and the Gram matrices ZZᵀ/ν, a 16 × ~5·10⁴ product per bin, are formed in
  float32, because Xe2 has no fast float64. The 16 × 16 transform L·(ZZᵀ/ν)·Lᵀ stays in float64
  on the CPU. Each run first recomputes a batch from the same normals in float64 and stops if the
  float32 result differs by more than 10⁻⁵ relative (measured: ~3·10⁻⁶). Without an XPU device
  the same draws run in numpy.
- **Batched likelihood (float64, CPU).** The MK likelihood evaluates all 72 bins in one batched
  pass instead of a per-bin loop. It is about 10× faster, equal to the loop to 1e-12 (sandbox
  suite `mk_core`), and the fit reproduces the previous one to 2·10⁻¹⁰ degrees.

Measured per draw: 41 ms on the Arc GPU against 414 ms in numpy, and 0.2 s per MK fit against
~2 s before. The likelihoods stay in float64 throughout; only the Gram products use the GPU.
**Device attestation (`scripts/gpu_attest.py`).** The sandbox cannot reach a GPU or NPU, but it
can check their output. The attestation script builds normals with a counter-based generator
(splitmix64 + Box–Muller, pure integer arithmetic, so the sandbox regenerates them bit for bit).
It runs ZZᵀ/ν at three real Planck bin sizes (ν up to ~5·10⁴) on each accelerator and records
the raw results in `tests/sandbox/jovin_tests_accel.py`. The `accel` suite recomputes everything
in float64 and checks each device's error and its accept/reject decision:

| Device | Max relative error vs float64 | Decision (limit 10⁻⁵) |
|---|---|---|
| Intel Arc 140V GPU, PyTorch XPU, float32 | 3.2e-06 | used |
| Intel AI Boost NPU, OpenVINO | 6.8e-04 | not used (reduced-precision arithmetic, 68× over the limit) |

The suite also asserts the GPU result is not a float64 copy (error > 10⁻⁹), so the receipt shows
the GPU's own float32 arithmetic. The GPU's random-number stream cannot be replayed in the
sandbox; its statistics are tested locally in `tests/test_accel.py`.

## Reproduce

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/xpu   # optional, Intel Arc GPU
.venv/Scripts/python -m cmbcst.data                 # ~17 GB: 8 NPIPE maps, 3 PLA products, beam tarball
.venv/Scripts/python scripts/run_reproduction.py    # masks -> alms -> spectra -> transfer -> analyse -> persistence
.venv/Scripts/python scripts/build_studio.py        # studio/index.html + README tables
.venv/Scripts/python -m pytest                      # local tests (ducc0, injection)
.venv/Scripts/python scripts/jovin_verify.py        # sandbox suites (needs the JoVin harness)
```

On Windows, Python 3.12 with numpy, scipy, astropy, ducc0 and camb is enough; healpy and
NaMaster are not needed.

## Method

- **Maps:** the eight PR4 NPIPE detector-split maps at N_side 2048. Each file header confirms its
  split, e.g. `HFI_SkyMap_100-1-4` is built from `npipe6v20A_100_map`.
- **Masks:** the polarized point-source union (100–353 GHz) × a CO(2-1) cut, giving f_sky 0.92.
  The second mask adds the PLA Galactic cut whose combination lands nearest f_sky 0.62.
- **Spectra:** masked spin-2 transforms (ducc0, ℓ ≤ 1520) and all 16×16 field cross-spectra, in
  72 bins of width 20 over 51 ≤ ℓ < 1491.
- **Pair angles:** tan 2Δα_ij = D_ij / S_ij, with D = C^{B_iE_j} − C^{E_iB_j} and
  S = C^{E_iE_j} + C^{B_iB_j}, fitted by reweighted least squares with Gaussian covariance. The
  joint 28×28 covariance comes from linear propagation. Beams and mode coupling cancel in the ratio.
- **Network:** GLS on d₀ with α_143A = 0, triangle closures d₁Δα, and the Hodge split into
  gradient and curl.
- **MK likelihood:** a numpy port of the reference model, with 4 dust-EB amplitudes from ψ_ℓ
  (353 GHz TE/TB), NPIPE QuickPol beam windows, and a simulated pseudo-C_ℓ transfer for the
  ΛCDM term. Fisher scoring gives the point estimate and covariance; a profile-likelihood scan
  over β confirms σ.
- **χ²_diff:** the network and MK patterns on the 7-dimensional differential subspace, keeping
  their cross-covariance from 2,000 Wishart draws of the bandpowers (drawn on the GPU when one
  is available; see *Hardware acceleration*).
- **Persistence:** 20° patches of the 143 GHz (A+B)/2 temperature map at the Galactic caps and the
  Cold Spot, PD₀ statistics, and 1,000 Gaussian skies from the measured spectrum.

## Differences from the paper

- **Spectra:** pseudo-C_ℓ divided by ⟨w²⟩ instead of NaMaster mode-coupling deconvolution. The
  pair angles are insensitive to this. For the MK theory term, a mask transfer measured from 8
  ΛCDM skies per mask is applied instead: ≈ 1 for EE, and 1.3–4.7 for BB (E→B leakage).
- **Masks:** rebuilt from PLA products, because the paper's masks are not distributed. Only the
  sky fractions were matched; nothing downstream was tuned.
- **Fitting:** Fisher scoring instead of MCMC. The profile likelihood (with ln det C) gives the
  same σ_β.
- **β precision:** σ_β is about twice the paper's, and β moves with the dust-EB model (0, 1 or 4
  amplitudes; `mk.beta_by_dust_eb_amplitudes` in `results/reproduction.json`). The rebuilt masks,
  the pseudo-C_ℓ approach and the ψ_ℓ template are the likely causes. The differential pattern,
  which is the paper's central test, is not affected.

## Layout

```
src/cmbcst/      network.py (cochain complex, GLS, closures), differential.py (pair angles),
                 mk_likelihood.py (MK port), spectra.py (ducc0 SHTs), masks.py, beams.py,
                 theory.py (CAMB, pixel window), persistence.py, accel.py (GPU bootstrap), data.py
scripts/         run_reproduction.py, build_studio.py, jovin_verify.py, gpu_attest.py,
                 run_sandbox_suites.py (sandbox suites under plain pytest, used by CI)
tests/           pytest (local) and tests/sandbox (JoVin suites)
studio/          template.html -> index.html
docs/            GitHub Pages copy of the studio (generated by build_studio.py)
results/         reproduction.json, persistence.json
data/evidence/   jovin_receipts.json
```

## Credits

The estimator, likelihood and analysis choices follow Lonappan, Keating & Arnold (2026).
`mk_likelihood.py` is a numpy port of the model in their MIT-licensed pipeline,
[antolonappan/cosmic_birefringence](https://github.com/antolonappan/cosmic_birefringence).
The data are ESA Planck PR4 (NPIPE) via NERSC and the Planck Legacy Archive.
