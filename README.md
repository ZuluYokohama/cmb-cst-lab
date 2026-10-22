# CMB-CST Lab

A first-pass, independent reproduction of the differential polarization-calibration network of
**Lonappan, Keating & Arnold (2026)**, [arXiv:2609.09149](https://arxiv.org/abs/2609.09149)
(ApJL 1009, L12), on the public Planck PR4 (NPIPE) maps. It is paired with a cochain/Hodge
reading of the same network and a persistent-homology look at the real sky.

**Studio:** `studio/index.html` is a self-contained page with the K₈ network, per-map angles,
pair spectra, triangle closures, the β comparison, persistence on the real sky, and the
verification receipts. Open it in a browser.

## Summary

1. **The estimator works on public data.** Eight NPIPE A/B maps (100–353 GHz) give 28
   birefringence-blind pair angles. A gauge-fixed network fit turns them into one angle per map,
   and a Minami–Komatsu (MK) fit of the same maps gives the same pattern.
2. **The network is a cochain complex.** The paper's incidence matrix is the coboundary d₀ on K₈
   (the paper notes A·1 = 0 is the gauge). With all 56 triangles filled, H¹(K₈) = 0, so whatever
   the fit does not explain is curl: triangle-closure failure. Its weighted size is exactly the
   network fit's residual χ², and the sandbox checks that identity.
3. **β agrees with the paper, at lower precision.** The MK β is consistent with the published
   value but carries about twice its error; see *Differences from the paper*.
4. **The real sky shows no Galactic-cap asymmetry in sublevel persistence**, and the Cold Spot
   patch is unremarkable in these statistics.

## Results

Every number in the first column is computed in this repository. Paper values are quoted for
comparison.

<!-- RESULTS:START -->
| Quantity | This repository | Paper |
|---|---|---|
| Sky fraction (masks "0" / "30") | 0.920 / 0.600 | 0.92 / 0.62 |
| Network residual χ² (21 dof), f_sky 0.92 | 28.8, PTE 0.12 | not reported |
| Network residual χ² (21 dof), f_sky 0.62 | 30.7, PTE 0.08 | not reported |
| Network vs MK, χ²_diff (7 dof) | 3.27, PTE 0.86 | 8.18, PTE 0.32 |
| Largest per-map pull | 143B +0.78 | 143B +1.8 |
| Pearson r over 28 pair angles | 0.984 | 0.98 |
| MK β, f_sky 0.92 | 0.25° ± 0.23° | 0.37° ± 0.11° |
| Anchored β, f_sky 0.92 | 0.24° ± 0.23° | 0.37° ± 0.12° |
| Anchored β, f_sky 0.62 | 0.23° ± 0.23° | 0.40° ± 0.13° |
| H¹(K₈), filled complex | b₁ = 0 (rank d₀ = 7, rank d₁ = 21) | not discussed |
| NGP − SGP mean PD₀ lifetime, 143 GHz | -0.001 (sims -0.000 ± 0.030, p = 0.98) | not in paper |
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

<!-- RECEIPTS:START -->
| Suite | Result | Receipt |
|---|---|---|
| `network` | pass 11/11 | `20261022T162235Z-2682b8ed3d8e` |
| `differential` | pass 7/7 | `20261022T162238Z-56a6af847c12` |
| `mk_core` | pass 7/7 | `20261022T162240Z-d35036cacbf4` |
| `persistence` | pass 6/6 | `20261022T162243Z-47da1c40da3c` |
<!-- RECEIPTS:END -->

The harness counts one more item than each file defines (its own collection check).

Steps that need ducc0 or the full-resolution maps cannot run in the sandbox. They are covered by
local `pytest`, including an end-to-end injection on synthetic HEALPix maps: per-map rotations
plus a common β are applied in map space, and the pipeline recovers them. Over 12 injected skies
(N_side 256, |b| > 20° cut) the pair-angle pulls have standard deviation 0.90 and mean −0.02, and
the network χ² averages 19.4 for 21 dof. The Gaussian errors are unbiased and about 10%
conservative under a cut sky.

## Reproduce

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
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
  their cross-covariance from 200 Wishart draws of the bandpowers.
- **Persistence:** 20° patches of the 143 GHz (A+B)/2 temperature map at the Galactic caps and the
  Cold Spot, PD₀ statistics, and 200 Gaussian skies from the measured spectrum.

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
                 theory.py (CAMB, pixel window), persistence.py, data.py
scripts/         run_reproduction.py, build_studio.py, jovin_verify.py
tests/           pytest (local) and tests/sandbox (JoVin suites)
studio/          template.html -> index.html
results/         reproduction.json, persistence.json
data/evidence/   jovin_receipts.json
```

## Credits

The estimator, likelihood and analysis choices follow Lonappan, Keating & Arnold (2026).
`mk_likelihood.py` is a numpy port of the model in their MIT-licensed pipeline,
[antolonappan/cosmic_birefringence](https://github.com/antolonappan/cosmic_birefringence).
The data are ESA Planck PR4 (NPIPE) via NERSC and the Planck Legacy Archive.
