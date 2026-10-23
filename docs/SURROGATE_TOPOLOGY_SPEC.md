# Exploratory topology beyond two-point summaries

The current user requests a search for additional structure beyond ordinary
methods. The lab already reports elevated PD0 counts in three temperature
patches, with Galactic contamination an unresolved explanation. We will test
that known observation against stronger, patch-specific surrogate controls.
This is exploratory analysis of an already-seen sky, not blind confirmation.

The established methods are [CMB persistent homology](https://arxiv.org/abs/1712.08159)
and [IAAFT surrogates](https://arxiv.org/abs/chao-dyn/9909041). This implementation
extends the latter to 2D patches. It does not claim invention of those methods.

## Fixed experiment

Use 143 GHz (A+B)/2 temperature, Gaussian smoothing 10 arcmin, ell <= 1000,
synthesis NSIDE512. Use the same fixed 128x128, 20-degree NGP, SGP and Cold Spot
patches as the earlier analysis, with open boundaries and per-patch standardization.
Do not cut compact-source holes into patches: those holes manufacture topology.

For each patch, generate 999 independently initialized 2D IAAFT surrogates, 500
iterations, seed 2026102307. Each iteration projects onto the observed Fourier
amplitudes, then maps ranks to the exact observed sorted pixel values. Retain
both final stages: S has exact 2D Fourier amplitudes, H has exact pixel histogram.
Their pixel ordering is the same (for unique values), so PD0 feature counts agree.
Both constraints cannot generally be satisfied exactly by one finite field.

Report three statistics per patch: finite PD0 count; total finite PD0 persistence
in H; total finite PD0 persistence in S. Use the existing two-sided Monte Carlo
tail convention with +1 correction and Holm correction across all nine tests.
Flag adjusted p <= .05 as an exploratory candidate only if quality/calibration
gates pass. Report all nine comparisons, not only the smallest p.

Require S amplitude relative RMS error <=1e-10, H histogram exactly equal,
H amplitude relative RMS error <=.01, S histogram Kolmogorov distance <=.01,
and matching S/H PD0 counts for every surrogate. Retain failures; do not discard
draws or tune the seed, iteration count, thresholds or patch selection afterward.
Also retain conventional mean/std/skewness/kurtosis and mismatch diagnostics.

Before examining new real-sky summaries, check 64 synthetic Gaussian fields of
size32x32 against 199 surrogates each, using the same 500 steps and fixed seeds.
Use isotropic smooth Gaussian fields and their monotonic exp(0.5*x) transforms
in alternating cases. Check family-wise Holm p across the three statistics.
More than eight false flags among 64 is a gross-calibration failure. This small
synthetic check does not prove exact significance on a real, anisotropic sky.
Check quality for these controls too and report its failure rate separately.

## Verification and acceptance

| Requirement | Observable check |
|---|---|
| Exact shipped PD0 behavior | Match existing diagrams on random and plateau fields, including boundaries |
| Marginal constraint | Sorted H equals sorted target bit for bit |
| Spectral constraint | S Fourier amplitude matches target within the fixed float64 tolerance |
| Accelerator parity | CUDA first projection/rank agrees with NumPy on identical deterministic inputs |
| Multiple comparisons | All three patches and all nine adjusted p-values retained |
| Honest null | Describe IAAFT as an algorithmic surrogate ensemble, not an exact conditional distribution |
| Source/data freeze | Save immutable plan with code/input hashes before synthetic and real evaluation |
| Scientific limits | Flat-patch Fourier power differs from spherical C_ell; no foreground/noise/instrument attribution |

The standard comparators are intentionally fixed or closely matched in the
surrogates. A topological difference would motivate a follow-up on phase/shape
structure; it would not establish new cosmology, uniquely identify compact
sources, or show superiority over every conventional higher-order statistic.
