# Rank-geometry follow-up to the failed matched-surrogate pilot

The frozen v1 pilot is retained under results/surrogate_topology_v1. It failed
its prespecified gross-calibration gate: 16 of 64 control examples flagged, all
16 in the 32 monotonic-Gaussian cases; zero of 32 Gaussian controls flagged. Sky
evaluation was blocked. These are surrogate-comparison flags before the separate
per-surrogate matching-quality requirements; they are not accepted sky findings.

This v2 protocol changes the scientific question explicitly: does the geometry
of pixel ranks differ from a Gaussian-copula surrogate, conditional on the
normal-score field's 2D Fourier amplitudes? It first maps empirical midranks to
normal quantiles. Strictly monotonic brightness changes then produce exactly the
same input; the skewness nuisance that broke v1 is removed by construction.

This is a method-development follow-up, not a repeat of v1 with looser thresholds.
V1 source, outputs and failure remain intact. V2 uses disjoint new control seeds,
root 49310702026. Its 64 held-out controls still alternate Gaussian fields and
exp(0.5*x) transforms, with 199 surrogates each. No v2 control or new sky statistic
has been inspected when this protocol is saved.

Keep all v1 settings otherwise: 500 iterations, 999 real surrogates per patch,
three fixed 128x128 temperature patches, two matched stages S/H, the same three
PD0 statistics, .01matching tolerances, .05Holm-adjusted candidate threshold,
nine sky comparisons, and at most 8 flags among 64 controls. Both exact-spectrum
and exact-histogram stages are retained. Run synthetic validation first; block
sky evaluation if its fixed gate fails. Report matching failures separately.

Save raw standardized patches as well as normal-score patches. No source holes
are introduced. NGP/SGP/ColdSpot projection geometry was checked without reading
pixel values: at NSIDE 512 each 128x128 patch samples 16384 distinct HEALPix pixels.

Interpretation is limited to Gaussianized patch two-point structure and rank
topology. This is not a test conditional on the original spherical C_ell, not a
proof of superiority over all conventional higher-order statistics, and not a
new cosmological detection. Flat-patch boundaries, nonstationarity, foregrounds,
noise and IAAFT's approximate algorithmic distribution remain material limits.
