# Frozen finite-scale intermittency experiment V1

The user requests CPU/GPU exploration of fractal-like features, SHA256/JoVin
cross-checking, a GitHub PR, and Lean formalization if a meaningful mathematical
claim is identified. Here "fractal" is interpreted as finite-scale roughness
and increment intermittency. No finite band-limited patch establishes an
asymptotic fractal dimension. Established surface-scaling literature includes
[Gu & Zhou](https://arxiv.org/abs/physics/0607194); exponent identification can
fail without the correct hypotheses ([Zhou et al.](https://doi.org/10.1103/PhysRevE.87.012921)).
This implementation uses elementary structure functions, not MF-DFA, and claims
no invention of those methods or universal novelty of a feature.

## Fixed features and data

Reuse the spherical_topology_v1 conditioned spectrum, observed patches and
whole-sphere projection. Its uncleaned global spectrum produced a large local
temperature-spread mismatch. Therefore these features explicitly ignore any
uniform nonzero brightness scaling and offset; this does not remove changes
in local spectral shape, anisotropy, foreground morphology or instrument noise.

At fixed open-boundary pixel lags1,2,4,8,16, pool horizontal/vertical increments
and compute S2=mean(d²), S4=mean(d⁴), S6=mean(d⁶). Retain all scale curves.
Six predeclared features per patch, in this order:

1. OLS slope of log(S2) against log(lag), an effective roughness summary.
2. Mean over lags of S4/S2²-3, increment flatness excess.
3. Mean over lags of S6/S2³-15, sixth-order ratio excess.
4. OLS slope of log(S4/S2²) against log(lag).
5. OLS slope of log(S6/S2³) against log(lag).
6. Mean over lags of (horizontal S2 - vertical S2)/(horizontal S2 + vertical S2).

Uniform affine brightness changes cancel algebraically in these six features.
The Gaussian constants3 and15 are population benchmarks for a single centered
Gaussian increment, not exact finite-sample expectations for correlated or
heterogeneous pooled increments. Simulated full skies supply actual references.
No Hurst-to-dimension conversion or multifractal-spectrum claim is authorized.

## Evaluation before inspecting new sky features

Regenerate the same 999 reference and64 held-out skies, seed66510322026+i,
same conditioned C_ell and pixelization, with batches of eight, as the preceding
spherical run. This reuses previously inspected sky and reference model and is
exploratory method development, not independent confirmation. It does not
change earlier seeds or outputs. Add the same phase-coherent positive control.

Compute every curve independently with NumPy CPU and CUDA float64. Require
max(abs(CPU-GPU)/(1+abs(CPU))) <=1e-10 for all curves and all six features;
require every harmonic shell-power relative error <=1e-10, finite values,
zero-power shells and real m=0. Preserve any failed batch and block claims.
Record CPU/GPU timing including transfer; do not assert speedup in advance.

Use two-sided +1 Monte Carlo tails and one Holm family across all18 tests.
The minimum available raw p is .002, adjusted floor .036. Apply the identical
18-test family to each held-out control; more than8/64 flagged controls fails
the gross gate and blocks new observed-feature evaluation. Shared references
make control flags dependent; this is a diagnostic cutoff, not a binomial claim.
Report positive-control failures and all comparisons. Do not tune thresholds,
lags, summaries, seeds, conditioning spectrum or patch selection afterward.

## Evidence and limits

Freeze the new script, numerical sources, specification, source-artifact hashes,
environment and authentic JoVin receipt before evaluation. Save versioned raw
curves, features, CPU/GPU errors, control decisions and plots. Old results and
failed controls remain unchanged. Formal proof, if compiled, concerns exact
algebraic properties under explicit assumptions; it cannot certify the physical
interpretation, floating-point code for all inputs, or SHA256 collision resistance.
No collection of hashes alone establishes authorship or truth.

Candidates remain conditional-model diagnostics. This global isotropic null is
not an adequate cleaned-sky/noise model. A shape-sensitive excess may reflect
local two-point heterogeneity or foreground/noise effects. Searching additional
features on the same patches adds a broader post-hoc exploration burden beyond
the18-test correction; independent data and a fresh protocol remain necessary.
