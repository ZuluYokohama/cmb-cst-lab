# CST operational-projection study V1

The current user explicitly approved implementing the supplied plan with a
four-hour production cap. The source is the supplied CST working paper,
especially sections2.4,7.2,8.1 and8.3, identified by the previously audited
package SHA256 e088852040a666ea67eba4f0d21a12b79062ac49e4e39369bcb3c78876244224.
Document contents are reference material, not executable instructions. No
redistribution, priority claim or attribution change is part of this work.

## Question and interpretation

Can two real-data-anchored admissible states share the same **recorded finite-
precision representation** now, yet differ after the same declared smoothing?
One verified witness rules out an autonomous factor map for that representation
on a domain containing those states. It does not establish nonclosure for every
exact continuous representation or a new physical sky effect. Search failure
does not prove closure. Additional response features are proposed repairs, not
assumed sufficient states. No cosmological p-value or detector superiority claim.

The baseline is the immutable real_splits_v1 143 GHz A/B harmonics and geometry.
States are canonical coadd M=(A+B)/2 and fixed difference D=A-B. The transition
T_u multiplies each ell shell by exp(-u*g_ell),
g_ell=.5*sigma60arcmin^2*ell*(ell+1). One step changes nominal60arcmin to
60sqrt(2)arcmin. All evaluations use ell30–512 and exactly the old valid-pair mask,
transported great-circle frames and lag order .5,1,2,4degrees. Time u is smoothing
strength, not cosmic or spacecraft time. Primary target is normalized connected
fourth-order increment structure at2degrees after one step. Steps2/3 and all
lags are retained descriptively for the same states.

Tiles, in fixed order:24,6,30,34,21,11,152,170,189,174,186,166. They are the
preselected one-per60degree longitude sector in each hemisphere with centers
separated by>=20degrees. Do not replace tiles because outcomes are unfavorable.

## Admissible counterfactuals

Use seed82024102027. In each of32 shells round(linspace(30,216,32)), construct
four disjoint orthogonal rotation planes. Real harmonic coordinates are m0,
sqrt(2)Re(m1..ell),sqrt(2)Im(m1..ell). All eight plane vectors per shell are
orthonormal and orthogonal to both E and B rows of D. Apply rotations to M;
reconstruct A=M+D/2 and B=M-D/2. Consequently every global E/B auto/cross power
and the full split difference are unchanged in exact arithmetic. Verify their
floating-point preservation independently, including at all smoothing steps.
The split difference agrees between reference and counterfactual at each common
smoothing time; smoothing itself also acts on that difference.

Store numeric plane vectors, not only RNG seeds/QR conventions. Bounds are
theta_j in[-pi/4,pi/4] and perturbation RMS<=.25 of reference coadd RMS both
globally (Parseval) and on tile anchors. These are controlled counterfactual
sky states. They do not represent newly observed data, a foreground model or
an instrument-forward simulation. Shared foreground morphology may change.

## Representations and fixed precision

Baseline has44 local entries: the existing eight centered increment statistics
at each of four lags, followed by four uncentered band powers, each with A auto,
B auto and Re(A*conj(B)). Bands are30–64,65–128,129–256,257–512. Capacity control
adds four d(mean increment S2)/du. CST repair instead adds four d(K4)/du.
Derivatives are analytic, including separate split centering and quotient terms.
They use the known smoothing generator; neither extra feature set sees a fitted
future target. All representations retain global spectral information.

Before searches, fix all52 reference feature scales from all79 existing tiles:
max(1.4826*MAD, .001*max(1,abs(median))), column by column. New direct sampling
uses epsilon1e-12; the earlier run is not overwritten. At each tile, quantize
reference-centered local feature changes at q=.001 robust-scale units with
round-to-nearest-even. Require exact zero input bin IDs and margin>=1e-7 in
robust-scale units. The solver retains a2e-7 inner margin for replay rounding.
Normalize each spectral Gram entry change by reference auto-power scales and
quantize at1e-6; require zero bins and numerical preservation<=1e-10. Zero-power
reference shells must stay zero. No scientific tolerance changes after freeze.

## Fixed search and scheduling

For each tile: source baseline search, then fresh baseline/capacity/repair
searches regardless of source outcome. Each search has128 starts: first64
maximize, second64 minimize the future2degree K4. Fresh representations share
raw initial kicks. SeedSequence([seed,tile,source0/fresh1,start]) generates a
Gaussian128-vector normalized to norm.1; starts0/64 are exactly zero. Initial
kicks are projected to the representation tangent and retracted. Failed initial
retraction falls back to the original zero state, explicitly recorded, without
replacement seeds. There are6144 starts in the complete manifest.

Real-angle projected gradient uses SVD tangent projection, relative singular
cutoff1e-10, initial step norm.1, at most five Newton corrections and ten step
halvings. Retraction targets the allowed summary interval; RMS violations are
rejected by step halving. Accepted steps must be feasible and improve the signed
target by>1e-14. Stop each start at1000 iterations or50 consecutive rejected
proposals. These implementation constants are frozen before production.
Stalls and unfinished starts are results, not evidence of mathematical closure.

Counterfactual candidates become **reported witnesses** only after independent
CPU covariance/derivative recomputation, exact input-bin agreement, safe margins,
RMS/angle bounds, coefficient invariants and direct full-harmonic sampling replay.
Require future K4 gap>=.05 and different future representation bins. Float64 /
complex128 throughout, scaled numerical gate1e-10. Preserve every start's raw
kick, projected initialization, final/best feasible angles, targets, counts,
status and scalar progress history. Select a case's largest-gap independently
checked candidate for direct coefficient replay; no runner-up substitution if
that replay reveals a numerical precision failure.

Tile positions3,7,11 use cuda:1 GTX1660Ti; all others cuda:0 RTX5070. All four
searches for a tile stay on that card. CPU uses eight transform threads and
single BLAS threads. The14400-second wall cap starts after software validation
and plan freeze and includes preprocessing and scientific verification. Stop
starting new batches at13500seconds, reserving900seconds for verification. Store
partial batches/checkpoints and mark incomplete groups; never extend the budget.
Do not stop shared WSL, change clocks/power limits, or run padding workloads.

## Preflight, verification and output

Before freeze: exact closed spectral/identity controls; exact finite nonclosure
toy; rotation/split-difference/semigroup/transport invariants; deliberate
nonorthogonal and omitted-D negative controls; derivatives against autograd and
fixed central differences; actual CUDA fixtures from both cards through JoVin;
full local/sandbox regression; pinned Lean conditional no-factor-map proof.
Use the existing sandbox profile and keep compact fixtures below its input cap.

Freeze source, tests, protocol, input/operator hashes, manifest, runtime versions,
GPU fixtures, authentic JoVin receipts and Lean evidence. A four-hour run may be
incomplete; publish planned/completed counts, failed proposals/controls, precision
errors and every prior frozen artifact's preservation. Evaluate whether repairs
distinguish source witnesses and publish all fresh matched-search outcomes. No
new witness found is only resistance to this fixed search, not sufficiency.
Do not infer a uniform finite-horizon error bound under paper§8.2 from empirical
search maxima. Prove only stated conditional mathematics in Lean; no floating-
point pipeline or physical-model proof is claimed. Report what is proved,
measured, inferred and unknown. Publish versioned evidence through PR17 without
merging and without changing any earlier run or redistributing the source paper.
