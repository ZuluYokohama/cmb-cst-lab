# First verified v2 baseline witness (study still running)

The completed source-baseline cell for tile 24 contains one reported fixed
cell maximum, start 56, with raw 2-degree K4 gap **0.05774000454822048**
after one identical smoothing step. It passes the original **0.050** tier as
well as the prospectively declared lower v2 tiers. All 128 starts in this cell
were directly replayed; the case records zero verification failures. This is
an interim result, not the final 48-cell comparison.

All 44 recorded current baseline bins match. The minimum input-bin margin is
0.00049913902635293698 robust-scale units against the frozen
1e-5 guard. Future primary K4 bin IDs differ safely, 232 versus 1007.
Maximum both-GPU scaled discrepancy is 4.229770417660232e-14,
below the fixed 1e-10 gate. Direct sampling uses epsilon 1e-12. The four-time
spectral and unchanged-split-difference invariant checks pass. Perturbation RMS
ratios are **0.0992126045 tile** and
**0.034723044 global**, each below 0.25.

| Smoothing step u | Reference raw K4 | Candidate raw K4 | Absolute difference |
|---:|---:|---:|---:|
| 0 | -0.10073566229649224 | -0.1007356686130268 | 6.3165345592919309e-09 |
| 1 | -0.083486398084474195 | -0.025746393536253718 | 0.05774000454822048 |
| 2 | 0.00060810843727014284 | 0.1049973485478633 | 0.10438924011059315 |
| 3 | 0.072331126173542926 | 0.20423521371652903 | 0.1319040875429861 |

Both the ordinary-power derivative augmentation and connected-fourth-order
augmentation distinguish this specific pair through safely changed coordinates.
This is not universal repair or a demonstration of superiority. Fresh matched
searches are still running; their outcomes must be independently verified.

The finding concerns representation sufficiency on bounded counterfactuals
anchored to Planck measurements. These are not two newly observed skies, a new
cosmological detection, a physical ceiling or a methodological-priority claim.
The paper's conditional no-factor argument is kernel-checked separately;
floating-point sky calculations and physical interpretation are not Lean proofs.

The [concrete recorded-bin Lean specialization](../formal/ClosureFibersV2Witness.lean)
has six axiom-free kernel-checked theorems. Its
[compiler and evidence-binding receipt](../data/evidence/closure_v2_first_witness_lean.json)
links the exact current/future tables to this saved pair. Equality of retained
global records and the interpretation of the table as measurements are explicit
assumptions; the proof does not certify the floating-point transforms.

Source freeze: `ab94c88c081a95e2f1d5828d700847346a1df86a`.
[Protocol](CLOSURE_FIBERS_V2_SPEC.md),
[complete cell audit](../results/closure_fibers_v2/certificates_v2/tile_24_source_baseline/case.json),
[selected row](../results/closure_fibers_v2/certificates_v2/tile_24_source_baseline/rows/start_0056.json),
[before/after arrays and parameters](../results/closure_fibers_v2/certificates_v2/tile_24_source_baseline/rows/start_0056.npz),
[reconstructed harmonics](../results/closure_fibers_v2/certificates_v2/tile_24_source_baseline/tile_24_source_baseline_maximum_candidate_alms.npz).

Selected-array SHA256: `cf61cc8890fd1dd44d1887c60bb7589461349811ecad60fc238d47109b58f136`.
Coefficient archive SHA256: `5660c971ea41fcaeab9de32f9a62c6b543981444c873490f9f41cb4471827068`.
Selected-row SHA256: `ddb83b1587aebbfe02aa148c734b44c4c83d740e6ee6a4db1cb531ac880fc6a1`.
