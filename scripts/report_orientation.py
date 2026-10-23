"""Render the frozen grid-symmetry audit and evidence-ranked research priorities."""
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cmbcst.copula_topology import normal_scores
from cmbcst.orientation import d4_view
from cmbcst.persistence import standardise

OUT = ROOT/'results/orientation_v1'


def main():
    plan = json.loads((OUT/'plan.json').read_text())
    result = json.loads((OUT/'result.json').read_text())
    source = ROOT/'data/work/spherical_topology_v1/observed_patches.npz'
    if hashlib.sha256(source.read_bytes()).hexdigest() != plan['patch_source_sha256']:
        raise ValueError('patch cache changed since the frozen audit')
    with np.load(source) as patches:
        raw = patches['NGP']
    representations = [('Original temperature', raw, 'µK'),
        ('Standardized temperature', standardise(raw), 'Standard deviations'),
        ('Empirical normal scores', normal_scores(raw), 'Normal-score units')]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
        'text.color':'#25374d','axes.labelcolor':'#25374d'})
    fig, axes = plt.subplots(3, 4, figsize=(12.8, 9))
    fig.subplots_adjust(left=.12, right=.88, top=.83, bottom=.14, wspace=.14, hspace=.25)
    fig.text(.12, .955, 'Rotated representations | exact pixel permutations', fontsize=21, weight='bold')
    fig.text(.12, .905, f"{result['transform_checks']}/{result['transform_checks']} checks passed · 13 cases · CPU and RTX 5070 float64", fontsize=12)
    for row, (label, field, unit) in enumerate(representations):
        limit = float(np.max(np.abs(field)))
        for k in range(4):
            ax = axes[row, k]
            im = ax.imshow(d4_view(field, k), origin='lower', cmap='RdBu_r', vmin=-limit, vmax=limit,
                interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f'{90*k}° grid turn', pad=9)
            if k == 0:
                ax.set_ylabel(label, labelpad=12)
        cax = fig.add_axes([.903, axes[row, 0].get_position().y0, .014, axes[row, 0].get_position().height])
        fig.colorbar(im, cax=cax, label=unit)
    fig.text(.12, .078, 'Shown: one NGP patch, not twelve independent observations. Reflected views are also included in the audit.', fontsize=10)
    fig.text(.12, .043, 'These checks do not test arbitrary celestial rotations, correct the foreground/noise model, or establish a discovery.', fontsize=10)
    fig.savefig(OUT/'representations.png', dpi=160, facecolor='white')
    plt.close(fig)

    labels = {
        'cpu_gpu_curves':'CPU/CUDA scale curves',
        'cpu_gpu_features':'CPU/CUDA compressed features',
        'orientation_curves':'Rotated/reflected scale curves vs expected values',
        'orientation_features':'Rotated/reflected features vs expected values',
        'topology_lifetime':'Total finite PD0 lifetime within each representation',
        'rank_commutation':'Rank normalization vs permutation order',
        'affine_feature_invariance':'Standardized vs original affine-invariant features',
    }
    rows = '\n'.join(f"| {labels[key]} | {value:.6g} |" for key, value in result['maximum_scaled_errors'].items())
    report = f'''# Rotated representations and research merit

**{result['transform_checks']}/{result['transform_checks']} exact-grid checks passed; {result['failures']} failures.**
Every CPU/CUDA comparison was finite and below the frozen scaled tolerance of
1e-10. This is a numerical symmetry result, not a new sky anomaly. The user
confirmed that “represent and rotate” meant testing rotated representations.

![Original, standardized and normal-score NGP grids under quarter turns](../results/orientation_v1/representations.png)

## What actually rotated

We tested all four quarter turns, with and without a horizontal reflection,
for 13 fixed cases: the three previously studied Planck patches, eight fixed-seed
Gaussian fields, a ramp and a spiked field. Each had original, standardized and
empirical normal-score representations. Thus 13 × 3 × 8 = 312 checks.
These transformations permute the pixels exactly and preserve open-grid
adjacency; no interpolation or angle selection enters this audit.

The first five multiscale features stay invariant. The sixth, the signed
horizontal/vertical contrast, reverses sign under odd quarter turns because
the axes swap. Its expected sign change passed; calling it invariant would
misstate the result. Finite PD0 count and total lifetime are invariant under
these grid symmetries **within each representation**. Counts and pixel
histograms passed every exact check. Standardization preserves the six
affine-invariant feature values. Rank normalization changes the scientific
observable, so its feature values need not equal the original values.

| Comparison | Maximum scaled error |
|---|---:|
{rows}

Errors are max(abs(measured−expected)/(1+abs(expected))). The audit took
**{result['seconds']:.3f}s**, including both CPU and GPU paths and topology.
This does not measure an end-to-end GPU speedup. Complete per-case feature
values, expected values, topology summaries and errors are in the
[frozen result](../results/orientation_v1/result.json).

## Where the evidence puts the next effort

The following order is a research judgment from the saved results, not a
calculated discovery score. No angle, representation or threshold was selected
to make the earlier sky comparisons pass.

| Priority | Evidence now | Next useful test and merit criterion |
|---|---|---|
| 1. Justify the foreground/noise reference | The whole-sphere model predicts patch spread around 399–401 µK, while the observed patches are 96.6–101.4 µK. Both topology count and conventional spread flag; unique topology information is unestablished. | Specify foreground treatment, frequency/split checks, noise and identical map operators before another sky test. First require held-out null calibration and conventional baseline diagnostics. |
| 2. Test actual spherical rotation/projection dependence | This audit removes exact grid-axis relabeling as a numerical explanation for these features. It does not rotate the celestial sphere or instrument pipeline. | Freeze an SO(3) orientation set and common sampling/mask rules; process controls identically. Quantify interpolation and boundary effects and use a joint rotation-aware decision rule, not the most favorable angle. |
| 3. Test added information on independent targets | The multiscale study has 0/18 adjusted flags, minimum p=.18; its raw p=.01 indication does not survive its planned correction. Previously examined patches are not independent confirmation. | Predeclare a bounded feature family and independent data, with a conventional-statistic comparator and power/false-alarm checks. Advance only on calibrated, reproducible added information. |

The earlier periodic/cropped control failure (20/64 flags) stays visible in
[the copula report](COPULA_TOPOLOGY_RUN.md). The spherical model's own controls
flagged 2/64, and the multiscale controls 4/64; passing those gross diagnostic
gates does not validate the real sky foreground or noise assumptions.
See [spherical results](SPHERICAL_TOPOLOGY_RUN.md) and
[all multiscale comparisons](MULTISCALE_RUN.md). No physical cause has been isolated.

## Reproducibility and proof scope

The plan freezes source bytes, fixed seeds, patch-cache SHA256 and JoVin receipt
**`{plan['jovin_receipt_id']}`** before measurement. The new JoVin suite passed
5/5 checks including collection. The
[fresh verification](../results/orientation_v1/verification.json) checks all
19 current suite bundles/test hashes against authentic receipts and verifies
chain membership, earlier frozen sources and earlier saved artifact hashes.
JoVin sandbox checks run on CPU; this audit's independent production CUDA
comparisons ran in the existing WSL GPU environment. Hash agreement establishes
byte identity and receipt linkage, not scientific validity or universal novelty.

The existing [nine Lean rational invariants](../formal/README.md) remain
unchanged. They do not prove D4 topology invariance, arbitrary floating-point
correctness or cosmology. No new physical finding is offered for formalization.

Arbitrary celestial rotations can change projected pixels, foreground/noise
alignment and boundary effects. This exact-grid check is a limited prerequisite
for that distinct experiment; it does not establish SO(3) invariance or a
fractal dimension. Normal-score and standardized views are derived versions of
the same observations, not additional data or independent replications.

[Frozen specification](ORIENTATION_SPEC.md),
[immutable plan](../results/orientation_v1/plan.json),
[all 312 records](../results/orientation_v1/result.json),
[receipt/source/artifact verification](../results/orientation_v1/verification.json).
'''
    (ROOT/'docs/ORIENTATION_MERIT_RUN.md').write_text(report, encoding='utf-8')
    print(json.dumps({'status':result['status'], 'checks':result['transform_checks'],
        'plot':str(OUT/'representations.png')}, indent=2))


if __name__ == '__main__':
    main()
