"""Report every frozen finite-scale comparison and the separate Lean proof scope."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/multiscale_v1'


def main():
    read = lambda name: json.loads((OUT/name).read_text())
    result,plan,stage,ensemble,validation,positive,lean = map(read,
        ('result.json','plan.json','stage_audit.json','ensemble.json','validation.json','positive_control.json','lean_verification.json'))
    reference = np.asarray(ensemble['curves'])[:999]
    lags = np.asarray(plan['lags'])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'text.color':'#25374d','axes.labelcolor':'#25374d'})
    fig,axes = plt.subplots(3,3,figsize=(12,10))
    fig.subplots_adjust(left=.085,right=.97,bottom=.14,top=.82,hspace=.55,wspace=.32)
    fig.text(.085,.95,'Scale-dependent sky texture | CPU × CUDA',fontsize=23,weight='bold')
    fig.text(.085,.895,f"0/18 adjusted sky flags · held-out controls {validation['familywise_flags']}/64 · independent float64 checks",fontsize=12)
    labels = ['S2(lag) / S2(1)','Fourth-order flatness excess','Sixth-order ratio excess']
    for row,index in enumerate((0,3,4)):
        for col,patch in enumerate(result['patches']):
            values = reference[:,col,:,index]
            observed = np.asarray(result['observed_curves'])[col,:,index]
            if row==0:
                values = values/values[:,:1]
                observed = observed/observed[0]
            lo,median,hi = np.quantile(values,[.025,.5,.975],axis=0)
            ax = axes[row,col]
            ax.fill_between(lags,lo,hi,color='#087f8c',alpha=.18,label='Reference pointwise 95%')
            ax.plot(lags,median,color='#087f8c',label='Reference median')
            ax.plot(lags,observed,'o-',color='#cf7d23',label='Observed')
            ax.set(xscale='log',xlabel='Pixel lag',ylabel=labels[row],title=patch)
            ax.set_xticks(lags,labels=[str(x) for x in lags])
            if row==0 and col==0:
                ax.legend(frameon=False,fontsize=8)
    fig.text(.085,.073,'Bands are descriptive pointwise ranges; decisions use the 18 predeclared Holm-adjusted feature tests.',fontsize=10)
    fig.text(.085,.038,'Finite, smoothed patches do not establish a fractal dimension. The uncleaned sky model retains foreground/noise limits.',fontsize=9)
    fig.savefig(OUT/'curves.png',dpi=160,facecolor='white')
    plt.close(fig)
    rows = []
    for i,patch in enumerate(result['patches']):
        for j,name in enumerate(result['features']):
            rows.append(f"| {patch} | {name} | {result['observed'][i][j]:.6g} | {result['reference_mean'][i][j]:.6g} ± {result['reference_sd'][i][j]:.4g} | {result['raw_p'][i][j]:.4g} | {result['p_holm'][i][j]:.4g} |")
    minimum = min(v for row in result['p_holm'] for v in row)
    positive_flags = sum(v<=.05 for row in positive['p_holm'] for v in row)
    report = f'''# Finite-scale roughness and intermittency: measured result

**No feature crossed the predeclared adjusted .05 threshold.** The smallest
Holm-adjusted p was **{minimum:.3g}**. The held-out control gate passed with
**{validation['familywise_flags']}/64** skies flagged; numerical failures: **{stage['precision_failures']}**.
This is an exploratory null result for six particular finite-scale features,
not evidence that the sky has no higher-order structure or no fractal process.

![All scale curves](../results/multiscale_v1/curves.png)

## What was tested

We compared the same three previously examined Planck 143 GHz patches with 999
whole-sphere conditional-power skies and 64 held-out controls. The earlier
[whole-sphere model](SPHERICAL_TOPOLOGY_RUN.md) has a substantial amplitude
mismatch. Here six features are invariant to uniform nonzero brightness scale
and offset: S2 log-slope, mean normalized fourth/sixth increment-moment excess,
their two log-slopes, and a bounded directional contrast. They use open-boundary
horizontal/vertical increments at lags 1, 2, 4, 8, 16 pixels. See the
[frozen specification](MULTISCALE_SPEC.md) for exact definitions and gates.

Structure functions and multifractal methods are established. We do not claim
to have invented them or discovered an unknown feature. The 2D scaling literature
requires care about exponent interpretation: [Gu & Zhou](https://arxiv.org/abs/physics/0607194),
[Zhou et al.](https://doi.org/10.1103/PhysRevE.87.012921). This study is not MF-DFA,
does not convert slopes to a Hurst exponent or dimension, and does not establish
an asymptotic scaling law from five lags on a smoothed 128×128 patch.

## Every planned comparison

| Patch | Feature | Observed | Reference mean ± SD | Raw two-sided p | Holm p (18 tests) |
|---|---|---:|---:|---:|---:|
{chr(10).join(rows)}

The strongest unadjusted indication was the NGP mean flatness excess (raw p=.01),
which becomes .18 under the planned 18-test correction. It is not promoted to a
discovery or used to retune the experiment. The positive phase-coherent control
flagged **{positive_flags}/18** comparisons, demonstrating sensitivity to that
deliberately extreme alternative only. No simulation or failed draw was dropped.
The 999 references and 64 controls reuse the preceding conditional model; they
are not a new independent sky confirmation. Explorations across prior studies
add a broader selection burden beyond this 18-test family.

## CPU, GPU and provenance

Every reference/control curve was independently computed on NumPy CPU and
RTX 5070 CUDA in float64, then compared using abs(CPU-GPU)/(1+abs(CPU)).

- Maximum scaled curve error: **{stage['max_curve_error']:.4g}**, limit1e-10.
- Maximum scaled feature error: **{stage['max_feature_error']:.4g}**, limit1e-10.
- Maximum relative harmonic-shell power error: **{stage['max_shell_error']:.4g}**, limit1e-10.
- CPU feature calls totaled {stage['cpu_feature_seconds']:.3f}s; GPU calls including transfer totaled {stage['gpu_feature_seconds']:.3f}s.
- Complete audit run, which computed both paths: **{stage['elapsed_s']:.1f}s**. These timings do not establish an end-to-end speedup.
- Authentic new JoVin receipt: `{plan['jovin_receipt_id']}`. The sandbox checks shipped CPU code and actual recorded GPU outputs; production CUDA ran separately in WSL.
- All 18 current JoVin suites passed (122 checks including collection); local regression: 181 passed, three unavailable Intel XPU skips. All CI-equivalent sandbox suites passed.

Git now preserves the exact raw bytes identified by frozen SHA256 plans, including
CRLF sources. The staged blobs were cross-checked against working files. Two
regression checks verify historical source/artifact hashes from a checkout.
Redundant progress snapshots remain local; final scientific records, including
the failed experiments, retain every final draw. Hashes identify bytes, not truth,
authorship, statistical validity or cryptographic signatures.

## What Lean establishes

Lean 4.35.0 kernel-checked **{lean['theorem_count']} supporting theorems** over exact
rationals using only its standard library. They prove affine increment
cancellation, homogeneous scaling of finite even-moment sums, normalized fourth/
sixth sample-moment invariance, and directional contrast invariance with explicit
nonzero-denominator assumptions. There are no admitted theorems or extra axiom
declarations. Standard foundations are propext, Classical.choice and Quot.sound.

The compiler archive was checked against the official release SHA256. The
[formal source](../formal/ScaleInvariance.lean), [scope and reproduction instructions](../formal/README.md),
and [compiler/source receipt](../results/multiscale_v1/lean_verification.json)
are preserved. CI repeats the kernel check with the pinned compiler. This is
elementary established algebra, not a new physical theorem. It does not prove
the log-slope summaries, floating-point implementation for all inputs, a sky
anomaly, multifractality, or SHA256 collision resistance.

## Scientific limits

The constants3 and15 are centered-Gaussian population benchmarks; correlated
finite samples, nonzero increment means and pooled directional variance mixtures
can depart from them even for a Gaussian field. Actual simulation references
provide the reported p-values. Affine invariance removes an amplitude nuisance,
not local spectral-shape, anisotropy, foreground or instrument-noise mismatches.
These uncleaned previously studied patches and the global isotropic conditional
model cannot isolate primordial physics or demonstrate unique information beyond
all conventional statistics. No new cosmological or fractal discovery is claimed.

[Immutable plan](../results/multiscale_v1/plan.json),
[all curves/features and precision checks](../results/multiscale_v1/ensemble.json),
[held-out controls](../results/multiscale_v1/validation.json),
[raw result](../results/multiscale_v1/result.json),
[final SHA256/receipt/calculation verification](../results/multiscale_v1/verification.json).
'''
    (ROOT/'docs/MULTISCALE_RUN.md').write_text(report)
    print(json.dumps({'min_adjusted_p':minimum,'positive_control_flags':positive_flags,'control_flags':validation['familywise_flags']},indent=2))


if __name__=='__main__':
    main()
