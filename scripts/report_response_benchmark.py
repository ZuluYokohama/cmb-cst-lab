"""Show all fixed power comparisons, false alarms and scientific limitations."""
import hashlib
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cmbcst.copula_topology import normal_scores
OUT = ROOT/'results/response_v1'
read = lambda p:json.loads(Path(p).read_text())


def interval(k,n):
    z = 1.959963984540054
    center = (k/n+z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/(1+z*z/n)
    return center-half,center+half


def main():
    plan,result,validation,stage,ensemble = [read(OUT/name) for name in
        ('plan.json','result.json','validation.json','stage_audit.json','ensemble.json')]
    comparisons = result['comparisons']
    colors = ['#087f8c','#cf7d23']
    names = ['Baseline','Baseline + responses']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'text.color':'#25374d','axes.labelcolor':'#25374d'})
    fig,axes = plt.subplots(2,3,figsize=(13,8.7))
    fig.subplots_adjust(left=.075,right=.97,bottom=.14,top=.80,wspace=.37,hspace=.55)
    fig.text(.075,.95,'Do controlled responses add detection power?',fontsize=23,weight='bold')
    fig.text(.075,.90,f"3,303 synthetic skies · matched null flags {validation['flags'][0]}/512 vs {validation['flags'][1]}/512",fontsize=12)
    fig.text(.075,.855,f"Frozen outcome: {result['status'].replace('_',' ')}",fontsize=11)
    for col,(kind,label) in enumerate((('quadratic','Local quadratic coupling'),('compact','Compact positive sources'),('filament','Windowed filament / edge'))):
        records = [r for r in comparisons if r['kind']==kind]
        ax = axes[0,col]
        for method in range(2):
            rates = np.array([r['rates'][method] for r in records])
            bounds = np.array([interval(r['flags'][method],r['draws']) for r in records]).T
            xx = np.array([r['amplitude'] for r in records])+(-.006 if method==0 else .006)
            ax.errorbar(xx,rates,yerr=np.maximum(0,np.stack([rates-bounds[0],bounds[1]-rates])),fmt='o-',
                color=colors[method],capsize=3,label=names[method])
        ax.set(title=label,xlabel='Injection / original local signal RMS',ylabel='Detection fraction',ylim=(0,1.05),xticks=[.15,.35])
        if col==0:
            ax.legend(frameon=False,fontsize=8,loc='best')
    ax = axes[1,0]
    rates = np.asarray(validation['rates'])
    bounds = np.asarray(validation['wilson_95']).T
    ax.bar([0,1],rates,color=colors,width=.55)
    ax.errorbar([0,1],rates,yerr=np.maximum(0,np.stack([rates-bounds[0],bounds[1]-rates])),fmt='none',ecolor='#25374d',capsize=4)
    ax.axhline(.05,color='#25374d',ls='--',lw=1,label='Nominal .05')
    ax.set(title='Held-out matched null',ylabel='False-alarm fraction',xticks=[0,1],xticklabels=['Baseline','+ responses'],ylim=(0,max(.12,float(bounds.max())*1.2)))
    ax.legend(frameon=False,fontsize=8)
    ax = axes[1,1]
    ax.bar([0,1],result['stripe_stress']['rates'],color=colors,width=.55)
    ax.set(title='Omitted stripe component',ylabel='Discrepancy detection fraction',xticks=[0,1],xticklabels=['Baseline','+ responses'],ylim=(0,1.05))
    ax.text(.03,.94,'Flags do not identify the physical cause',transform=ax.transAxes,fontsize=8,va='top')
    ax = axes[1,2]
    gain = np.array([r['gain'] for r in comparisons])
    bounds = np.array([r['paired_gain_bootstrap_95'] for r in comparisons]).T
    yy = np.arange(6)
    ax.barh(yy,gain,color=[colors[1] if r['provisional_gain'] else '#85939f' for r in comparisons])
    ax.errorbar(gain,yy,xerr=np.maximum(0,np.stack([gain-bounds[0],bounds[1]-gain])),fmt='none',ecolor='#25374d',capsize=3)
    ax.axvline(0,color='#25374d',lw=1)
    ax.set(title='All six paired comparisons',xlabel='Added detection fraction',yticks=yy,
        yticklabels=[r['group'].replace('_',' ') for r in comparisons])
    fig.text(.075,.067,'Intervals: descriptive 95%; paired gain decisions use all six exact paired tests with Holm correction and a ≥.10 gain.',fontsize=9)
    fig.text(.075,.035,'Synthetic forward model with modeled foreground/noise variation. No observational Planck hypothesis test or cosmological discovery.',fontsize=9)
    fig.savefig(OUT/'benchmark.png',dpi=160,facecolor='white')
    plt.close(fig)

    source = ROOT/'data/work/response_v1/examples.npz'
    if hashlib.sha256(source.read_bytes()).hexdigest()!=result['example_cache_sha256']:
        raise ValueError('example cache changed')
    with np.load(source) as examples:
        fields = examples['matched_null']
    rows = [('Original / 30 arcmin',fields[0],'µK'),('Smoothed / 120 arcmin',fields[2],'µK'),
        ('Normal scores / 30 arcmin',np.stack([normal_scores(f) for f in fields[0]]),'Normal-score units')]
    fig,axes = plt.subplots(3,4,figsize=(12.8,9))
    fig.subplots_adjust(left=.12,right=.88,bottom=.14,top=.83,wspace=.14,hspace=.25)
    fig.text(.12,.955,'Same simulated sphere | controlled sampling changes',fontsize=21,weight='bold')
    fig.text(.12,.905,'First held-out matched-null draw; fixed 20° patch center; direct spherical-harmonic evaluation',fontsize=11)
    limit_raw = float(np.max(np.abs(fields)))
    for row,(label,views,unit) in enumerate(rows):
        limit = limit_raw if row<2 else float(np.max(np.abs(views)))
        for col,angle in enumerate(plan['angles_degrees']):
            ax = axes[row,col]
            im = ax.imshow(views[col],origin='lower',cmap='RdBu_r',vmin=-limit,vmax=limit,interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            if row==0:
                ax.set_title(f'{angle:g}° sampling turn')
            if col==0:
                ax.set_ylabel(label,labelpad=12)
        position = axes[row,0].get_position()
        cax = fig.add_axes([.903,position.y0,.014,position.height])
        fig.colorbar(im,cax=cax,label=unit)
    fig.text(.12,.078,'Rotations are about one fixed center. Square footprints and directional increments change; no exact invariance is assumed.',fontsize=9)
    fig.text(.12,.043,'All displayed maps derive from one synthetic observation with the same foreground/noise realization.',fontsize=10)
    fig.savefig(OUT/'sampling.png',dpi=160,facecolor='white')
    plt.close(fig)

    table = '\n'.join(f"| {r['group']} | {r['flags'][0]}/128 | {r['flags'][1]}/128 | {100*r['gain']:+.2f} pp | [{100*r['paired_gain_bootstrap_95'][0]:+.2f}, {100*r['paired_gain_bootstrap_95'][1]:+.2f}] pp | {r['raw_p']:.5g} | {r['p_holm']:.5g} | {'yes' if r['provisional_gain'] else 'no'} |" for r in comparisons)
    gains = [r['group'] for r in comparisons if r['provisional_gain']]
    outcome = ('Provisional added detection power appeared in '+', '.join(gains)+'.') if gains else 'No comparison met the complete predeclared added-power rule.'
    weakest = min(comparisons,key=lambda row:row['gain'])
    loss_note = (f"The largest measured loss was {weakest['group']}: baseline {weakest['flags'][0]}/128 versus augmented {weakest['flags'][1]}/128, a {100*weakest['gain']:.2f} percentage-point change.") if weakest['gain']<0 else ''
    calibration_note = ('The augmented Wilson 95% interval lies above the nominal 5% rate. Passing the gross cutoff must not be described as precise false-alarm calibration; finite-reference variability and model/score adequacy still need assessment.') if validation['wilson_95'][1][0]>.05 else ''
    timing = ensemble['timings']
    report = f'''# Controlled spherical response benchmark V1

**Outcome: {result['status']}.** {outcome}
This is a synthetic method-development result. No observational Planck
hypothesis test was performed, and no new physical or fractal discovery is claimed.

{loss_note}

![Every power comparison and held-out check](../results/response_v1/benchmark.png)

## The proposed method

The user authorized combining controlled feature responses with a test of added
information. We measured geometry at four spherical sampling orientations,
three smoothing scales and two representations of the same synthetic sky.
The 90 fixed response contrasts summarize how six established finite-scale
features and three excursion Euler characteristics change under those operators.
The 17-value baseline already includes moments, two-point increment statistics,
the six finite-scale features and the three Euler values in the original view.

Ridge regression, fitted only to 768 null training draws, predicts responses
from baseline features. The augmented detector scores the joint baseline and
response residual vector, using fixed covariance shrinkage. Residualization
alone does not prove independence, nuisance immunity or added information;
the held-out comparison measures the latter within this benchmark.

![Spherical orientations and representations of the first validation draw](../results/response_v1/sampling.png)

The locations are actual gnomonic coordinates on a sphere, at 0,22.5,45,67.5°
around one fixed center, evaluated directly from band-limited harmonics rather
than by rotating a raster. Rotated square footprints and directional increments
change. This is a response experiment, not an exact symmetry test or coverage
of all SO(3) orientations. All components receive the same smoothing/sampling.

## Forward model and withheld observations

All 3303 skies use a fiducial CAMB lensed TT spectrum through ell 256, with Gaussian
harmonic directions conditional on fixed shell powers. This is not a full
non-Gaussian lensed CMB simulator. Each observation includes an explicit synthetic
Galactic-band foreground proxy with amplitude Uniform(0,60)µK and anisotropic
band-limited noise with amplitude Uniform(5,25)µK. These vary in training,
calibration and matched validation. The three injected alternatives are local
quadratic coupling, compact positive sources and a windowed filament/edge.
Their modified signal shell powers are restored before nuisance addition.
Full formulas, seeds and limits are in the [frozen protocol](RESPONSE_SPEC.md).

This controlled forward model replaces the inadequate observational reference
**for the benchmark only**. The previous Planck model's local spread mismatch
remains unresolved. None of these synthetic nuisance templates establishes a
physical Planck noise/foreground fit; observational testing remains withheld.

## Calibration and every planned comparison

Training 768, independent calibration 999, matched null 512, six alternative groups
of 128, and an omitted-stripe stress test 256. Both methods use upper-tail
plus-one Monte Carlo p-values against their own calibration score distribution,
with nominal p<=.05. They share data and calibration draws, allowing paired
comparison; the finite realized false-alarm rates need not be identical.

- Baseline null flags: **{validation['flags'][0]}/512 ({validation['rates'][0]:.3%})**;
  Wilson 95% [{validation['wilson_95'][0][0]:.3%}, {validation['wilson_95'][0][1]:.3%}].
- Augmented null flags: **{validation['flags'][1]}/512 ({validation['rates'][1]:.3%})**;
  Wilson 95% [{validation['wilson_95'][1][0]:.3%}, {validation['wilson_95'][1][1]:.3%}].
- Gross calibration gate: **{validation['status']}**, at most 40 flags for each
  method. Nuisance-amplitude subgroup counts are retained in validation.json.

{calibration_note}

| Injection | Baseline detected | + responses detected | Gain | Paired bootstrap 95% | Exact paired p | Holm p (6) | Provisional gain |
|---|---:|---:|---:|---|---:|---:|---|
{table}

The gain rule requires the matched-null gate, all numerical checks, at least
10 percentage points of added detection and Holm p<=.05 across all six paired
tests. Intervals condition on the fitted models and reference; they do not
include uncertainty from retraining/recalibration. No failed draw was dropped,
and no seed, feature, amplitude, model penalty or threshold was retuned.

The omitted stripe component flagged **{result['stripe_stress']['flags'][0]}/256**
baseline and **{result['stripe_stress']['flags'][1]}/256** augmented cases.
These are model-discrepancy detections, not matched-null false alarms or
identification of primordial physics. This limited stress test cannot cover
unknown unknowns in general or isolate physical causes.

## Numerical execution and evidence

- Numerical/finite-value failures: **{stage['numerical_failures']}/{stage['nonfinite_failures']}**.
- Maximum scaled baseline CPU/CUDA difference: **{stage['max_baseline_error']:.6g}**.
- Maximum scaled response difference: **{stage['max_response_error']:.6g}**.
- Maximum scaled score difference: **{stage['score_scaled_error']:.6g}**.
- Maximum relative signal-shell power error: **{stage['max_signal_shell_error']:.6g}**.
- Fixed tolerance 1e-10; covariance condition numbers: {result['condition_numbers']}.
- Complete audit: **{stage['elapsed_s']:.1f}s**, including both feature paths.
  Forward CPU operators: {timing['forward_cpu_seconds']:.2f}s; CPU rank preprocessing:
  {timing['rank_cpu_seconds']:.2f}s; CPU feature calls: {timing['feature_cpu_seconds']:.2f}s;
  CUDA feature calls including transfers: {timing['feature_gpu_seconds_including_transfer']:.2f}s.
  These timings do not establish an end-to-end GPU speedup.

Spherical synthesis, shared rank preprocessing and training fit run on CPU;
CUDA independently computes moment curves, excursion complexes, response
aggregation and final scores. The earlier multiscale compression step is shared
CPU code. JoVin's new 9/9 suite checks shipped NumPy kernels and a real recorded
CUDA fixture; production CUDA runs separately in WSL. The spherical forward
class is source-frozen and locally tested, rather than claimed to execute in
the NumPy-only JoVin sandbox.

New authentic JoVin receipt: `{plan['jovin_receipt_id']}`. All 20 current suites
match their authentic receipts (136 checks including collection); regression 185
passed, three unavailable Intel XPU skips; all 20 CI-equivalent CPU suites passed.
A [separate 30-case CPU replay](../results/response_v1/replay.json) regenerates
three fixed indices per role using the frozen kernels and stored fiducial C_ell.
It is separate execution, not an independent physical simulator.
The [final verification](../results/response_v1/verification.json) recomputes
training transformations, scores, tails, paired tests, Holm decisions, intervals
and all frozen bindings, with fresh authentic receipt-chain membership checks.

The CUDA development record predates a corrected quarter-turn assertion in a
development test; its recorded fixture values and numerical source hashes remain
unchanged. The actual JoVin receipt binds the corrected tests used at freeze.
Earlier scientific artifacts and the nine Lean rational invariants remain intact.
No Lean proof establishes detection power or validity of this synthetic model.

## What would justify advancing it

This comparison covers the specified 17 baseline summaries and six synthetic
alternatives. Broader added-information or novelty claims require benchmarks
against bispectrum/scattering methods, more realistic nuisance models, sensitivity
to changes in the training distribution and genuinely withheld observational
checks. Frequency and detector splits share the same sky, so they are useful
contamination checks rather than independent universes.

Relevant prior work includes [nuisance-hardened compression](https://arxiv.org/abs/1903.01473),
[spherical scattering](https://arxiv.org/abs/2407.07007) and
[model-misspecification tests](https://arxiv.org/abs/2412.15100). This pilot uses
ridge residualization, not the first paper's efficient-score construction;
the response features are not a scattering transform. Any useful novelty in
the combination remains a hypothesis requiring a focused literature comparison.

[Plan](../results/response_v1/plan.json),
[conditioning](../results/response_v1/conditioning.json),
[all features/scores/draw parameters](../results/response_v1/ensemble.json),
[fitted model](../results/response_v1/model.json),
[validation](../results/response_v1/validation.json),
[result](../results/response_v1/result.json),
[stage audit](../results/response_v1/stage_audit.json),
[verification](../results/response_v1/verification.json).
'''
    (ROOT/'docs/RESPONSE_BENCHMARK_RUN.md').write_text(report,encoding='utf-8')
    print(json.dumps({'status':result['status'],'provisional_gain_groups':gains,'report':'docs/RESPONSE_BENCHMARK_RUN.md'}))


if __name__ == '__main__':
    main()
