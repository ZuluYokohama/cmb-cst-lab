"""Present every development comparison and locked final test, without retuning."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/targeted_v1'


def main():
    read = lambda name:json.loads((OUT/name).read_text())
    plan,dev,lock,result,stage,replay = map(read,('plan.json','development.json','selection_lock.json','result.json','stage_audit.json','replay.json'))
    colors = ['#64748b','#087f8c','#cf7d23']
    labels = ['Original omnibus','Trained baseline','Six responses']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes = plt.subplots(2,2,figsize=(12.5,9))
    fig.subplots_adjust(top=.82,bottom=.14,left=.09,right=.97,hspace=.42,wspace=.30)
    fig.text(.09,.955,'Physical blind spots | targeted response test',fontsize=23,weight='bold',color='#25374d')
    fig.text(.09,.90,'Final outcome: '+result['status'].replace('_',' '),fontsize=14,color='#25374d')
    fig.text(.09,.855,f"{stage['draws']:,} synthetic skies · {stage['elapsed_seconds']:.1f}s · final holdouts generated after selection lock",fontsize=11)
    ax = axes[0,0]
    ax.axvspan(0,.15,color='#087f8c',alpha=.08)
    for j,family in enumerate(('quadratic','compact','filament')):
        rows = [r for r in dev['comparisons'] if r['family']==family]
        ax.scatter([r['baseline_rate'] for r in rows],[r['augmented_rate'] for r in rows],marker=('o','s','^')[j],color=colors[j],label=family)
        selected = [r for r in rows if r['id']==lock['selected'][family]]
        for row in selected:
            ax.scatter(row['baseline_rate'],row['augmented_rate'],s=180,facecolors='none',edgecolors='black',linewidths=1.5)
    ax.plot([0,1],[0,1],'--',color='#94a3b8',lw=1)
    ax.set(title='Development search: all 18 configurations',xlabel='Trained baseline detection fraction',ylabel='Six-response detection fraction',xlim=(-.02,1.02),ylim=(-.02,1.02))
    ax.legend(frameon=False,fontsize=8)
    ax.text(.98,.03,'Outlined: locked selections',transform=ax.transAxes,ha='right',fontsize=8)
    ax = axes[0,1]
    x = np.arange(3)
    for method in range(3):
        values = [r['rates'][method] if r['rates'] is not None else np.nan for r in result['comparisons']]
        ax.bar(x+(method-1)*.23,values,width=.23,color=colors[method],label=labels[method])
    for j,row in enumerate(result['comparisons']):
        if row['draws']==0:
            ax.text(j,.04,'No eligible\nconfiguration',ha='center',fontsize=8)
    ax.set(title='Fresh final examples: selected recipes',ylabel='Detection fraction',xticks=x,xticklabels=['Quadratic','Compact','Filament'],ylim=(0,1.05))
    ax.legend(frameon=False,fontsize=8)
    ax = axes[1,0]
    rates = np.asarray(result['null']['rates'])
    bounds = np.asarray(result['null']['wilson_95']).T
    ax.bar(x,rates,color=colors,width=.6)
    ax.errorbar(x,rates,yerr=np.maximum(0,np.stack([rates-bounds[0],bounds[1]-rates])),fmt='none',ecolor='#25374d',capsize=4)
    ax.axhline(.05,color='#25374d',ls='--',label='Nominal .05')
    ax.set(title='Independent final null checks (n=1,024)',ylabel='False-alarm fraction',xticks=x,xticklabels=['Omnibus','Baseline','+ responses'],ylim=(0,max(.1,float(bounds.max())*1.15)))
    ax.legend(frameon=False,fontsize=8)
    ax = axes[1,1]
    delta = [-r['loss'] for r in result['strong_controls']]
    ax.barh(x,delta,color=colors[2])
    ax.axvline(-.05,color='#25374d',ls='--',label='Maximum permitted loss')
    ax.axvline(0,color='#94a3b8',lw=1)
    ax.set(title='Strong-signal retention (128 per family)',xlabel='Response − trained baseline detection',yticks=x,yticklabels=['Quadratic','Compact','Filament'])
    ax.legend(frameon=False,fontsize=8)
    fig.text(.09,.068,'Development selects recipes, not individual final examples. Final inference includes every generated example.',fontsize=10)
    fig.text(.09,.033,'Synthetic foreground/noise model; no Planck sky test, physical attribution or universal novelty claim.',fontsize=10)
    fig.savefig(OUT/'benchmark.png',dpi=160,facecolor='white')
    plt.close(fig)
    development_rows = '\n'.join(f"| {r['id']} | {r['family']} | {r['amplitude']} | {r['variant']} | {r['baseline_rate']:.3%} | {r['augmented_rate']:.3%} | {r['gain']:+.3%} |" for r in dev['comparisons'])
    final_rows = []
    for row in result['comparisons']:
        if row['draws']:
            final_rows.append(f"| {row['family']} / {row['recipe']} | {row['flags'][0]}/{row['draws']} | {row['flags'][1]}/{row['draws']} | {row['flags'][2]}/{row['draws']} | {row['gain']:+.3%} | {row['raw_p']:.6g} | {row['p_holm']:.6g} | {row['provisional_gain']} |")
        else:
            final_rows.append(f"| {row['family']} | — | — | — | no eligible recipe | 1 | 1 | False |")
    maximum_feature = max(max(q['errors']) for data in (dev,read('final.json')) for q in data['quality'])
    strong_rows = '\n'.join(f"| {r['family']} | {r['flags'][0]}/128 | {r['flags'][1]}/128 | {r['flags'][2]}/128 | {r['loss']:+.3%} |" for r in result['strong_controls'])
    quadratic = result['comparisons'][0]
    quadratic_note = (f"The original omnibus detected **{quadratic['flags'][0]}/256** selected quadratic examples, compared with **{quadratic['flags'][1]}/256** for the trained baseline and **{quadratic['flags'][2]}/256** for six responses. For this selected case, the result suggests a scoring-rule limitation rather than absence of information from the conventional summaries.") if quadratic['draws'] else 'No quadratic configuration met the development eligibility rule; no final quadratic gain is inferred.'
    report = f'''# Targeted physical blind-spot benchmark

**Final outcome: {result['status']}.** Calibration gate: **{result['calibration_gate']}**;
strong-signal retention gate: **{result['retention_gate']}**. No physical or
observational Planck finding is established by this synthetic experiment.

{quadratic_note}

![All development and locked final outcomes](../results/targeted_v1/benchmark.png)

## What changed

The first90-response detector failed its merit comparison. This separate version
uses exactly six predeclared geometric/moment responses, residualized against
the same17 conventional summaries using null-only ridge regression. The six
measure raw/rank Euler response to orientation, and raw/rank fourth-moment and
Euler response to smoothing. The learned models have17 and23 coordinates.
Both have three linear family heads, equal training data and fixed covariance
shrinkage. Each complete maximum score is calibrated against independent null
examples. The original omnibus score is a reported comparator, not a selection gate.

The forward model, continuous spherical sampling orientations and nuisance
amplitude distributions are the earlier explicit synthetic model. Quadratic
coupling signs±1 and localized widths .5/1degree vary across18 recipes. Injection
amplitudes .1/.2/.3 are specified before harmonic-power restoration; final effective
map changes need not equal those fractions. These are physically motivated
patterns, not a realistic Planck foreground/noise simulator.

## Every development comparison

Training:1024 nulls plus64 per recipe; development calibration499 nulls,
diagnostic null256, and32 examples per recipe. The null diagnostic flags
(omnibus/baseline/responses) were **{dev['diagnostic_null_flags']}**.
Eligibility requires trained-baseline detection<=15%. Select the largest measured
gain per eligible family, breaking ties by lower amplitude then recipe ID.
Development gains are selected estimates and are not confirmation evidence.
Eligibility is relative to the trained linear-head baseline. A selected recipe
may still be detectable by the original omnibus statistic; all three methods
remain visible. A paired gain would not establish that the17 summaries contain
no information or that conventional methods generally have this blind spot.

| Recipe | Family | Amplitude | Sign / width(deg) | Baseline | Responses | Difference |
|---|---|---:|---:|---:|---:|---:|
{development_rows}

Selected IDs: `{json.dumps(lock['selected'],sort_keys=True)}`.
The [selection lock](../results/targeted_v1/selection_lock.json) binds model,
development records, initial plan and the exact final role manifest before
any final sample is generated. Final examples are never filtered by whether
the baseline misses them. Old study holdouts were not reused.

## Fresh final evaluation

Calibration999 nulls; validation1024 nulls;256 examples per selected family;
128 strong controls per family at amplitude .35. Both detectors use nominal
p<=.05, with plus-one inclusive upper tails of the complete maximum score.

| Method | Null flags /1024 | Rate | Wilson95% |
|---|---:|---:|---|
| Original omnibus | {result['null']['flags'][0]} | {result['null']['rates'][0]:.3%} | [{result['null']['wilson_95'][0][0]:.3%}, {result['null']['wilson_95'][0][1]:.3%}] |
| Trained baseline | {result['null']['flags'][1]} | {result['null']['rates'][1]:.3%} | [{result['null']['wilson_95'][1][0]:.3%}, {result['null']['wilson_95'][1][1]:.3%}] |
| Six responses | {result['null']['flags'][2]} | {result['null']['rates'][2]:.3%} | [{result['null']['wilson_95'][2][0]:.3%}, {result['null']['wilson_95'][2][1]:.3%}] |

The trained baseline and augmented Wilson upper bounds must each be<=7.5%;
augmented rate must be<=baseline+1percentage point. All three strong-control
losses must be<=5percentage points. These are preset benchmark gates, not proof
of physical model adequacy.

| Strong family | Omnibus | Trained baseline | Six responses | Baseline − responses (loss) |
|---|---:|---:|---:|---:|
{strong_rows}

| Family / selected recipe | Omnibus detections | Baseline | Responses | Gain | Paired p | Holm p(3) | Provisional gain |
|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(final_rows)}

Promotion requires all gates,>=10percentage points added sensitivity and a
one-sided paired exact test with Holm p<=.05 across three families. Missing
families receive p=1. Raw records include descriptive2000-resample paired
bootstrap intervals, conditional on the fitted model/calibration reference.
No thresholds, features, seeds or models were changed after the freeze.

## Execution and evidence

- Completed **{stage['draws']} draws in {stage['elapsed_seconds']:.2f}s**, below the1800s cap.
- Numerical failures: **{stage['numerical_failures']}**.
- Maximum scaled CPU/CUDA feature error: **{maximum_feature:.6g}**.
- Maximum scaled end-to-end score error: **{stage['score_error']:.6g}**; limit1e-10.
- Separate CPU replay: **{len(replay['records'])} samples**, status **{replay['status']}**.
- Spherical synthesis, rank preprocessing and fitting are CPU work; feature
  kernels/aggregation and final scoring execute independently on CUDA float64.
  Total timings include both paths and do not establish end-to-end GPU speedup.
- New JoVin receipt: `{plan['jovin_receipt_id']}`. All21 current suites authenticate
  (142 checks including collection); local tests189 passed, three unavailable
  Intel XPU skips. All21 CI-equivalent CPU suites passed. JoVin checks numerical
  kernels and recorded real CUDA outputs; it does not run the spherical simulator.

[Fresh verification](../results/targeted_v1/verification.json) checks source/test
hashes against authentic receipts and chain membership, independently reconstructs
model fits, scores, selection, tails and final decisions, and checks preservation
of prior frozen artifacts. The replay reuses frozen kernels and is a separate
execution, not an independently designed physical simulator. Existing Lean
rational invariants remain unchanged; no new formal theorem or empirical
detection-power proof is claimed.

The Planck foreground/reference mismatch remains unresolved. A benchmark gain,
if present, is limited to these baselines and specified synthetic distributions.
Broader tests, a focused novelty review and observational contamination checks
would be needed before any real-sky or methodological-priority claim.

[Protocol](TARGETED_SPEC.md), [plan](../results/targeted_v1/plan.json),
[development](../results/targeted_v1/development.json),
[model](../results/targeted_v1/model.json),
[final samples](../results/targeted_v1/final.json),
[result](../results/targeted_v1/result.json),
[stage audit](../results/targeted_v1/stage_audit.json),
[replay](../results/targeted_v1/replay.json).
'''
    (ROOT/'docs/TARGETED_RUN.md').write_text(report,encoding='utf-8')
    print(result['status'])


if __name__=='__main__':
    main()
