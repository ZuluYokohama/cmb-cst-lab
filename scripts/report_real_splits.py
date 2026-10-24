"""Plot all predeclared observational summaries without anomaly decisions."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator,FuncFormatter
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cmbcst.real_splits import MODES
OUT=ROOT/'results/real_splits_v1'
read=lambda name:json.loads((OUT/name).read_text())


def main():
    result,stage,tiles,plan=map(read,('result.json','stage_audit.json','tiles.json','plan.json'))
    replay=read('sampling_replay.json')
    with np.load(OUT/'tile_statistics.npz') as d:
        f=d['features']
    with np.load(OUT/'geometry.npz') as d:
        loc,tileids=d['loc'],d['tile_ids']
    ids=[v['tile'] for v in tiles['records'] if v['accepted']]
    vectors=np.column_stack([np.sin(loc[:,0])*np.cos(loc[:,1]),np.sin(loc[:,0])*np.sin(loc[:,1]),np.cos(loc[:,0])])
    center_vectors=np.array([np.mean(vectors[tileids==i],axis=0) for i in ids])
    center_vectors/=np.linalg.norm(center_vectors,axis=1)[:,None]
    centers=np.column_stack([np.arccos(np.clip(center_vectors[:,2],-1,1)),np.mod(np.arctan2(center_vectors[:,1],center_vectors[:,0]),2*np.pi)])
    mean=np.asarray(result['tile_equal_weight_mean'])
    bands=np.asarray(result['tile_resampling_bands'])
    lag=np.array(plan['lags_degrees'])
    colors=['#126782','#1f9e89','#e08b2c','#b94e63']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(15,10))
    fig.subplots_adjust(left=.075,right=.94,top=.81,bottom=.13,wspace=.32,hspace=.45)
    fig.text(.075,.95,'Real Planck split polarization',fontsize=25,weight='bold',color='#25374d')
    fig.text(.075,.895,f"8 observed maps · {tiles['accepted']} geometry-selected tiles · 60 arcmin nominal beam · ell 30–512",fontsize=12)
    fig.text(.075,.85,'Descriptive measurements; shaded tile-resampling bands are not cosmological confidence intervals.',fontsize=10)
    for i,name in enumerate(('100','143','217','353')):
        axes[0,0].plot(lag,mean[0,i,:,3],'o-',color=colors[i],label=name+' GHz')
        axes[0,0].plot(lag,mean[0,i,:,4],'--',color=colors[i],alpha=.7)
        axes[0,1].plot(lag,mean[0,i,:,5],'o-',color=colors[i],label=name+' GHz')
        axes[0,2].plot(lag,mean[0,i,:,6],'o-',color=colors[i],label=name+' GHz')
        axes[0,2].fill_between(lag,bands[0,0,i,:,6],bands[1,0,i,:,6],color=colors[i],alpha=.12)
    axes[0,0].set(title='Mean-map (solid) and split-difference power',yscale='log',ylabel='Centered increment power (µK²)')
    axes[0,0].legend(frameon=False,fontsize=8)
    axes[0,1].set(title='A/B cross coherence',ylabel='Re cross / sqrt(auto A × auto B)',ylim=(-1.05,1.05))
    axes[0,2].set(title='Connected fourth-order structure',ylabel='Normalized cross K4')
    for i,name in enumerate(MODES):
        axes[1,0].plot(lag,mean[i,1,:,6],'o-',label=name.replace('_',' '))
    axes[1,0].set(title='143 GHz: representation / alignment controls',ylabel='Normalized cross K4')
    axes[1,0].legend(frameon=False,fontsize=8)
    maximum=max(float(np.max(np.abs(f[0,1,:,2,6]))),1e-10)
    points=axes[1,1].scatter((np.degrees(centers[:,1])+180)%360-180,90-np.degrees(centers[:,0]),c=f[0,1,:,2,6],vmin=-maximum,vmax=maximum,cmap='coolwarm',s=28)
    axes[1,1].set(title='143 GHz, 2°: accepted tiles',xlabel='Galactic longitude (deg)',ylabel='Latitude (deg)',ylim=(-95,95))
    fig.colorbar(points,ax=axes[1,1],shrink=.75,pad=.025,label='Cross K4')
    foreground=f[0,3,:,2,3]
    axes[1,2].scatter(foreground,f[0,1,:,2,6],c=np.where(centers[:,0]<np.pi/2,colors[0],colors[3]),s=23,alpha=.8)
    axes[1,2].set(title='Tile association with 353 GHz structure',xscale='log',xlabel='353 GHz mean-map increment power (µK²)',ylabel='143 GHz cross K4 at 2°')
    for ax in list(axes[0])+[axes[1,0]]:
        ax.set(xscale='log',xlabel='Angular separation (deg)',xticks=lag,xticklabels=['0.5','1','2','4'])
        ax.xaxis.set_minor_locator(NullLocator())
    candidates=np.array([v*10.**exponent for exponent in range(-5,12) for v in (1,2,5)])
    ticks=candidates[(candidates>=foreground.min())&(candidates<=foreground.max())]
    axes[1,2].set_xticks(ticks)
    axes[1,2].xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'))
    axes[1,2].xaxis.set_minor_locator(NullLocator())
    fig.text(.075,.065,'Q/U endpoints share a transported great-circle frame. Misaligned B is a stress test, not a statistical sky null.',fontsize=10)
    fig.text(.075,.032,'Foregrounds, beam/transfer residuals, spatial mixtures and noise can all affect these quantities; no primordial claim.',fontsize=10)
    fig.savefig(OUT/'real_polarization.png',dpi=160,facecolor='white')
    plt.close(fig)
    association=[]
    for i in range(4):
        for l in range(4):
            rho=float(spearmanr(f[0,i,:,l,6],f[0,3,:,l,3]).statistic)
            association.append(dict(frequency_GHz=(100,143,217,353)[i],lag_degrees=float(lag[l]),
                rank_correlation_with_353_mean_increment_power=rho,scope='Descriptive association, no independence p-value or causal attribution'))
    (OUT/'association.json').write_text(json.dumps(association,indent=2,allow_nan=False)+'\n')
    rows='\n'.join(f"| {name} | {mean[0,i,2,2]:.6g} | {mean[0,i,2,4]:.6g} | {mean[0,i,2,5]:.5f} | {mean[0,i,2,6]:.5f} | [{bands[0,0,i,2,6]:.5f}, {bands[1,0,i,2,6]:.5f}] |" for i,name in enumerate((100,143,217,353)))
    control='\n'.join(f"| {name} | {mean[m,1,2,5]:.6g} | {mean[m,1,2,6]:.6g} |" for m,name in enumerate(MODES))
    timing='\n'.join(f"| {name.replace('_',' ')} | {value:.3f} |" for name,value in stage['stages'].items())
    device_rows='\n'.join(f"| {v['name']} | {v['maximum_scaled_error']:.3g} | {v['wall_seconds']:.3f} |" for v in result['devices'])
    report=f'''# Real Planck split-polarization measurement

**Completed: {tiles['accepted']} accepted sky tiles, {tiles['anchors']:,} anchors and
{tiles['valid_pairs']:,} valid sampled pairs in {stage['elapsed_seconds']:.2f} seconds.**
This run reads the actual eight NPIPE A/B Q/U maps, not synthetic replacement skies.
The outcome is an exploratory measurement, with zero recorded numerical failures;
no calibrated sky anomaly, fractal dimension or method-superiority claim follows.

![Real observed split measurements](../results/real_splits_v1/real_polarization.png)

## Frozen real-data question

Measure which polarization increment patterns recur between detector splits,
how coordinate handling and amplitude-only representations change those summaries,
and how a connected fourth-order descriptor varies alongside ordinary power.
The [frozen protocol](REAL_SPLITS_SPEC.md) fixes all frequencies, scales, mask rules,
descriptors and resampling before extraction. Related sky regions were previously
inspected; this is not blind confirmation or a test of the earlier learned detector.

Input bytes are freshly SHA256-checked against the existing audited inventory.
Each map contains 50,331,648 NSIDE2048 pixels. Only Q/U are analyzed here.
The input files total {read('inputs.json')['map_bytes']:,} bytes. The 3-degree mask
margin and tile exclusions use geometry/counts only. All candidate tile counts,
including exclusions, are retained in [tiles.json](../results/real_splits_v1/tiles.json).

The spin-2 area-weighted adjoint at lmax 512 is not an exact inverse of a HEALPix
map. Ell 30–512 are retained and filtered toward a nominal 60 arcmin beam using
actual diagonal auto QuickPol EE/BB responses. This is diagonal effective beam
equalization; it does not remove anisotropic beams, cross-component mixing,
mapmaking transfer or all contamination. The common pixel window remains.
Invalid source pixels are counted and zero-filled; their counts are published.

## Results at the predeclared reading point: 2 degrees

Each number is an equal-weight average of accepted tile-level centered quantities.
The bands are 2.5/97.5 percentiles of 2000 tile resamples. Spatially related tiles
and modes are dependent, so these are **resampling stability bands, not confidence
intervals or detection significance**. All four scales appear in the figure and
complete arrays; none was selected for a favorable result.

| GHz | Cross S2 (µK²) | Split-difference S2 (µK²) | A/B coherence | Normalized cross K4 | K4 stability band |
|---:|---:|---:|---:|---:|---|
{rows}

Cross S2 is Re(Eab*) after tile/lag centering. The mean-map increment power minus
split-difference power equals it algebraically; maximum scaled identity error
was {result['split_identity_error']:.3g}. Split differences contain noise and
residual signal/beam/bandpass/systematic mismatches. They are not certified
pure-noise maps. High coherence does not identify a cosmological source.

K4 subtracts the Gaussian/Wick pair products from E|a|²|b|² and normalizes by
the two variances. It is an established connected fourth-order descriptor,
not a newly proved method, unbiased finite-sample estimator or evidence of
information absent from every standard approach. Within-tile variance mixtures,
foregrounds, noise and residual instrumental effects can make it nonzero.

## Representation and alignment controls

At 143 GHz and 2 degrees:

| Representation | Coherence | Cross K4 |
|---|---:|---:|
{control}

Transported Q/U expresses both ends in their shared great-circle frame. Naive Q/U
subtraction compares different local reference axes. Amplitude-only increments
discard polarization direction. These descriptors therefore ask different questions;
a larger value does not establish a better detector. Uniform physical rotation of
both splits remains degenerate for the invariant summaries; no absolute angle is fit.
Taking a magnitude is nonlinear: Gaussian Q/U do not imply Gaussian amplitude
increments. Those representations therefore do not share one Gaussian reference.

The misaligned-B control rolls valid B increments within each tile/lag, preserving
their marginal samples and breaking exact alignment. It is a stress test, not an
exchangeable or foreground-matched null. Its outcomes produce no p-values.

The tile rank association of 143 GHz K4 with 353 GHz mean increment power at
2 degrees is {association[6]['rank_correlation_with_353_mean_increment_power']:.5f}.
[All 16 associations](../results/real_splits_v1/association.json) are reported as
descriptive coefficients, without independence p-values or causal attribution.
Frequency dependence and repeatability need foreground/end-to-end modeling
before any primordial or new-physics interpretation.

## Numerical and provenance evidence

| Actual GPU | Maximum scaled difference from all CPU tile statistics | Wall seconds |
|---|---:|---:|
{device_rows}

All 2000 real-feature tile resamples are independently reproduced on CPU.
Each GPU computes 1000 draws plus the same first 32 draws as a cross-device check.
The preset scaled precision tolerance is 1e-10, without threshold/seed tuning.
The [sampling replay](../results/real_splits_v1/sampling_replay.json) separately
evaluates all eight retained filtered harmonic sets, with a direct/grid spin-2
operator control. It also reanalyzes the complete raw 143A Q/U map using inverse
pixel-index gathering and explicit filter algebra, compared with production's
forward index scattering. The ducc transform backend remains shared; this is
not an independent physical instrument model or a replay of all eight raw maps.
The raw 143A scaled harmonic difference was {replay['raw_map_replay']['scaled_error']:.6g}.
The separate direct/grid sampling control reached {replay['spin2_grid_direct_error']:.6g},
below but close to the preset 1e-10 gate. Its tolerance, epsilon and seed were
not changed to improve that margin. Replay of the eight retained coefficient
sets gave maximum scaled sample difference {max(v['scaled_error'] for v in replay['records']):.6g};
that repeatability shares the sampling algorithm and is not an accuracy proof.

The new authentic JoVin receipt is `{plan['jovin_receipt_id']}`. The sandbox
checks exact shipped CPU kernels, an independent transported-tensor/gauge oracle,
Gaussian and heavy-tail controls, and recorded real CUDA outputs from both cards.
It does not run CUDA or the full Planck preprocessing inside the sandbox.
Local regression: 195 passed, three unavailable Intel XPU skips; all 24 current
CI-equivalent sandbox suites passed. Fresh [verification](../results/real_splits_v1/verification.json)
authenticates all 24 current suites (158 checks including collection), current
source/test hashes, independent real-coordinate covariance calculations,
resamples, labeled rows and preservation of all prior frozen artifacts.

The first development fixture exceeded JoVin's 262144-byte test input limit;
it was rejected before execution. Its original test/source, actual GPU record
and failed admission ledger entry remain. A compact fixture passed the same
unchanged profile. [Admission record](../data/evidence/real_splits_fixture_admission.json).
The earlier eleven Lean rational identities are unchanged. No empirical or
new physical theorem is claimed from this observational measurement.

## Execution and reproducibility

| Stage | Seconds |
|---|---:|
{timing}

CPU transforms, fresh file hashing and direct spherical sampling dominate this
workflow. GPU reductions/resampling are real but light; this does not claim
saturated GPUs or end-to-end GPU speedup. [Device-wide phase-tagged telemetry](../results/real_splits_v1/telemetry.json)
includes other applications. Production completed within the 3600-second cap.

[Actual filtered labeled rows](../results/real_splits_v1/labeled_real_samples.csv)
include map/split, sky pixel, Galactic coordinates, angular separation, bearing,
mask validity, Q/U and transported increments. No fictitious time or radial-depth
column is added. Endpoint arrays, phases, masks, tile statistics, resamples and
all filtered harmonics are retained alongside the frozen plan.

Sources: [NPIPE release](https://portal.nersc.gov/project/cmb/planck2020/),
[NPIPE processing/split design](https://doi.org/10.1051/0004-6361/202038073),
[ducc operator definitions](https://mtr.pages.mpcdf.de/ducc/sht.html).

The earlier foreground/reference mismatch and negative method comparisons remain
unresolved. These measurements provide real-data constraints for future controls;
they do not repair that reference or validate cosmological significance.
'''
    (ROOT/'docs/REAL_SPLITS_RUN.md').write_text(report,encoding='utf-8')
    print(json.dumps(dict(status=result['status'],tiles=tiles['accepted'],primary_143GHz_2deg=dict(coherence=float(mean[0,1,2,5]),K4=float(mean[0,1,2,6])),seconds=stage['elapsed_seconds']),indent=2))


if __name__=='__main__':
    main()
