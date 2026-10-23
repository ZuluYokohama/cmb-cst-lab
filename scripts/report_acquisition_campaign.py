"""Render measured hardware use, reconstruction diagnostics and indexed samples."""
import json
import csv
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cmbcst.acquisition import batch_reference,MODES
OUT = ROOT/'results/acquisition_v1'
read = lambda name:json.loads((OUT/name).read_text())


def main():
    plan,result,stage,telemetry = map(read,('plan.json','result.json','stage_audit.json','telemetry.json'))
    devices = [read(f'device_cuda{i}.json') for i in (0,1)]
    samples = telemetry['samples']
    colors = ['#087f8c','#cf7d23']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes = plt.subplots(2,2,figsize=(12.5,8.5))
    fig.subplots_adjust(left=.085,right=.97,top=.81,bottom=.13,wspace=.28,hspace=.4)
    fig.text(.085,.95,'Measured GPU campaign | two cards, resident arrays',fontsize=22,weight='bold',color='#25374d')
    fig.text(.085,.89,f"{result['total_skies']:,} skies · {result['detector_pixel_samples']:,} detector-pixel samples · float64/complex128",fontsize=11)
    fig.text(.085,.845,f"Campaign wall time {stage['elapsed_seconds']:.1f}s; one-hour cap. Device-wide counters include other applications.",fontsize=10)
    statistics = []
    for i,record in enumerate(devices):
        series = [(s['seconds'],d) for s in samples for d in s.get('devices',[]) if d['index']==i and s['seconds']<=record['elapsed_seconds']]
        if not series:
            statistics.append(dict(device=i,available=False))
            continue
        seconds = np.array([s[0] for s in series])
        rows = [s[1] for s in series]
        name = record['name'].replace('NVIDIA GeForce ','')
        axes[0,0].plot(seconds,[r['utilization'] for r in rows],color=colors[i],label=name,lw=1)
        axes[0,1].plot(seconds,[r['power_W'] for r in rows],color=colors[i],label=name,lw=1)
        axes[0,1].axhline(rows[0]['power_limit_W'],color=colors[i],ls='--',alpha=.45)
        axes[1,0].plot(seconds,[r['memory_used_MiB']/1024 for r in rows],color=colors[i],label=name,lw=1)
        production = [b for b in record['batches'] if b['role']=='production']
        count = sum(b['count'] for b in production)
        gpu_seconds = sum(b['gpu_event_ms'] for b in production)/1000
        throughput = count/gpu_seconds
        axes[1,1].bar(i,throughput,color=colors[i],width=.55)
        statistics.append(dict(device=i,name=name,available=True,samples=len(rows),
            mean_utilization=float(np.mean([r['utilization'] for r in rows])),
            median_utilization=float(np.median([r['utilization'] for r in rows])),
            fraction_at_least_90_percent=float(np.mean([r['utilization']>=90 for r in rows])),
            mean_power_W=float(np.mean([r['power_W'] for r in rows])),peak_power_W=max(r['power_W'] for r in rows),
            peak_device_memory_MiB=max(r['memory_used_MiB'] for r in rows),
            peak_temperature_C=max(r['temperature_C'] for r in rows),
            gpu_event_production_seconds=gpu_seconds,production_skies_per_event_second=throughput))
    axes[0,0].set(title='Sampled GPU utilization',xlabel='Seconds',ylabel='Device busy (%)',ylim=(0,105))
    axes[0,0].legend(frameon=False)
    axes[0,1].set(title='Measured power (dashed: device limit)',xlabel='Seconds',ylabel='Watts')
    axes[1,0].set(title='Device memory in use',xlabel='Seconds',ylabel='GiB')
    axes[1,1].set(title='Production throughput from CUDA events',ylabel='Skies / event-second',xticks=[0,1],xticklabels=['RTX 5070','GTX 1660 Ti'])
    fig.text(.085,.055,'Work includes Gaussian generation, FFTs, eight-channel forward observations and four reconstructions; no padding loops.',fontsize=9)
    fig.savefig(OUT/'hardware.png',dpi=160,facecolor='white')
    plt.close(fig)
    (OUT/'performance.json').write_text(json.dumps(dict(devices=statistics,scope='Measured counters; not an end-to-end CPU speedup comparison'),indent=2)+'\n')

    all_metrics = []
    for i in (0,1):
        with np.load(OUT/f'production_cuda{i}.npz') as data:
            all_metrics.append(data['mse'].mean(0))
    mean_mse = sum(all_metrics[i]*plan['devices'][i]['production'] for i in (0,1))/sum(d['production'] for d in plan['devices'])
    with np.load(OUT/'audit_input_cuda0.npz') as data:
        # Display-only replay of the published primary audit input with Q/U=0.
        example = batch_reference(data['sky_white'],data['noise_white'],i_only=True,return_maps=True)
        polarized = batch_reference(data['sky_white'],data['noise_white'],return_maps=True)
    fig,axes = plt.subplots(2,3,figsize=(14,9))
    fig.subplots_adjust(left=.07,right=.90,top=.83,bottom=.14,wspace=.35,hspace=.45)
    fig.text(.07,.955,'Geometry changes what the detector reports',fontsize=24,weight='bold',color='#25374d')
    fig.text(.07,.90,'Top: intensity-only replay of published audit inputs. Bottom: all production skies and basis response.',fontsize=11)
    ps = [np.hypot(example['recovered'][0,i,1],example['recovered'][0,i,2]) for i in (0,2,3)]
    maximum = max(float(p.max()) for p in ps)
    for col,(p,label) in enumerate(zip(ps,('Correct geometry','Ignored pointing offsets','Assumed circular beams'))):
        im = axes[0,col].imshow(p,origin='lower',cmap='magma',vmin=0,vmax=maximum)
        axes[0,col].set(title=label,xticks=[],yticks=[])
    cax = fig.add_axes([.925,.56,.012,.25])
    fig.colorbar(im,cax=cax,label='Polarization amplitude (µK)')
    ax = axes[1,0]
    for mode in range(4):
        ax.plot(range(4),np.sqrt(mean_mse[mode]),'o-',label=MODES[mode].replace('_',' '))
    ax.set(title='All production: RMSE against known target',ylabel='µK (log scale)',yscale='log',xticks=range(4),xticklabels=['I','Q','U','|P|'])
    ax.legend(frameon=False,fontsize=8)
    ax = axes[1,1]
    correct,roll = polarized['recovered'][0,:2]
    for array,label,style in ((correct[1],'Q correct','-'),(roll[1],'Q wrong roll','--'),
        (np.hypot(correct[1],correct[2]),'|P| correct','-'),(np.hypot(roll[1],roll[2]),'|P| wrong roll',':')):
        ax.plot(array[256],style,lw=1,label=label)
    ax.set(title='A roll error can hide in amplitude',xlabel='Pixel x at y=256',ylabel='µK')
    ax.legend(frameon=False,fontsize=7)
    ax = axes[1,2]
    for i,record in enumerate(devices):
        ax.bar(np.arange(3)+(i-.5)*.3,record['checks']['noise_variance_ratio'],width=.3,color=colors[i],label=('RTX 5070','GTX 1660 Ti')[i])
    ax.axhspan(.98,1.02,color='#64748b',alpha=.1)
    ax.axhline(1,color='#64748b',ls='--',lw=1)
    ax.set(title='Correct noise MSE / analytic prediction',xticks=range(3),xticklabels=['I','Q','U'],ylim=(.95,1.05))
    ax.legend(frameon=False,fontsize=8)
    fig.text(.07,.067,'A local periodic tangent-plane model with known beams/pointing. These expected artifacts do not diagnose real Planck data.',fontsize=10)
    fig.text(.07,.033,'Uniform basis rotation preserves Q²+U²; independent position/beam mistakes can create intensity-to-polarization leakage.',fontsize=10)
    fig.savefig(OUT/'geometry.png',dpi=160,facecolor='white')
    plt.close(fig)
    cases = '\n'.join(f"| {name} | "+' | '.join(f'{x:.6g}' for x in np.sqrt(mean_mse[i]))+' |' for i,name in enumerate(MODES))
    control_mse = []
    for i in (0,1):
        with np.load(OUT/f'intensity_only_cuda{i}.npz') as data:
            control_mse.append(data['mse'].mean(0))
    control_rows = '\n'.join(f"| {name} | {np.sqrt(np.mean(control_mse,axis=0)[i,3]):.6g} |" for i,name in enumerate(MODES))
    index_receipt = [r for r in json.loads((ROOT/'data/evidence/jovin_receipts.json').read_text()) if r['suite']=='acquisition_index'][-1]['receipt_id']
    with (OUT/'labeled_samples_corrected.csv').open(newline='') as stream:
        example_rows = list(csv.DictReader(stream))[:4]
    indexed_rows = '\n'.join(f"| {r['sample_index']} | {r['time_index']} | {r['channel']} | {float(r['look_x_unwrapped_deg']):.6f} | {float(r['look_y_unwrapped_deg']):.6f} | {float(r['incidence_deg']):g} | {float(r['polarization_deg']):g} | {float(r['observed_uK']):.5f} |" for r in example_rows)
    diagnostic_rows = '\n'.join(f"| {statistics[i]['name']} | {r['checks']['noiseless_max_RMSE_uK']:.3g} | {max(r['cpu_audit']['errors'].values()):.3g} | {r['checks']['max_roll_amplitude_error']:.3g} | "+', '.join(f'{v:.6f}' for v in r['checks']['noise_variance_ratio'])+' |' for i,r in enumerate(devices))
    hardware_rows = '\n'.join(f"| {s['name']} | {plan['devices'][i]['production']:,} | {devices[i]['batch']} | {s['mean_utilization']:.2f}% | {s['fraction_at_least_90_percent']:.2%} | {s['mean_power_W']:.1f}W | {devices[i]['peak_allocated_bytes']/2**30:.3f}GiB |" if s['available'] else f'| GPU{i} | telemetry unavailable |' for i,s in enumerate(statistics))
    report = f'''# GPU acquisition geometry: measured result

**{result['status']}: {result['total_skies']:,} synthetic skies in {stage['elapsed_seconds']:.2f}s.**
The campaign processed {result['detector_pixel_samples']:,} detector-pixel values
and four reconstructions of each observation, within the 3600 s cap. This is a
controlled acquisition-model test, not a new cosmological result or actual Planck TOD.
The detector-pixel count is the equivalent size of the eight-channel observation
grids, processed spectrally. Those full grids are not all retained on disk.

![Measured hardware work](../results/acquisition_v1/hardware.png)

## Hardware use

| GPU | Production skies | Batch | Mean sampled busy | Samples at ≥90% busy | Mean power | Peak allocated tensors |
|---|---:|---:|---:|---:|---:|---:|
{hardware_rows}

The device counters were sampled once per second and include other applications.
CUDA event times and allocated/reserved memory are retained per batch/device.
Busy percentage is not percentage of theoretical FLOPS, memory capacity or power
limit. This FP64 workload need not draw maximum rated power; no clocks or power
settings were changed and no meaningless loops padded the run. Throughput is
for this declared model, not a comparison against the previous, different pipeline.

## Measurement geometry and indexed channels

Each raster position produces eight detector readings from a known local I/Q/U
sky. Arrival position, oblique beam incidence, ellipse orientation, pointing
offsets, gain and polarization orientation are separate metadata. A calibrated
Fourier-space inverse estimates three components at a common beam resolution.
The [32 corrected labeled sample rows](../results/acquisition_v1/labeled_samples_corrected.csv)
include sample/time indices, channel, angular x/y, detector look coordinates,
incidence/beam/polarization angles and µK readings. The 1 ms raster cadence is
artificial. Angular position is this model's spatial index; radial depth is not inferred.

The first four readings share raster-time index 0 and pixel (0,0); their
detector look positions differ. Values below use unwrapped local coordinates:

| Sample | Time index | Channel | Look x (deg) | Look y (deg) | Incidence (deg) | Polarization angle (deg) | Reading (µK) |
|---:|---:|---:|---:|---:|---:|---:|---:|
{indexed_rows}

**Coordinate-label correction:** the frozen Fourier phase samples at nominal
position minus the pointing offset. The original CSV labeled nominal-plus-offset
coordinates as pointing. Its bytes are preserved. The corrected view retains
those old values under explicit registration labels and supplies actual look
coordinates, with and without periodic wrapping. An independent analytic cosine
test resolves the sign convention; forward/inverse agreement alone cannot do so.
See the [erratum](ACQUISITION_COORDINATE_ERRATUM.md) and
[coordinate audit](../results/acquisition_v1/coordinate_export_audit.json).
The simulation, observations and reconstruction metrics were not changed.

True incidence angles range 0–60 degrees. Beam major sigma is 1.2/cos(incidence)
pixels, minor 1.2; target beam sigma 3 pixels. Four inverses assume correct geometry,
a +10 degree roll offset, zero pointing offsets, or circular beams. See the
[frozen protocol](ACQUISITION_SPEC.md) for the complete equations and limitations.

![Reconstruction and noise diagnostics](../results/acquisition_v1/geometry.png)

## Measured reconstruction errors

Production pooled RMSE against the known common-resolution target, in µK:

| Assumed geometry | I | Q | U | Polarization amplitude |
|---|---:|---:|---:|---:|
{cases}

Each wrong-assumption case reconstructs the same observations and noise as the
correct case. The top image is a display-only intensity-only replay of the
published primary audit input, separate from the 256-per-device intensity-only
control cohorts. It illustrates false polarization from pointing/beam errors;
cohort statistics are preserved in NPZ files and the final verification.

Measured reconstructed polarization RMS in the pooled 512 intensity-only control
skies, including detector noise, in µK (the true Q/U are zero):

| Assumed geometry | Polarization RMS |
|---|---:|
{control_rows}

Uniform roll changes Q/U while leaving their amplitude invariant. The maximum
scaled amplitude differences were {[d['checks']['max_roll_amplitude_error'] for d in devices]}.
This known degeneracy explains why an amplitude-only feature cannot identify
an absolute orientation error in this constructed setting. It is not a newly
discovered physical effect or an explanation of previous Planck discrepancies.

| GPU | Noise-free max I/Q/U RMSE (µK) | Max scaled CPU audit error | Max scaled roll-amplitude difference | Noise MSE / prediction, I/Q/U |
|---|---:|---:|---:|---|
{diagnostic_rows}

Noise-free absolute recovery and scaled precision limits were 1e-10;
the preset relative noise-variance gate was 2%.
Rank-deficient orientation coverage is rejected in the independently checked
CPU/JoVin negative control; no pseudoinverse manufactures a reconstruction.

## Provenance, numerical checks and formal scope

All production metrics are finite. The two full-resolution audit inputs are
published with the result. Each is independently computed on NumPy CPU and
checked against its producing GPU, including full maps and observations. The
primary input is also reconstructed on the secondary GPU; scaled metric
difference **{result['cross_gpu_scaled_error']:.6g}**, limit 1e-10. Production GPU
draws are not all recomputed on CPU: the audit scope is two full-resolution
inputs, small real-device fixtures, static geometry checks, every-batch roll
invariance and the aggregate exact-noise prediction. GPU RNG reproducibility
is limited to recorded device/backend/batch shapes; published audit inputs
support checks independent of RNG implementation.

The new JoVin receipt `{plan['jovin_receipt_id']}` checks shipped CPU kernels
and recorded actual CUDA outputs from both cards. The separate coordinate receipt
is `{index_receipt}`. All 23 current suites verify
(152 checks including collection); local tests 192 passed, three unavailable
Intel XPU skips; all 23 CI-equivalent sandbox suites passed. The sandbox does
not execute CUDA. Fresh [verification](../results/acquisition_v1/verification.json)
authenticates receipts/chain membership and checks current source/test hashes,
all prior frozen artifacts, production/control metrics, audit inputs and formal
source/compiler records. No SHA match proves physical validity or novelty.

Two new exact rational [Lean identities](../formal/PolarizationGeometry.lean)
prove preservation of polarization norm and a co-rotated projection under
c²+s²=1. They were kernel-checked with the same official SHA-pinned Lean 4.35.0.
They do not prove floating-point inversion, valid optics or real-world calibration.
The earlier nine rational invariants remain unchanged.

A development-only CUDA workspace OOM occurred before scientific freeze. Its
[failure record](../data/evidence/acquisition_workspace_failure.json) and archived
source are preserved. Static solves were bounded in chunks of 256 with CPU conditioning
diagnostics; no scientific thresholds or model assumptions were tuned to pass.

This periodic tangent-plane instrument uses a fixed local Stokes basis and
synthetic red Gaussian fields, not full spherical spin-2 CMB physics or a
validated Planck/MWD instrument. Real timeline analysis would need pointing,
beam/bandpass/calibration, noise and coordinate-basis provenance. The earlier
observational foreground/reference mismatch remains unresolved.

[Plan](../results/acquisition_v1/plan.json),
[result](../results/acquisition_v1/result.json),
[sampled telemetry](../results/acquisition_v1/telemetry.json),
[performance summary](../results/acquisition_v1/performance.json),
[primary audit input](../results/acquisition_v1/audit_input_cuda0.npz),
[secondary audit input](../results/acquisition_v1/audit_input_cuda1.npz).
'''
    (ROOT/'docs/ACQUISITION_RUN.md').write_text(report,encoding='utf-8')
    print(json.dumps(dict(status=result['status'],hardware=statistics),indent=2))


if __name__=='__main__':
    main()
