"""Frozen, read-only observational split measurement; never reuse old nulls."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cmbcst import real_splits as r
from cmbcst.real_splits_data import pair_geometry,prepare_observations,sha
OUT,WORK = ROOT/'results/real_splits_v1',ROOT/'data/work/real_splits_v1'


def save(path,value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def freeze():
    import torch,ducc0
    from jovin_verify import SUITES,build_bundle
    OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
    if (OUT/'plan.json').exists():
        raise FileExistsError('preserve frozen run; do not launch duplicate')
    receipt = [v for v in json.loads((ROOT/'data/evidence/jovin_receipts.json').read_text()) if v['suite']=='real_splits'][-1]
    if not all(receipt[k] for k in ('passed','receipt_verified','chain_valid')) or receipt['bundle_sha256']!=hashlib.sha256(build_bundle('real_splits')[0].encode()).hexdigest() or receipt['tests_sha256']!=sha(ROOT/SUITES['real_splits'][1]):
        raise ValueError('current kernels/tests lack authentic matching JoVin receipt')
    paths = ['src/cmbcst/real_splits.py','src/cmbcst/real_splits_data.py','src/cmbcst/spectra.py','src/cmbcst/theory.py','src/cmbcst/data.py',
        'scripts/run_real_splits.py','scripts/prepare_real_splits.py','tests/test_real_splits.py','tests/sandbox/jovin_tests_real_splits.py','docs/REAL_SPLITS_SPEC.md']
    plan = dict(created_utc=now(),scope='Exploratory real Planck A/B polarization; no calibrated anomaly or novelty claim',
        maps=['100A','100B','143A','143B','217A','217B','353A','353B'],nside_input=2048,lmax=512,lmin=30,
        common_beam_arcmin=60,beam_method='sqrt(diagonal auto QuickPol EE/BB) effective equalization only',threads=8,
        anchor_nside=64,tile_nside=4,lags_degrees=[.5,1.,2.,4.],bearings_degrees=[0.,45.,90.,135.],
        anchor_min_latitude_degrees=45,path_min_latitude_degrees=40,mask_margin_degrees=3,
        min_pairs_per_tile_lag=128,bootstrap_draws=2000,seed=82024102026,compute_cap_seconds=3600,
        modes=r.MODES,statistics=r.STATISTICS,maximum_scaled_error=1e-10,
        source_sha256={name:sha(ROOT/name) for name in paths},
        input_inventory_sha256=sha(ROOT/'data/evidence/planck_dataset.json'),
        input_audit_sha256=sha(ROOT/'data/evidence/planck_input_audit.json'),
        mask_sha256=sha(ROOT/'data/work/rtx5070_real/mask_30.npy'),fixture_sha256=sha(ROOT/'data/evidence/real_splits_cuda_v1.json'),
        jovin_receipt_id=receipt['receipt_id'],jovin_bundle_sha256=receipt['bundle_sha256'],
        python=sys.version,numpy=np.__version__,torch=str(torch.__version__),cuda=torch.version.cuda,ducc0=ducc0.__version__,
        devices=[torch.cuda.get_device_name(i) for i in (0,1)])
    with (OUT/'plan.json').open('x') as stream:
        json.dump(plan,stream,indent=2)
    return plan


def main():
    import torch
    plan = freeze();start=time.perf_counter()
    stage = dict(status='running',started_utc=now(),plan_sha256=sha(OUT/'plan.json'),stages={})
    save(OUT/'stage_audit.json',stage)
    stopped=threading.Event();telemetry=[];phase=['starting']
    def budget():
        if time.perf_counter()-start>=3600:
            raise TimeoutError('fixed real-data compute cap exceeded')
    def progress(name,key=None):
        phase[0]=name
        record=dict(phase=name,map=key,seconds=time.perf_counter()-start)
        save(OUT/'progress.json',record)
        print(json.dumps(record),flush=True)
    def monitor():
        while not stopped.is_set():
            try:
                query=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
                telemetry.append(dict(seconds=time.perf_counter()-start,phase=phase[0],exit_code=query.returncode,
                    devices=[dict(index=int(v[0]),utilization=float(v[1]),memory_MiB=float(v[2]),power_W=float(v[3]),temperature_C=float(v[4])) for v in csv.reader(query.stdout.splitlines()) if len(v)==5]))
            except Exception as exc:
                telemetry.append(dict(seconds=time.perf_counter()-start,phase=phase[0],error=str(exc)))
            stopped.wait(1.)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:
        progress('mask_geometry');tick=time.perf_counter()
        g=pair_geometry(ROOT)
        if g['mask_sha256']!=plan['mask_sha256']:
            raise ValueError('mask changed after freeze')
        np.savez_compressed(OUT/'geometry.npz',**{k:v for k,v in g.items() if k!='mask_sha256'})
        stage['stages']['geometry_seconds']=time.perf_counter()-tick
        tick=time.perf_counter()
        anchor,endpoint,provenance=prepare_observations(ROOT,WORK,g,progress,budget)
        np.savez_compressed(OUT/'observations.npz',anchor=anchor,endpoint=endpoint)
        save(OUT/'inputs.json',dict(maps=provenance,mask_sha256=g['mask_sha256'],
            map_bytes=sum(v['bytes'] for v in provenance),limits='Zero-filling, area-weighted adjoint and diagonal effective beams; no full instrument inversion'))
        stage['stages']['hash_read_transform_sample_seconds']=time.perf_counter()-tick
        budget();progress('cpu_statistics');tick=time.perf_counter()
        views=r.increment_views(anchor,endpoint,g['phase0'],g['phase1'])
        features,tiles=r.tile_statistics(views,g['valid'],g['tile_ids'])
        np.savez_compressed(OUT/'tile_statistics.npz',features=features)
        save(OUT/'tiles.json',dict(records=tiles,accepted=sum(v['accepted'] for v in tiles),
            valid_pairs=int(g['valid'].sum()),anchors=len(g['loc'])))
        stage['stages']['cpu_statistics_seconds']=time.perf_counter()-tick
        progress('gpu_statistics')
        def device_stats(index):
            with torch.cuda.device(index):
                torch.cuda.reset_peak_memory_stats(index)
                tick=time.perf_counter();begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record()
                actual,actual_tiles=r.tile_statistics(views,g['valid'],g['tile_ids'],device=f'cuda:{index}')
                end.record();end.synchronize()
                error=float(np.max(np.abs(actual-features)/(1+np.abs(features))))
                if error>1e-10 or not np.isfinite(actual).all() or actual_tiles!=tiles:
                    np.savez_compressed(OUT/f'failed_gpu{index}.npz',features=actual)
                    raise ValueError('real-data CPU/GPU precision failure')
                np.savez_compressed(OUT/f'gpu{index}_statistics.npz',features=actual)
                return dict(device=index,name=plan['devices'][index],maximum_scaled_error=error,
                    wall_seconds=time.perf_counter()-tick,event_ms=begin.elapsed_time(end),peak_allocated_bytes=torch.cuda.max_memory_allocated(index))
        tick=time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as pool:
            devices=list(pool.map(device_stats,(0,1)))
        stage['stages']['both_gpu_statistics_wall_seconds']=time.perf_counter()-tick
        identity=float(np.max(np.abs(features[...,3]-features[...,4]-features[...,2])/(1+np.abs(features[...,2]))))
        if not np.isfinite(features).all() or identity>1e-10 or np.max(np.abs(features[...,5]))>1+1e-10:
            raise ValueError('nonfinite, coherence or split identity failure')
        budget();progress('tile_resampling');tick=time.perf_counter()
        ix=np.random.default_rng(82024102026).integers(0,features.shape[2],size=(2000,features.shape[2]))
        reference=r.bootstrap_tiles(features,ix)
        def device_boot(index):
            with torch.cuda.device(index):
                first=index*1000
                value=r.bootstrap_tiles(features,ix[first:first+1000],f'cuda:{index}')
                common=r.bootstrap_tiles(features,ix[:32],f'cuda:{index}')
                error=float(np.max(np.abs(value-reference[first:first+1000])/(1+np.abs(reference[first:first+1000]))))
                common_error=float(np.max(np.abs(common-reference[:32])/(1+np.abs(reference[:32]))))
                if max(error,common_error)>1e-10 or not np.isfinite(value).all():
                    raise ValueError('GPU tile resampling precision failure')
                return value,dict(device=index,maximum_scaled_error=error,common32_cpu_error=common_error)
        with ThreadPoolExecutor(max_workers=2) as pool:
            boot=list(pool.map(device_boot,(0,1)))
        result_boot=np.concatenate([v[0] for v in boot])
        np.savez_compressed(OUT/'resampling.npz',indices=ix,statistics=result_boot)
        stage['stages']['resampling_seconds']=time.perf_counter()-tick
        budget()
        means=features.mean(axis=2)
        bands=np.quantile(result_boot,[.025,.975],axis=0)
        result=dict(status='exploratory_measurement_complete',completed_utc=now(),plan_sha256=sha(OUT/'plan.json'),
            accepted_tiles=features.shape[2],tile_equal_weight_mean=means.tolist(),tile_resampling_bands=bands.tolist(),
            devices=devices,resampling_devices=[v[1] for v in boot],split_identity_error=identity,
            accepted_physical_claims=[],accepted_method_superiority_claims=[],
            limits='Descriptive real-data repeatability; foregrounds, beams, transfer, residual noise and spatial dependence unresolved; no p-values or fractal dimension')
        save(OUT/'result.json',result)
        # Real observed rows, one anchor and two directions, all eight maps.
        from cmbcst import DETECTORS
        with (OUT/'labeled_real_samples.csv').open('x',newline='') as stream:
            writer=csv.writer(stream)
            writer.writerow(['sample_index','map','anchor_pixel_nside64_ring','galactic_latitude_deg','galactic_longitude_deg','lag_deg','bearing_deg','valid_pair','Q_anchor_uK','U_anchor_uK','Q_end_uK','U_end_uK','transported_delta_real_uK','transported_delta_imag_uK'])
            j=int(np.flatnonzero(g['valid'][2,0])[0])
            for m,key in enumerate(DETECTORS):
                for direction in (0,1):
                    value=views[0,m,2,direction,j]
                    writer.writerow([m*2+direction,key,int(g['anchor_pixel'][j]),90-np.degrees(g['loc'][j,0]),np.degrees(g['loc'][j,1]),2.,direction*45.,bool(g['valid'][2,direction,j]),anchor[m,j].real,anchor[m,j].imag,endpoint[m,2,direction,j].real,endpoint[m,2,direction,j].imag,value.real,value.imag])
        stage.update(status=result['status'],completed_utc=now(),elapsed_seconds=time.perf_counter()-start,numerical_failures=0,result_sha256=sha(OUT/'result.json'))
        save(OUT/'stage_audit.json',stage)
        print(json.dumps(dict(status=stage['status'],seconds=stage['elapsed_seconds'],tiles=features.shape[2],cpu_gpu_errors=[v['maximum_scaled_error'] for v in devices])),flush=True)
    except Exception:
        stage.update(status='execution_failed',completed_utc=now(),elapsed_seconds=time.perf_counter()-start,error=traceback.format_exc())
        save(OUT/'stage_audit.json',stage)
        raise
    finally:
        stopped.set();thread.join(timeout=6)
        save(OUT/'telemetry.json',dict(samples=telemetry,scope='Device-wide sampled counters; CPU transform stages dominate; other applications may contribute'))


if __name__=='__main__':
    main()
