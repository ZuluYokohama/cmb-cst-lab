"""Frozen, concurrent two-GPU acquisition campaign with telemetry and CPU audits."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor,wait
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
from cmbcst.acquisition import AcquisitionCUDA,batch_reference,detector_table,MODES
from run_response_benchmark import save,sha,now,scaled_error
from jovin_verify import SUITES,build_bundle
OUT = ROOT/'results/acquisition_v1'
WORK = ROOT/'data/work/acquisition_v1'
SEED = 986004152026
COUNTS = (196608,65536)


def freeze():
    import torch
    OUT.mkdir(parents=True,exist_ok=True)
    WORK.mkdir(parents=True,exist_ok=True)
    if (OUT/'plan.json').exists():
        raise FileExistsError('preserve frozen campaign; no duplicate/budget restart')
    profile = json.loads((ROOT/'data/evidence/acquisition_gpu_preflight.json').read_text())
    proof = json.loads((ROOT/'data/evidence/acquisition_lean.json').read_text())
    if proof['status']!='passed' or proof['source_sha256']!=sha(ROOT/'formal/PolarizationGeometry.lean'):
        raise ValueError('supporting proof not checked')
    record = [r for r in json.loads((ROOT/'data/evidence/jovin_receipts.json').read_text()) if r['suite']=='acquisition'][-1]
    if not (record['passed'] and record['chain_valid'] and record['receipt_verified']
        and record['bundle_sha256']==hashlib.sha256(build_bundle('acquisition')[0].encode()).hexdigest()
        and record['tests_sha256']==sha(ROOT/SUITES['acquisition'][1])):
        raise ValueError('current acquisition kernels/tests lack authentic receipt')
    sources = ['src/cmbcst/acquisition.py','scripts/prepare_acquisition.py','scripts/run_acquisition_campaign.py',
        'scripts/verify_acquisition_lean.py','formal/PolarizationGeometry.lean','docs/ACQUISITION_SPEC.md',
        'tests/test_acquisition.py','tests/sandbox/jovin_tests_acquisition.py']
    for name,digest in profile['source_sha256'].items():
        if sha(ROOT/name)!=digest:
            raise ValueError('preflight source changed')
    devices = []
    for index,p in enumerate(profile['profiles']):
        if torch.cuda.get_device_name(index)!=p['name']:
            raise ValueError('device identity changed')
        devices.append(dict(index=index,name=p['name'],batch=p['batch'],production=COUNTS[index],
            noiseless=256,intensity_only=256,preflight_peak=max(x['peak_allocated_bytes'] for x in p['candidates'] if x['batch']==p['batch'])))
    plan = dict(created_utc=now(),scope='Local tangent-plane instrument model; no Planck TOD or radial-depth reconstruction',
        base_seed=SEED,grid_size=512,field_degrees=20.,devices=devices,total_skies=263168,
        detectors=8,modes=MODES,noise_sigma_uK=.5,compute_cap_seconds=3600,precision='float64/complex128',
        maximum_scaled_error=1e-10,noise_variance_relative_gate=.02,
        source_sha256={name:sha(ROOT/name) for name in sources},
        preflight_sha256=sha(ROOT/'data/evidence/acquisition_gpu_preflight.json'),
        proof_sha256=sha(ROOT/'data/evidence/acquisition_lean.json'),
        jovin_receipt_id=record['receipt_id'],jovin_bundle_sha256=record['bundle_sha256'],
        torch=str(torch.__version__),cuda=torch.version.cuda,numpy=np.__version__)
    with (OUT/'plan.json').open('x') as stream:
        json.dump(plan,stream,indent=2)
    return plan


def timeline_rows(audit):
    d = detector_table()
    rows = []
    for pixel in range(4):
        y,x = divmod(pixel,512)
        for h in range(8):
            rows.append(dict(sample_index=8*pixel+h,time_index=pixel,time_seconds=.001*pixel,channel=h,
                pixel_y=y,pixel_x=x,angular_x_deg=(x-256)*20/512,angular_y_deg=(y-256)*20/512,
                pointing_x_deg=((x-256)+d['offset_x_pixels'][h])*20/512,
                pointing_y_deg=((y-256)+d['offset_y_pixels'][h])*20/512,
                incidence_deg=d['incidence_deg'][h],beam_angle_deg=d['beam_angle_deg'][h],
                polarization_deg=d['polarization_deg'][h],observed_uK=audit['observed'][0,h,y,x]))
    with (OUT/'labeled_samples.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run():
    import torch
    torch.set_num_threads(8)
    plan = freeze()
    start = time.perf_counter()
    stopped = threading.Event()
    telemetry = []
    progress = {}
    progress_lock = threading.Lock()
    stage = dict(status='running',started_utc=now(),plan_sha256=sha(OUT/'plan.json'))
    save(OUT/'stage_audit.json',stage)

    def budget():
        if stopped.is_set() or time.perf_counter()-start>=3600:
            raise TimeoutError('campaign cap or peer failure; preserve incomplete records')

    def monitor():
        while not stopped.is_set():
            try:
                result = subprocess.run(['nvidia-smi','--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,power.draw,power.limit,temperature.gpu',
                    '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
                sample = dict(utc=now(),seconds=time.perf_counter()-start,returncode=result.returncode,devices=[])
                for row in csv.reader(result.stdout.splitlines()):
                    if len(row)==8:
                        sample['devices'].append(dict(index=int(row[0]),name=row[1].strip(),utilization=float(row[2]),
                            memory_utilization=float(row[3]),memory_used_MiB=float(row[4]),power_W=float(row[5]),
                            power_limit_W=float(row[6]),temperature_C=float(row[7])))
                telemetry.append(sample)
            except Exception as exc:
                telemetry.append(dict(utc=now(),seconds=time.perf_counter()-start,error=str(exc)))
            stopped.wait(1.)

    def worker(item):
        index,batch = item['index'],item['batch']
        device = f'cuda:{index}'
        batches,files,checks = [],[],{}
        with torch.cuda.device(index):
            model = AcquisitionCUDA(512,device)
            torch.cuda.reset_peak_memory_stats(index)
            audit_record = None
            for role,count,offset in (('noiseless',256,1000000),('intensity_only',256,2000000),('production',item['production'],0)):
                metrics = np.empty((count,4,4),dtype=np.float64)
                done = 0
                first = seed = None
                try:
                    for first in range(0,count,batch):
                        budget()
                        seed = SEED+offset+index*(100000 if role=='production' else 10000)+first//batch
                        generator = torch.Generator(device=device).manual_seed(seed)
                        tick = time.perf_counter()
                        begin,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        begin.record()
                        white = torch.randn((batch,3,512,512),dtype=torch.float64,device=device,generator=generator)
                        noise = torch.randn((batch,8,512,512),dtype=torch.float64,device=device,generator=generator)
                        value = model.run(white,noise,sigma=0. if role=='noiseless' else .5,i_only=role=='intensity_only')
                        end.record(); end.synchronize()
                        if value['roll_amplitude_error']>1e-10:
                            raise ValueError('roll-amplitude precision failure')
                        metrics[first:first+batch] = value['mse']
                        done += batch
                        batches.append(dict(role=role,first_index=first,count=batch,seed=seed,
                            gpu_event_ms=begin.elapsed_time(end),wall_seconds=time.perf_counter()-tick,
                            roll_amplitude_error=value['roll_amplitude_error']))
                        if role=='production' and first==0:
                            sw,nw = white[:1].cpu().numpy(),noise[:1].cpu().numpy()
                            input_path = OUT/f'audit_input_cuda{index}.npz'
                            np.savez_compressed(input_path,sky_white=sw,noise_white=nw)
                            reference = batch_reference(sw,nw,return_maps=True)
                            actual = model.run(white[:1],noise[:1],return_maps=True)
                            errors = {key:scaled_error(actual[key],reference[key]) for key in ('mse','truth','recovered','observed')}
                            errors['production_batch_metric'] = scaled_error(value['mse'][0],actual['mse'][0])
                            if max(errors.values())>1e-10:
                                raise ValueError('full-resolution CPU/GPU audit failure')
                            audit_record = dict(input_sha256=sha(input_path),seed=seed,errors=errors,passed=True,
                                reference_mse=reference['mse'].tolist(),gpu_mse=actual['mse'].tolist())
                            np.savez_compressed(WORK/f'audit_maps_cuda{index}.npz',**{key:actual[key] for key in ('truth','recovered','observed')})
                            if index==0:
                                timeline_rows(actual)
                        del white,noise
                        if first//batch%64==0:
                            print(f"{device} {role} {done}/{count} ({time.perf_counter()-start:.1f}s)",flush=True)
                        with progress_lock:
                            progress[device] = dict(role=role,completed=done,total=count,elapsed_seconds=time.perf_counter()-start)
                        save(WORK/f'checkpoint_cuda{index}.json',progress[device])
                        budget()
                except Exception:
                    save(OUT/f'failure_cuda{index}.json',dict(role=role,first_index=first,seed=seed,error=traceback.format_exc()))
                    if 'white' in locals() and 'noise' in locals():
                        try:
                            np.savez_compressed(WORK/f'failed_input_cuda{index}.npz',sky_white=white[:1].cpu().numpy(),noise_white=noise[:1].cpu().numpy())
                        except Exception:
                            pass
                    raise
                finally:
                    path = OUT/f'{role}_cuda{index}.npz'
                    np.savez_compressed(path,mse=metrics[:done])
                    files.append(path.name)
                if role=='noiseless':
                    checks['noiseless_max_RMSE_uK'] = float(np.sqrt(metrics[:,0,:3]).max())
                if role=='production':
                    ratio = metrics[:,0,:3].mean(0)/model.predicted_noise_variance
                    checks.update(noise_variance_ratio=ratio.tolist(),predicted_noise_variance=model.predicted_noise_variance.tolist())
            checks.update(noiseless_passed=checks['noiseless_max_RMSE_uK']<=1e-10,
                noise_variance_passed=max(abs(x-1) for x in checks['noise_variance_ratio'])<=.02,
                max_roll_amplitude_error=max(row['roll_amplitude_error'] for row in batches))
            record = dict(device=device,name=item['name'],batch=batch,checks=checks,batches=batches,
                cpu_audit=audit_record,conditions=model.conditions,files=files,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(index),peak_reserved_bytes=torch.cuda.max_memory_reserved(index),
                completed_utc=now(),elapsed_seconds=time.perf_counter()-start)
            save(OUT/f'device_cuda{index}.json',record)
            return record

    monitor_thread = threading.Thread(target=monitor,daemon=True)
    monitor_thread.start()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker,item) for item in plan['devices']]
            while True:
                complete,_ = wait(futures,timeout=2)
                with progress_lock:
                    save(OUT/'progress.json',dict(devices=dict(progress),elapsed_seconds=time.perf_counter()-start))
                failures = [f.exception() for f in futures if f.done() and f.exception() is not None]
                if failures:
                    stopped.set()
                    raise failures[0]
                if len(complete)==2:
                    break
            records = [f.result() for f in futures]
        budget()
        # Same complete primary audit input on the second GPU: independent device execution.
        with np.load(OUT/'audit_input_cuda0.npz') as inputs:
            with torch.cuda.device(1):
                cross_model = AcquisitionCUDA(512,'cuda:1')
                cross = cross_model.run(torch.as_tensor(inputs['sky_white'],device='cuda:1'),torch.as_tensor(inputs['noise_white'],device='cuda:1'))
                cross_error = scaled_error(cross['mse'],records[0]['cpu_audit']['gpu_mse'])
                del cross_model
        if cross_error>1e-10:
            raise ValueError('cross-GPU precision failure')
        budget()
        passed = all(r['checks']['noiseless_passed'] and r['checks']['noise_variance_passed'] for r in records)
        result = dict(status='geometry_checks_passed' if passed else 'model_checks_failed',completed_utc=now(),
            devices=[r['checks'] for r in records],cross_gpu_scaled_error=cross_error,total_skies=263168,
            detector_pixel_samples=263168*8*512*512,accepted_cosmological_claims=[],
            limits='Declared periodic tangent-plane Stokes instrument, not actual Planck TOD, radial depth, validated optics or a new physical discovery',
            plan_sha256=sha(OUT/'plan.json'))
        save(OUT/'result.json',result)
        stage.update(status=result['status'],completed_utc=now(),elapsed_seconds=time.perf_counter()-start,
            numerical_failures=0,result_sha256=sha(OUT/'result.json'))
        save(OUT/'stage_audit.json',stage)
        print(json.dumps(dict(status=result['status'],seconds=stage['elapsed_seconds'],cross_gpu_error=cross_error)),flush=True)
    except Exception as exc:
        stopped.set()
        stage.update(status='incomplete' if isinstance(exc,TimeoutError) else 'execution_failed',completed_utc=now(),
            elapsed_seconds=time.perf_counter()-start,error=traceback.format_exc())
        save(OUT/'stage_audit.json',stage)
        raise
    finally:
        stopped.set(); monitor_thread.join(timeout=6)
        save(OUT/'telemetry.json',dict(samples=telemetry,scope='Sampled device-wide counters; other applications may contribute'))


if __name__=='__main__':
    run()
