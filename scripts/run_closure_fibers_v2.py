"""One separately frozen batched search with pipelined all-start direct replay."""
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cmbcst.closure_fibers import features_numpy, robust_scales, projection_indices
from cmbcst.closure_fibers_data import sample_reference, extract_tile_reference, build_family, input_hashes, sha, replay_candidate
from cmbcst.closure_search import TILES, STAGES, manifest, initial_kicks
from cmbcst.closure_search_v2 import projected_batch
from cmbcst.closure_gpu_v2 import ClosureBatchCUDA
from cmbcst.closure_certification_v2 import certify_case_v2, TIERS

OUT = ROOT/'results/closure_fibers_v2'
WORK = ROOT/'data/work/closure_fibers_v2'
PROFILE = ROOT/'data/evidence/closure_v2_preflight_final_hybrid.json'
CAP, RESERVE = 14400., 900.


class BudgetCPULock:
    """Do not start a queued factorization after the search reserve boundary."""
    def __init__(self, lock, check):
        self.lock, self.check = lock, check

    def __enter__(self):
        self.check()
        while not self.lock.acquire(timeout=.05):
            self.check()
        try:
            self.check()
        except BaseException:
            self.lock.release()
            raise
        return self

    def __exit__(self, *exception):
        self.lock.release()


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def save_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def v1_bindings():
    paths = sorted((ROOT/'results/closure_fibers_v1').rglob('*'))
    paths += [ROOT/'docs/CLOSURE_FIBERS_SPEC.md']
    return {p.relative_to(ROOT).as_posix():sha(p) for p in paths if p.is_file()}


def source_paths():
    names = ['closure_fibers.py','closure_fibers_data.py','closure_fibers_oracle.py',
        'closure_jacobian.py','closure_search.py','closure_search_v2.py',
        'closure_gpu_v2.py','closure_certification_v2.py']
    paths = ['src/cmbcst/'+name for name in names]
    paths += ['scripts/run_closure_fibers_v2.py','scripts/prepare_closure_fibers_v2.py',
        'docs/CLOSURE_FIBERS_V2_SPEC.md','formal/OperationalClosure.lean',
        'tests/sandbox/jovin_tests_closure_fibers_v2.py']
    paths += ['tests/'+name for name in ('test_closure_gpu_v2.py',
        'test_closure_search_v2.py','test_closure_certification_v2.py','test_closure_driver_v2.py',
        'test_closure_reporting_v2.py')]
    return paths


def freeze():
    import torch, ducc0
    from jovin_verify import SUITES, build_bundle
    OUT.mkdir(parents=True, exist_ok=True); WORK.mkdir(parents=True, exist_ok=True)
    if (OUT/'plan.json').exists():
        raise FileExistsError('preserve v2 freeze; duplicate launch/restart forbidden')
    profile = json.loads(PROFILE.read_text())
    if profile['status'] != 'passed' or not profile.get('source_unchanged'):
        raise ValueError('v2 actual-device preflight did not pass')
    if any(sha(ROOT/p) != h for p,h in profile['source_sha256'].items()):
        raise ValueError('v2 preflight no longer binds current sources')
    proof_path = ROOT/'data/evidence/closure_lean.json'
    proof = json.loads(proof_path.read_text())
    if proof['status'] != 'passed' or proof['source_sha256'] != sha(ROOT/'formal/OperationalClosure.lean'):
        raise ValueError('pinned conditional proof binding failed')
    history = json.loads((ROOT/'data/evidence/jovin_receipts.json').read_text())
    receipt = [r for r in history if r['suite']=='closure_fibers_v2'][-1]
    bundle_hash = hashlib.sha256(build_bundle('closure_fibers_v2')[0].encode()).hexdigest()
    if (not all(receipt[k] for k in ('passed','receipt_verified','chain_valid'))
        or receipt['bundle_sha256'] != bundle_hash
        or receipt['tests_sha256'] != sha(ROOT/SUITES['closure_fibers_v2'][1])):
        raise ValueError('v2 current source/test JoVin receipt missing or mismatched')
    authentication_path=ROOT/'data/evidence/closure_v2_jovin_authentication.json'
    authentication=json.loads(authentication_path.read_text())
    if (authentication['status']!='passed' or set(authentication['suites'])!=set(SUITES)
        or not all(v['matched'] for v in authentication['suites'].values())):
        raise ValueError('all current JoVin suites must authenticate before freeze')
    for name,(_,tests) in SUITES.items():
        bound=authentication['suites'][name]
        if (bound['bundle_sha256']!=hashlib.sha256(build_bundle(name)[0].encode()).hexdigest()
            or bound['tests_sha256']!=sha(ROOT/tests)):
            raise ValueError('JoVin authentication is stale for '+name)
    paths = source_paths()
    hashes = {p:sha(ROOT/p) for p in paths}
    for p,h in hashes.items():
        if hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+p], cwd=ROOT)).hexdigest()!=h:
            raise ValueError('commit scientific source before freeze: '+p)
    kicks = {}
    for tile in TILES:
        for phase in ('source','fresh'):
            raw, signs = initial_kicks(tile,phase)
            kicks[f'tile_{tile}_{phase}'] = raw
            kicks[f'signs_{tile}_{phase}'] = signs
    save_npz(OUT/'initializations.npz', **kicks)
    plan = dict(created_utc=now(), source_revision=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(), seed=82024102027,
        scope='Prospectively frozen v2 finite-precision counterfactual test, informed by v1; no cosmological detection',
        selected_tiles=list(TILES), cases=[manifest(t,s,r) for t in TILES for s,r in STAGES],
        planned_starts=6144, starts_per_case=128, planes=128,
        shells=np.rint(np.linspace(30,216,32)).astype(int).tolist(),
        batch_starts={'cuda:0':128,'cuda:1':64},
        tiers=list(TIERS), tier_units='raw dimensionless normalized K4, not robust-scale bin widths',
        local_quantum=.001, certificate_guard=1e-5, search_guard=1e-5, center_tolerance=1e-6,
        active_rms_threshold=.24, rms_ratio_limit=.25, rms_ratio_names=['tile','global'],
        angle_limit=float(np.pi/4), svd_relative_cutoff=1e-10, max_iterations=1000,
        max_stalls=50, step_norm=.1, newton_corrections=5, step_halvings=10,
        svd_driver=None, svd_backend='cpu',
        cap_seconds=CAP, compute_cap_seconds=CAP, verification_reserve_seconds=RESERVE,
        clock_gate='max(monotonic elapsed, UTC elapsed); terminate only this study process',
        cpu_verification_workers=1, threads=8, blas_threads=1,
        sampling_epsilon=1e-12, numerical_gate=1e-10, spectral_quantum=1e-6,
        source_sha256=hashes, input_sha256=input_hashes(ROOT),
        v1_frozen_sha256=v1_bindings(), initialization_sha256=sha(OUT/'initializations.npz'),
        preflight_path=PROFILE.relative_to(ROOT).as_posix(), preflight_sha256=sha(PROFILE),
        jovin_receipt_id=receipt['receipt_id'], jovin_bundle_sha256=bundle_hash,
        jovin_authentication_path=authentication_path.relative_to(ROOT).as_posix(),
        jovin_authentication_sha256=sha(authentication_path),
        proof_path=proof_path.relative_to(ROOT).as_posix(), proof_sha256=sha(proof_path),
        python=sys.version, numpy=np.__version__, torch=str(torch.__version__),
        cuda=torch.version.cuda, ducc0=ducc0.__version__, devices=[torch.cuda.get_device_name(i) for i in (0,1)])
    with (OUT/'plan.json').open('x',encoding='utf-8') as stream:
        json.dump(plan,stream,indent=2,allow_nan=False)
    return plan


def main():
    import torch
    torch.set_num_threads(8)
    plan=freeze(); tick=time.perf_counter(); utc_start=datetime.now(timezone.utc)
    stop=threading.Event(); abort=threading.Event(); lock=threading.RLock(); cpu_heavy=threading.Lock(); progress={}; telemetry=[]
    records=[]; certificates=[]; verification_futures=[]
    stage=dict(status='running', started_utc=utc_start.isoformat(), plan_sha256=sha(OUT/'plan.json'), stages={})
    def clocks():
        elapsed=time.perf_counter()-tick
        utc_elapsed=(datetime.now(timezone.utc)-utc_start).total_seconds()
        return dict(elapsed_seconds=elapsed, utc_elapsed_seconds=utc_elapsed,
            budget_elapsed_seconds=max(elapsed,utc_elapsed))
    def budget(search=False):
        if abort.is_set():
            raise RuntimeError('peer execution failed; preserve v2 checkpoints')
        if clocks()['budget_elapsed_seconds'] >= CAP-(RESERVE if search else 0):
            raise TimeoutError('v2 search verification reserve reached' if search else 'v2 production cap reached')
    def write_stage():
        with lock: save(OUT/'stage_audit.json',stage)
    def update(key,value):
        with lock:
            progress[key]={**value, **clocks()}
            save(OUT/'progress.json',dict(devices=dict(progress), **clocks()))
    search_cpu_lock=BudgetCPULock(cpu_heavy,lambda:budget(True))
    def monitor():
        while not stop.is_set():
            item=clocks()
            try:
                result=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw,temperature.gpu',
                    '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
                item['devices']=[dict(index=int(r[0]),utilization=float(r[1]),memory_MiB=float(r[2]),
                    power_W=float(r[3]),temperature_C=float(r[4])) for r in csv.reader(result.stdout.splitlines()) if len(r)==5]
            except Exception as exc: item['error']=str(exc)
            telemetry.append(item)
            if len(telemetry)%30==0: save(OUT/'telemetry.json',dict(samples=telemetry,scope='Device-wide sampled counters'))
            stop.wait(1.)
    def deadline():
        while not stop.is_set():
            timing=clocks()
            if timing['budget_elapsed_seconds']>=CAP:
                with lock:
                    stage.update(status='incomplete_hard_cap', completed_utc=now(), **timing)
                    save(OUT/'hard_cap.json',stage); write_stage()
                os._exit(124)
            stop.wait(.25)
    monitor_thread=threading.Thread(target=monitor,daemon=True); monitor_thread.start()
    threading.Thread(target=deadline,daemon=True).start(); write_stage()
    try:
        preprocessing=time.perf_counter(); update('cpu',dict(phase='preprocessing'))
        shared=sample_reference(ROOT,WORK,budget)
        accepted=[int(t) for t in shared['accepted_tiles']]
        references=np.stack([features_numpy(extract_tile_reference(shared,t),np.zeros(0)) for t in accepted])
        scales=robust_scales(references)
        save_npz(OUT/'reference_features.npz',tile_ids=accepted,features=references,scales=scales)
        save(OUT/'normalization_lock.json',dict(created_utc=now(),plan_sha256=sha(OUT/'plan.json'),
            reference_features_sha256=sha(OUT/'reference_features.npz'),accepted_tiles=accepted,
            rule='max(1.4826*MAD, .001*max(1,abs(median)))'))
        (OUT/'operators').mkdir(exist_ok=True)
        for tile in TILES:
            budget(); build_family(ROOT,tile,WORK,budget,shared)
            for name in (f'planes_{tile}.npz',f'family_{tile}.json'):
                shutil.copyfile(WORK/name,OUT/'operators'/name)
        stage['stages']['preprocessing_seconds']=time.perf_counter()-preprocessing; write_stage()
        def verify_case(case):
            budget(); update('cpu',dict(phase='direct_verification',case=case['name']))
            family=build_family(ROOT,case['tile'],WORK,budget,shared)
            with np.load(OUT/case['arrays'],allow_pickle=False) as source:
                arrays={k:source[k] for k in source.files}
            def direct_replay(*args,**kwargs):
                budget()
                with cpu_heavy:
                    budget()
                    return replay_candidate(*args,**kwargs)
            try:
                certificate=certify_case_v2(ROOT,OUT,WORK,case,family,shared,scales,arrays,budget,
                    replay_function=direct_replay)
                compact=OUT/'certificates_v2'/case['name']/'case.json'
                if compact.exists():
                    certificate=json.loads(compact.read_text())
            except Exception as exc:
                certificate=dict(status='incomplete' if isinstance(exc,TimeoutError) else 'execution_failed',
                    case=case, error=traceback.format_exc(),tier_counts={f'{t:.3f}':0 for t in TIERS})
                save(OUT/'certificates_v2'/case['name']/'worker_failure.json',certificate)
            with lock: certificates.append(certificate)
            return certificate
        def worker(index,cpu_pool):
            device=f'cuda:{index}'; batch=plan['batch_starts'][device]
            with torch.cuda.device(index):
                for tile in TILES:
                    if manifest(tile,*STAGES[0])['device']!=index: continue
                    for phase,representation in STAGES:
                        case={**manifest(tile,phase,representation),'name':f'tile_{tile}_{phase}_{representation}'}
                        if clocks()['budget_elapsed_seconds']>=CAP-RESERVE:
                            with lock: records.append({**case,'status':'not_started_reserve','starts_started':0,'starts_completed':0})
                            continue
                        started=time.perf_counter(); family=build_family(ROOT,tile,WORK,lambda:budget(True),shared)
                        model=ClosureBatchCUDA(family,device); ix=projection_indices(representation)
                        ref=features_numpy(family,np.zeros(128)); ref_t=torch.as_tensor(ref,device=device)
                        scale_t=torch.as_tensor(scales,device=device)
                        feature=lambda x:(model.features(x)[...,ix]-ref_t[ix])/scale_t[ix]
                        jacobian=lambda x:model.feature_jacobian(x,indices=ix)/scale_t[ix,None]
                        raw,signs=initial_kicks(tile,phase)
                        arrays=dict(raw_kicks=raw,signs=signs,started=np.zeros(128,dtype=np.int64),
                            theta=np.zeros((128,128)),initial_theta=np.zeros((128,128)),best_theta=np.zeros((128,128)),
                            status=np.full(128,'not_started',dtype='<U64'),iterations=np.zeros(128,dtype=np.int64),
                            stalls=np.zeros(128,dtype=np.int64),gpu_features=np.full((128,52),np.nan),
                            gpu_all_features=np.full((128,4,52),np.nan),gpu_target=np.full(128,np.nan),
                            gpu_ratios=np.full((128,2),np.nan))
                        history=[]; errors=[]; path=OUT/'cases'/f"{case['name']}.npz"
                        def persist(): save_npz(path,**arrays)
                        for first in range(0,128,batch):
                            if clocks()['budget_elapsed_seconds']>=CAP-RESERVE: break
                            ids=np.arange(first,min(first+batch,128)); arrays['started'][ids]=1; arrays['status'][ids]='active'
                            latest={}
                            def checkpoint(snapshot):
                                latest.clear(); latest.update(snapshot)
                                checkpoint_path=OUT/'checkpoints'/f"{case['name']}.npz"
                                save_npz(checkpoint_path,checkpoint_start_ids=ids,
                                    checkpoint_first_start=np.array(first,dtype=np.int64),
                                    checkpoint_phase=np.array(snapshot.get('checkpoint_phase','unknown')),
                                    **{k:v for k,v in snapshot.items() if isinstance(v,np.ndarray)})
                                save(OUT/'checkpoints'/f"{case['name']}.json",dict(first_start=int(first),
                                    checkpoint_sha256=sha(checkpoint_path),history=snapshot.get('history',[]),
                                    phase=snapshot.get('checkpoint_phase'),**clocks()))
                                update(device,dict(phase='search',case=case['name'],first_start=int(first),
                                    iteration=int(np.max(snapshot.get('iterations',[0])))))
                            try:
                                result=projected_batch(feature,model.target,model.ratios,raw[ids],signs[ids],device=device,
                                    budget=lambda:budget(True),checkpoint=checkpoint,jacobian_function=jacobian,
                                    target_gradient_function=model.target_gradient,rms_gradient_function=model.rms_gradient,
                                    svd_driver=plan['svd_driver'],svd_backend=plan['svd_backend'],cpu_lock=search_cpu_lock)
                            except Exception as exc:
                                errors.append(dict(first_start=int(first),type=type(exc).__name__,error=traceback.format_exc()))
                                result=latest or {}
                            failure_payload={key:value for key,value in result.items()
                                if key.startswith('failure_') and isinstance(value,np.ndarray)}
                            if failure_payload:
                                save_npz(OUT/'failures'/f"{case['name']}_start_{first}.npz",**failure_payload)
                            for key,value in result.items():
                                if key.startswith('failure_'): continue
                                if not isinstance(value,np.ndarray) or value.shape[:1]!=(len(ids),): continue
                                if key in ('raw_kicks','signs','started'): continue
                                if key not in arrays: arrays[key]=np.zeros((128,)+value.shape[1:],dtype=value.dtype)
                                arrays[key][ids]=value
                            arrays['status'][ids]=np.where(arrays['status'][ids]=='active','incomplete',arrays['status'][ids])
                            with torch.no_grad():
                                a=torch.as_tensor(arrays['initial_theta'][ids],device=device)
                                b=torch.as_tensor(arrays['theta'][ids],device=device)
                                zero=model.target(torch.zeros((1,128),dtype=torch.float64,device=device))
                                yi=model.target(a); yf=model.target(b)
                                best=torch.where((torch.abs(yi-zero)>torch.abs(yf-zero))[:,None],a,b)
                                arrays['best_theta'][ids]=best.cpu().numpy()
                                all_values=torch.stack([model.features(best,u) for u in range(4)],dim=1).cpu().numpy()
                                arrays['gpu_all_features'][ids]=all_values; arrays['gpu_features'][ids]=all_values[:,0]
                                arrays['gpu_target'][ids]=all_values[:,1,22]
                                arrays['gpu_ratios'][ids]=model.ratios(best).cpu().numpy()
                            history += [{**h,'first_start':int(first)} for h in result.get('history',[])]; persist()
                            print(json.dumps(dict(event='search_batch_saved',case=case['name'],first_start=int(first),**clocks())),flush=True)
                            if errors and errors[-1]['type']=='TimeoutError': break
                        persist()
                        complete=int(np.sum(np.isin(arrays['status'],['stalled','iteration_cap'])))
                        numeric_failures=int(np.sum((arrays['started']>0)&~np.isin(arrays['status'],['stalled','iteration_cap','incomplete'])))
                        record={**case,'status':'complete' if complete==128 and not errors else ('execution_failed' if numeric_failures or any(e['type']!='TimeoutError' for e in errors) else 'incomplete'),
                            'starts_started':int(arrays['started'].sum()),'starts_completed':complete,'errors':errors,
                            'terminal_failure_starts':numeric_failures,
                            'seconds':time.perf_counter()-started,'arrays':path.relative_to(OUT).as_posix(),
                            'arrays_sha256':sha(path),'rms_ratio_names':['tile','global']}
                        save(path.with_suffix('.json'),record); save(path.with_name(path.stem+'_history.json'),dict(records=history))
                        with lock: records.append(record)
                        if record['starts_started']:
                            with lock: verification_futures.append(cpu_pool.submit(verify_case,record))
                        del model
        searching=time.perf_counter()
        with ThreadPoolExecutor(max_workers=1) as cpu_pool:
            with ThreadPoolExecutor(max_workers=2) as gpu_pool:
                workers=[gpu_pool.submit(worker,i,cpu_pool) for i in (0,1)]
                while True:
                    done,_=wait(workers,timeout=1)
                    errors=[f.exception() for f in done if f.exception() is not None]
                    if errors:
                        abort.set()
                        raise errors[0]
                    if len(done)==2: break
                    budget()
                for future in workers: future.result()
            stage['stages']['search_wall_seconds']=time.perf_counter()-searching; write_stage()
            save(OUT/'cases.json',dict(records=records))
            while len(wait(verification_futures,timeout=1)[0])<len(verification_futures): budget()
            for future in verification_futures: future.result()
        budget()
        preserved=all(sha(ROOT/p)==h for p,h in plan['v1_frozen_sha256'].items())
        failures=sum(sum(e['type']!='TimeoutError' for e in r.get('errors',[])) for r in records)
        for r in records:
            if r.get('arrays'):
                with np.load(OUT/r['arrays'],allow_pickle=False) as a:
                    failures+=int(np.sum((a['started']>0)&~np.isin(a['status'],['stalled','iteration_cap','incomplete'])))
        failures+=sum(c.get('verification_failures',0)+int(c.get('status') in ('execution_failed','precision_failure','verification_failed')) for c in certificates)
        completed=sum(r['starts_completed'] for r in records); begun=sum(r['starts_started'] for r in records)
        tiers={f'{t:.3f}':sum(c.get('tier_counts',{}).get(f'{t:.3f}',0) for c in certificates) for t in TIERS}
        verified_all=len(certificates)==48 and all(c.get('status')=='complete' and not c.get('unverified_start_ids') for c in certificates)
        status=('complete' if not failures else 'complete_with_failures') if completed==6144 and verified_all else ('incomplete_with_failures' if failures else 'incomplete')
        if not preserved: status='preservation_failed'
        result=dict(status=status,planned_starts=6144,starts_started=begun,starts_completed=completed,
            scientific_outcome='verified_tier_witnesses' if any(tiers.values()) else 'no_verified_tier_witness',
            tier_counts=tiers,numerical_or_execution_failures=failures,v1_preserved=preserved,
            cases=records,certificates=certificates,plan_sha256=sha(OUT/'plan.json'),
            scientific_completed_utc=now(),**clocks())
        save(OUT/'result.json',result)
        stage.update(status=status,completed_utc=now(),scientific_completed_utc=result['scientific_completed_utc'],
            result_sha256=sha(OUT/'result.json'),**clocks()); write_stage()
        print(json.dumps(dict(event='scientific_run_terminal',status=status,tier_counts=tiers,**clocks())),flush=True)
    except Exception:
        stage.update(status='incomplete' if clocks()['budget_elapsed_seconds']>=CAP-RESERVE else 'execution_failed',
            completed_utc=now(),error=traceback.format_exc(),**clocks()); write_stage(); raise
    finally:
        stop.set(); monitor_thread.join(timeout=6)
        save(OUT/'telemetry.json',dict(samples=telemetry,scope='Device-wide counters; other applications may contribute'))


if __name__=='__main__':
    main()
