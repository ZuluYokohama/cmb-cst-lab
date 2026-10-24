"""One frozen four-hour CST search, preserving partial starts and every failure."""
from concurrent.futures import ThreadPoolExecutor,wait
from datetime import datetime,timezone
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

BLAS_THREAD_VARIABLES=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')
for name in BLAS_THREAD_VARIABLES:
    os.environ[name]='1'

import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cmbcst.closure_fibers import features_numpy,robust_scales,projection_indices,ClosureCUDA
from cmbcst.closure_fibers_data import sample_reference,extract_tile_reference,build_family,input_hashes,sha
from cmbcst.closure_search import TILES,STAGES,manifest,initial_kicks,projected_batch
from cmbcst.closure_certification import certify_case
OUT=ROOT/'results/closure_fibers_v1'
WORK=ROOT/'data/work/closure_fibers_v1'
PROFILE=ROOT/'data/evidence/closure_gpu_preflight_v3.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def save_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,**arrays)
    temporary.replace(path)


def freeze():
    import torch,ducc0
    from jovin_verify import SUITES,build_bundle
    OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
    if (OUT/'plan.json').exists():
        raise FileExistsError('preserve scientific freeze: do not duplicate or restart budget')
    preflight=json.loads(PROFILE.read_text())
    if preflight['status']!='passed' or any(sha(ROOT/name)!=digest for name,digest in preflight['source_sha256'].items()):
        raise ValueError('current scientific source lacks matching actual GPU preflight')
    proof=json.loads((ROOT/'data/evidence/closure_lean.json').read_text())
    if proof['status']!='passed' or proof['source_sha256']!=sha(ROOT/'formal/OperationalClosure.lean'):
        raise ValueError('conditional proof has not passed pinned Lean')
    history=json.loads((ROOT/'data/evidence/jovin_receipts.json').read_text())
    receipt=[r for r in history if r['suite']=='closure_fibers'][-1]
    if not all(receipt[k] for k in ('passed','receipt_verified','chain_valid')) or receipt['bundle_sha256']!=hashlib.sha256(build_bundle('closure_fibers')[0].encode()).hexdigest() or receipt['tests_sha256']!=sha(ROOT/SUITES['closure_fibers'][1]):
        raise ValueError('current source/test bundle lacks authentic JoVin receipt')
    paths=['src/cmbcst/'+name for name in ('closure_fibers.py','closure_fibers_data.py','closure_fibers_oracle.py','closure_search.py','closure_certification.py','closure_jacobian.py')]
    paths+=['scripts/run_closure_fibers.py','scripts/prepare_closure_fibers.py','scripts/verify_closure_lean.py','formal/OperationalClosure.lean','docs/CLOSURE_FIBERS_SPEC.md','tests/sandbox/jovin_tests_closure_fibers.py']
    paths+=['tests/'+name for name in ('test_closure_fibers.py','test_closure_fibers_data.py','test_closure_oracle.py','test_closure_search.py','test_closure_certification.py')]
    cases=[manifest(tile,stage,representation) for tile in TILES for stage,representation in STAGES]
    kicks={}
    for tile in TILES:
        for stage in ('source','fresh'):
            raw,signs=initial_kicks(tile,stage)
            kicks[f'tile_{tile}_{stage}']=raw
            kicks[f'signs_{tile}_{stage}']=signs
    np.savez_compressed(OUT/'initializations.npz',**kicks)
    plan=dict(created_utc=now(),scope='CST finite-precision counterfactual closure, not a new cosmological detection',
        base_revision='6797b5634ca0bb54d4c0f625ff37e4c765bfeccd',seed=82024102027,
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        selected_tiles=list(TILES),cases=cases,planned_starts=6144,starts_per_case=128,
        planes=128,shells=np.rint(np.linspace(30,216,32)).astype(int).tolist(),
        local_quantum=.001,spectral_quantum=1e-6,certificate_guard=1e-7,search_guard=2e-7,
        rms_ratio_limit=.25,angle_limit=float(np.pi/4),witness_gap=.05,
        max_iterations=1000,max_stalls=50,svd_relative_cutoff=1e-10,step_norm=.1,newton_corrections=5,step_halvings=10,
        compute_cap_seconds=14400,verification_reserve_seconds=900,batch_starts=4,
        sampling_epsilon=1e-12,numerical_gate=1e-10,threads=8,
        blas_threads={name:int(os.environ[name]) for name in BLAS_THREAD_VARIABLES},
        input_sha256=input_hashes(ROOT),source_sha256={name:sha(ROOT/name) for name in paths},
        initialization_sha256=sha(OUT/'initializations.npz'),preflight_sha256=sha(PROFILE),
        preflight_path=str(PROFILE.relative_to(ROOT)),
        proof_sha256=sha(ROOT/'data/evidence/closure_lean.json'),jovin_receipt_id=receipt['receipt_id'],jovin_bundle_sha256=receipt['bundle_sha256'],
        real_operator_preflight_sha256=sha(ROOT/'data/evidence/closure_real_operator_preflight.json'),
        numpy=np.__version__,torch=str(torch.__version__),cuda=torch.version.cuda,ducc0=ducc0.__version__,python=sys.version,
        devices=[torch.cuda.get_device_name(i) for i in (0,1)])
    with (OUT/'plan.json').open('x') as stream:
        json.dump(plan,stream,indent=2,allow_nan=False)
    return plan


def main():
    import torch
    from cmbcst.closure_jacobian import feature_jacobian,ClosureTargetCUDA
    torch.set_num_threads(8)
    plan=freeze();start=time.perf_counter()
    stop=threading.Event();abort=threading.Event();progress_lock=threading.Lock();stage_lock=threading.RLock();progress={};telemetry=[];records=[]
    stage=dict(status='running',started_utc=now(),plan_sha256=sha(OUT/'plan.json'),stages={})
    def write_stage(value):
        with stage_lock:
            save(OUT/'stage_audit.json',value)
    write_stage(stage)
    def budget():
        if abort.is_set():
            raise RuntimeError('peer execution failed; preserve remaining work')
        if time.perf_counter()-start>=14400:
            raise TimeoutError('scientific four-hour cap reached')
    def search_budget():
        if abort.is_set():
            raise RuntimeError('peer execution failed; preserve search checkpoint')
        if time.perf_counter()-start>=13500:
            raise TimeoutError('scientific verification reserve reached')
    def update(key,value):
        with progress_lock:
            progress[key]={**value,'seconds':time.perf_counter()-start}
            save(OUT/'progress.json',dict(devices=dict(progress),elapsed_seconds=time.perf_counter()-start))
    def hard_deadline():
        while not stop.is_set():
            elapsed=time.perf_counter()-start
            if elapsed>=14400:
                # This terminates this single study process, never shared WSL.
                # Checkpoints/finished cases survive; no unverified candidate is
                # promoted merely because its solver completed before timeout.
                marker={**stage,'status':'incomplete_hard_cap','elapsed_seconds':elapsed,
                    'completed_utc':now(),'reason':'Own study process reached the fixed four-hour wall cap'}
                save(OUT/'hard_cap.json',marker)
                write_stage(marker)
                os._exit(124)
            stop.wait(min(.25,max(14400-elapsed,0.)))
    def monitor():
        while not stop.is_set():
            elapsed=time.perf_counter()-start
            try:
                query=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
                telemetry.append(dict(seconds=elapsed,devices=[dict(index=int(r[0]),utilization=float(r[1]),memory_MiB=float(r[2]),power_W=float(r[3]),temperature_C=float(r[4])) for r in csv.reader(query.stdout.splitlines()) if len(r)==5]))
            except Exception as exc:
                telemetry.append(dict(seconds=elapsed,error=str(exc)))
            if len(telemetry)%30==0:
                save(OUT/'telemetry.json',dict(samples=telemetry,scope='Device-wide counters; periodic checkpoint, other applications may contribute'))
            stop.wait(1.)
    monitor_thread=threading.Thread(target=monitor,daemon=True);monitor_thread.start()
    deadline_thread=threading.Thread(target=hard_deadline,daemon=True);deadline_thread.start()
    try:
        tick=time.perf_counter();update('cpu',dict(phase='reference_preprocessing'))
        shared=sample_reference(ROOT,WORK,budget)
        accepted=[int(tile) for tile in shared['accepted_tiles']]
        reference_features=np.stack([features_numpy(extract_tile_reference(shared,tile),np.zeros(0)) for tile in accepted])
        scales=robust_scales(reference_features)
        np.savez_compressed(OUT/'reference_features.npz',tile_ids=accepted,features=reference_features,scales=scales)
        save(OUT/'normalization_lock.json',dict(created_utc=now(),plan_sha256=sha(OUT/'plan.json'),
            reference_features_sha256=sha(OUT/'reference_features.npz'),accepted_tiles=accepted,
            rule='max(1.4826*MAD, .001*max(1,abs(median)))',scope='Fixed before any scientific search starts'))
        (OUT/'operators').mkdir(exist_ok=True)
        for tile in TILES:
            budget();update('cpu',dict(phase='family_preprocessing',tile=tile))
            family=build_family(ROOT,tile,WORK,budget,shared)
            for suffix in (f'planes_{tile}.npz',f'family_{tile}.json'):
                shutil.copyfile(WORK/suffix,OUT/'operators'/suffix)
        stage['stages']['preprocessing_seconds']=time.perf_counter()-tick
        write_stage(stage)
        def worker(index):
            local_records=[]
            with torch.cuda.device(index):
                for case in plan['cases']:
                    if case['device']!=index:
                        continue
                    if abort.is_set():
                        local_records.append({**case,'status':'not_started_peer_failure','starts_completed':0,'starts_started':0})
                        continue
                    if time.perf_counter()-start>=13500:
                        local_records.append({**case,'status':'not_started_reserve','starts_completed':0,'starts_started':0})
                        continue
                    tile=case['tile'];name=f"tile_{tile}_{case['stage']}_{case['representation']}"
                    family=build_family(ROOT,tile,WORK,search_budget,shared)
                    model=ClosureCUDA(family,f'cuda:{index}')
                    ref=features_numpy(family,np.zeros(128))
                    ix=projection_indices(case['representation'])
                    device_ix=torch.as_tensor(ix,device=index)
                    ref_tensor=torch.as_tensor(ref,device=index);scale_tensor=torch.as_tensor(scales,device=index)
                    feature=lambda theta:(model.features(theta)[device_ix]-ref_tensor[device_ix])/scale_tensor[device_ix]
                    target_model=ClosureTargetCUDA(model,1,2)
                    target=target_model.value
                    fast_jac=lambda theta:feature_jacobian(model,theta)[...,device_ix,:]/scale_tensor[device_ix,None]
                    kicks,signs=initial_kicks(tile,case['stage'])
                    arrays=dict(theta=np.zeros((128,128)),best_theta=np.zeros((128,128)),initial_theta=np.zeros((128,128)),
                        raw_kicks=kicks,signs=signs,started=np.zeros(128,dtype=np.int64),
                        iterations=np.zeros(128,dtype=np.int64),stalls=np.zeros(128,dtype=np.int64),rejected=np.zeros(128,dtype=np.int64),
                        nonfinite_trials=np.zeros(128,dtype=np.int64),rank=np.zeros(128,dtype=np.int64),
                        status=np.full(128,'not_started',dtype='<U32'),gpu_features=np.full((128,52),np.nan),
                        gpu_target=np.full(128,np.nan),gpu_ratios=np.full((128,2),np.nan))
                    counters=('nonfinite_constraint_trials','nonfinite_ratio_trials','nonfinite_target_trials',
                        'nonfinite_jacobian_evaluations','nonfinite_gradient_evaluations','proposal_evaluations','retraction_corrections')
                    arrays.update({key:np.zeros(128,dtype=np.int64) for key in counters})
                    arrays['initial_retraction_succeeded']=np.zeros(128,dtype=bool)
                    directory=OUT/'cases';directory.mkdir(exist_ok=True)
                    histories=[];case_error=None;case_tick=time.perf_counter()
                    def persist():
                        save_npz(directory/f'{name}.npz',**arrays)
                    for first in range(0,128,4):
                        if time.perf_counter()-start>=13500:
                            break
                        ids=np.arange(first,min(first+4,128))
                        arrays['started'][ids]=1;arrays['status'][ids]='active'
                        latest={}
                        def checkpoint(snapshot):
                            latest.clear();latest.update(snapshot)
                            save_npz(OUT/'checkpoints'/f'{name}.npz',**{k:v for k,v in snapshot.items() if isinstance(v,np.ndarray)})
                            save(OUT/'checkpoints'/f'{name}.json',dict(first_start=int(first),history=snapshot['history'],
                                checkpoint_phase=snapshot.get('checkpoint_phase'),interruption=snapshot.get('interruption'),
                                initialization_complete=snapshot.get('initialization_complete'),seconds=time.perf_counter()-start))
                            update(f'cuda:{index}',dict(phase='search',tile=tile,stage=case['stage'],representation=case['representation'],
                                first_start=int(first),iteration=int(np.max(snapshot['iterations'])),completed=int(np.sum(np.isin(arrays['status'],['stalled','iteration_cap'])))))
                        try:
                            result=projected_batch(feature,target,model.ratios,kicks[ids],signs[ids],device=f'cuda:{index}',
                                budget=search_budget,checkpoint=checkpoint,jacobian_function=fast_jac,
                                target_gradient_function=target_model.gradient)
                        except Exception as exc:
                            if latest:
                                result=latest
                                result.setdefault('theta',np.zeros((len(ids),128)))
                            else:
                                result=dict(theta=np.zeros((len(ids),128)),initial_theta=np.zeros((len(ids),128)),iterations=np.zeros(len(ids),dtype=int),
                                    stalls=np.zeros(len(ids),dtype=int),status=np.full(len(ids),'incomplete',dtype='<U32'),history=[])
                            case_error=dict(type=type(exc).__name__,error=traceback.format_exc(),first_start=int(first))
                        if np.shape(result.get('initial_theta'))!=(len(ids),128):
                            result['initial_theta']=np.zeros((len(ids),128))
                        for key in ('theta','initial_theta','iterations','stalls','rejected','nonfinite_trials','rank','status','initial_retraction_succeeded',*counters):
                            if key in result:
                                arrays[key][ids]=result[key]
                        with torch.no_grad():
                            initial_tensor=torch.as_tensor(arrays['initial_theta'][ids],device=index)
                            final_tensor=torch.as_tensor(arrays['theta'][ids],device=index)
                            original_target=float(target_model.value(torch.zeros(128,dtype=torch.float64,device=index)).item())
                            yi=target_model.value(initial_tensor).cpu().numpy()
                            yf=target_model.value(final_tensor).cpu().numpy()
                            use_initial=np.abs(yi-original_target)>np.abs(yf-original_target)
                            best=np.where(use_initial[:,None],arrays['initial_theta'][ids],arrays['theta'][ids])
                            arrays['best_theta'][ids]=best
                            bt=torch.as_tensor(best,device=index)
                            arrays['gpu_features'][ids]=model.features(bt).cpu().numpy()
                            arrays['gpu_target'][ids]=target_model.value(bt).cpu().numpy()
                            arrays['gpu_ratios'][ids]=model.ratios(bt).cpu().numpy()
                        for h in result.get('history',[]):
                            histories.append({**h,'first_start':int(first)})
                        persist()
                        print(json.dumps(dict(device=index,tile=tile,stage=case['stage'],representation=case['representation'],
                            starts_finished=int(np.sum(np.isin(arrays['status'],['stalled','iteration_cap']))),seconds=time.perf_counter()-start)),flush=True)
                        if case_error:
                            break
                    persist()
                    completed=int(np.sum(np.isin(arrays['status'],['stalled','iteration_cap'])))
                    failed=int(np.sum(np.isin(arrays['status'],['nonfinite_jacobian','nonfinite_gradient'])))
                    record={**case,'status':'complete' if completed==128 else ('execution_failed' if failed or (case_error and case_error['type']!='TimeoutError') else 'incomplete'),
                        'starts_started':int(arrays['started'].sum()),'starts_completed':completed,'seconds':time.perf_counter()-case_tick,
                        'terminal_failure_starts':failed,'nonfinite_trial_count':int(arrays['nonfinite_trials'].sum()),
                        'arrays':str((directory/f'{name}.npz').relative_to(OUT)),'error':case_error}
                    save(directory/f'{name}.json',record)
                    save(directory/f'{name}_history.json',dict(records=histories))
                    local_records.append(record)
                    del model
            return local_records
        tick=time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(worker,i) for i in (0,1)]
            while True:
                done,_=wait(futures,timeout=2)
                failures=[f.exception() for f in done if f.exception() is not None]
                if failures:
                    abort.set()
                    raise failures[0]
                if len(done)==2:
                    break
            records=[r for f in futures for r in f.result()]
        stage['stages']['search_seconds']=time.perf_counter()-tick
        save(OUT/'cases.json',dict(records=records))
        tick=time.perf_counter();certificates=[]
        for record in records:
            if not record.get('arrays') or not record.get('starts_started'):
                continue
            budget();update('cpu',dict(phase='scientific_verification',tile=record['tile'],stage=record['stage'],representation=record['representation']))
            with np.load(OUT/record['arrays'],allow_pickle=False) as source:
                arrays={key:source[key] for key in source.files}
            family=build_family(ROOT,record['tile'],WORK,budget,shared)
            try:
                certificate=certify_case(ROOT,OUT,WORK,record,family,shared,scales,arrays,budget)
            except TimeoutError:
                raise
            except Exception:
                certificate=dict(status='precision_failure',case=record,error=traceback.format_exc())
                save(OUT/'certificates'/f"tile_{record['tile']}_{record['stage']}_{record['representation']}_failure.json",certificate)
            certificates.append(certificate)
        stage['stages']['scientific_verification_seconds']=time.perf_counter()-tick
        budget()
        witnesses=sum(c['status']=='verified_witness' for c in certificates)
        failed_starts=sum(r.get('terminal_failure_starts',0) for r in records)
        failures=failed_starts+sum(bool(r.get('error')) and r['error']['type']!='TimeoutError' for r in records)+sum(c['status']=='precision_failure' for c in certificates)
        completed=sum(r['starts_completed'] for r in records);begun=sum(r['starts_started'] for r in records)
        all_terminal=completed+failed_starts==6144
        status=('complete' if not failures else 'complete_with_failures') if all_terminal else ('incomplete_with_failures' if failures else 'incomplete')
        result=dict(status=status,scientific_outcome='verified_finite_precision_collision' if witnesses else 'no_verified_witness_found',
            planned_starts=6144,starts_started=begun,starts_completed=completed,verified_witnesses=witnesses,
            terminal_failure_starts=failed_starts,
            numerical_or_execution_failures=failures,cases=records,certificates=certificates,
            plan_sha256=sha(OUT/'plan.json'),completed_utc=now(),
            limits='Counterfactual finite-precision representation test; no continuous closure, universal repair, cosmological or priority claim')
        save(OUT/'result.json',result)
        stage.update(status=status,completed_utc=now(),elapsed_seconds=time.perf_counter()-start,
            result_sha256=sha(OUT/'result.json'),numerical_failures=failures)
        write_stage(stage)
        print(json.dumps(dict(status=status,outcome=result['scientific_outcome'],completed=completed,witnesses=witnesses,seconds=stage['elapsed_seconds'])),flush=True)
    except Exception:
        stage.update(status='incomplete' if time.perf_counter()-start>=13500 else 'execution_failed',
            completed_utc=now(),elapsed_seconds=time.perf_counter()-start,error=traceback.format_exc())
        write_stage(stage)
        raise
    finally:
        stop.set();monitor_thread.join(timeout=6)
        save(OUT/'telemetry.json',dict(samples=telemetry,scope='Device-wide counters, other applications may contribute'))


if __name__=='__main__':
    main()
