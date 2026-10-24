"""Two-stage, source-frozen physical blind-spot benchmark with a hard budget."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cmbcst import targeted as t, response, persistence
from cmbcst.targeted_sky import TargetedSky
from run_response_benchmark import save,sha,now,scaled_error,represent
from jovin_verify import SUITES,build_bundle

OUT = ROOT/'results/targeted_v1'
WORK = ROOT/'data/work/targeted_v1'
SEED = 876003142026


def recipes():
    result = []
    for family,prefix in zip(t.FAMILIES,('q','c','f')):
        variants = (-1,1) if family=='quadratic' else (.5,1.)
        for amplitude in (.1,.2,.3):
            for variant in variants:
                j = len([r for r in result if r['family']==family])
                result.append(dict(id=f'{prefix}{j:02d}',family=family,amplitude=amplitude,variant=variant))
    return result


def development_roles():
    bank = recipes()
    return ([dict(name='train_null',count=1024,offset=0,recipe=None)]
        +[dict(name='train_'+r['id'],count=64,offset=10000+1000*j,recipe=r) for j,r in enumerate(bank)]
        +[dict(name='dev_cal',count=499,offset=40000,recipe=None),dict(name='dev_null',count=256,offset=41000,recipe=None)]
        +[dict(name='dev_'+r['id'],count=32,offset=50000+1000*j,recipe=r) for j,r in enumerate(bank)])


def final_roles(selected):
    bank = {r['id']:r for r in recipes()}
    roles = [dict(name='final_cal',count=999,offset=80000,recipe=None),dict(name='final_null',count=1024,offset=82000,recipe=None)]
    for j,family in enumerate(t.FAMILIES):
        if selected[family] is not None:
            roles.append(dict(name='final_'+family,count=256,offset=90000+1000*j,recipe=bank[selected[family]]))
    for j,family in enumerate(t.FAMILIES):
        recipe = dict(id='strong_'+family,family=family,amplitude=.35,variant=1 if family=='quadratic' else .5)
        roles.append(dict(name='strong_'+family,count=128,offset=100000+1000*j,recipe=recipe))
    return roles


def freeze():
    import torch,ducc0,camb
    OUT.mkdir(parents=True,exist_ok=True)
    WORK.mkdir(parents=True,exist_ok=True)
    if (OUT/'plan.json').exists():
        raise FileExistsError('preserve existing targeted run; do not restart its budget')
    record = [r for r in json.loads((ROOT/'data/evidence/jovin_receipts.json').read_text()) if r['suite']=='targeted'][-1]
    if not (record['passed'] and record['receipt_verified'] and record['chain_valid']
        and record['bundle_sha256']==hashlib.sha256(build_bundle('targeted')[0].encode()).hexdigest()
        and record['tests_sha256']==sha(ROOT/SUITES['targeted'][1])):
        raise ValueError('current detector lacks matching JoVin receipt')
    sources = ['scripts/run_targeted_benchmark.py','scripts/record_targeted_cuda.py','docs/TARGETED_SPEC.md',
        'src/cmbcst/targeted.py','src/cmbcst/targeted_sky.py','tests/test_targeted.py','tests/sandbox/jovin_tests_targeted.py']
    old = json.loads((ROOT/'results/response_v1/plan.json').read_text())['source_sha256']
    hashes = {**old,**{name:sha(ROOT/name) for name in sources}}
    if any(sha(ROOT/name)!=digest for name,digest in hashes.items()):
        raise ValueError('previous frozen numerical source changed')
    plan = dict(created_utc=now(),scope='Synthetic physical blind-spot pilot; no observational Planck test',
        base_seed=SEED,compute_limit_seconds=1800,batch_size=8,maximum_draws=6682,
        recipes=recipes(),development_roles=development_roles(),response_indices=t.RESPONSE_INDICES,
        source_sha256=hashes,jovin_receipt_id=record['receipt_id'],jovin_bundle_sha256=record['bundle_sha256'],
        hardware_fixture_sha256=sha(ROOT/'data/evidence/targeted_cuda_development.json'),
        lmax=256,nside=128,threads=8,device=torch.cuda.get_device_name(0),numpy=np.__version__,
        torch=str(torch.__version__),cuda=torch.version.cuda,ducc0=ducc0.__version__,camb=camb.__version__,
        scaled_tolerance=1e-10,nominal_alpha=.05,null_wilson_upper_limit=.075,
        maximum_null_rate_excess=.01,maximum_strong_loss=.05,minimum_gain=.1,holm_family=3)
    with (OUT/'plan.json').open('x') as stream:
        json.dump(plan,stream,indent=2)
    return plan


def check_lock():
    lock = json.loads((OUT/'selection_lock.json').read_text())
    for key,name in (('plan_sha256','plan.json'),('model_sha256','model.json'),('development_sha256','development.json')):
        if lock[key]!=sha(OUT/name):
            raise ValueError('final selection binding changed')
    if lock['final_roles']!=final_roles(lock['selected']):
        raise ValueError('final role manifest changed')
    return lock


def measured_scores(b,r,gb,gr,model):
    cpu = t.score_targeted(b,r,model)
    gpu = t.score_targeted_cuda(gb,gr,model)
    error = scaled_error(gpu,cpu)
    if error>1e-10:
        np.savez(WORK/'failed_scores.npz',baseline=b,response=r,cpu=cpu,gpu=gpu)
        raise ValueError('CPU/CUDA score failure')
    return cpu,gpu,error


def evaluate_final(data,model,selected):
    roles = np.array([r['role'] for r in data['draws']])
    cal = roles=='final_cal'
    scores = np.asarray(data['scores'])
    p = response.upper_tail_pvalues(scores[~cal],scores[cal])
    flags = np.zeros_like(scores,dtype=bool)
    flags[~cal] = p<=.05
    data['pvalues'] = [None if iscal else [] for iscal in cal]
    for index,value in zip(np.flatnonzero(~cal),p):
        data['pvalues'][index] = value.tolist()
    null = flags[roles=='final_null']
    intervals = [t.wilson_interval(null[:,i]) for i in range(3)]
    rates = null.mean(0)
    calibration_gate = intervals[1][1]<=.075 and intervals[2][1]<=.075 and rates[2]<=rates[1]+.01
    strong = []
    for family in t.FAMILIES:
        f = flags[roles=='strong_'+family]
        strong.append(dict(family=family,draws=len(f),flags=f.sum(0).tolist(),rates=f.mean(0).tolist(),
            loss=float(f[:,1].mean()-f[:,2].mean())))
    retention_gate = all(row['loss']<=.05 for row in strong)
    comparisons = []
    for j,family in enumerate(t.FAMILIES):
        f = flags[roles=='final_'+family]
        if len(f):
            wins,losses,pvalue = response.paired_win_pvalue(f[:,1],f[:,2])
            delta = f[:,2].astype(float)-f[:,1].astype(float)
            boot = delta[np.random.default_rng(843600+j).integers(0,len(f),size=(2000,len(f)))].mean(1)
            row = dict(family=family,recipe=selected[family],draws=len(f),flags=f.sum(0).tolist(),rates=f.mean(0).tolist(),
                gain=float(delta.mean()),wins=wins,losses=losses,raw_p=pvalue,bootstrap_95=np.quantile(boot,[.025,.975]).tolist())
        else:
            row = dict(family=family,recipe=None,draws=0,flags=None,rates=None,gain=None,wins=0,losses=0,raw_p=1.,bootstrap_95=None)
        comparisons.append(row)
    adjusted = persistence.holm(np.array([row['raw_p'] for row in comparisons]))
    for row,pvalue in zip(comparisons,adjusted):
        row.update(p_holm=float(pvalue),provisional_gain=bool(calibration_gate and retention_gate and row['gain'] is not None and row['gain']>=.1 and pvalue<=.05))
    status = ('calibration_failed' if not calibration_gate else 'retention_failed' if not retention_gate else
        'provisional_synthetic_gain' if any(row['provisional_gain'] for row in comparisons) else
        'no_eligible_blind_spots' if all(value is None for value in selected.values()) else 'no_validated_gain')
    return dict(status=status,completed_utc=now(),methods=['original_omnibus','trained_baseline','six_responses'],
        null=dict(draws=len(null),flags=null.sum(0).tolist(),rates=rates.tolist(),wilson_95=intervals),
        calibration_gate=bool(calibration_gate),retention_gate=bool(retention_gate),strong_controls=strong,
        comparisons=comparisons,accepted_cosmological_claims=[],observational_planck_test_performed=False)


def run():
    start = time.perf_counter()
    plan = freeze()
    stage = dict(status='running',started_utc=now(),plan_sha256=sha(OUT/'plan.json'))
    completed = 0
    save(OUT/'stage_audit.json',stage)
    budget = lambda:t.ensure_budget(time.perf_counter()-start)
    try:
        sky = TargetedSky()
        save(OUT/'conditioning.json',dict(cl_TT_uK2=sky.cl.tolist(),scope='Same synthetic nuisance model as response_v1'))

        def generate(roles,phase):
            nonlocal completed
            b,r,gb,gr,draws,quality = [],[],[],[],[],[]
            for role in roles:
                for first in range(0,role['count'],8):
                    raw,parameters = [],[]
                    try:
                        for index in range(first,min(first+8,role['count'])):
                            budget()
                            field,param,q = sky.draw_recipe(SEED+role['offset']+index,role['recipe'])
                            if not q['passed']:
                                np.save(WORK/'failed_signal_field.npy',field)
                                raise ValueError('signal harmonic precision failure')
                            param.update(role=role['name'],index=index,signal_quality=q)
                            parameters.append(param)
                            raw.append(field)
                        fields = represent(np.stack(raw))
                        tick = time.perf_counter()
                        cb,fullr = response.extract_response_reference(fields)
                        cr = t.six_responses(fullr)
                        cpu_seconds = time.perf_counter()-tick
                        tick = time.perf_counter()
                        ub,ur_full = response.extract_response_cuda(fields)
                        ur = t.six_responses(ur_full)
                        gpu_seconds = time.perf_counter()-tick
                        errors = [scaled_error(ub,cb),scaled_error(ur_full,fullr)]
                        if max(errors)>1e-10:
                            raise ValueError('feature CPU/CUDA precision failure')
                    except Exception:
                        if raw:
                            np.savez(WORK/'failed_batch.npz',fields=np.asarray(raw))
                        save(OUT/'failure_context.json',dict(phase=phase,role=role,index=first,error=traceback.format_exc()))
                        raise
                    b.extend(cb); r.extend(cr); gb.extend(ub); gr.extend(ur); draws.extend(parameters)
                    completed += len(parameters)
                    quality.append(dict(role=role['name'],first_index=first,count=len(raw),errors=errors,finite=True,
                        cpu_seconds=cpu_seconds,gpu_seconds_including_transfer=gpu_seconds))
                    np.savez(WORK/f"{role['name']}_{first:04d}.npz",baseline=cb,response=cr,gpu_baseline=ub,gpu_response=ur)
                    save(WORK/'checkpoint.json',dict(completed=completed,phase=phase,role=role['name'],next_index=first+len(raw),
                        elapsed_seconds=time.perf_counter()-start,plan_sha256=sha(OUT/'plan.json')))
                    budget()
                    if len(quality)%16==0:
                        save(OUT/'progress.json',dict(phase=phase,completed=completed,elapsed_seconds=time.perf_counter()-start))
                        print(f'{phase}: completed {completed} ({time.perf_counter()-start:.1f}s)',flush=True)
            return dict(draws=draws,baseline=np.asarray(b),response=np.asarray(r),gpu_baseline=np.asarray(gb),gpu_response=np.asarray(gr),quality=quality)

        development = generate(plan['development_roles'],'development')
        role_names = np.array([row['role'] for row in development['draws']])
        train = np.array([name.startswith('train_') for name in role_names])
        labels = ['null' if row['recipe'] is None else next(r['family'] for r in recipes() if r['id']==row['recipe'])
            for row,is_train in zip(development['draws'],train) if is_train]
        model = t.fit_targeted(development['baseline'][train],development['response'][train],labels)
        if max(head['condition_number'] for head in model['detectors'])>1e8:
            raise ValueError('ill-conditioned detector')
        save(OUT/'model.json',model)
        ds,dg,de = measured_scores(development['baseline'],development['response'],development['gpu_baseline'],development['gpu_response'],model)
        development.update(scores=ds,gpu_scores=dg,score_error=de)
        cal = role_names=='dev_cal'
        rows = []
        for recipe in recipes():
            p = response.upper_tail_pvalues(ds[role_names=='dev_'+recipe['id']],ds[cal])
            f = p<=.05
            rows.append(dict(**recipe,baseline_rate=float(f[:,1].mean()),augmented_rate=float(f[:,2].mean()),
                gain=float(f[:,2].mean()-f[:,1].mean()),flags=f.sum(0).tolist()))
        devnull = response.upper_tail_pvalues(ds[role_names=='dev_null'],ds[cal])<=.05
        development.update(comparisons=rows,diagnostic_null_flags=devnull.sum(0).tolist())

        def serial(data):
            return {key:value.tolist() if isinstance(value,np.ndarray) else value for key,value in data.items()}

        save(OUT/'development.json',serial(development))
        selected = t.choose_blind_spots(rows)
        lock = dict(created_utc=now(),selected=selected,final_roles=final_roles(selected),plan_sha256=sha(OUT/'plan.json'),
            model_sha256=sha(OUT/'model.json'),development_sha256=sha(OUT/'development.json'),
            elapsed_seconds=time.perf_counter()-start)
        budget()
        with (OUT/'selection_lock.json').open('x') as stream:
            json.dump(lock,stream,indent=2)
        check_lock()
        print('selection locked: '+json.dumps(selected),flush=True)
        final = generate(lock['final_roles'],'final')
        check_lock()
        fs,fg,fe = measured_scores(final['baseline'],final['response'],final['gpu_baseline'],final['gpu_response'],model)
        final.update(scores=fs,gpu_scores=fg,score_error=fe)
        budget()
        result = evaluate_final(final,model,selected)
        save(OUT/'final.json',serial(final))
        result.update(plan_sha256=sha(OUT/'plan.json'),lock_sha256=sha(OUT/'selection_lock.json'),
            final_sha256=sha(OUT/'final.json'),model_sha256=sha(OUT/'model.json'))
        budget()
        save(OUT/'result.json',result)
        stage.update(status=result['status'],completed_utc=now(),elapsed_seconds=time.perf_counter()-start,draws=completed,
            numerical_failures=0,score_error=max(de,fe),result_sha256=sha(OUT/'result.json'))
        save(OUT/'stage_audit.json',stage)
        print(json.dumps(dict(status=result['status'],selected=selected,null_flags=result['null']['flags'],
            calibration_gate=result['calibration_gate'],retention_gate=result['retention_gate'],elapsed_seconds=stage['elapsed_seconds'])),flush=True)
    except Exception as exc:
        stage.update(status='incomplete' if isinstance(exc,TimeoutError) else 'execution_failed',completed_utc=now(),
            elapsed_seconds=time.perf_counter()-start,draws=completed,error=traceback.format_exc())
        save(OUT/'stage_audit.json',stage)
        raise


if __name__=='__main__':
    run()
