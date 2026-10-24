"""Independent reconstruction of targeted fits, selection and confirmatory decisions."""
import argparse
import ast
from bisect import bisect_left
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import numpy as np
from jovin_verify import REPO,SUITES,build_bundle,configuration,_harness
OUT = REPO/'results/targeted_v1'
read = lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha = lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
FAMILIES = ('quadratic','compact','filament')


def close(a,b):
    a,b = np.asarray(a),np.asarray(b)
    return bool(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all() and np.max(np.abs(a-b)/(1+np.abs(b)))<=1e-10)


def tails(scores,reference):
    columns = [sorted(row) for row in reference.T]
    n = len(reference)
    return np.array([[(n+1-bisect_left(columns[j],x))/(n+1) for j,x in enumerate(row)] for row in scores])


def wilson(flags):
    n,p,z = len(flags),float(np.mean(flags)),1.959963984540054
    center = (p+z*z/(2*n))/(1+z*z/n)
    radius = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [center-radius,center+radius]


def main():
    command,env = configuration(argparse.Namespace(harness=None,harness_python=None,runtime='wsl:JovinHarness',data_dir=None))
    ret,chain = _harness(command,env,['receipts','verify'])
    history = read(REPO/'data/evidence/jovin_receipts.json')
    suites = {}
    for name,(_,tests) in SUITES.items():
        record = [r for r in history if r['suite']==name][-1]
        sr,authentic = _harness(command,env,['receipts','show',record['receipt_id']])
        mr,membership = _harness(command,env,['receipts','verify',record['receipt_id']])
        ok = (record['passed'] and record['receipt_verified'] and record['chain_valid'] and sr==mr==0
            and authentic==record['receipt'] and membership.get('valid') is True and membership.get('chain')=='linked'
            and hashlib.sha256(build_bundle(name)[0].encode()).hexdigest()==record['bundle_sha256']
            and sha(REPO/tests)==record['tests_sha256'])
        suites[name] = dict(matched=bool(ok),receipt_id=record['receipt_id'],total=record['total'])
    frozen = {str(p.relative_to(REPO)):all(sha(REPO/name)==digest for name,digest in read(p).get('source_sha256',{}).items()) for p in (REPO/'results').glob('*/plan.json')}
    prior = {str(p.relative_to(REPO)):all(sha(p.parent/name)==digest for name,digest in read(p).get('artifact_sha256',{}).items()) for p in (REPO/'results').glob('*/verification.json') if p.parent!=OUT}
    plan,dev,lock,model,final,result,stage,replay = [read(OUT/name) for name in
        ('plan.json','development.json','selection_lock.json','model.json','final.json','result.json','stage_audit.json','replay.json')]
    checks = {}
    b,r = np.asarray(dev['baseline']),np.asarray(dev['response'])
    roles = np.array([row['role'] for row in dev['draws']])
    train = np.array([name.startswith('train_') for name in roles])
    null = roles=='train_null'
    recipe_map = {row['id']:row for row in plan['recipes']}
    labels = np.array(['null' if row['recipe'] is None else recipe_map[row['recipe']]['family'] for row in dev['draws']])
    bm,rm = b[null].mean(0),r[null].mean(0)
    bs,rs = b[null].std(0,ddof=1),r[null].std(0,ddof=1)
    bs,rs = np.where(bs>1e-12,bs,1.),np.where(rs>1e-12,rs,1.)
    bz,rz = (b-bm)/bs,(r-rm)/rs
    coefficient = np.linalg.solve(bz[null].T@bz[null]+.1*1024*np.eye(17),bz[null].T@rz[null])
    joint = np.c_[bz,rz-bz@coefficient]
    checks['null_only_preprocessing'] = all(close(x,y) for x,y in ((bm,model['baseline_mean']),(rm,model['response_mean']),
        (bs,model['baseline_scale']),(rs,model['response_scale']),(coefficient,model['coefficients']))) and model['training_draws']==1024
    cb = .9*np.cov(bz[null],rowvar=False)+.1*np.eye(17)
    pb = np.asarray(model['baseline_precision'])
    checks['baseline_inverse_identity'] = close(cb@pb,np.eye(17))
    cj = .9*np.cov(joint[null],rowvar=False)+.1*np.eye(23)
    checks['residual_fit_metadata'] = close(cj@np.asarray(model['joint_precision']),np.eye(23)) and close(joint[null].mean(0),model['joint_center'])
    omni = (bz[null]@pb*bz[null]).sum(1)
    checks['omnibus_standardization'] = close([omni.mean(),omni.std(ddof=1)],[model['omnibus_mean'],model['omnibus_scale']])
    heads_ok = []
    for x,head in zip((bz,joint),model['detectors']):
        center = x[null].mean(0)
        cov = np.cov(x[null],rowvar=False)
        regularized = .9*cov+.1*np.eye(x.shape[1])
        weights = np.stack([np.linalg.solve(regularized,x[train&(labels==family)].mean(0)-center) for family in FAMILIES],axis=1)
        scale = ((x[null]-center)@weights).std(0,ddof=1)
        heads_ok.append(close(center,head['center']) and close(weights,head['weights']) and close(scale,head['scale'])
            and close([np.linalg.cond(regularized)],[head['condition_number']]) and head['condition_number']<=1e8)
    checks['training_only_equal_family_heads'] = all(heads_ok) and model['training_counts']=={'null':1024,'quadratic':384,'compact':384,'filament':384}

    def rescore(data):
        base,resp = np.asarray(data['baseline']),np.asarray(data['response'])
        z = (base-bm)/bs
        x = np.c_[z,(resp-rm)/rs-z@coefficient]
        scores = [((z@pb*z).sum(1)-model['omnibus_mean'])/model['omnibus_scale']]
        for values,head in zip((z,x),model['detectors']):
            scores.append(np.max(((values-head['center'])@head['weights'])/head['scale'],axis=1))
        return np.column_stack(scores)

    ds,fs = rescore(dev),rescore(final)
    checks['every_cpu_gpu_score'] = close(ds,dev['scores']) and close(fs,final['scores']) and close(ds,dev['gpu_scores']) and close(fs,final['gpu_scores'])
    cal = roles=='dev_cal'
    rows_ok = []
    expected_selection = {}
    for recipe in plan['recipes']:
        flags = tails(ds[roles=='dev_'+recipe['id']],ds[cal])<=.05
        row = next(r for r in dev['comparisons'] if r['id']==recipe['id'])
        rows_ok.append(row['flags']==flags.sum(0).tolist() and close([row['baseline_rate'],row['augmented_rate'],row['gain']],
            [flags[:,1].mean(),flags[:,2].mean(),flags[:,2].mean()-flags[:,1].mean()]))
    for family in FAMILIES:
        eligible = [r for r in dev['comparisons'] if r['family']==family and r['baseline_rate']<=.15]
        expected_selection[family] = sorted(eligible,key=lambda r:(-r['gain'],r['amplitude'],r['id']))[0]['id'] if eligible else None
    checks['all_development_comparisons_and_selection'] = all(rows_ok) and expected_selection==lock['selected'] and len(dev['comparisons'])==18
    checks['development_null_diagnostic'] = dev['diagnostic_null_flags']==(tails(ds[roles=='dev_null'],ds[cal])<=.05).sum(0).tolist()
    final_names = np.array([r['role'] for r in final['draws']])
    cal = final_names=='final_cal'
    p = tails(fs[~cal],fs[cal])
    flags = np.zeros(fs.shape,dtype=bool)
    flags[~cal] = p<=.05
    checks['entire_maximum_score_calibration'] = close([v for v in final['pvalues'] if v is not None],p) and all((v is None)==c for v,c in zip(final['pvalues'],cal))
    nflags = flags[final_names=='final_null']
    rates = nflags.mean(0)
    intervals = [wilson(nflags[:,i]) for i in range(3)]
    gate = bool(intervals[1][1]<=.075 and intervals[2][1]<=.075 and rates[2]<=rates[1]+.01)
    checks['final_null_counts_intervals_gate'] = result['null']['draws']==1024 and result['null']['flags']==nflags.sum(0).tolist() and close(result['null']['rates'],rates) and close(result['null']['wilson_95'],intervals) and result['calibration_gate']==gate
    strong_ok,losses = [],[]
    raw,gains,paired_ok = [],[],[]
    for j,family in enumerate(FAMILIES):
        strong = flags[final_names=='strong_'+family]
        loss = strong[:,1].mean()-strong[:,2].mean()
        row = result['strong_controls'][j]
        strong_ok.append(row['family']==family and row['draws']==128 and row['flags']==strong.sum(0).tolist() and close(row['rates'],strong.mean(0)) and row['loss']==loss)
        losses.append(loss)
        pair = flags[final_names=='final_'+family]
        row = result['comparisons'][j]
        if len(pair):
            wins = int(np.sum(pair[:,2]&~pair[:,1])); lost = int(np.sum(pair[:,1]&~pair[:,2])); n = wins+lost
            pv = sum(math.comb(n,k) for k in range(wins,n+1))/2**n if n else 1.
            delta = pair[:,2].astype(float)-pair[:,1].astype(float)
            ci = np.quantile(delta[np.random.default_rng(843600+j).integers(0,256,size=(2000,256))].mean(1),[.025,.975])
            gain = float(delta.mean())
            paired_ok.append(len(pair)==256 and row['flags']==pair.sum(0).tolist() and close(row['rates'],pair.mean(0)) and row['wins']==wins and row['losses']==lost and row['gain']==gain and abs(row['raw_p']-pv)<1e-12 and close(row['bootstrap_95'],ci))
        else:
            pv,gain = 1.,None
            paired_ok.append(row['recipe'] is None and row['draws']==0 and row['flags'] is None and row['gain'] is None and row['raw_p']==1.)
        paired_ok[-1] &= row['family']==family and row['recipe']==expected_selection[family]
        raw.append(pv); gains.append(gain)
    retention = all(loss<=.05 for loss in losses)
    adjusted = [0.]*3
    running = 0.
    for rank,index in enumerate(sorted(range(3),key=lambda i:raw[i])):
        running = max(running,(3-rank)*raw[index]); adjusted[index] = min(1.,running)
    decisions = [bool(gate and retention and g is not None and g>=.1 and p<=.05) for g,p in zip(gains,adjusted)]
    expected_status = ('calibration_failed' if not gate else 'retention_failed' if not retention else 'provisional_synthetic_gain' if any(decisions) else 'no_eligible_blind_spots' if all(r is None for r in expected_selection.values()) else 'no_validated_gain')
    checks['strong_retention'] = all(strong_ok) and result['retention_gate']==retention
    checks['all_paired_tests_bootstrap_and_holm'] = all(paired_ok) and close(adjusted,[row['p_holm'] for row in result['comparisons']])
    checks['promotion_and_scope'] = result['status']==stage['status']==expected_status and decisions==[row['provisional_gain'] for row in result['comparisons']] and result['accepted_cosmological_claims']==[] and result['observational_planck_test_performed'] is False
    all_draws = dev['draws']+final['draws']
    all_seeds = []
    domain_ok = []
    for data,manifest in ((dev,plan['development_roles']),(final,lock['final_roles'])):
        expected = [(role['name'],i,plan['base_seed']+role['offset']+i,None if role['recipe'] is None else role['recipe']['id']) for role in manifest for i in range(role['count'])]
        observed = [(d['role'],d['index'],d['seed'],d['recipe']) for d in data['draws']]
        domain_ok.append(expected==observed)
        all_seeds.extend(d['seed'] for d in data['draws'])
    checks['all_unfiltered_disjoint_draws'] = all(domain_ok) and len(set(all_seeds))==len(all_draws)==stage['draws'] and len(dev['draws'])==3507 and stage['draws']<=6682
    checks['signal_and_feature_precision'] = all(d['signal_quality']['passed'] and d['signal_quality']['finite'] and d['signal_quality']['zero_shells_exact'] and d['signal_quality']['mzero_real'] and d['signal_quality']['shell_relative_error']<=1e-10 for d in all_draws) and all(q['finite'] and max(q['errors'])<=1e-10 for data in (dev,final) for q in data['quality'])
    checks['budget_and_no_numerical_failures'] = 0<stage['elapsed_seconds']<1800 and plan['compute_limit_seconds']==1800 and stage['numerical_failures']==0 and stage['score_error']==max(dev['score_error'],final['score_error']) and stage['score_error']<=1e-10
    checks['separate_replay_domain'] = replay['status']=='passed' and len(replay['records'])==2*(len(plan['development_roles'])+len(lock['final_roles'])) and all(row['passed'] and max(row['errors'])<=1e-10 for row in replay['records'])
    fixture = read(REPO/'data/evidence/targeted_cuda_development.json')
    tree = ast.parse((REPO/'tests/sandbox/jovin_tests_targeted.py').read_text())
    recorded = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='GPU_ATTESTATION' for target in node.targets))
    checks['hardware_record_and_current_fixture'] = recorded==fixture['scores'] and all(sha(REPO/name)==digest for name,digest in fixture['source_sha256'].items())
    bindings = {'lock_'+key:lock[key]==sha(OUT/name) for key,name in (('plan_sha256','plan.json'),('model_sha256','model.json'),('development_sha256','development.json'))}
    bindings.update({'result_'+key:result[key]==sha(OUT/name) for key,name in (('plan_sha256','plan.json'),('model_sha256','model.json'),('lock_sha256','selection_lock.json'),('final_sha256','final.json'))})
    bindings.update(stage_result=stage['result_sha256']==sha(OUT/'result.json'),replay_source=replay['source_sha256']==sha(REPO/'scripts/replay_targeted_samples.py'),
        replay_plan=replay['plan_sha256']==sha(OUT/'plan.json'),replay_lock=replay['lock_sha256']==sha(OUT/'selection_lock.json'),
        hardware_fixture=plan['hardware_fixture_sha256']==sha(REPO/'data/evidence/targeted_cuda_development.json'),
        receipt=plan['jovin_receipt_id']==suites['targeted']['receipt_id'])
    tests,sandbox = REPO/'data/targeted_tests.log',REPO/'data/targeted_sandbox_tests.log'
    tests_summary = tests.read_text().lower()
    tests_ok = (re.search(r'\b189 passed, 3 skipped\b',tests_summary) is not None
        and 'failed' not in tests_summary and 'error' not in tests_summary)
    sandbox_ok = sandbox.read_text().count('[PASS]')==21 and '[FAIL]' not in sandbox.read_text()
    ok = ret==0 and chain.get('valid') is True and len(suites)==21 and all(v['matched'] for v in suites.values()) and all(frozen.values()) and all(prior.values()) and all(checks.values()) and all(bindings.values()) and tests_ok and sandbox_ok
    audit = dict(status='passed' if ok else 'failed',verified_utc=datetime.now(timezone.utc).isoformat(),
        scope='Authentic receipts and independent saved-data calculations, not proof of physical validity or universal novelty',
        verification_environment=dict(python=sys.version,numpy=np.__version__,platform=sys.platform),
        jovin_suites=suites,chain=chain,frozen_sources=frozen,prior_artifacts_preserved=prior,calculation_checks=checks,bindings=bindings,
        local_tests=dict(passed=189,skipped=3,matched=tests_ok,log_sha256=sha(tests)),
        ci_equivalent_sandbox_suites=dict(passed=21,matched=sandbox_ok,log_sha256=sha(sandbox)),
        artifact_sha256={name:sha(OUT/name) for name in ('plan.json','conditioning.json','model.json','development.json','selection_lock.json','final.json','result.json','stage_audit.json','replay.json','benchmark.png')},
        report_sha256=sha(REPO/'docs/TARGETED_RUN.md'),supporting_source_sha256={name:sha(REPO/name) for name in
            ('scripts/verify_targeted_evidence.py','scripts/report_targeted_benchmark.py','scripts/replay_targeted_samples.py')})
    (OUT/'verification.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(dict(status=audit['status'],scientific_outcome=result['status'],suites=len(suites),checks=sum(r['total'] for r in suites.values()),
        failed_calculations=[key for key,value in checks.items() if not value],failed_bindings=[key for key,value in bindings.items() if not value]),indent=2))
    if not ok:
        raise SystemExit(1)


if __name__=='__main__':
    main()
