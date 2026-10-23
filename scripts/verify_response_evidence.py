"""Fresh authentic receipts, independent saved-data calculations and frozen bindings."""
import argparse
import ast
from bisect import bisect_left
from datetime import datetime, timezone
import hashlib
import gzip
import json
import math
from pathlib import Path
import sys
import numpy as np
from jovin_verify import REPO, SUITES, build_bundle, configuration, _harness

OUT = REPO/'results/response_v1'
sha = lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
read = lambda path:json.loads(Path(path).read_text(encoding='utf-8'))


def close(a,b):
    a,b = np.asarray(a),np.asarray(b)
    return bool(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        and np.max(np.abs(a-b)/(1+np.abs(b)))<=1e-10)


def main():
    command,env = configuration(argparse.Namespace(harness=None,harness_python=None,runtime='wsl:JovinHarness',data_dir=None))
    ret,chain = _harness(command,env,['receipts','verify'])
    history = read(REPO/'data/evidence/jovin_receipts.json')
    suites = {}
    for suite,(_,tests) in SUITES.items():
        record = [r for r in history if r['suite']==suite][-1]
        show_ret,authentic = _harness(command,env,['receipts','show',record['receipt_id']])
        member_ret,membership = _harness(command,env,['receipts','verify',record['receipt_id']])
        matched = (record['passed'] and record['receipt_verified'] and record['chain_valid']
            and hashlib.sha256(build_bundle(suite)[0].encode()).hexdigest()==record['bundle_sha256']
            and sha(REPO/tests)==record['tests_sha256'] and show_ret==0 and authentic==record['receipt']
            and member_ret==0 and membership.get('valid') is True and membership.get('chain')=='linked')
        suites[suite] = {'matched':bool(matched),'receipt_id':record['receipt_id'],'total':record['total']}
    frozen = {str(path.relative_to(REPO)):all(sha(REPO/name)==digest for name,digest in read(path).get('source_sha256',{}).items())
        for path in (REPO/'results').glob('*/plan.json')}
    preserved = {str(path.relative_to(REPO)):all(sha(path.parent/name)==digest for name,digest in read(path).get('artifact_sha256',{}).items())
        for path in (REPO/'results').glob('*/verification.json') if path.parent!=OUT}
    plan,conditioning,model,ensemble,validation,result,stage,replay = [read(OUT/name) for name in
        ('plan.json','conditioning.json','model.json','ensemble.json','validation.json','result.json','stage_audit.json','replay.json')]
    b,r,saved_scores = [np.asarray(ensemble[name]) for name in ('baseline_features','response_features','scores')]
    draws = ensemble['draws']
    roles = np.array([d['group'] for d in draws])
    train,cal,null = roles=='training',roles=='calibration',roles=='matched_null'
    held = ~(train|cal)
    checks = {'feature_dimensions':b.shape==(3303,17) and r.shape==(3303,90) and saved_scores.shape==(3303,2),
        'all_finite_features_scores':bool(np.isfinite(b).all() and np.isfinite(r).all() and np.isfinite(saved_scores).all()),
        'fixed_design':plan['total_draws']==3303 and plan['ridge']==plan['shrinkage']==.1
            and plan['candidate_p']==.05 and plan['minimum_power_gain']==.1 and plan['max_null_flags']==40
            and plan['power_family_size']==6 and plan['base_seed']==765902132026
            and plan['angles_degrees']==[0.,22.5,45.,67.5] and plan['smoothing_arcmin']==[30.,60.,120.]}
    expected_groups = [('training',768,0,'null',0.),('calibration',999,10000,'null',0.),('matched_null',512,20000,'null',0.),
        ('quadratic_0.15',128,30000,'quadratic',.15),('quadratic_0.35',128,31000,'quadratic',.35),
        ('compact_0.15',128,40000,'compact',.15),('compact_0.35',128,41000,'compact',.35),
        ('filament_0.15',128,50000,'filament',.15),('filament_0.35',128,51000,'filament',.35),('stripe_stress',256,60000,'stripe_stress',0.)]
    expected = [(name,index,765902132026+offset+index,kind,amp) for name,n,offset,kind,amp in expected_groups for index in range(n)]
    checks['complete_fixed_seed_role_domain'] = len(draws)==len(expected) and all(
        (d['group'],d['index'],d['seed'],d['kind'],d['amplitude'])==item for d,item in zip(draws,expected)) and len({d['seed'] for d in draws})==3303
    checks['all_signal_power_checks'] = all(d['signal_quality']['passed'] and d['signal_quality']['finite']
        and d['signal_quality']['zero_shells_exact'] and d['signal_quality']['mzero_real']
        and 0<=d['signal_quality']['shell_relative_error']<=1e-10 for d in draws)
    checks['all_nuisance_draw_bounds'] = all(0<=d['foreground_amplitude_uK']<=60 and 5<=d['noise_amplitude_uK']<=25
        and d['original_local_signal_rms_uK']>0 and d['observed_base_std_uK']>0 for d in draws)
    q = ensemble['batch_quality']
    checks['all_batch_numerics'] = sum(row['count'] for row in q)==3303 and all(row['passed'] and row['finite']
        and 0<=row['baseline_scaled_error']<=1e-10 and 0<=row['response_scaled_error']<=1e-10 for row in q)
    checks['stage_precision_aggregates'] = stage['draws']==3303 and stage['numerical_failures']==stage['nonfinite_failures']==0
    checks['stage_precision_aggregates'] &= (stage['max_baseline_error']==max(row['baseline_scaled_error'] for row in q)
        and stage['max_response_error']==max(row['response_scaled_error'] for row in q)
        and stage['max_signal_shell_error']==max(d['signal_quality']['shell_relative_error'] for d in draws)
        and 0<=stage['score_scaled_error']==result['score_scaled_error']<=1e-10)

    bm,rm = b[train].mean(0),r[train].mean(0)
    bs,rs = b[train].std(0,ddof=1),r[train].std(0,ddof=1)
    bs,rs = np.where(bs>1e-12,bs,1.),np.where(rs>1e-12,rs,1.)
    bz,rz = (b-bm)/bs,(r-rm)/rs
    coefficients = np.linalg.solve(bz[train].T@bz[train]+.1*768*np.eye(17),bz[train].T@rz[train])
    joint = np.concatenate([bz,rz-bz@coefficients],axis=1)
    center = joint[train].mean(0)
    joint -= center
    cb = .9*np.cov(bz[train],rowvar=False)+.1*np.eye(17)
    cj = .9*np.cov(joint[train],rowvar=False)+.1*np.eye(107)
    pb,pj = np.asarray(model['baseline_precision']),np.asarray(model['joint_precision'])
    checks['training_only_fit_reconstructed'] = all(close(actual,expected) for actual,expected in
        ((model['baseline_mean'],bm),(model['response_mean'],rm),(model['baseline_scale'],bs),
         (model['response_scale'],rs),(model['coefficients'],coefficients),(model['joint_center'],center))) and model['training_draws']==768
    checks['precision_covariance_inverse_identities'] = close(cb@pb,np.eye(17)) and close(cj@pj,np.eye(107))
    checks['conditioning_recomputed'] = close(model['condition_numbers'],[np.linalg.cond(cb),np.linalg.cond(cj)]) and max(model['condition_numbers'])<=1e8
    scores = np.column_stack([(bz@pb*bz).sum(1),(joint@pj*joint).sum(1)])
    checks['all_scores_recomputed'] = close(saved_scores,scores)
    references = [sorted(saved_scores[cal,col]) for col in range(2)]
    pvalues = np.array([[(1000-bisect_left(references[col],row[col]))/1000 for col in range(2)] for row in saved_scores[held]])
    checks['all_heldout_pvalues_recomputed'] = close([p for p in ensemble['held_out_pvalues'] if p is not None],pvalues)
    checks['no_training_calibration_decisions'] = all((p is None)==(not h) for p,h in zip(ensemble['held_out_pvalues'],held))
    flags = np.zeros((3303,2),dtype=bool)
    flags[held] = pvalues<=.05
    counts = flags[null].sum(0)
    gate = bool(np.all(counts<=40))
    checks['matched_null_gate_recomputed'] = validation['draws']==512 and counts.tolist()==validation['flags'] and close(counts/512,validation['rates']) and validation['status']==('passed' if gate else 'failed')
    z = 1.959963984540054
    intervals = []
    for count in counts:
        center_w = (count/512+z*z/1024)/(1+z*z/512)
        half = z*np.sqrt(count/512*(1-count/512)/512+z*z/(4*512**2))/(1+z*z/512)
        intervals.append([center_w-half,center_w+half])
    checks['null_wilson_intervals_recomputed'] = close(intervals,validation['wilson_95'])
    subgroup_ok = []
    for key,parameter,threshold in (('foreground','foreground_amplitude_uK',30.),('noise','noise_amplitude_uK',15.)):
        for side in ('low','high'):
            mask = null & np.array([(d[parameter]<threshold)==(side=='low') for d in draws])
            row = validation['nuisance_subgroups'][key+'_'+side]
            subgroup_ok.append(row['draws']==int(mask.sum()) and row['flags']==flags[mask].sum(0).tolist())
    checks['all_nuisance_subgroups'] = all(subgroup_ok)
    paired_ok,raw_p,gains = [],[],[]
    for index,(name,n,_,_,_) in enumerate(expected_groups[3:9]):
        row = result['comparisons'][index]
        pair = flags[roles==name]
        wins = int(np.sum(pair[:,1]&~pair[:,0]))
        losses = int(np.sum(pair[:,0]&~pair[:,1]))
        discordant = wins+losses
        p = sum(math.comb(discordant,k) for k in range(wins,discordant+1))/2**discordant if discordant else 1.
        difference = pair[:,1].astype(float)-pair[:,0].astype(float)
        gain = float(difference.mean())
        rng = np.random.default_rng(833600+index)
        interval = np.quantile(difference[rng.integers(0,n,size=(2000,n))].mean(1),[.025,.975])
        paired_ok.append(row['group']==name and row['draws']==n and row['flags']==pair.sum(0).tolist()
            and close(row['rates'],pair.mean(0)) and row['combined_only_wins']==wins and row['baseline_only_losses']==losses
            and abs(row['raw_p']-p)<=1e-12 and row['gain']==gain and close(row['paired_gain_bootstrap_95'],interval))
        raw_p.append(p)
        gains.append(gain)
    adjusted = [0.]*6
    running = 0.
    for rank,index in enumerate(sorted(range(6),key=lambda i:raw_p[i])):
        running = max(running,(6-rank)*raw_p[index])
        adjusted[index] = min(1.,running)
    decisions = [gate and p<=.05 and gain>=.1 for p,gain in zip(adjusted,gains)]
    checks['all_six_paired_tests_bootstrap_and_holm'] = len(result['comparisons'])==6 and all(paired_ok) and close(adjusted,[r['p_holm'] for r in result['comparisons']])
    checks['all_gain_decisions'] = decisions==[r['provisional_gain'] for r in result['comparisons']]
    expected_status = 'calibration_failed' if not gate else ('provisional_synthetic_gain' if any(decisions) else 'no_validated_gain')
    checks['honest_outcome_and_scope'] = result['status']==stage['status']==expected_status and result['accepted_cosmological_claims']==[] and result['observational_planck_test_performed'] is False
    stress = flags[roles=='stripe_stress']
    checks['omitted_stripe_counts'] = result['stripe_stress']['draws']==256 and result['stripe_stress']['flags']==stress.sum(0).tolist() and close(result['stripe_stress']['rates'],stress.mean(0))
    checks['separate_30_case_replay'] = replay['status']=='passed' and len(replay['records'])==30 and all(row['passed'] and max(row[key] for key in ('baseline_error','response_error','parameter_error','example_error'))<=1e-10 for row in replay['records'])
    initial_replay = read(OUT/'replay_initial.json')
    checks['initial_replay_preserved_before_whitespace_cleanup'] = (initial_replay['status']=='passed'
        and initial_replay['source_sha256']==hashlib.sha256(gzip.decompress((OUT/'replay_initial_source.py.gz').read_bytes())).hexdigest()
        and len(initial_replay['records'])==len(replay['records'])==30
        and all(a['passed'] and (a['group'],a['index'],a['seed'])==(b['group'],b['index'],b['seed'])
            for a,b in zip(initial_replay['records'],replay['records'])))
    fixture_record = read(REPO/'data/evidence/response_cuda_development.json')
    tree = ast.parse((REPO/'tests/sandbox/jovin_tests_response.py').read_text())
    fixture = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='GPU_ATTESTATION' for target in node.targets))
    checks['actual_cuda_fixture_preserved'] = fixture==fixture_record['fixture'] and all(sha(REPO/name)==digest for name,digest in fixture_record['source_sha256'].items())
    bindings = {key:result[key]==sha(OUT/name) for key,name in
        (('plan_sha256','plan.json'),('conditioning_sha256','conditioning.json'),('model_sha256','model.json'),('ensemble_sha256','ensemble.json'),('validation_sha256','validation.json'))}
    bindings.update({'stage_result':stage['result_sha256']==sha(OUT/'result.json'),
        'replay_plan':replay['plan_sha256']==sha(OUT/'plan.json'),'replay_ensemble':replay['ensemble_sha256']==sha(OUT/'ensemble.json'),
        'replay_source':replay['source_sha256']==sha(REPO/'scripts/replay_response_samples.py'),
        'response_receipt':plan['jovin_receipt_id']==suites['response']['receipt_id'] and suites['response']['matched'],
        'conditioning_cache':conditioning['conditioning_cache_sha256']==sha(REPO/'data/work/response_v1/conditioning.npz'),
        'example_cache':result['example_cache_sha256']==sha(REPO/'data/work/response_v1/examples.npz')})
    bindings.update({name:sha(REPO/name)==digest for name,digest in plan['source_artifacts_sha256'].items()})
    tests,sandbox = REPO/'data/response_tests.log',REPO/'data/response_sandbox_tests.log'
    tests_ok = '185 passed, 3 skipped' in tests.read_text()
    sandbox_ok = sandbox.read_text().count('[PASS]')==20 and '[FAIL]' not in sandbox.read_text()
    ok = (ret==0 and chain.get('valid') is True and len(suites)==20 and all(row['matched'] for row in suites.values())
        and all(frozen.values()) and all(preserved.values()) and all(checks.values()) and all(bindings.values()) and tests_ok and sandbox_ok)
    audit = {'verified_utc':datetime.now(timezone.utc).isoformat(),'status':'passed' if ok else 'failed',
        'scope':'Authentic receipt membership, source/artifact identity and independent saved-data calculations; a verified negative benchmark is not a method improvement',
        'verification_environment':{'platform':sys.platform,'python':sys.version,'numpy':np.__version__},
        'jovin_suites':suites,'chain':chain,'frozen_sources':frozen,'prior_artifacts_preserved':preserved,
        'calculation_checks':checks,'bindings':bindings,
        'local_tests':{'passed':185,'skipped':3,'matched':tests_ok,'log_sha256':sha(tests)},
        'ci_equivalent_sandbox_suites':{'passed':20,'matched':sandbox_ok,'log_sha256':sha(sandbox)},
        'artifact_sha256':{name:sha(OUT/name) for name in ('plan.json','conditioning.json','model.json','ensemble.json','validation.json','result.json','stage_audit.json','replay.json','replay_initial.json','replay_initial_source.py.gz','benchmark.png','sampling.png')},
        'report_sha256':sha(REPO/'docs/RESPONSE_BENCHMARK_RUN.md'),
        'supporting_source_sha256':{name:sha(REPO/name) for name in ('scripts/verify_response_evidence.py','scripts/report_response_benchmark.py','scripts/replay_response_samples.py')}}
    (OUT/'verification.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({'status':audit['status'],'scientific_outcome':result['status'],'suites':len(suites),
        'checks':sum(row['total'] for row in suites.values()),'failed_calculations':[name for name,ok in checks.items() if not ok],
        'failed_bindings':[name for name,ok in bindings.items() if not ok]},indent=2))
    if not ok:
        raise SystemExit(1)


if __name__=='__main__':
    main()
