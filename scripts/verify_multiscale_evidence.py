"""Fresh authentic receipt and independent multiscale result verification."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from jovin_verify import REPO, SUITES, build_bundle, configuration, _harness

OUT = REPO/'results/multiscale_v1'
read = lambda path:json.loads(Path(path).read_text())
sha = lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
flat = lambda values:[value for row in values for value in row]
close = lambda a,b:len(a)==len(b) and all(abs(x-y)/(1+abs(y))<=1e-10 for x,y in zip(a,b))


def probabilities(observed,reference):
    raw = []
    for row in range(3):
        for col in range(6):
            values = [sky[row][col] for sky in reference]
            value = observed[row][col]
            raw.append(min(1.,2*min((1+sum(v<=value for v in values))/1000,
                                   (1+sum(v>=value for v in values))/1000)))
    adjusted = [0.]*18
    running = 0.
    for rank,index in enumerate(sorted(range(18),key=lambda i:raw[i])):
        running = max(running,(18-rank)*raw[index])
        adjusted[index] = min(1.,running)
    return raw,adjusted


def compress(curves):
    t = [math.log(lag) for lag in (1,2,4,8,16)]
    mean_t = sum(t)/5
    centered = [v-mean_t for v in t]
    denominator = sum(v*v for v in centered)
    slope = lambda values:sum(math.log(y)*x for x,y in zip(centered,values))/denominator
    return [slope([c[0] for c in curves]),sum(c[3] for c in curves)/5,
        sum(c[4] for c in curves)/5,slope([c[3]+3 for c in curves]),
        slope([c[4]+15 for c in curves]),sum(c[5] for c in curves)/5]


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
    preserved = {}
    for path in (REPO/'results').glob('*/verification.json'):
        if path.parent==OUT:
            continue
        preserved[str(path.relative_to(REPO))] = all(sha(path.parent/name)==digest for name,digest in read(path).get('artifact_sha256',{}).items())
    plan,result,ensemble,validation,positive,stage,lean = [read(OUT/name) for name in
        ('plan.json','result.json','ensemble.json','validation.json','positive_control.json','stage_audit.json','lean_verification.json')]
    reference = ensemble['features'][:999]
    records = validation['records']
    q = ensemble['batch_quality']
    calculations = {'shape':len(ensemble['features'])==len(ensemble['curves'])==1063,
        'finite_values':all(math.isfinite(v) for sky in ensemble['curves'] for patch in sky for row in patch for v in row),
        'all_curve_compressions':all(close(features,compress(curves)) for sky,curve_sky in zip(ensemble['features'],ensemble['curves']) for features,curves in zip(sky,curve_sky)),
        'all_cpu_gpu_precision':sum(item['count'] for item in q)==1063 and all(item['passed'] and item['finite']
            and item['curve_scaled_error']<=1e-10 and item['feature_scaled_error']<=1e-10 and item['max_shell_error']<=1e-10 for item in q)}
    control_checks = []
    for record,features in zip(records,ensemble['features'][999:]):
        raw,adjusted = probabilities(features,reference)
        control_checks.append(record['features']==features and close(flat(record['raw_p']),raw)
            and close(flat(record['p_holm']),adjusted) and record['flag']==(min(adjusted)<=.05))
    calculations['all_control_decisions'] = len(control_checks)==64 and all(control_checks)
    flags = sum(r['flag'] for r in records)
    calculations['calibration_gate'] = flags==validation['familywise_flags'] and validation['status']==('passed' if flags<=8 else 'failed')
    raw,adjusted = probabilities(result['observed'],reference)
    calculations['all_sky_pvalues'] = close(flat(result['raw_p']),raw) and close(flat(result['p_holm']),adjusted)
    calculations['no_threshold_flags'] = not result['candidate_flags'] and all(v>.05 for v in adjusted)
    calculations['observed_compression'] = all(close(f,compress(c)) for f,c in zip(result['observed'],result['observed_curves']))
    raw,adjusted = probabilities(positive['features'],reference)
    calculations['positive_control'] = positive['quality']['passed'] and close(flat(positive['raw_p']),raw) and close(flat(positive['p_holm']),adjusted)
    bindings = {name:sha(REPO/name)==digest for name,digest in plan['source_artifacts_sha256'].items()}
    for key,name in (('plan_sha256','plan.json'),('ensemble_sha256','ensemble.json'),('validation_sha256','validation.json')):
        bindings[key] = result[key]==sha(OUT/name)
    bindings['stage_result'] = stage['result_sha256']==sha(OUT/'result.json')
    bindings['lean_source'] = lean['status']=='passed' and lean['exit_code']==0 and lean['theorem_count']==9 and lean['source_sha256']==sha(REPO/'formal/ScaleInvariance.lean')
    bindings['lean_numerical_module'] = lean['numerical_module_sha256']==sha(REPO/'src/cmbcst/multiscale.py')
    tests = REPO/'data/multiscale_tests.log'
    sandbox = REPO/'data/multiscale_sandbox_tests.log'
    tests_ok = '181 passed, 3 skipped' in tests.read_text()
    sandbox_ok = sandbox.read_text().count('[PASS]')==18 and '[FAIL]' not in sandbox.read_text()
    ok = ret==0 and chain.get('valid') is True and all(r['matched'] for r in suites.values()) and all(frozen.values()) and all(preserved.values()) and all(calculations.values()) and all(bindings.values()) and tests_ok and sandbox_ok
    artifacts = ['plan.json','ensemble.json','validation.json','positive_control.json','result.json','stage_audit.json','lean_verification.json','curves.png']
    audit = {'verified_utc':datetime.now(timezone.utc).isoformat(),'status':'passed' if ok else 'failed',
        'scope':'Byte identity, authentic local receipt membership and independent calculations; not proof of scientific truth or universal novelty',
        'jovin_suites':suites,'chain':chain,'frozen_sources':frozen,'prior_artifacts_preserved':preserved,
        'calculation_checks':calculations,'bindings':bindings,
        'local_tests':{'passed':181,'skipped':3,'summary_matched':tests_ok,'log_sha256':sha(tests)},
        'ci_equivalent_sandbox_suites':{'passed':18,'matched':sandbox_ok,'log_sha256':sha(sandbox)},
        'artifact_sha256':{name:sha(OUT/name) for name in artifacts},'report_sha256':sha(REPO/'docs/MULTISCALE_RUN.md')}
    (OUT/'verification.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({'status':audit['status'],'suites':len(suites),'checks':sum(r['total'] for r in suites.values()),'calculations':calculations},indent=2))
    if not ok:
        raise SystemExit(1)


if __name__=='__main__':
    main()
