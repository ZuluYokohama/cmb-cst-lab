"""Check authentic current JoVin receipts and all frozen grid-audit bindings."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from jovin_verify import REPO, SUITES, build_bundle, configuration, _harness

OUT = REPO/'results/orientation_v1'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scaled_error(measured, expected):
    if len(measured) != len(expected) or not all(math.isfinite(v) for v in measured+expected):
        return math.inf
    return max(abs(a-b)/(1+abs(b)) for a,b in zip(measured, expected))


def main():
    command, env = configuration(argparse.Namespace(harness=None, harness_python=None,
        runtime='wsl:JovinHarness', data_dir=None))
    ret, chain = _harness(command, env, ['receipts', 'verify'])
    history = read(REPO/'data/evidence/jovin_receipts.json')
    suites = {}
    for suite, (_, tests) in SUITES.items():
        record = [r for r in history if r['suite'] == suite][-1]
        show_ret, authentic = _harness(command, env, ['receipts', 'show', record['receipt_id']])
        member_ret, membership = _harness(command, env, ['receipts', 'verify', record['receipt_id']])
        matched = (record['passed'] and record['receipt_verified'] and record['chain_valid']
            and hashlib.sha256(build_bundle(suite)[0].encode()).hexdigest() == record['bundle_sha256']
            and sha(REPO/tests) == record['tests_sha256'] and show_ret == 0 and authentic == record['receipt']
            and member_ret == 0 and membership.get('valid') is True and membership.get('chain') == 'linked')
        suites[suite] = {'matched':bool(matched), 'receipt_id':record['receipt_id'], 'total':record['total']}
    frozen = {str(path.relative_to(REPO)):all(sha(REPO/name) == digest
        for name, digest in read(path).get('source_sha256', {}).items())
        for path in (REPO/'results').glob('*/plan.json')}
    preserved = {str(path.relative_to(REPO)):all(sha(path.parent/name) == digest
        for name, digest in read(path).get('artifact_sha256', {}).items())
        for path in (REPO/'results').glob('*/verification.json') if path.parent != OUT}

    plan, result = read(OUT/'plan.json'), read(OUT/'result.json')
    tolerance = 1e-10
    cases = ['NGP', 'SGP', 'ColdSpot'] + [f'Gaussian{i}' for i in range(8)] + ['ramp', 'spike']
    representations = ['raw', 'standardized', 'normal_scores']
    expected_domain = {(name, rep, k, reflected) for name in cases for rep in representations
        for k in range(4) for reflected in (False, True)}
    records = result['records']
    observed_domain = {(r['case'], r['representation'], r['quarter_turns'], r['reflected']) for r in records}
    by_key = {(r['case'], r['representation'], r['quarter_turns'], r['reflected']):r for r in records}
    required_errors = {'cpu_gpu_curves', 'cpu_gpu_features', 'orientation_curves',
        'orientation_features', 'topology_lifetime', 'rank_commutation'}
    checks = {
        'frozen_domain':plan['representations'] == representations and plan['quarter_turns'] == [0,1,2,3]
            and plan['reflections'] == [False,True] and plan['shape'] == [128,128]
            and plan['gaussian_seeds'] == list(range(710395,710403)) and plan['spike_seed'] == 710410
            and plan['scaled_tolerance'] == tolerance,
        'all_312_distinct_checks':len(records) == 312 and observed_domain == expected_domain
            and result['cases'] == 13 and result['transform_checks'] == 312,
        'all_expected_error_fields':all(set(r['errors']) == required_errors
            | ({'affine_feature_invariance'} if r['representation'] == 'standardized' else set()) for r in records),
        'all_precision_and_finiteness':all(all(math.isfinite(v) and 0 <= v <= tolerance for v in r['errors'].values())
            and len(r['features']) == len(r['expected_features']) == 6
            and all(math.isfinite(v) for v in r['features']+r['expected_features']+r['finite_PD0']) for r in records),
        'all_exact_checks':all(r['exact_checks'] == {'histogram_preserved':True, 'topology_count_preserved':True} for r in records),
        'status_and_no_physical_claims':result['status'] == 'passed' and result['failures'] == 0
            and all(r['passed'] is True for r in records) and result['accepted_cosmological_claims'] == [],
        'positive_elapsed_time':math.isfinite(result['seconds']) and result['seconds'] > 0,
    }
    covariance, topology, affine, ranks = [], [], [], []
    for r in records:
        base = by_key[(r['case'], r['representation'], 0, False)]
        sign = -1 if r['quarter_turns'] % 2 else 1
        expected = base['features'][:5]+[sign*base['features'][5]]
        covariance.append(scaled_error(r['expected_features'], expected) <= tolerance
            and scaled_error(r['features'], expected) <= tolerance
            and abs(scaled_error(r['features'], r['expected_features'])-r['errors']['orientation_features']) <= tolerance)
        topology.append(scaled_error(r['finite_PD0'], base['finite_PD0']) <= tolerance
            and r['finite_PD0'][0] == by_key[(r['case'], 'raw', 0, False)]['finite_PD0'][0])
        if r['representation'] == 'standardized':
            raw = by_key[(r['case'], 'raw', 0, False)]['features']
            expected_affine = raw[:5]+[sign*raw[5]]
            affine.append(scaled_error(r['features'], expected_affine) <= tolerance)
        ranks.append(r['errors']['rank_commutation'] <= tolerance)
    checks.update({'independent_feature_covariances':all(covariance),
        'independent_topology_invariance':all(topology),
        'all_104_affine_comparisons':len(affine) == 104 and all(affine),
        'all_rank_commutation_checks':all(ranks),
        'all_error_aggregates':set(result['maximum_scaled_errors']) == required_errors | {'affine_feature_invariance'}
            and all(result['maximum_scaled_errors'][key] == max(r['errors'].get(key, 0.) for r in records)
                for key in result['maximum_scaled_errors'])})
    orientation_receipt = [r for r in history if r['suite'] == 'orientation'][-1]
    bindings = {'plan':result['plan_sha256'] == sha(OUT/'plan.json'),
        'cached_patches':plan['patch_source_sha256'] == sha(REPO/'data/work/spherical_topology_v1/observed_patches.npz'),
        'authentic_orientation_receipt':plan['jovin_receipt_id'] == orientation_receipt['receipt_id']
            and plan['jovin_bundle_sha256'] == orientation_receipt['bundle_sha256'] and suites['orientation']['matched']}
    tests = REPO/'data/orientation_tests.log'
    sandbox = REPO/'data/orientation_sandbox_tests.log'
    tests_ok = '182 passed, 3 skipped' in tests.read_text()
    sandbox_ok = sandbox.read_text().count('[PASS]') == 19 and '[FAIL]' not in sandbox.read_text()
    ok = (ret == 0 and chain.get('valid') is True and len(suites) == 19 and all(r['matched'] for r in suites.values())
        and all(frozen.values()) and all(preserved.values()) and all(checks.values()) and all(bindings.values())
        and tests_ok and sandbox_ok)
    audit = {'verified_utc':datetime.now(timezone.utc).isoformat(), 'status':'passed' if ok else 'failed',
        'scope':'Current source/test byte identity, authentic receipt membership and saved numerical invariance checks; not SO(3) invariance, scientific truth or novelty',
        'jovin_suites':suites, 'chain':chain, 'frozen_sources':frozen, 'prior_artifacts_preserved':preserved,
        'calculation_checks':checks, 'bindings':bindings,
        'local_tests':{'passed':182, 'skipped':3, 'summary_matched':tests_ok, 'log_sha256':sha(tests)},
        'ci_equivalent_sandbox_suites':{'passed':19, 'matched':sandbox_ok, 'log_sha256':sha(sandbox)},
        'artifact_sha256':{name:sha(OUT/name) for name in ('plan.json','result.json','representations.png')},
        'report_sha256':sha(REPO/'docs/ORIENTATION_MERIT_RUN.md'),
        'supporting_source_sha256':{name:sha(REPO/name) for name in
            ('scripts/report_orientation.py','scripts/verify_orientation_evidence.py',
             'tests/sandbox/jovin_tests_orientation.py','tests/test_orientation.py')}}
    (OUT/'verification.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps({'status':audit['status'], 'suites':len(suites),
        'checks':sum(r['total'] for r in suites.values()), 'calculation_checks':checks,
        'bindings':bindings, 'local_tests':tests_ok, 'sandbox_suites':sandbox_ok}, indent=2))
    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
