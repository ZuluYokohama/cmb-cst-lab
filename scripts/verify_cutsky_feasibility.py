"""Audit saved H1 bytes, statuses and arithmetic; authenticate current JoVin.

No NaMaster/ducc operation, harmonic reconstruction, feature extraction, family
generation or certificate kernel runs here. Reported Gram/RMS certificates are
bound to their input arrays and current source; this is a post-run evidence
audit, not another scientific verification or physical-model certification.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from report_cutsky_feasibility import ROOT, OUT, VERIFIED, TERMINAL, read, load_arrays, sha, array_sha, models_array_sha, collect_run
from verify_closure_fibers import Checks, authenticate
from jovin_verify import SUITES


def close(a, b, gate=1e-10):
    a, b = np.asarray(a), np.asarray(b)
    return bool(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
                and np.max(np.abs(a-b)/(1+np.abs(b)), initial=0.) <= gate)


def same_scalar(a, b):
    return bool(isinstance(a, (int, float)) and isinstance(b, (int, float)) and np.isfinite(a) and np.isfinite(b)
                and abs(a-b) <= 1e-18+1e-9*abs(b))


def lock_error(a, b, scales):
    return float(np.max(np.abs(np.asarray(a)-np.asarray(b))/scales))


def input_hashes(out):
    record = read(out/'inputs/manifest.json', read(out/'inputs/input_manifest.json', {}))
    for key in ('sha256', 'input_sha256', 'artifact_sha256', 'files'):
        if isinstance(record.get(key), dict):
            return record[key]
    return record if record and all(isinstance(v, str) for v in record.values()) else {}


def input_path(out, name):
    path = Path(name)
    return out/path if path.parts and path.parts[0] == 'inputs' else out/'inputs'/path


def check_row(out, entry, item, prepared, profile_arrays, family_arrays, plan, checks):
    row, path = item['row'], item['path']
    expected = entry['expected']; prefix = expected['name']+f":{row.get('start_id')}"
    start = int(row['start_id']); tolerance = expected['tolerance']
    checks.add(prefix+':identity', row['case'] == expected['name'] and row['profile'] == expected['profile']
               and 0 <= start < plan['starts_per_case'] and row['tolerance'] == tolerance
               and row['global_start_id'] == expected['profile_index']*plan['starts_per_case']+start)
    checks.add(prefix+':status_domain', row['status'] in (*TERMINAL, 'in_progress', 'incomplete', 'candidate_found'))
    views = (entry['record'] or {}).get('rows', [])
    view = next((v for v in views if v.get('start_id') == start), None)
    if view:
        checks.add(prefix+':case_audit_hash', view['audit_sha256'] == sha(path)
                   and view['status'] == row['status'] and view['global_start_id'] == row['global_start_id'])
    solver = row.get('solver')
    if solver is None:
        checks.add(prefix+':unfinished_without_solver', row['status'] in ('in_progress', 'incomplete'))
        return
    checks.add(prefix+':solver_limits', 0 <= solver['iterations'] <= plan['max_iterations']
               and 0 <= solver['stalls'] <= plan['max_stalls']
               and solver['minimum_rms'] == plan['minimum_harmonic_rms']
               and solver['maximum_rms'] == plan['maximum_rms'] and solver['tolerance'] == tolerance
               and solver['restore_target'] == min(1e-12, tolerance/10))
    directions = read(out/'inputs/directions.json', {})
    checks.add(prefix+':direction_manifest', solver['seed_manifest'] == directions.get(str(row['global_start_id'])))
    if not row.get('arrays'):
        checks.add(prefix+':no_false_verified_array_absence', row['status'] != VERIFIED)
        return
    array_path = out/row['arrays']
    checks.add(prefix+':array_hash', array_path.exists() and sha(array_path) == row['arrays_sha256'])
    a = load_arrays(array_path)
    if row['status'] not in (VERIFIED, 'verification_failed'):
        checks.add(prefix+':partial_verification_explicit', row['status'] in ('candidate_found', 'incomplete'))
        return
    cert = row['certificate']; metrics = cert.get('metrics', {})
    checks.add(prefix+':H1_scope', not cert.get('counterexample_verified') and not cert.get('universal_closure_verified')
               and not cert.get('statistical_sufficiency_verified') and not cert.get('physical_model_validated')
               and 'future_pair' not in cert)
    if row['status'] == 'verification_failed' and cert.get('status') == 'invalid_or_unsupported_input':
        checks.add(prefix+':explicit_invalid_candidate_rejection', not cert['accepted'] and bool(cert.get('error')))
        return
    nalm = (plan['lmax']+1)*(plan['lmax']+2)//2
    npix = 12*plan['nside']**2
    nfeatures = 12*(len(plan['band_edges'])-1)
    checks.add(prefix+':finite_shapes', a['candidate_alms'].shape == (2, 2, nalm)
               and a['candidate_maps'].shape == (2, 2, npix) and a['theta'].shape == (plan['n_planes'],)
               and a['direct_features'].shape == a['reference_features'].shape == a['scales'].shape == (nfeatures,)
               and all(np.isfinite(v).all() for v in a.values()))
    scales = a['scales']; direct = a['direct_features']; reference = a['reference_features']
    checks.add(prefix+':reference_scales_and_parameters', np.all(scales > 0)
               and np.array_equal(scales, profile_arrays['scales'])
               and close(reference, profile_arrays['reference_features']) and np.array_equal(a['theta'], solver['theta']))
    feature_error = lock_error(direct, reference, scales)
    checks.add(prefix+':realized_lock_arithmetic', same_scalar(metrics['realized_feature_lock_error'], feature_error))
    checks.add(prefix+':saved_predicted_features', 'predicted_features' in a)
    if 'predicted_features' in a:
        checks.add(prefix+':direct_model_error_arithmetic', same_scalar(row['direct_model_scaled_error'], lock_error(a['predicted_features'], direct, scales)))
    checks.add(prefix+':both_actual_gpu_arrays', a.get('actual_gpu_features', np.empty(0)).shape == (2, nfeatures)
               and set(row['actual_both_gpu_errors']) == {'cuda:0', 'cuda:1'})
    gpu_errors = {f'cuda:{d}': lock_error(a['actual_gpu_features'][d], direct, scales) for d in (0, 1)}
    checks.add(prefix+':gpu_error_arithmetic', all(same_scalar(row['actual_both_gpu_errors'][key], value) for key, value in gpu_errors.items()))
    checks.add(prefix+':independent_error_arithmetic', same_scalar(metrics['direct_feature_error'],
               max(row['direct_model_scaled_error'], *gpu_errors.values())))
    checks.add(prefix+':certificate_limits', cert['limits']['minimum_harmonic_rms'] == plan['minimum_harmonic_rms']
               and cert['limits']['maximum_rms'] == plan['maximum_rms'] and cert['limits']['tolerance'] == tolerance)
    checks.add(prefix+':amplitude_records', close(metrics['harmonic_coadd_perturbation_ratio'], solver['rms_ratio'])
               and same_scalar(metrics['maximum_abs_angle'], float(np.max(np.abs(a['theta']))))
               and metrics['observed_map_rms_assessed'] is True)
    expected_refs = dict(reference_alms=family_arrays['reference_alms'], candidate_alms=a['candidate_alms'],
        reference_features=reference, candidate_features=direct, scales=scales, angle_values=a['theta'],
        reference_maps=prepared['reference_maps'], candidate_maps=a['candidate_maps'], observed_map_weights=profile_arrays['mask']**2)
    checks.add(prefix+':certificate_array_bindings', all(cert['array_references'][key]['sha256'] == array_sha(value)
                                                      for key, value in expected_refs.items()))
    original_D = prepared['reference_maps'][0]-prepared['reference_maps'][1]
    candidate_D = a['candidate_maps'][0]-a['candidate_maps'][1]
    derr = float(np.max(np.abs(candidate_D-original_D)/(1+np.abs(original_D))))
    checks.add(prefix+':observed_D_arithmetic', same_scalar(row['observed_split_difference_scaled_error'], derr))
    current = (direct-reference)/scales
    bins = np.rint(current/cert['limits']['quantum']).astype(np.int64)
    margin = cert['limits']['quantum']/2-np.abs(current-cert['limits']['quantum']*bins)
    checks.add(prefix+':current_bin_records', np.array_equal(metrics['current_candidate_bins']['bins'], bins)
               and close(metrics['current_candidate_bins']['margins'], margin))
    numeric_ok = (all(cert['checks'].values()) and feature_error <= tolerance
        and metrics['normalized_full_gram_error'] <= tolerance and metrics['full_split_difference_scaled_error'] <= tolerance
        and plan['minimum_harmonic_rms'] <= metrics['harmonic_coadd_perturbation_ratio'] <= plan['maximum_rms']
        and metrics['observed_coadd_perturbation_ratio'] <= plan['maximum_rms']
        and metrics['maximum_abs_angle'] <= plan['angle_limit'] and derr <= tolerance
        and max(gpu_errors.values()) <= tolerance and row['direct_model_scaled_error'] <= tolerance)
    checks.add(prefix+':honest_candidate_promotion', (row['status'] == VERIFIED) == bool(numeric_ok)
               and cert['accepted'] == all(cert['checks'].values()))


def validate_saved(out=OUT):
    out = Path(out); run = collect_run(out)
    plan, stage, result, summary = run['plan'], run['stage'], run['result'], run['summary']
    checks = Checks()
    checks.add('frozen_protocol', plan['format'] == 'cmbcst.cutsky_feasibility.v1'
        and plan['planned_profiles'] == 6 and plan['planned_cases'] == 12 and plan['planned_starts'] == 48
        and plan['starts_per_case'] == 4 and plan['tolerances'] == [1e-10, 1e-12]
        and plan['minimum_harmonic_rms'] == .01 and plan['maximum_rms'] == .25
        and plan['compute_cap_seconds'] == 1800 and plan['verification_reserve_seconds'] == 180
        and plan['n_planes'] == 128 and plan['threads'] == 8 and plan['blas_threads'] == 1)
    checks.add('plan_stage_binding', stage.get('plan_sha256') == sha(out/'plan.json'))
    checks.attempt('current_source_hashes', lambda: bool(plan['source_sha256'])
        and all(sha(ROOT/name) == value for name, value in plan['source_sha256'].items()))
    checks.attempt('prior_frozen_bytes_preserved', lambda: bool(plan['prior_frozen_sha256'])
        and all(sha(ROOT/name) == value for name, value in plan['prior_frozen_sha256'].items()))
    profile_path = ROOT/plan['preflight_path']; preflight = read(profile_path, {})
    fixture_path = profile_path.with_name(profile_path.stem+'_fixture.json')
    preflight_arrays = profile_path.with_name(profile_path.stem+'_arrays.npz')
    checks.attempt('actual_library_preflight_bindings', lambda: preflight['status'] == 'passed'
        and sha(profile_path) == plan['preflight_sha256']
        and sha(fixture_path) == plan['preflight_fixture_sha256'] == preflight['compact_fixture_sha256']
        and sha(preflight_arrays) == plan['preflight_arrays_sha256'] == preflight['arrays_sha256']
        and all(sha(ROOT/name) == value for name, value in preflight['source_sha256'].items())
        and preflight['versions'] == plan['versions'] and preflight['versions']['pymaster'] == '3.0.1')
    def fixture_binding():
        text = (ROOT/'tests/sandbox/jovin_tests_integration_cutsky_v1.py').read_text(encoding='utf-8')
        tree = ast.parse(text)
        assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'GPU_ATTESTATION' for t in n.targets))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'quadratic_fixture')
        import hashlib
        return (ast.literal_eval(assignment.value) == read(fixture_path)
            and hashlib.sha256(ast.get_source_segment(text, function).encode()).hexdigest() == preflight['fixture_function_sha256']
            and len(preflight['devices']) == 2 and {x['device'] for x in preflight['devices']} == {'cuda:0', 'cuda:1'}
            and all(all(0 <= value <= 1e-10 for value in d['errors'].values()) for d in preflight['devices'])
            and 0 <= preflight['direct_reconstruction_scaled_error'] <= 1e-10
            and 0 <= preflight['observed_reference_control']['scaled_feature_error'] <= 1e-10)
    checks.attempt('recorded_actual_gpu_fixture_binding', fixture_binding)
    manifest_hashes = input_hashes(out)
    required_inputs = {'splits.fits', 'splits.h5', 'family.npz', 'prepared_arrays.npz', 'directions.json', 'provenance.json'}
    inputs_required = summary['starts_started'] > 0 or str(stage.get('status')).startswith('complete')
    checks.add('input_manifest_complete', (not inputs_required and not manifest_hashes)
               or required_inputs <= {Path(name).name for name in manifest_hashes})
    checks.attempt('actual_input_artifact_hashes', lambda: (not inputs_required and not manifest_hashes)
                   or bool(manifest_hashes) and all(sha(input_path(out, name)) == value for name, value in manifest_hashes.items()))
    provenance = read(out/'inputs/provenance.json', {})
    prepared_path = out/'inputs/prepared_arrays.npz'
    family_path = out/'inputs/family.npz'
    prepared = load_arrays(prepared_path) if prepared_path.exists() else None
    family = load_arrays(family_path) if family_path.exists() else None
    if prepared is not None and family is not None:
        checks.add('original_reference_and_residual_bindings', models_array_sha(prepared['reference_maps']) == provenance['reference_maps_sha256']
            and models_array_sha(prepared['residual_maps']) == provenance['residual_maps_sha256']
            and models_array_sha(prepared['reference_maps'][0]) == provenance['input']['a_sha256']
            and models_array_sha(prepared['reference_maps'][1]) == provenance['input']['b_sha256']
            and np.array_equal(prepared['reference_alms'], family['reference_alms']))
        checks.add('canonical_family_array_hashes', all(array_sha(family[name]) == value for name, value in provenance['family']['array_sha256'].items()))
        checks.add('empirical_noise_scope_explicit', provenance['physical_noise_model_validated'] is False
            and provenance['spectral_invariant_scope'] == 'canonical retained alms only')
    else:
        checks.add('unfinished_input_preprocessing_explicit', summary['starts_started'] == 0 and not str(stage.get('status')).startswith('complete'))
    for entry in run['entries']:
        name = entry['expected']['name']; p = entry['profile_record']
        if p is None:
            checks.add(name+':unprocessed_profile_explicit', entry['started'] == 0)
            continue
        path = out/p['arrays']
        checks.add(name+':profile_array_hash', sha(path) == p['arrays_sha256'])
        pa = load_arrays(path); op = p['operator']; profile = op['profile']
        checks.add(name+':supported_observing_profile', profile['representation'] == 'coupled_binned_pseudo_cl'
            and profile['pairs'] == ['AA', 'BB', 'AB'] and profile['channels'] == ['EE', 'EB', 'BE', 'BB']
            and profile['templates'] is None and not profile['anisotropic_weights']
            and profile['mask_lmax'] == profile['lmax'] == plan['lmax']
            and profile['band_edges'] == plan['band_edges'] and op['versions']['pymaster'] == plan['versions']['pymaster'])
        checks.add(name+':operator_mask_hash', array_sha(pa['mask']) == op['mask_sha256']
            and op['source_sha256'] == plan['source_sha256']['src/cmbcst/integration/namaster.py'])
        def operator_digest():
            import hashlib
            content = {key: value for key, value in op.items() if key != 'operator_sha256'}
            return hashlib.sha256(json.dumps(content, sort_keys=True, allow_nan=False).encode()).hexdigest() == op['operator_sha256']
        checks.attempt(name+':operator_manifest_hash', operator_digest)
        checks.add(name+':profile_saved_arithmetic', close(p['positive_pixel_fraction'], float(np.mean(pa['mask'] > 0)))
            and close(p['mean_weight'], float(np.mean(pa['mask'])))
            and close(p['mean_squared_weight'], float(np.mean(pa['mask']**2)))
            and 0 <= p['zero_model_scaled_error'] <= min(plan['tolerances']))
        if entry['record']:
            record = entry['record']
            checks.add(name+':case_count_arithmetic', record['starts'] == entry['started']
                and record['verified'] == entry['verified'] and record['tolerance'] == entry['expected']['tolerance'])
        for item in entry['rows']:
            checks.attempt(name+f":row:{item['row'].get('start_id')}", lambda item=item: check_row(out, entry, item, prepared, pa, family, plan, checks) is None)
    timing = [stage.get(key) for key in ('elapsed_seconds', 'utc_elapsed_seconds', 'budget_elapsed_seconds')]
    checks.add('terminal_execution_status', stage.get('status') in ('complete', 'complete_with_failures', 'incomplete', 'incomplete_hard_cap'))
    checks.add('both_clocks_recorded', all(isinstance(x, (float, int)) and np.isfinite(x) and x >= 0 for x in timing))
    if all(isinstance(x, (float, int)) for x in timing):
        checks.add('budget_clock_arithmetic', close(timing[2], max(timing[:2])))
        checks.add('cap_outcome_honest', timing[2] <= plan['compute_cap_seconds'] or str(stage['status']).startswith('incomplete'))
    if stage.get('completed_utc'):
        checks.attempt('utc_interval_arithmetic', lambda: abs((datetime.fromisoformat(stage['completed_utc'])-datetime.fromisoformat(stage['started_utc'])).total_seconds()-timing[1]) < 1.)
    else:
        checks.add('missing_end_utc_explicit', stage.get('status') == 'incomplete_hard_cap' and (out/'hard_cap.json').exists())
    if result:
        checks.add('terminal_result_bindings', result['plan_sha256'] == sha(out/'plan.json')
                   and sha(out/'result.json') == stage['result_sha256'] and result['status'] == stage['status'])
        checks.add('result_count_arithmetic', result['starts_started'] == summary['starts_started']
            and result['verified_candidates'] == summary['verified_candidates']
            and result['numerical_or_verification_failures'] == summary['numerical_or_verification_failures']
            and result['completed_cases'] == summary['closed_cases'])
        checks.add('no_missing_work_success', result['status'] != 'complete' or
            (summary['starts_started'] == summary['starts_terminal'] == 48 and summary['closed_cases'] == 12))
        checks.add('none_found_is_inconclusive', result.get('scientific_outcome') != 'none_found_in_budget' or
            (summary['verified_candidates'] == 0 and result['status'] == 'complete'))
        checks.add('terminal_input_manifest_binding', result.get('input_manifest_sha256') ==
                   (sha(out/'inputs/manifest.json') if (out/'inputs/manifest.json').exists() else None))
        actual_rows = {(e['expected']['name'], item['row']['start_id']): item for e in run['entries'] for item in e['rows']}
        views = result.get('rows', [])
        checks.add('terminal_row_manifest', len(views) == len(actual_rows)
            and {(v['case'], v['start_id']) for v in views} == set(actual_rows)
            and all(v['audit_sha256'] == actual_rows[(v['case'], v['start_id'])]['audit_sha256']
                    and v['status'] == actual_rows[(v['case'], v['start_id'])]['row']['status'] for v in views))
    else:
        checks.add('missing_result_explicit', str(stage.get('status')).startswith('incomplete'))
    checks.add('saved_report_summary', read(out/'report_summary.json') == summary)
    return checks, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=OUT)
    parser.add_argument('--report', type=Path, default=ROOT/'docs/CUTSKY_FEASIBILITY_RUN.md')
    args = parser.parse_args(); out = args.run_dir
    if not out.name.startswith('cutsky_feasibility_'):
        raise ValueError('only separate cut-sky output directories are supported')
    checks, run = validate_saved(out)
    chain_ok, suites, chain = authenticate()
    checks.add('all_current_authentic_jovin', chain_ok and len(suites) == len(SUITES) == 27 and all(x['matched'] for x in suites.values()))
    suite = suites.get('integration_cutsky_v1', {})
    checks.add('freeze_integration_receipt_binding', suite.get('receipt_id') == run['plan']['jovin_receipt_id']
        and suite.get('bundle_sha256') == run['plan']['jovin_bundle_sha256']
        and suite.get('tests_sha256') == run['plan']['jovin_tests_sha256'])
    checks.add('report_and_plot_presence', args.report.exists() and all((out/name).exists() for name in ('coverage.png', 'rms_residual.png')))
    okay = all(checks.values.values())
    files = [p for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ('verification.json', 'progress.json')]
    audit = dict(status='passed' if okay else 'failed', metadata_integrity='passed' if okay else 'failed',
        verified_utc=datetime.now(timezone.utc).isoformat(), scientific_summary=run['summary'], scope=__doc__,
        arithmetic_checks=checks.values, errors=checks.errors, jovin_suites=suites, chain=chain,
        jovin_suite_count=len(suites), jovin_test_count=sum(x.get('total', 0) for x in suites.values()),
        frozen_runtime_metadata=run['plan']['versions'], runtime_comparison='Recorded Linux environment only; no optional scientific packages required on the Windows metadata verifier',
        artifact_sha256={str(p.relative_to(out)).replace('\\', '/'): sha(p) for p in files},
        report_sha256=sha(args.report) if args.report.exists() else None,
        supporting_script_sha256={name: sha(ROOT/name) for name in ('scripts/report_cutsky_feasibility.py', 'scripts/verify_cutsky_feasibility.py')},
        preflight_reference=dict(path=run['plan']['preflight_path'], sha256=run['plan']['preflight_sha256']),
        source_provenance_limits='Byte identity and receipt membership do not establish observing-model or physical-noise validity')
    (out/'verification.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(metadata_integrity=audit['metadata_integrity'], scientific_summary=run['summary'],
        failed=[k for k, value in checks.values.items() if not value], errors=checks.errors), indent=2))
    return 0 if okay else 1


if __name__ == '__main__':
    raise SystemExit(main())
