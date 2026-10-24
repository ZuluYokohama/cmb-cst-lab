"""Authenticate saved V2 bytes and arithmetic without extending the science run.

No Torch, ducc transform, feature evaluator or optimizer is called. Numerical
failures and incomplete coverage remain scientific outcomes, distinct from an
honest metadata audit passing. All current JoVin receipts are freshly read and
authenticated; no test suite or old V1 verifier main is executed here.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cmbcst.closure_fibers import projection_indices, robust_scales
from report_closure_fibers_v2 import collect_run, read, load_npz, TIERS, SUCCESS, FAILURE, CHECKPOINT_PHASES
from verify_closure_fibers import Checks, authenticate, sha, close, exact
from report_closure_fibers import pytest_summary
from jovin_verify import SUITES

OUT = ROOT/'results/closure_fibers_v2'
GATE = 1e-10
GUARD = 1e-5
Q = .001


def array_sha(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def error(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return float('inf')
    return float(np.max(np.abs(a-b)/(1+np.abs(b)), initial=0.))


def same_error(saved, computed):
    return saved is None if not np.isfinite(computed) else close(saved, computed)


def bins(value, reference, scales, rep):
    ix = projection_indices(rep)
    z = (np.asarray(value)[ix]-np.asarray(reference)[ix])/scales[ix]
    if not np.isfinite(z).all() or np.any(np.abs(z/Q) >= 2.**63):
        raise ValueError('invalid saved bin arithmetic')
    integer = np.rint(z/Q).astype(np.int64)
    margin = Q/2-np.abs(z-Q*integer)
    return integer, margin


def safe_input(value, reference, scales, rep):
    integer, margin = bins(value, reference, scales, rep)
    return bool(np.all(integer == 0) and np.min(margin) >= GUARD)


def safe_future(reference_future, candidate, reference, scales, rep):
    a, am = bins(reference_future, reference, scales, rep)
    b, bm = bins(candidate, reference, scales, rep)
    return bool(np.any((a != b) & (am >= GUARD) & (bm >= GUARD)))


def validate_checkpoint(entry, checks):
    """Audit an atomic checkpoint independently of an interrupted JSON write."""
    info = entry['checkpoint_info']
    if not info['present']:
        return
    prefix = entry['name']+':checkpoint'
    checks.add(prefix+':self_identifiers', info['valid'] and info['self_ids_valid'] and info['id_source'] == 'npz')
    checks.add(prefix+':phase', info['phase_valid'] and info['phase'] in CHECKPOINT_PHASES and info['phase_source'] == 'npz')
    checks.add(prefix+':companion_consistent_or_recovered', info['json_matches_npz'] or
               (info['pair_recovered'] and not info['contradictory_json'] and bool(info['issues'])))
    ids = np.asarray(info['start_ids'], dtype=int)
    overlay = np.asarray(entry['checkpoint_overlay'], dtype=int)
    checks.add(prefix+':overlay_global_ids', set(overlay.tolist()) <= set(ids.tolist())
               and len(np.unique(overlay)) == len(overlay))
    arrays = entry['arrays']
    checks.add(prefix+':overlay_counted_as_started', not len(overlay) or np.all(arrays['started'][overlay] > 0))
    checks.add(prefix+':changed_measurements_unverified', all(
        key not in arrays or not len(overlay) or np.isnan(arrays[key][overlay]).all()
        for key in ('gpu_features', 'gpu_target', 'gpu_ratios', 'gpu_all_features')))


def _row(out, entry, view, scales, locked_reference, checks):
    prefix = f"row:{entry['name']}:{view['start_id']}"
    path = out/view.get('audit_path', f"certificates_v2/{entry['name']}/rows/start_{view['start_id']:04d}.json")
    checks.add(prefix+':audit_hash', path.exists() and (not view.get('audit_sha256') or sha(path) == view['audit_sha256']))
    record = read(path)
    if record is None:
        return None
    checks.add(prefix+':identity', record['start_id'] == view['start_id']
               and record.get('binding_sha256') == entry['certificate']['binding_sha256'])
    if not record.get('completed'):
        checks.add(prefix+':incomplete_explicit', not record.get('certified') and record['status'] == 'incomplete')
        return record
    if not record.get('arrays'):
        checks.add(prefix+':failure_explicit', record['status'] == 'verification_failure'
                   and bool(record.get('error')) and not record.get('certified'))
        return record
    array_path = out/record['arrays']
    checks.add(prefix+':arrays_hash', sha(array_path) == record['arrays_sha256'])
    a = load_npz(array_path)
    if record['status'] == 'verification_failure':
        checks.add(prefix+':preserved_execution_failure', not record.get('certified') and bool(record.get('error')))
        return record
    direct, matrix, reference = (a[k] for k in ('direct_features', 'matrix_features', 'reference_features'))
    checks.add(prefix+':shapes_finite', all(a[k].shape == (4, 52) and np.isfinite(a[k]).all()
               for k in ('direct_features', 'matrix_features', 'reference_features', 'direct_complex_features'))
               and a['theta'].shape == (128,) and np.isfinite(a['theta']).all()
               and a['perturbation_ratios'].shape == a['matrix_perturbation_ratios'].shape == (2,))
    checks.add(prefix+':reference_and_parameters', close(reference[0], locked_reference)
               and array_sha(a['theta']) == record['theta_sha256']
               and exact(a['theta'], entry['saved_arrays']['best_theta'][record['start_id']])
               and exact(a['rms_ratio_names'], ['tile', 'global'])
               and record.get('rms_ratio_names') == ['tile', 'global'])
    stored_error = dict(matrix_vs_direct=error(matrix, direct),
                        independent_direct_covariance=error(a['direct_complex_features'], direct),
                        matrix_vs_direct_rms=error(a['matrix_perturbation_ratios'], a['perturbation_ratios']),
                        archived_gpu_current_features=error(a['archived_gpu_features'], matrix[0]),
                        archived_gpu_future_raw_k4=error(a['archived_gpu_target'], matrix[1, 22]),
                        archived_gpu_rms_ratios=error(a['archived_gpu_ratios'], a['matrix_perturbation_ratios']))
    if 'archived_gpu_all_features' in a:
        stored_error['archived_gpu_all_times_features'] = error(a['archived_gpu_all_features'], matrix)
    checks.add(prefix+':error_arithmetic', all(same_error(record['errors'].get(k), v) for k, v in stored_error.items()))
    checks.add(prefix+':archived_search_gpu', exact(a['archived_gpu_features'], entry['saved_arrays']['gpu_features'][record['start_id']])
               and exact(a['archived_gpu_target'], entry['saved_arrays']['gpu_target'][record['start_id']])
               and exact(a['archived_gpu_ratios'], entry['saved_arrays']['gpu_ratios'][record['start_id']]))
    rep = entry['case']['representation']
    integer, margin = bins(direct[0], reference[0], scales, rep)
    checks.add(prefix+':input_bin_records', exact(integer, record['input_bins']['bins'])
               and close(margin, record['input_bins']['margins'])
               and close(float(margin.min()), record['input_bins']['minimum_margin']))
    input_ok = all(safe_input(value, reference[0], scales, rep)
                   for value in (direct[0], matrix[0], a['direct_complex_features'][0], a['archived_gpu_features']))
    checks.add(prefix+':input_safe_arithmetic', record.get('input_routes_all_safe') == input_ok)
    ratios = a['perturbation_ratios']
    angle_ok = bool(np.max(np.abs(a['theta'])) <= np.pi/4)
    rms_ok = bool(np.isfinite(ratios).all() and np.min(ratios) >= 0 and np.max(ratios) <= .25)
    checks.add(prefix+':bound_records', record['angle_bound_passed'] == angle_ok
               and record['rms_bound_passed'] == rms_ok and close(record['perturbation_ratios'], ratios))
    gaps = np.abs(direct[:, 22]-reference[:, 22])
    checks.add(prefix+':raw_gap_arithmetic', close(gaps, record['raw_k4_gaps'])
               and close(gaps[1], record['raw_primary_gap']))
    invariants = record['invariant_checks']
    checks.add(prefix+':all_time_invariants_present', len(invariants) == 4
               and record['sampling_epsilon'] == 1e-12 and record['sampled_steps'] == [0, 1, 2, 3])
    invariant_ok = bool(len(invariants) == 4 and all(v.get('passed') is True
                        and 0 <= v['normalized_spectral_gram_error'] <= GATE
                        and 0 <= v['split_difference_scaled_error'] <= GATE
                        and v.get('spectral_bins_zero') is True
                        and v.get('zero_reference_auto_shells_exact') is True
                        and v.get('unmodified_removed_shells') is True and v.get('mzero_real') is True
                        and close(v['spectral_minimum_margin'], .5e-6-v['normalized_spectral_gram_error'])
                        for v in invariants))
    dual_ok = True
    if record.get('is_fixed_cell_maximum'):
        checks.add(prefix+':both_gpu_arrays', all(a.get(f'gpu{d}_features', np.empty(0)).shape == (4, 52) for d in (0, 1)))
        dual_errors = {f'gpu{d}': error(a.get(f'gpu{d}_features', np.empty(0)), direct) for d in (0, 1)}
        checks.add(prefix+':both_gpu_error_arithmetic', all(same_error(record.get('actual_both_gpu_errors', {}).get(k), v)
                                                          for k, v in dual_errors.items()))
        dual_ok = bool(max(dual_errors.values()) <= GATE
                       and all(safe_input(a[f'gpu{d}_features'][0], reference[0], scales, rep) for d in (0, 1)))
        coefficients = out/record['coefficients']
        checks.add(prefix+':full_coefficient_binding', sha(coefficients) == record['coefficients_sha256']
                   and array_sha(load_npz(coefficients)['alms']) == record['candidate_harmonics_sha256'])
        for representation in ('baseline', 'capacity', 'repair'):
            qb, qm = bins(direct[0], reference[0], scales, representation)
            distinguish = bool(np.any((qb != 0) & (qm >= GUARD)))
            checks.add(prefix+':augmentation:'+representation,
                       record.get('distinguished_by', {}).get(representation) == distinguish)
    finite_precision = bool(max(stored_error.values()) <= GATE and invariant_ok and dual_ok)
    certified = bool(finite_precision and input_ok and angle_ok and rms_ok)
    checks.add(prefix+':qualification_honest', record['certified'] == certified)
    checks.add(prefix+':raw_tier_arithmetic', all(record['tiers'][f'{t:.3f}'] == bool(certified and gaps[1] >= t) for t in TIERS))
    safe = safe_future(reference[1], direct[1], reference[0], scales, rep)
    if record.get('is_fixed_cell_maximum') and all(a.get(f'gpu{d}_features', np.empty(0)).shape == (4, 52)
                                                 and np.isfinite(a[f'gpu{d}_features']).all() for d in (0, 1)):
        safe = bool(safe and all(safe_future(reference[1], a[f'gpu{d}_features'][1], reference[0], scales, rep) for d in (0, 1)))
    checks.add(prefix+':safe_future_arithmetic', record['safe_primary_future_bin_difference'] == safe)
    return {**record, '_matrix_primary_gap': float(abs(matrix[1, 22]-reference[1, 22])),
            '_matrix_feasible': safe_input(matrix[0], reference[0], scales, rep)
            and np.max(np.abs(a['theta'])) <= np.pi/4
            and np.isfinite(a['matrix_perturbation_ratios']).all()
            and np.min(a['matrix_perturbation_ratios']) >= 0
            and np.max(a['matrix_perturbation_ratios']) <= .25}


def validate_saved(out=OUT, root=ROOT):
    out, root = Path(out), Path(root)
    run = collect_run(out)
    plan, stage, summary = run['plan'], run['stage'], run['summary']
    checks = Checks()
    expected_tiles = [24, 6, 30, 34, 21, 11, 152, 170, 189, 174, 186, 166]
    checks.add('v2_protocol', plan['planned_starts'] == 6144 and plan['starts_per_case'] == plan['planes'] == 128
               and len(plan['cases']) == 48 and plan['selected_tiles'] == expected_tiles
               and plan['local_quantum'] == .001 and plan['spectral_quantum'] == 1e-6
               and plan.get('certificate_guard', plan.get('input_guard', GUARD)) == GUARD
               and plan['sampling_epsilon'] == 1e-12 and plan['numerical_gate'] == GATE
               and plan['threads'] == 8 and plan['compute_cap_seconds'] == 14400
               and plan['verification_reserve_seconds'] == 900
               and plan['seed'] == 82024102027)
    checks.add('plan_stage_binding', stage.get('plan_sha256') == sha(out/'plan.json'))
    checks.attempt('current_source_hashes', lambda: bool(plan['source_sha256'])
                   and all(sha(root/name) == value for name, value in plan['source_sha256'].items()))
    checks.attempt('input_hashes', lambda: len(plan['input_sha256']) == 4
                   and all(sha(root/name) == value for name, value in plan['input_sha256'].items()))
    frozen = plan.get('v1_frozen_sha256', {})
    checks.attempt('all_v1_bytes_preserved', lambda: bool(frozen)
                   and all(sha(root/name) == value for name, value in frozen.items()))
    checks.add('v1_entire_declared_tree', bool(frozen)
               and {str(p.relative_to(root)).replace('\\', '/') for p in (root/'results/closure_fibers_v1').rglob('*') if p.is_file()}
               | {'docs/CLOSURE_FIBERS_SPEC.md'} <= set(frozen))
    checks.attempt('initialization_hash', lambda: sha(out/'initializations.npz') == plan['initialization_sha256'])
    initial = load_npz(out/'initializations.npz')
    reference = load_npz(out/'reference_features.npz')
    normalization = read(out/'normalization_lock.json', {})
    checks.add('frozen_reference_scales', reference is not None and reference['features'].shape == (79, 52)
               and reference['tile_ids'].shape == (79,) and len(np.unique(reference['tile_ids'])) == 79
               and close(reference['scales'], robust_scales(reference['features'])))
    checks.attempt('normalization_lock', lambda: normalization['plan_sha256'] == sha(out/'plan.json')
                   and normalization['reference_features_sha256'] == sha(out/'reference_features.npz'))
    refmap = {} if reference is None else {int(tile): row for tile, row in zip(reference['tile_ids'], reference['features'])}
    scales = np.ones(52) if reference is None else reference['scales']
    for tile in expected_tiles:
        meta_path = out/'operators'/f'family_{tile}.json'
        plane_path = out/'operators'/f'planes_{tile}.npz'
        used = any(e['starts_started'] for e in run['entries'] if e['case']['tile'] == tile)
        checks.add(f'operator:{tile}:present', not used or (meta_path.exists() and plane_path.exists()))
        if meta_path.exists():
            def operator(meta_path=meta_path, plane_path=plane_path, tile=tile):
                meta = read(meta_path)
                config = meta['configuration']
                numerical = load_npz(plane_path)
                return (sha(plane_path) == meta['planes_sha256'] and config['tile'] == tile and config['seed'] == plan['seed']
                        and config['source']['input_sha256'] == plan['input_sha256']
                        and config['source']['epsilon'] == 1e-12 and config['source']['threads'] == 8
                        and numerical['shells'].shape == (128,) and np.isfinite(numerical['u']).all()
                        and np.isfinite(numerical['v']).all()
                        and max(meta['plane_checks'].values()) <= GATE)
            checks.attempt(f'operator:{tile}:bindings', operator)
    for entry in run['entries']:
        name, a, case = entry['name'], entry['saved_arrays'], entry['case']
        validate_checkpoint(entry, checks)
        if a is None:
            checks.add(name+':missing_arrays_explicit', entry['certificate'] is None
                       and (not str(stage.get('status')).startswith('complete') or entry['starts_started'] == 0))
            continue
        checks.add(name+':archive_shapes', a['best_theta'].shape == (128, 128)
                   and a['gpu_features'].shape == (128, 52) and a['gpu_target'].shape == (128,)
                   and a['gpu_ratios'].shape == (128, 2) and a['started'].shape == a['status'].shape == (128,))
        checks.add(name+':case_archive_hash', not entry['record'].get('arrays_sha256')
                   or sha(out/'cases'/f'{name}.npz') == entry['record']['arrays_sha256'])
        checks.add(name+':locked_initializations', initial is not None
                   and exact(a['raw_kicks'], initial[f"tile_{case['tile']}_{case['stage']}"])
                   and exact(a['signs'], initial[f"signs_{case['tile']}_{case['stage']}"]))
        checks.add(name+':bounded_iterations', np.all(a['iterations'] <= 1000) and np.all(a['stalls'] <= 50)
                   and np.all(a['iterations'] >= 0) and np.all(a['stalls'] >= 0))
        cert = entry['certificate']
        if cert is None:
            checks.add(name+':missing_verification_explicit', not str(stage.get('status')).startswith('complete')
                       or entry['starts_started'] == 0)
            continue
        directory = out/'certificates_v2'/name
        binding = read(directory/'binding.json')
        if binding is None and cert.get('status') in ('execution_failed', 'verification_failed', 'incomplete'):
            checks.add(name+':worker_failure_explicit', bool(cert.get('error')) and not any(cert.get('tier_counts', {}).values()))
            continue
        def binding_ok():
            payload = binding['binding']
            digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            source_by_name = {Path(name).name: value for name, value in plan['source_sha256'].items()}
            operator = read(out/'operators'/f"family_{case['tile']}.json", {})
            operator_hashes = payload['operator_sha256']
            operator_ok = all((value == operator.get('family_sha256') if name == f"family_{case['tile']}.npz"
                               else (out/'operators'/name).exists() and sha(out/'operators'/name) == value)
                              for name, value in operator_hashes.items())
            return (digest == binding['binding_sha256'] == cert['binding_sha256']
                    and all(array_sha(a[key]) == value for key, value in payload['archive_array_sha256'].items())
                    and payload['scales_sha256'] == array_sha(scales)
                    and payload['input_guard_robust_units'] == GUARD and payload['direct_epsilon'] == 1e-12
                    and payload['tiers_raw_k4'] == list(TIERS)
                    and operator_ok
                    and all(source_by_name.get(name) == value for name, value in payload['source_sha256'].items())
                    and payload.get('scientific_input_sha256', plan['input_sha256']) == plan['input_sha256'])
        checks.attempt(name+':certificate_binding', binding_ok)
        rows = []
        for view in cert.get('rows', []):
            checks.attempt(name+f":row:{view['start_id']}", lambda view=view: rows.append(
                _row(out, entry, view, scales, refmap[case['tile']], checks)) is None)
        rows = [r for r in rows if r]
        completed_ids = {r['start_id'] for r in rows if r.get('completed')}
        started_ids = set(np.flatnonzero(a['started'] > 0).tolist())
        checks.add(name+':row_coverage_honest', completed_ids <= started_ids and cert['starts_checked'] == len(completed_ids)
                   and (cert['status'] != 'complete' or completed_ids == started_ids))
        selection = read(directory/'selection.json', {})
        checks.add(name+':fixed_selection_binding', selection.get('start_id') == cert.get('maximum_selected_start')
                   and (selection.get('maximum_gap') is None or close(selection['maximum_gap'], cert['maximum_discovered_feasible_gap'])))
        candidates = [r for r in rows if r.get('_matrix_feasible')]
        if candidates and cert['status'] == 'complete' and all(r.get('arrays') for r in rows):
            chosen = min(candidates, key=lambda r: (-r['_matrix_primary_gap'], r['start_id']))
            checks.add(name+':maximum_no_substitution', chosen['start_id'] == cert['maximum_selected_start'])
        maximum = next((r for r in rows if r['start_id'] == cert.get('maximum_selected_start')), None)
        if cert.get('maximum') is not None and maximum is not None:
            keys = ('start_id', 'status', 'certified', 'raw_primary_gap', 'input_bins', 'errors',
                    'perturbation_ratios', 'distinguished_by', 'coefficients_sha256',
                    'actual_both_gpu_errors', 'safe_primary_future_bin_difference')
            checks.add(name+':embedded_maximum_matches_row', all(cert['maximum'].get(k) == maximum.get(k) for k in keys))
        witness = bool(maximum and maximum.get('certified') and maximum.get('actual_both_gpu_errors')
                       and max(maximum['actual_both_gpu_errors'].values()) <= GATE
                       and maximum.get('safe_primary_future_bin_difference'))
        checks.add(name+':cell_maximum_tiers', all(cert['tier_counts'][f'{t:.3f}'] == int(witness and maximum['tiers'][f'{t:.3f}']) for t in TIERS)
                   and cert['reported_cell_maximum_no_factor_witness'] == bool(witness and any(maximum['tiers'].values())))
    elapsed = stage.get('elapsed_seconds')
    utc_elapsed = stage.get('utc_elapsed_seconds')
    budget_elapsed = stage.get('budget_elapsed_seconds')
    checks.add('terminal_status', stage.get('status') in ('complete', 'complete_with_failures', 'incomplete',
               'incomplete_with_failures', 'incomplete_hard_cap', 'execution_failed'))
    checks.add('both_clocks_recorded', all(isinstance(x, (float, int)) and np.isfinite(x) and x >= 0
                                         for x in (elapsed, utc_elapsed, budget_elapsed)))
    if all(isinstance(x, (float, int)) for x in (elapsed, utc_elapsed, budget_elapsed)):
        checks.add('budget_uses_larger_clock', close(budget_elapsed, max(elapsed, utc_elapsed)))
        checks.add('cap_honest', budget_elapsed <= 14400 or str(stage['status']).startswith('incomplete'))
    checks.attempt('utc_interval_arithmetic', lambda: abs((datetime.fromisoformat(stage['completed_utc'])
                   - datetime.fromisoformat(stage['started_utc'])).total_seconds()-utc_elapsed) < 1.)
    if run['result']:
        result = run['result']
        checks.add('final_result_bindings', result['plan_sha256'] == sha(out/'plan.json')
                   and stage['result_sha256'] == sha(out/'result.json') and result['status'] == stage['status'])
        checks.add('final_coverage_arithmetic', all(result[k] == summary[k] for k in
                   ('planned_starts', 'starts_started', 'starts_completed')))
        checks.add('final_cell_tier_arithmetic', result['tier_counts'] == summary['tier_counts'])
    else:
        checks.add('no_false_completion_without_result', not str(stage['status']).startswith('complete'))
    proof_path = root/'data/evidence/closure_lean.json'
    proof = read(proof_path, {})
    checks.attempt('reused_lean_binding', lambda: proof['status'] == 'passed' and proof['theorem_count'] == 7
                   and proof['source_sha256'] == sha(root/'formal/OperationalClosure.lean')
                   and proof['archive_sha256'] == '7289597a44543246c9227b8cae501767e095ce17f6d448b95fc4a8dc97268f6d'
                   and proof['compiler_sha256'] == 'bf8d54e4714cc4b03d3f6bb34c83b7202b87e49c8bfcbff6895c085bb90ceb38'
                   and (not plan.get('proof_sha256') or sha(proof_path) == plan['proof_sha256']))
    profile_path = root/plan.get('preflight_path', 'data/evidence/closure_v2_preflight.json')
    profile = read(profile_path, {})
    checks.attempt('preflight_source_and_output_binding', lambda: profile['status'] == 'passed'
                   and sha(profile_path) == plan['preflight_sha256']
                   and all(sha(root/name) == value for name, value in profile['source_sha256'].items())
                   and len(profile['devices']) == 2)
    def preflight_fixture_binding():
        fixture_path = profile_path.with_name(profile_path.stem+'_fixture.json')
        fixture = read(fixture_path)
        tree = ast.parse((root/'tests/sandbox/jovin_tests_closure_fibers_v2.py').read_text(encoding='utf-8'))
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == 'GPU_ATTESTATION' for target in node.targets))
        attested = ast.literal_eval(assignment.value)
        return (sha(fixture_path) == profile['compact_fixture_sha256'] and fixture == attested
                and len(fixture['devices']) == 2 and {d['device'] for d in fixture['devices']} == {'cuda:0', 'cuda:1'}
                and all(all(np.isfinite(v) and 0 <= v <= GATE for v in device['errors'].values())
                        for device in fixture['devices']))
    checks.attempt('sandbox_actual_output_binding', preflight_fixture_binding)
    checks.add('saved_report_summary', read(out/'report_summary.json', {}) == summary)
    return checks, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=OUT)
    parser.add_argument('--report', type=Path, default=ROOT/'docs/CLOSURE_FIBERS_V2_RUN.md')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--tests-log', type=Path, default=ROOT/'data/closure_v2_tests.log')
    parser.add_argument('--sandbox-log', type=Path, default=ROOT/'data/closure_v2_sandbox_tests.log')
    args = parser.parse_args()
    out = args.run_dir
    output = args.output or out/'verification.json'
    if out.resolve() == (ROOT/'results/closure_fibers_v1').resolve() or (ROOT/'results/closure_fibers_v1').resolve() in output.resolve().parents:
        raise ValueError('V1 evidence is immutable')
    checks, run = validate_saved(out)
    tests = pytest_summary(args.tests_log)
    checks.add('local_regression_log', tests['available'] and tests['passed'] > 0 and tests['failed'] == tests['errors'] == 0)
    text = args.sandbox_log.read_text(errors='replace') if args.sandbox_log.exists() else ''
    names = re.findall(r'^\[PASS\]\s+([^\s:]+)', text, re.MULTILINE)
    checks.add('all_plain_sandbox_suites', len(names) == len(SUITES) and set(names) == set(SUITES) and '[FAIL]' not in text)
    chain_ok, suites, chain = authenticate()
    checks.add('all_current_authentic_jovin', chain_ok and len(suites) == len(SUITES) and all(x['matched'] for x in suites.values()))
    plan = run['plan']
    checks.add('freeze_jovin_receipt', suites.get('closure_fibers_v2', {}).get('receipt_id') == plan.get('jovin_receipt_id')
               and suites.get('closure_fibers_v2', {}).get('bundle_sha256') == plan.get('jovin_bundle_sha256'))
    checks.add('report_and_plots_present', args.report.exists() and all((out/name).exists() for name in ('coverage.png', 'tier_heatmap.png', 'hardware.png')))
    okay = all(checks.values.values())
    artifacts = [p for p in sorted(out.rglob('*')) if p.is_file() and p.resolve() != output.resolve() and p.name != 'progress.json']
    verification = dict(status='passed' if okay else 'failed', metadata_integrity='passed' if okay else 'failed',
                        verified_utc=datetime.now(timezone.utc).isoformat(), scientific_status=run['summary']['scientific_status'],
                        scientific_summary=run['summary'], scope=__doc__, arithmetic_checks=checks.values,
                        errors=checks.errors, jovin_suites=suites, chain=chain, jovin_suite_count=len(suites),
                        jovin_test_count=sum(v.get('total', 0) for v in suites.values()),
                        artifact_sha256={str(p.relative_to(out)).replace('\\', '/'): sha(p) for p in artifacts},
                        report_sha256=sha(args.report) if args.report.exists() else None,
                        supporting_script_sha256={name: sha(ROOT/name) for name in
                            ('scripts/report_closure_fibers_v2.py', 'scripts/verify_closure_fibers_v2.py')},
                        local_tests={**tests, 'log_sha256': sha(args.tests_log) if args.tests_log.exists() else None},
                        plain_sandbox_tests=dict(suites=names, log_sha256=sha(args.sandbox_log) if args.sandbox_log.exists() else None))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(verification, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=verification['status'], scientific_status=verification['scientific_status'],
                         tier_counts=run['summary']['tier_counts'], suites=len(suites),
                         failed=[k for k, v in checks.values.items() if not v], errors=checks.errors), indent=2))
    return 0 if okay else 1


if __name__ == '__main__':
    raise SystemExit(main())
