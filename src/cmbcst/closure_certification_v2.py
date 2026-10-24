"""Resumable, all-start certification for the separately frozen V2 study.

Thresholds concern the raw dimensionless normalized K4 coordinate, never its
robust scale or bin width. The fixed discovered cell maximum is not replaced by
a runner-up if its independent numerical verification fails. Search maxima are
observed lower bounds; none of these checks establishes a worst-case bound.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from .closure_fibers import features_numpy, projection_indices, ratios_numpy
from .closure_fibers_oracle import oracle_features
from .closure_fibers_data import (
    EPSILON, SIGMA, harmonic_layout, infer_lmax, invariants, replay_candidate, sha,
)

TIERS = (.010, .015, .025, .050)
QUANTUM = .001
GUARD = 1e-5
TOLERANCE = 1e-10
TARGET_INDEX = 22
RMS_RATIO_NAMES = ('tile', 'global')


def scaled_error(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return float('inf')
    return float(np.max(np.abs(a-b)/(1+np.abs(b)), initial=0.))


def input_bins(values, reference, scales, representation):
    """Reference-centered bins with margins in robust-scale units."""
    ix = projection_indices(representation)
    value, ref, scale = (np.asarray(x, dtype=np.float64) for x in (values, reference, scales))
    if (value.shape != (52,) or ref.shape != (52,) or scale.shape != (52,)
            or not all(np.isfinite(x).all() for x in (value, ref, scale))
            or np.any(scale <= 0)):
        raise ValueError('finite 52-coordinate features and positive scales required')
    z = (value[ix]-ref[ix])/scale[ix]
    if np.any(np.abs(z/QUANTUM) >= 2.**63):
        raise ValueError('feature bin exceeds int64 range')
    bins = np.rint(z/QUANTUM).astype(np.int64)
    margins = QUANTUM/2-np.abs(z-QUANTUM*bins)
    return dict(matched=bool(np.all(bins == 0) and np.min(margins) >= GUARD),
                minimum_margin=float(np.min(margins)), bins=bins.tolist(),
                margins=margins.tolist(), guard_robust_units=GUARD)


def future_bin_difference(reference_future, candidate_future, reference, scales, representation):
    left = input_bins(reference_future, reference, scales, representation)
    right = input_bins(candidate_future, reference, scales, representation)
    a, b = np.asarray(left['bins']), np.asarray(right['bins'])
    safe = ((a != b) & (np.asarray(left['margins']) >= GUARD)
            & (np.asarray(right['margins']) >= GUARD))
    return dict(bins_differ=bool(np.any(a != b)), safe_difference=bool(np.any(safe)),
                reference_bins=left['bins'], candidate_bins=right['bins'],
                reference_margins=left['margins'], candidate_margins=right['margins'])


def raw_tiers(gap, certified):
    """Raw K4 thresholds: robust normalization must never enter this function."""
    if not np.isfinite(gap) or gap < 0:
        raise ValueError('finite nonnegative raw K4 gap required')
    return {f'{threshold:.3f}': bool(certified and gap >= threshold) for threshold in TIERS}


def _array_sha(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(x) for x in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, (np.bool_, np.integer)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(_clean(record), indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def _npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _gpu_features(family, theta):
    """Actual all-feature evaluation on both devices; no CPU fallback."""
    import torch
    from .closure_fibers import ClosureCUDA
    values = {}
    for device in (0, 1):
        with torch.cuda.device(device), torch.no_grad():
            model = ClosureCUDA(family, f'cuda:{device}')
            tensor = torch.as_tensor(theta, dtype=torch.float64, device=device)
            values[f'gpu{device}'] = np.stack([model.features(tensor, u).cpu().numpy()
                                             for u in range(4)])
            del model
    return values


def _time_invariants(shared, replay):
    reference, candidate = shared['reference_alms'], replay['candidate_alms']
    ell, _ = harmonic_layout(infer_lmax(reference))
    rate = .5*SIGMA**2*ell*(ell+1)
    return [invariants(reference*np.exp(-u*rate), candidate*np.exp(-u*rate))
            for u in range(4)]


def _binding(case, family, shared, scales, archive, work):
    from . import closure_fibers, closure_fibers_oracle, closure_fibers_data
    sources = [Path(__file__), Path(closure_fibers.__file__),
               Path(closure_fibers_oracle.__file__), Path(closure_fibers_data.__file__)]
    operator_files = [Path(work)/f'{prefix}_{case["tile"]}.{suffix}'
                      for prefix, suffix in (('family', 'json'), ('family', 'npz'), ('planes', 'npz'))]
    record = dict(case=case, source_sha256={p.name: sha(p) for p in sources},
                  operator_sha256={p.name: sha(p) for p in operator_files if p.exists()},
                  family_array_sha256={k: _array_sha(v) for k, v in sorted(family.items())},
                  reference_alms_sha256=_array_sha(shared['reference_alms']),
                  scientific_input_sha256=shared.get('metadata', {}).get('source', {}).get('input_sha256', {}),
                  scales_sha256=_array_sha(scales),
                  archive_array_sha256={k: _array_sha(v) for k, v in sorted(archive.items())},
                  tiers_raw_k4=list(TIERS), quantum=QUANTUM, input_guard_robust_units=GUARD,
                  tolerance=TOLERANCE, direct_epsilon=EPSILON)
    record['array_sha256_rule'] = 'SHA256(dtype.str ASCII || shape tuple ASCII || C-contiguous raw bytes)'
    encoded = json.dumps(_clean(record), sort_keys=True, separators=(',', ':')).encode()
    return record, hashlib.sha256(encoded).hexdigest()


def _archived_errors(archive, index, matrix, ratios):
    errors = dict(current_features=scaled_error(archive['gpu_features'][index], matrix[0]),
                  future_raw_k4=scaled_error(archive['gpu_target'][index], matrix[1, TARGET_INDEX]),
                  rms_ratios=scaled_error(archive['gpu_ratios'][index], ratios))
    if 'gpu_all_features' in archive:
        errors['all_times_features'] = scaled_error(archive['gpu_all_features'][index], matrix)
    return errors


def _row_audit(root, work, case, family, shared, scales, archive, index,
               matrix, reference, budget, replay_function):
    theta = np.asarray(archive['best_theta'][index], dtype=np.float64)
    replay = replay_function(root, case['tile'], work, theta, shared=shared, budget_callback=budget)
    budget()
    direct = replay['direct_family']
    direct_values = np.stack([oracle_features(direct, np.zeros(0), u) for u in range(4)])
    complex_values = np.stack([features_numpy(direct, np.zeros(0), u) for u in range(4)])
    ratios = np.asarray(replay['perturbation_ratios'], dtype=np.float64)
    matrix_ratios = ratios_numpy(family, theta)
    archived_errors = _archived_errors(archive, index, matrix, matrix_ratios)
    errors = dict(matrix_vs_direct=scaled_error(matrix, direct_values),
                  independent_direct_covariance=scaled_error(complex_values, direct_values),
                  matrix_vs_direct_rms=scaled_error(matrix_ratios, ratios),
                  **{'archived_gpu_'+key: value for key, value in archived_errors.items()})
    bins = input_bins(direct_values[0], reference[0], scales, case['representation'])
    input_routes = dict(direct_real=bins,
                        direct_complex=input_bins(complex_values[0], reference[0], scales, case['representation']),
                        matrix_real=input_bins(matrix[0], reference[0], scales, case['representation']),
                        archived_search_gpu=input_bins(archive['gpu_features'][index], reference[0], scales,
                                                       case['representation']))
    invariant_checks = _time_invariants(shared, replay)
    finite = bool(np.isfinite(direct_values).all() and ratios.shape == (2,)
                  and np.isfinite(ratios).all() and np.all(ratios >= 0))
    angle_ok = bool(np.max(np.abs(theta), initial=0.) <= np.pi/4)
    rms_ok = bool(finite and np.max(ratios, initial=0.) <= .25)
    preservation_ok = bool(replay['invariants']['passed'] and all(c['passed'] for c in invariant_checks))
    precision_ok = bool(finite and max(errors.values()) <= TOLERANCE and preservation_ok)
    input_routes_safe = bool(all(item['matched'] for item in input_routes.values()))
    feasible = bool(input_routes_safe and angle_ok and rms_ok and preservation_ok)
    certified = bool(precision_ok and feasible)
    gaps = np.abs(direct_values[:, TARGET_INDEX]-reference[:, TARGET_INDEX])
    future_bins = [future_bin_difference(reference[u], direct_values[u], reference[0], scales,
                                         case['representation']) for u in range(1, 4)]
    record = dict(start_id=int(index), status='certified_endpoint' if certified else
                  ('precision_failure' if not precision_ok else 'ineligible_endpoint'),
                  completed=True, certified=certified, raw_k4_gaps=gaps.tolist(),
                  raw_primary_gap=float(gaps[1]), tiers=raw_tiers(float(gaps[1]), certified),
                  input_bins=bins, input_route_checks=input_routes,
                  input_routes_all_safe=input_routes_safe,
                  angle_bound_passed=angle_ok, rms_bound_passed=rms_ok,
                  perturbation_ratios=ratios.tolist(), invariant_checks=invariant_checks,
                  rms_ratio_names=list(RMS_RATIO_NAMES),
                  reconstruction_invariants=replay['invariants'], errors=errors,
                  numerical_preservation_passed=preservation_ok, future_bins=future_bins,
                  sampling_epsilon=EPSILON, sampled_steps=[0, 1, 2, 3],
                  candidate_harmonics_sha256=_array_sha(replay['candidate_alms']),
                  theta_sha256=_array_sha(theta),
                  array_sha256_rule='SHA256(dtype.str ASCII || shape tuple ASCII || C-contiguous raw bytes)',
                  qualification_scope='Independent CPU direct replay and archived assigned search GPU; both GPUs only if explicitly recorded',
                  safe_primary_future_bin_difference=bool(future_bins[0]['safe_difference']))
    arrays = dict(theta=theta, matrix_features=matrix, direct_features=direct_values,
                  direct_complex_features=complex_values, reference_features=reference,
                  perturbation_ratios=ratios, matrix_perturbation_ratios=matrix_ratios,
                  archived_gpu_features=archive['gpu_features'][index],
                  rms_ratio_names=np.asarray(RMS_RATIO_NAMES),
                  archived_gpu_target=archive['gpu_target'][index],
                  archived_gpu_ratios=archive['gpu_ratios'][index])
    if 'gpu_all_features' in archive:
        arrays['archived_gpu_all_features'] = archive['gpu_all_features'][index]
    return record, arrays, replay


def certify_case_v2(root, out, work, case, family, shared, scales, archive, budget=None,
                    *, gpu_evaluator=None, replay_function=None):
    """Verify every started start's saved best endpoint, checkpointing each row.

    Archive keys: best_theta[S,P], started[S], actual gpu_features[S,52],
    gpu_target[S] (raw future K4), gpu_ratios[S,2]. Optional actual
    gpu_all_features[S,4,52] is additionally checked. The supplied best_theta
    must already be the runner's frozen best endpoint for each started start.

    Optional callables exist for synthetic tests and orchestration; production
    uses the unchanged V1 direct harmonic replay and actual both-device Torch
    evaluations. A timeout preserves a resumable, explicitly incomplete case.
    """
    tick = time.perf_counter()
    root, out, work = Path(root), Path(out), Path(work)
    budget = (lambda: None) if budget is None else budget
    replay_function = replay_candidate if replay_function is None else replay_function
    gpu_evaluator = _gpu_features if gpu_evaluator is None else gpu_evaluator
    if EPSILON != 1e-12:
        raise ValueError('V2 direct sampling epsilon must remain exactly 1e-12')
    scales = np.asarray(scales, dtype=np.float64)
    theta = np.asarray(archive['best_theta'], dtype=np.float64)
    started = np.asarray(archive['started']) > 0
    if (theta.ndim != 2 or started.shape != theta.shape[:1]
            or np.shape(archive['gpu_features']) != (len(theta), 52)
            or np.shape(archive['gpu_target']) != (len(theta),)
            or np.shape(archive['gpu_ratios']) != (len(theta), 2)
            or ('gpu_all_features' in archive and np.shape(archive['gpu_all_features']) != (len(theta), 4, 52))):
        raise ValueError('invalid V2 endpoint archive shapes')
    if scales.shape != (52,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('52 finite positive frozen robust scales required')
    name = f"tile_{int(case['tile'])}_{case['stage']}_{case['representation']}"
    directory = out/'certificates_v2'/name
    directory.mkdir(parents=True, exist_ok=True)
    binding, digest = _binding(case, family, shared, scales, archive, work)
    lock = directory/'binding.json'
    if lock.exists():
        if json.loads(lock.read_text())['binding_sha256'] != digest:
            raise ValueError('preserve V2 checkpoint: source/input/operator/archive binding changed')
    else:
        _json(lock, dict(binding_sha256=digest, binding=binding))
    ids = np.flatnonzero(started)
    record = dict(status='incomplete', case=case, binding_sha256=digest,
                  starts_started=len(ids), starts_checked=0, rows=[], maximum_selected_start=None,
                  maximum_discovered_feasible_gap=None, tier_counts={f'{x:.3f}': 0 for x in TIERS},
                  note='Raw K4 tiers; finite discovered maxima are not worst-case upper bounds')
    checkpoint = directory/'case.json'
    if checkpoint.exists():
        previous = json.loads(checkpoint.read_text())
        if previous.get('binding_sha256') != digest:
            raise ValueError('case checkpoint binding changed')
        for old in previous.get('rows', []):
            if old.get('audit_sha256'):
                audit = out/old['audit_path']
                if not audit.exists() or sha(audit) != old['audit_sha256']:
                    raise ValueError('row checkpoint audit changed')
    def persist():
        record['starts_checked'] = sum(bool(row.get('completed')) for row in record['rows'])
        record['row_cpu_direct_raw_tier_counts'] = {
            f'{t:.3f}': sum(bool(row.get('tiers', {}).get(f'{t:.3f}'))
                            for row in record['rows']) for t in TIERS}
        maximum = next((r for r in record['rows'] if r['start_id'] == record['maximum_selected_start']), None)
        dual_verified = bool(maximum and maximum.get('certified')
                             and maximum.get('actual_both_gpu_errors')
                             and max(maximum['actual_both_gpu_errors'].values()) <= TOLERANCE)
        witness = bool(dual_verified and maximum.get('safe_primary_future_bin_difference'))
        record['tier_counts'] = {f'{t:.3f}': int(witness and maximum['tiers'][f'{t:.3f}'])
                                 for t in TIERS}
        record['quantized_cell_maximum_collision_verified'] = witness
        record['reported_cell_maximum_no_factor_witness'] = bool(witness and any(maximum['tiers'].values()))
        record['rms_ratio_names'] = list(RMS_RATIO_NAMES)
        record['seconds_this_invocation'] = time.perf_counter()-tick
        compact_keys = ('start_id', 'status', 'completed', 'certified', 'raw_primary_gap',
                        'tiers', 'is_fixed_cell_maximum', 'audit_path', 'audit_sha256',
                        'arrays', 'arrays_sha256', 'coefficients', 'coefficients_sha256')
        view = {**record, 'rows': [{k: row[k] for k in compact_keys if k in row}
                                   for row in record['rows']],
                'row_audits': 'Full errors, feature arrays, margins and invariants are bound by per-row audit SHA256s'}
        _json(checkpoint, view)
    persist()
    try:
        budget()
        reference = np.stack([oracle_features(family, np.zeros(theta.shape[1]), u) for u in range(4)])
        reference_complex = np.stack([features_numpy(family, np.zeros(theta.shape[1]), u) for u in range(4)])
        reference_error = scaled_error(reference_complex, reference)
        if reference_error > TOLERANCE:
            raise FloatingPointError('independent reference covariance disagrees')
        matrix_values, eligible, gaps = {}, [], {}
        for index in ids:
            budget()
            if not np.isfinite(theta[index]).all():
                continue
            matrix = np.stack([oracle_features(family, theta[index], u) for u in range(4)])
            matrix_values[int(index)] = matrix
            ratios = ratios_numpy(family, theta[index])
            match = input_bins(matrix[0], reference[0], scales, case['representation'])
            gap = float(abs(matrix[1, TARGET_INDEX]-reference[1, TARGET_INDEX]))
            gaps[int(index)] = gap
            if (match['matched'] and np.isfinite(ratios).all() and np.all(ratios >= 0)
                    and np.max(ratios, initial=0.) <= .25
                    and np.max(np.abs(theta[index]), initial=0.) <= np.pi/4):
                eligible.append(int(index))
        selected = min(eligible, key=lambda i: (-gaps[i], i)) if eligible else None
        selection = dict(start_id=selected, maximum_gap=gaps.get(selected),
                         rule='Independent matrix feasible maximum; tie by start ID; no precision-failure substitution')
        selection_path = directory/'selection.json'
        if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
            raise ValueError('fixed V2 maximum selection changed on resume')
        if not selection_path.exists():
            _json(selection_path, selection)
        record.update(maximum_selected_start=selected, maximum_discovered_feasible_gap=gaps.get(selected),
                      provisional_feasible_endpoints=len(eligible), independent_reference_error=reference_error)
        persist()
        for index in ids:
            index = int(index)
            row_path = directory/'rows'/f'start_{index:04d}.json'
            arrays_path = directory/'rows'/f'start_{index:04d}.npz'
            if row_path.exists():
                old = json.loads(row_path.read_text())
                if old.get('binding_sha256') != digest:
                    raise ValueError('row checkpoint binding changed')
                if old.get('completed'):
                    if old.get('arrays_sha256') and (not arrays_path.exists() or sha(arrays_path) != old['arrays_sha256']):
                        raise ValueError('row checkpoint arrays changed')
                    if old.get('coefficients'):
                        coefficients = out/old['coefficients']
                        if not coefficients.exists() or sha(coefficients) != old['coefficients_sha256']:
                            raise ValueError('maximum coefficient artifact changed')
                    old.update(audit_path=str(row_path.relative_to(out)), audit_sha256=sha(row_path))
                    record['rows'].append(old)
                    persist()
                    continue
            budget()
            row, arrays = {}, None
            try:
                if index not in matrix_values:
                    raise ValueError('nonfinite saved endpoint cannot be reconstructed')
                row, arrays, replay = _row_audit(root, work, case, family, shared, scales,
                                                archive, index, matrix_values[index], reference,
                                                budget, replay_function)
                if index == selected:
                    coefficients = directory/f'{name}_maximum_candidate_alms.npz'
                    _npz(coefficients, alms=replay['candidate_alms'])
                    row.update(coefficients=str(coefficients.relative_to(out)), coefficients_sha256=sha(coefficients))
                    budget()
                    gpu_values = gpu_evaluator(family, theta[index])
                    budget()
                    if set(gpu_values) != {'gpu0', 'gpu1'}:
                        raise ValueError('fixed maximum requires actual all-time features from both GPUs')
                    gpu_errors = {key: scaled_error(value, arrays['direct_features'])
                                  for key, value in gpu_values.items()}
                    arrays.update({key+'_features': value for key, value in gpu_values.items()})
                    row['actual_both_gpu_errors'] = gpu_errors
                    gpu_bins = {key: input_bins(np.asarray(value)[0], reference[0], scales, case['representation'])
                                for key, value in gpu_values.items()}
                    row['actual_both_gpu_input_bins'] = gpu_bins
                    gpu_future_bins = {key: future_bin_difference(reference[1], np.asarray(value)[1], reference[0],
                                                                  scales, case['representation'])
                                       for key, value in gpu_values.items()}
                    row['actual_both_gpu_primary_future_bins'] = gpu_future_bins
                    if max(gpu_errors.values()) > TOLERANCE or not all(x['matched'] for x in gpu_bins.values()):
                        row.update(status='precision_failure', certified=False)
                        row['tiers'] = raw_tiers(row['raw_primary_gap'], False)
                    row['safe_primary_future_bin_difference'] = bool(
                        row['safe_primary_future_bin_difference']
                        and all(x['safe_difference'] for x in gpu_future_bins.values()))
                    separation = {name: input_bins(arrays['direct_features'][0], reference[0], scales, name)
                                  for name in ('baseline', 'capacity', 'repair')}
                    row['separation_details'] = separation
                    row['distinguished_by'] = {
                        name: bool(np.any((np.asarray(item['bins']) != 0)
                                          & (np.asarray(item['margins']) >= GUARD)))
                        for name, item in separation.items()}
                    row['is_fixed_cell_maximum'] = True
            except TimeoutError:
                row.update(start_id=index, status='incomplete', completed=False, certified=False,
                           binding_sha256=digest, is_fixed_cell_maximum=index == selected,
                           tiers={f'{t:.3f}': False for t in TIERS})
                if arrays is not None:
                    _npz(arrays_path, **arrays)
                    row.update(arrays=str(arrays_path.relative_to(out)), arrays_sha256=sha(arrays_path))
                _json(row_path, row)
                row.update(audit_path=str(row_path.relative_to(out)), audit_sha256=sha(row_path))
                record['rows'].append(row)
                persist()
                raise
            except Exception as exc:
                row.update(start_id=index, status='verification_failure', completed=True, certified=False,
                           error_type=type(exc).__name__, error=traceback.format_exc(),
                           is_fixed_cell_maximum=index == selected,
                           tiers={f'{t:.3f}': False for t in TIERS})
            if arrays is not None:
                _npz(arrays_path, **arrays)
                row.update(arrays=str(arrays_path.relative_to(out)), arrays_sha256=sha(arrays_path))
            row['binding_sha256'] = digest
            _json(row_path, row)
            row.update(audit_path=str(row_path.relative_to(out)), audit_sha256=sha(row_path))
            record['rows'].append(row)
            persist()
        maximum = next((r for r in record['rows'] if r['start_id'] == selected), None)
        if maximum is None:
            outcome = 'no_provisionally_feasible_endpoint'
        elif not maximum.get('certified'):
            outcome = 'fixed_maximum_failed_verification_no_substitution'
        elif any(maximum['tiers'].values()) and maximum.get('safe_primary_future_bin_difference'):
            outcome = 'certified_cell_maximum_no_factor_witness'
        elif any(maximum['tiers'].values()):
            outcome = 'certified_raw_k4_threshold_endpoint_without_safe_future_bin_difference'
        else:
            outcome = 'certified_maximum_below_all_raw_k4_tiers'
        record.update(status='complete', outcome=outcome, maximum=maximum,
                      certified_endpoints=sum(bool(r.get('certified')) for r in record['rows']),
                      verification_failures=sum(r['status'] in ('precision_failure', 'verification_failure')
                                                for r in record['rows']))
    except TimeoutError as exc:
        record.update(status='incomplete', interruption=str(exc),
                      unverified_start_ids=[int(i) for i in ids
                                            if int(i) not in {r['start_id'] for r in record['rows'] if r.get('completed')}])
    except Exception:
        record.update(status='verification_failed', error=traceback.format_exc())
        persist()
        raise
    persist()
    return _clean(record)
