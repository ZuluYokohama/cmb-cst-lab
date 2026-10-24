"""Independent finite-candidate checks and scoped evidence statuses.

Only NumPy and the standard library are used. No family/search implementation,
transform backend or previous frozen numerical module is imported. Supplied
replay measurements and trust anchors are evidence assertions, not proofs of
runtime independence, signature validity or correct sky physics.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


RESULTS = ('finite_feasible_candidate_verified', 'counterexample_verified',
           'none_found_in_budget', 'incomplete', 'verification_failed',
           'invalid_or_unsupported_input')


def evidence_status(result, *, execution_status='complete', numerical_status='not_assessed',
                    metadata_integrity='not_assessed', statistical_validation='not_assessed'):
    """Keep execution, numerical evidence, closure and physics as separate axes."""
    if result not in RESULTS or execution_status not in ('complete', 'incomplete', 'failed', 'not_started'):
        raise ValueError('unknown evidence or execution status')
    if numerical_status not in ('passed', 'failed', 'not_assessed', 'unavailable'):
        raise ValueError('unknown numerical status')
    if result in ('finite_feasible_candidate_verified', 'counterexample_verified'):
        if execution_status != 'complete' or numerical_status != 'passed':
            raise ValueError('a verified result requires completed numerical verification')
    if result == 'none_found_in_budget' and execution_status != 'complete':
        raise ValueError('an unfinished search cannot report a completed finite search')
    return dict(status=result, scientific_result=result, execution_status=execution_status,
                numerical_status=numerical_status, metadata_integrity=metadata_integrity,
                statistical_validation=statistical_validation,
                closure_status='counterexample_in_declared_precision' if result == 'counterexample_verified' else
                    ('inconclusive' if result == 'none_found_in_budget' else 'not_established'),
                universal_closure_verified=False, statistical_sufficiency_verified=False,
                physical_model_validated=False)


def _sha_string(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdefABCDEF' for c in value):
        raise ValueError('a complete hexadecimal SHA-256 reference is required')
    return value.lower()


def provenance_reference(name, payload, *, expected_sha256=None, source_uri=None, trust_anchor=None):
    """Hash actual bytes/arrays, retaining a separately declared trust anchor.

    Array framing is dtype.str, shape tuple and contiguous bytes; raw byte
    payloads use ordinary file/byte SHA-256. This function verifies no signature
    or ledger membership and makes no authentication claim from an anchor name.
    """
    if not isinstance(name, str) or not name:
        raise ValueError('a nonempty provenance name is required')
    if source_uri is not None and (not isinstance(source_uri, str) or not source_uri):
        raise ValueError('source_uri must be a nonempty string when provided')
    if trust_anchor is not None:
        if (not isinstance(trust_anchor, dict) or any(not isinstance(trust_anchor.get(k), str)
                or not trust_anchor[k] for k in ('kind', 'identifier'))):
            raise ValueError('trust anchor needs explicit kind and identifier')
        trust_anchor = json.loads(json.dumps(trust_anchor, allow_nan=False))
    digest = hashlib.sha256()
    if isinstance(payload, (bytes, bytearray, memoryview)):
        digest.update(bytes(payload)); details = dict(payload_kind='raw_bytes', byte_count=len(payload))
    else:
        value = np.ascontiguousarray(payload)
        if value.dtype.kind not in 'biufcSU':
            raise ValueError('object arrays cannot be stable provenance payloads')
        digest.update(value.dtype.str.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
        details = dict(payload_kind='array', dtype=value.dtype.str, shape=list(value.shape),
                       framing='dtype.str ASCII || shape tuple ASCII || C-contiguous bytes')
    expected = _sha_string(expected_sha256) if expected_sha256 is not None else None
    actual = digest.hexdigest()
    return dict(name=name, sha256=actual, expected_sha256=expected,
                expected_hash_matches=None if expected is None else actual == expected,
                source_uri=source_uri, trust_anchor=trust_anchor,
                anchor_status='caller_declared' if trust_anchor is not None else 'unanchored',
                authentication_status='not_established_by_this_helper', **details)


def scaled_error(value, reference):
    value, reference = np.asarray(value), np.asarray(reference)
    if value.shape != reference.shape or not np.isfinite(value).all() or not np.isfinite(reference).all():
        return float('inf')
    return float(np.max(np.abs(value-reference)/(1+np.abs(reference)), initial=0.))


def _harmonics(value):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'biufc':
        raise ValueError('numeric complete m-major harmonic coefficients required')
    value = np.asarray(raw, dtype=np.complex128)
    if value.ndim != 3 or value.shape[:2] != (2, 2) or not np.isfinite(value).all():
        raise ValueError('finite [2 splits,2 E/B,nalm] harmonics required')
    length = value.shape[-1]
    lmax = int(round((np.sqrt(8*length+1)-3)/2))
    if lmax < 0 or (lmax+1)*(lmax+2)//2 != length:
        raise ValueError('harmonic storage must contain every m-major mode')
    ell = np.concatenate([np.arange(m, lmax+1) for m in range(lmax+1)])
    order = np.concatenate([np.full(lmax+1-m, m, dtype=int) for m in range(lmax+1)])
    if np.any(value[..., order == 0].imag != 0):
        raise ValueError('real-field m=0 coefficients must be real')
    return value, ell, order


def harmonic_gram(alms):
    """Full split/E/B real spectral Gram, independently contracted per shell."""
    value, ell, order = _harmonics(alms)
    result = []
    for shell in range(int(ell.max())+1):
        selected = value[..., ell == shell].reshape(4, shell+1)
        coordinates = np.concatenate((selected[:, :1].real,
            np.sqrt(2.)*selected[:, 1:].real, np.sqrt(2.)*selected[:, 1:].imag), axis=1)
        result.append((coordinates@coordinates.T)/(2*shell+1))
    return np.asarray(result)


def _real_array(value, label):
    raw = np.asarray(value)
    # Object arrays can hide numpy complex scalars whose float conversion
    # merely warns and discards the imaginary part. Reject before conversion.
    if raw.dtype.kind not in 'biuf' or np.iscomplexobj(raw):
        raise ValueError(label+' must contain real numeric values, not complex or object values')
    return np.asarray(raw, dtype=np.float64)


def _real_scalar(value, label):
    raw = np.asarray(value)
    result = _real_array(raw, label)
    if result.shape != () or raw.dtype.kind == 'b' or not np.isfinite(result):
        raise ValueError(label+' must be a finite real scalar')
    return float(result)


def _features(reference, candidate, scales):
    values = []
    for raw, label in ((reference, 'reference features'), (candidate, 'candidate features'), (scales, 'feature scales')):
        values.append(_real_array(raw, label))
    reference, candidate, scales = values
    if (reference.ndim < 1 or not reference.size or candidate.shape != reference.shape
            or scales.shape != reference.shape or not all(np.isfinite(x).all() for x in values)
            or np.any(scales <= 0)):
        raise ValueError('matching finite realized feature arrays and positive fixed scales required')
    return reference, candidate, scales


def _bin_record(value, center, scales, quantum, guard):
    normalized = (value-center)/scales
    if not np.isfinite(normalized).all() or np.max(np.abs(normalized/quantum), initial=0.) >= 2.**63:
        raise ValueError('recorded feature bins exceed finite integer range')
    bins = np.rint(normalized/quantum).astype(np.int64)
    margins = quantum/2-np.abs(normalized-quantum*bins)
    return dict(bins=bins.tolist(), margins=margins.tolist(), minimum_margin=float(margins.min()),
                all_safe=bool(np.all(margins >= guard)))


def _replay_error(reference, candidate, replay_reference, replay_candidate, reported):
    errors = []
    if reported is not None:
        value = _real_scalar(reported, 'independent replay error')
        if value < 0:
            raise ValueError('independent replay error must be a finite nonnegative scalar')
        errors.append(value)
    supplied = replay_reference is not None or replay_candidate is not None
    if supplied:
        if replay_reference is None or replay_candidate is None:
            raise ValueError('both independent replay arrays are required')
        checked_reference = _real_array(replay_reference, 'independent reference features')
        checked_candidate = _real_array(replay_candidate, 'independent candidate features')
        errors.extend([scaled_error(checked_reference, reference), scaled_error(checked_candidate, candidate)])
    return (max(errors) if errors else None), ('independent_arrays' if supplied else
                                              ('reported_scalar' if errors else 'missing'))


def _map_ratio(reference, candidate, weights):
    reference = _real_array(reference, 'reference observed Q/U map')
    candidate = _real_array(candidate, 'candidate observed Q/U map')
    if (reference.ndim != 3 or reference.shape[:2] != (2, 2) or candidate.shape != reference.shape
            or not np.isfinite(reference).all() or not np.isfinite(candidate).all()):
        raise ValueError('matching finite [2 splits,2 Q/U,npix] map arrays required')
    weight = np.ones(reference.shape[-1]) if weights is None else _real_array(weights, 'observed map weights')
    if (weight.shape != reference.shape[-1:] or not np.isfinite(weight).all()
            or np.any(weight < 0) or not np.any(weight > 0)):
        raise ValueError('finite nonnegative common pixel weights with positive sum required')
    mean, changed = reference.mean(axis=0), candidate.mean(axis=0)
    denominator = float(np.sum(mean*mean*weight))
    if denominator <= 0:
        raise ValueError('positive weighted reference coadd map power required')
    return float(np.sqrt(np.sum((changed-mean)**2*weight)/denominator))


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (complex, np.complexfloating)):
        return dict(rejected_complex_real=_clean(float(value.real)), rejected_complex_imag=_clean(float(value.imag)))
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def certify_candidate(reference_alms, candidate_alms, reference_features, candidate_features, scales, *,
                      angle_values, min_rms=.01, max_rms=.25, tolerance=1e-10,
                      direct_feature_error=None, independent_replay=True, angle_limit=np.pi/4,
                      provenance=None, replayed_reference_features=None, replayed_candidate_features=None,
                      reference_maps=None, candidate_maps=None, weights=None,
                      future_reference_features=None, future_candidate_features=None,
                      future_direct_feature_error=None, replayed_future_reference_features=None,
                      replayed_future_candidate_features=None, quantum=.001, bin_guard=1e-5,
                      transition_id=None, representation_id=None):
    """Qualify one finite H1 candidate from independently reconstructed arrays.

    Realized feature locks use max(abs(candidate-reference)/scales). Full Gram
    errors use each shell's reference geometric-mean auto-power. D error uses
    abs(delta D)/(1+abs(reference D)), over the complete retained arrays.
    RMS is perturbation/reference coadd RMS on every retained harmonic mode;
    optional observed-map weights add a separate upper RMS bound. No local RMS
    claim is made when maps are absent. Minimum RMS applies to harmonics only.

    A supplied independent error or replay-array pair is mandatory. Its source
    independence is an explicit caller assertion, not established by this code.
    Optional future arrays require their own independent evidence before any
    counterexample is reported. No future input means no closure claim.
    """
    report = dict(certificate_kind='H1_finite_candidate_feasibility', passed=False, accepted=False,
                  finite_feasibility_verified=False, counterexample_verified=False,
                  checks={}, metrics={}, array_references={},
                  limits=dict(minimum_harmonic_rms=min_rms, maximum_rms=max_rms, tolerance=tolerance,
                              angle_limit=angle_limit, quantum=quantum, bin_guard=bin_guard),
                  independent_replay_assumption='Caller asserts independently reconstructed arrays/errors; implementation independence is not proved here',
                  scope='Finite retained-harmonic and declared realized-feature constraints only; no physical probability, universal manifold, statistical sufficiency or closure pass')
    try:
        controls = np.asarray([_real_scalar(value, label) for value, label in
            ((min_rms, 'minimum RMS'), (max_rms, 'maximum RMS'), (tolerance, 'numerical tolerance'),
             (angle_limit, 'angle limit'), (quantum, 'feature quantum'), (bin_guard, 'bin guard'))])
        min_rms, max_rms, tolerance, angle_limit, quantum, bin_guard = controls.tolist()
        if (not np.isfinite(controls).all() or not .01 <= min_rms <= max_rms <= .25
                or not 0 < tolerance <= 1e-10 or not 0 < angle_limit <= np.pi/4
                or quantum <= 0 or not 1e-5 <= bin_guard < quantum/2):
            raise ValueError('finite positive, ordered, prospectively fixed certificate controls required')
        if type(independent_replay) is not bool:
            raise ValueError('independent_replay must be an explicit boolean')
        ref, ell, order = _harmonics(reference_alms)
        candidate, ell_candidate, _ = _harmonics(candidate_alms)
        if ref.shape != candidate.shape or not np.array_equal(ell, ell_candidate):
            raise ValueError('reference and candidate harmonic layouts differ')
        fref, fcan, scale = _features(reference_features, candidate_features, scales)
        angles = _real_array(angle_values, 'rotation parameters')
        if angles.ndim != 1 or not len(angles) or not np.isfinite(angles).all():
            raise ValueError('a finite nonempty real parameter vector is required')
        if provenance is not None and not isinstance(provenance, dict):
            raise ValueError('provenance must be a plain JSON reference mapping')
        report['provenance'] = json.loads(json.dumps(provenance or {}, allow_nan=False))
        arrays = dict(reference_alms=ref, candidate_alms=candidate,
                      reference_features=fref, candidate_features=fcan, scales=scale, angle_values=angles)
        for name, value in arrays.items():
            report['array_references'][name] = provenance_reference(name, value)
        with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
            before, after = harmonic_gram(ref), harmonic_gram(candidate)
            diagonal = np.diagonal(before, axis1=1, axis2=2)
            gram_scale = np.sqrt(np.maximum(diagonal, 0))[:, :, None]*np.sqrt(np.maximum(diagonal, 0))[:, None, :]
            gram_error = float(np.max(np.abs(after-before)/np.maximum(gram_scale, np.finfo(float).tiny)))
            zero_auto = diagonal == 0
            zero_preserved = bool(np.all(np.diagonal(after, axis1=1, axis2=2)[zero_auto] == 0))
            dref, dcan = ref[0]-ref[1], candidate[0]-candidate[1]
            difference_error = scaled_error(dcan, dref)
            zero_difference_preserved = bool(np.any(dref != 0) or np.all(dcan == 0))
            coadd = (ref[0]+ref[1])/2
            delta = (candidate[0]+candidate[1])/2-coadd
            mode_weights = np.where(order == 0, 1., 2.)
            power = float(np.sum(np.abs(coadd)**2*mode_weights))
            if power <= 0:
                raise ValueError('positive reference coadd harmonic RMS required')
            ratio = float(np.sqrt(np.sum(np.abs(delta)**2*mode_weights)/power))
            feature_error = float(np.max(np.abs(fcan-fref)/scale))
        direct_error, replay_kind = _replay_error(fref, fcan, replayed_reference_features,
                                                 replayed_candidate_features, direct_feature_error)
        for name, value in (('replayed_reference_features', replayed_reference_features),
                            ('replayed_candidate_features', replayed_candidate_features)):
            if value is not None:
                report['array_references'][name] = provenance_reference(name, value)
        current_ref_bins = _bin_record(fref, fref, scale, quantum, bin_guard)
        current_can_bins = _bin_record(fcan, fref, scale, quantum, bin_guard)
        current_bins_match = bool(current_ref_bins['bins'] == current_can_bins['bins']
                                  and current_ref_bins['all_safe'] and current_can_bins['all_safe'])
        report['checks'] = dict(full_harmonic_gram=gram_error <= tolerance,
            zero_reference_auto_modes=zero_preserved, full_split_difference=difference_error <= tolerance,
            zero_reference_split_difference=zero_difference_preserved,
            realized_feature_locks=feature_error <= tolerance,
            current_recorded_bins_match=current_bins_match,
            angle_bounds=bool(np.max(np.abs(angles)) <= angle_limit),
            nontrivial_harmonic_rms=bool(ratio >= min_rms), harmonic_rms_upper_bound=bool(ratio <= max_rms),
            independent_replay_evidence=bool(independent_replay and direct_error is not None and direct_error <= tolerance),
            provenance_hash_references=all(not isinstance(v, dict) or v.get('expected_hash_matches') is not False
                                           for v in report['provenance'].values()))
        report['metrics'] = dict(normalized_full_gram_error=gram_error,
            full_split_difference_scaled_error=difference_error, realized_feature_lock_error=feature_error,
            direct_feature_error=direct_error, independent_evidence_kind=replay_kind,
            harmonic_coadd_perturbation_ratio=ratio, maximum_abs_angle=float(np.max(np.abs(angles))),
            retained_lmax=int(ell.max()), current_reference_bins=current_ref_bins,
            current_candidate_bins=current_can_bins, observed_map_rms_assessed=False,
            rms_scope='All retained harmonic coadd modes; not an observed/local-map bound unless separately recorded')
        if reference_maps is not None or candidate_maps is not None or weights is not None:
            if reference_maps is None or candidate_maps is None:
                raise ValueError('both observed map arrays are required with optional map weights')
            map_ratio = _map_ratio(reference_maps, candidate_maps, weights)
            report['metrics'].update(observed_map_rms_assessed=True, observed_coadd_perturbation_ratio=map_ratio)
            report['checks']['observed_map_rms_upper_bound'] = bool(map_ratio <= max_rms)
            for name, value in (('reference_maps', reference_maps), ('candidate_maps', candidate_maps)):
                report['array_references'][name] = provenance_reference(name, value)
            if weights is not None:
                report['array_references']['observed_map_weights'] = provenance_reference('observed_map_weights', weights)
        feasible = bool(all(report['checks'].values()))
        report.update(passed=feasible, accepted=feasible, finite_feasibility_verified=feasible)
        result = 'finite_feasible_candidate_verified' if feasible else 'verification_failed'
        supplied_future = future_reference_features is not None or future_candidate_features is not None
        if supplied_future:
            if future_reference_features is None or future_candidate_features is None:
                raise ValueError('both future feature arrays are required')
            future_ref, future_can, _ = _features(future_reference_features, future_candidate_features, scale)
            future_error, future_kind = _replay_error(future_ref, future_can,
                replayed_future_reference_features, replayed_future_candidate_features, future_direct_feature_error)
            first = _bin_record(future_ref, fref, scale, quantum, bin_guard)
            second = _bin_record(future_can, fref, scale, quantum, bin_guard)
            a, b = np.asarray(first['bins']), np.asarray(second['bins'])
            safe = (a != b) & (np.asarray(first['margins']) >= bin_guard) & (np.asarray(second['margins']) >= bin_guard)
            transition_declared = bool(isinstance(transition_id, str) and transition_id
                                       and isinstance(representation_id, str) and representation_id)
            future_verified = bool(independent_replay and future_error is not None and future_error <= tolerance
                                   and transition_declared)
            collision = bool(feasible and current_bins_match and future_verified and np.any(safe))
            report['future_pair'] = dict(status='counterexample_verified' if collision else
                ('not_independently_verified' if not future_verified else 'no_safe_separation_for_this_pair'),
                reference=first, candidate=second, direct_feature_error=future_error,
                independent_evidence_kind=future_kind, safe_changed_coordinates=int(np.sum(safe)),
                transition_id=transition_id, representation_id=representation_id,
                transition_assumption='Caller asserts both future arrays use the same declared transition and representation; no operator is executed here')
            for name, value in (('future_reference_features', future_ref), ('future_candidate_features', future_can)):
                report['array_references'][name] = provenance_reference(name, value)
            for name, value in (('replayed_future_reference_features', replayed_future_reference_features),
                                ('replayed_future_candidate_features', replayed_future_candidate_features)):
                if value is not None:
                    report['array_references'][name] = provenance_reference(name, value)
            report['counterexample_verified'] = collision
            if collision:
                result = 'counterexample_verified'
        report.update(evidence_status(result, numerical_status='passed' if feasible else 'failed',
            metadata_integrity='references_recorded_unanchored' if all(report['checks'].values()) else 'not_qualified'))
    except (ValueError, TypeError, OverflowError, FloatingPointError) as exc:
        report.update(passed=False, accepted=False, finite_feasibility_verified=False, counterexample_verified=False,
                      error_type=type(exc).__name__, error=str(exc))
        report.update(evidence_status('invalid_or_unsupported_input', numerical_status='failed'))
    report['rejection_reasons'] = [name for name, passed in report['checks'].items() if not passed]
    return _clean(report)
