"""Independent synthetic certificate controls; no transform, sky or GPU job."""
import hashlib
import json
import warnings

import numpy as np
import pytest

from cmbcst.integration.certificates import (
    certify_candidate, evidence_status, harmonic_gram, provenance_reference,
)


def pack_shell_two(rows):
    """Known real l=2 coordinates into complete m-major complex storage."""
    out = np.zeros((*np.shape(rows)[:-1], 6), dtype=np.complex128)
    out[..., 2] = rows[..., 0]
    out[..., 4] = (rows[..., 1]+1j*rows[..., 3])/np.sqrt(2.)
    out[..., 5] = (rows[..., 2]+1j*rows[..., 4])/np.sqrt(2.)
    return out


def fixture(angle=.1):
    mean = np.array([[1., .4, .2, .3, .1], [.2, 1., .3, .1, -.2]])
    difference = np.array([[0., 0., 0., 0., .2], [0., 0., 0., 0., -.3]])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    changed = mean.copy()
    changed[:, :2] = mean[:, :2]@rotation
    before_rows = np.stack([mean+difference/2, mean-difference/2])
    after_rows = np.stack([changed+difference/2, changed-difference/2])
    return pack_shell_two(before_rows), pack_shell_two(after_rows), before_rows


def certify(reference=None, candidate=None, **kwargs):
    before, after, _ = fixture()
    defaults = dict(angle_values=np.array([.1]), direct_feature_error=0.)
    defaults.update(kwargs)
    return certify_candidate(before if reference is None else reference,
                             after if candidate is None else candidate,
                             [4., 2., 1.], [4., 2., 1.], [1., 1., 1.], **defaults)


def test_feasible_point_is_not_a_closure_or_physics_certificate():
    record = certify()
    assert record['status'] == 'finite_feasible_candidate_verified'
    assert record['passed'] and record['finite_feasibility_verified']
    assert not record['counterexample_verified'] and not record['universal_closure_verified']
    assert not record['statistical_sufficiency_verified'] and not record['physical_model_validated']
    assert .01 <= record['metrics']['harmonic_coadd_perturbation_ratio'] <= .25
    assert not record['metrics']['observed_map_rms_assessed']
    assert record['array_references']['candidate_alms']['shape'] == [2, 2, 6]
    json.dumps(record, allow_nan=False)


def test_full_four_field_gram_matches_known_real_coordinate_gram():
    reference, candidate, rows = fixture()
    expected = rows.reshape(4, 5)@rows.reshape(4, 5).T/5
    assert np.allclose(harmonic_gram(reference)[2], expected, rtol=0, atol=2e-16)
    assert np.allclose(harmonic_gram(candidate), harmonic_gram(reference), rtol=0, atol=3e-16)


def test_full_spectral_preservation_does_not_replace_fixed_split_difference():
    reference, _, rows = fixture()
    # Apply an orthogonal map to ALL four fields, rotating the D direction too.
    rotation = np.eye(5)
    ids = np.ix_([0, 4], [0, 4])
    rotation[ids] = [[np.cos(.1), -np.sin(.1)], [np.sin(.1), np.cos(.1)]]
    candidate = pack_shell_two(rows@rotation)
    record = certify(reference, candidate)
    assert record['checks']['full_harmonic_gram']
    assert not record['checks']['full_split_difference']
    assert record['status'] == 'verification_failed' and not record['accepted']


def test_nonorthogonal_perturbation_and_realized_feature_drift_fail():
    reference, candidate, _ = fixture()
    changed = candidate.copy()
    changed[:, 0] *= 1.01
    assert not certify(reference, changed)['checks']['full_harmonic_gram']
    record = certify_candidate(reference, candidate, [1.], [1.+1e-8], [1.],
                               angle_values=[.1], direct_feature_error=0.)
    assert record['checks']['current_recorded_bins_match']
    assert not record['checks']['realized_feature_locks']


def test_trivial_candidate_missing_replay_and_inaccurate_replay_are_rejected():
    reference, _, _ = fixture()
    assert not certify(reference, reference)['checks']['nontrivial_harmonic_rms']
    missing = certify(direct_feature_error=None)
    assert missing['status'] == 'verification_failed'
    assert not missing['checks']['independent_replay_evidence']
    assert not certify(independent_replay=False)['accepted']
    assert not certify(direct_feature_error=2e-10)['accepted']
    independent = certify(direct_feature_error=None,
                          replayed_reference_features=[4., 2., 1.],
                          replayed_candidate_features=[4., 2., 1.])
    assert independent['accepted'] and independent['metrics']['independent_evidence_kind'] == 'independent_arrays'
    assert 'replayed_candidate_features' in independent['array_references']


@pytest.mark.parametrize('change', [dict(tolerance=1e-8), dict(min_rms=.001),
                                  dict(max_rms=.5), dict(angle_limit=np.pi),
                                  dict(bin_guard=1e-7)])
def test_h1_profile_cannot_be_relaxed_to_obtain_a_passing_point(change):
    record = certify(**change)
    assert record['status'] == 'invalid_or_unsupported_input' and not record['accepted']


def test_nonfinite_or_invalid_harmonics_angles_and_scales_are_not_certified():
    reference, candidate, _ = fixture()
    for bad in (candidate[..., :-1], candidate.copy()):
        if bad.shape == candidate.shape:
            bad[0, 0, 2] = np.nan
        record = certify(reference, bad)
        assert record['status'] == 'invalid_or_unsupported_input'
        json.dumps(record, allow_nan=False)
    candidate[0, 0, 2] += 1j*1e-8
    assert not certify(reference, candidate)['accepted']
    assert not certify(angle_values=[.1+0j])['accepted']
    assert not certify(angle_values=[1.])['accepted']
    record = certify_candidate(reference, reference, [1.], [1.], [0.], angle_values=[.1], direct_feature_error=0.)
    assert record['status'] == 'invalid_or_unsupported_input'


def test_gram_and_rms_certification_are_stable_under_small_unit_rescaling():
    reference, candidate, _ = fixture()
    one = certify(reference, candidate)
    small = certify(reference*1e-12, candidate*1e-12)
    assert one['accepted'] and small['accepted']
    assert abs(one['metrics']['harmonic_coadd_perturbation_ratio']-
               small['metrics']['harmonic_coadd_perturbation_ratio']) < 1e-15


def test_optional_weighted_observed_map_rms_is_separate_and_enforced():
    reference = np.ones((2, 2, 4))
    assert certify(reference_maps=reference, candidate_maps=reference*1.1)['metrics']['observed_map_rms_assessed']
    assert not certify(reference_maps=reference, candidate_maps=reference*1.4)['accepted']
    outside = reference.copy()
    outside[..., -1] = 10.
    record = certify(reference_maps=reference, candidate_maps=outside, weights=[1., 1., 1., 0.])
    assert record['accepted'] and record['metrics']['observed_coadd_perturbation_ratio'] == 0
    assert not certify(reference_maps=reference, weights=np.ones(4))['accepted']


def test_future_pair_needs_its_own_independent_evidence_and_declared_operator():
    options = dict(future_reference_features=[4.001, 2., 1.], future_candidate_features=[4.003, 2., 1.])
    assert not certify(**options)['counterexample_verified']
    assert not certify(**options, future_direct_feature_error=0.)['counterexample_verified']
    record = certify(**options, future_direct_feature_error=0., transition_id='fixed-T-v1', representation_id='R-v1')
    assert record['status'] == 'counterexample_verified' and record['counterexample_verified']
    assert record['future_pair']['safe_changed_coordinates'] == 1
    assert not record['universal_closure_verified']
    unsafe = certify(future_reference_features=[4.001, 2., 1.], future_candidate_features=[4.0015, 2., 1.],
                     future_direct_feature_error=0., transition_id='fixed-T-v1', representation_id='R-v1')
    assert unsafe['finite_feasibility_verified'] and not unsafe['counterexample_verified']


def test_completed_no_witness_search_never_turns_into_a_closure_pass():
    status = evidence_status('none_found_in_budget', numerical_status='passed')
    assert status['closure_status'] == 'inconclusive' and not status['universal_closure_verified']
    with pytest.raises(ValueError):
        evidence_status('none_found_in_budget', execution_status='incomplete')
    with pytest.raises(ValueError):
        evidence_status('counterexample_verified', numerical_status='failed')


def test_hash_identity_is_real_but_authentication_is_not_inferred():
    payload = b'actual saved bytes'
    digest = hashlib.sha256(payload).hexdigest()
    ref = provenance_reference('operator', payload, expected_sha256=digest,
        source_uri='artifact:operator-v1', trust_anchor={'kind': 'receipt', 'identifier': 'external-receipt'})
    assert ref['sha256'] == digest and ref['expected_hash_matches'] is True
    assert ref['anchor_status'] == 'caller_declared'
    assert ref['authentication_status'] == 'not_established_by_this_helper'
    bad = provenance_reference('operator', payload, expected_sha256='0'*64)
    assert not certify(provenance={'operator': bad})['accepted']
    with pytest.raises(ValueError):
        provenance_reference('operator', payload, trust_anchor={'verified': True})
    with pytest.raises(ValueError):
        provenance_reference('operator', payload, expected_sha256='short')


def strict_arguments():
    reference, candidate, _ = fixture()
    future_ref, future_can = [4.0012, 2., 1.], [4.0032, 2., 1.]
    return dict(reference_alms=reference, candidate_alms=candidate,
                reference_features=[4., 2., 1.], candidate_features=[4., 2., 1.], scales=[1., 1., 1.],
                angle_values=[.1], direct_feature_error=0.,
                replayed_reference_features=[4., 2., 1.], replayed_candidate_features=[4., 2., 1.],
                reference_maps=np.ones((2, 2, 4)), candidate_maps=np.full((2, 2, 4), 1.1), weights=np.ones(4),
                future_reference_features=future_ref, future_candidate_features=future_can,
                replayed_future_reference_features=future_ref, replayed_future_candidate_features=future_can,
                future_direct_feature_error=0., transition_id='strict-T-v1', representation_id='strict-R-v1')


@pytest.mark.parametrize('field', ['reference_features', 'candidate_features', 'scales', 'angle_values',
    'replayed_reference_features', 'replayed_candidate_features', 'reference_maps', 'candidate_maps', 'weights',
    'future_reference_features', 'future_candidate_features',
    'replayed_future_reference_features', 'replayed_future_candidate_features'])
@pytest.mark.parametrize('imaginary', [0., 1e-3])
def test_complex_real_quantity_arrays_are_rejected_before_promotion(field, imaginary):
    args = strict_arguments()
    args[field] = np.asarray(args[field], dtype=np.complex128)+imaginary*1j
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        record = certify_candidate(**args)
    assert record['status'] == 'invalid_or_unsupported_input'
    assert not record['accepted'] and not record['finite_feasibility_verified'] and not record['counterexample_verified']
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize('field', ['min_rms', 'max_rms', 'tolerance', 'angle_limit', 'quantum', 'bin_guard',
                                  'direct_feature_error', 'future_direct_feature_error'])
def test_complex_scalar_controls_and_replay_errors_are_rejected_without_casting(field):
    args = strict_arguments()
    defaults = dict(min_rms=.01, max_rms=.25, tolerance=1e-10, angle_limit=np.pi/4,
                    quantum=.001, bin_guard=1e-5, direct_feature_error=0., future_direct_feature_error=0.)
    args[field] = np.complex128(defaults[field]+0j)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        record = certify_candidate(**args)
    assert record['status'] == 'invalid_or_unsupported_input' and not record['accepted']
    json.dumps(record, allow_nan=False)


def test_object_array_cannot_hide_complex_scalars_but_harmonics_remain_complex():
    args = strict_arguments()
    good = certify_candidate(**args)
    assert good['counterexample_verified'] and good['accepted']
    assert np.iscomplexobj(args['candidate_alms'])
    for field in ('candidate_features', 'angle_values', 'weights', 'replayed_candidate_features'):
        bad = strict_arguments()
        values = np.asarray(bad[field], dtype=object)
        values.flat[0] = np.complex128(1.+1j)
        bad[field] = values
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            record = certify_candidate(**bad)
        assert record['status'] == 'invalid_or_unsupported_input' and not record['accepted']
