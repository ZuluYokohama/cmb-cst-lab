"""Synthetic certification contracts; never reconstruct or search a real sky."""
import json

import numpy as np
import pytest

from cmbcst import closure_certification_v2 as cert


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    calls, dual_calls = [], []
    failure = {'direct_angle': None, 'gpu_angle': None}

    def values(family, theta, u=0):
        angle = float(family.get('candidate_angle', np.asarray(theta)[0] if len(theta) else 0.))
        result = np.zeros(52)
        result[22] = u*angle
        if family.get('direct') and failure['direct_angle'] == angle:
            result[1] += 1e-4
        return result

    def ratios(family, theta):
        return np.array([.10, .02])

    def replay(root, tile, work, theta, shared=None, budget_callback=None):
        if budget_callback:
            budget_callback()
        angle = float(theta[0])
        calls.append(angle)
        return dict(direct_family=dict(candidate_angle=angle, direct=True),
                    candidate_alms=np.array([angle+0j, 0j]),
                    invariants={'passed': True}, perturbation_ratios=np.array([.10, .02]))

    def gpu(family, theta):
        angle = float(theta[0])
        dual_calls.append(angle)
        actual = np.stack([values(family, theta, u) for u in range(4)])
        other = actual.copy()
        if failure['gpu_angle'] == angle:
            other[3, 51] = .001
        return {'gpu0': actual, 'gpu1': other}

    monkeypatch.setattr(cert, 'oracle_features', values)
    monkeypatch.setattr(cert, 'features_numpy', values)
    monkeypatch.setattr(cert, 'ratios_numpy', ratios)
    monkeypatch.setattr(cert, '_time_invariants', lambda shared, replay: [{'passed': True} for _ in range(4)])
    family = {'fixture': np.array(7.)}
    shared = {'reference_alms': np.zeros(2, dtype=np.complex128)}
    case = dict(tile=24, stage='fresh', representation='baseline', device=0)

    def run(angles, budget=None, scales=None, mutate=None):
        angles = np.asarray(angles, dtype=np.float64)
        archive = dict(best_theta=angles[:, None], started=np.ones(len(angles), dtype=int),
                       gpu_features=np.zeros((len(angles), 52)), gpu_target=angles.copy(),
                       gpu_ratios=np.tile([.10, .02], (len(angles), 1)))
        if mutate:
            mutate(archive)
        return cert.certify_case_v2(tmp_path, tmp_path/'out', tmp_path/'work', case,
                                    family, shared, np.ones(52) if scales is None else scales,
                                    archive, budget, replay_function=replay, gpu_evaluator=gpu)

    return run, calls, dual_calls, failure, tmp_path


def test_every_started_endpoint_replayed_and_maximum_verified_below_old_threshold(sandbox):
    run, calls, dual, _, root = sandbox
    result = run([.009, .017, .027])
    assert result['status'] == 'complete'
    assert calls == [.009, .017, .027]
    assert dual == [.027]
    assert result['maximum_selected_start'] == 2
    assert result['tier_counts'] == {'0.010': 1, '0.015': 1, '0.025': 1, '0.050': 0}
    assert result['row_cpu_direct_raw_tier_counts'] == {'0.010': 2, '0.015': 2, '0.025': 1, '0.050': 0}
    assert result['reported_cell_maximum_no_factor_witness']
    assert result['rms_ratio_names'] == ['tile', 'global']
    assert (root/'out'/result['maximum']['coefficients']).exists()
    for row in result['rows']:
        assert len(row['invariant_checks']) == 4
        assert row['sampling_epsilon'] == 1e-12
        assert len(row['raw_k4_gaps']) == 4
        with np.load(root/'out'/row['arrays']) as arrays:
            assert arrays['direct_features'].shape == (4, 52)
            assert arrays['rms_ratio_names'].tolist() == ['tile', 'global']


def test_precision_failure_of_fixed_maximum_never_substitutes_runner_up(sandbox):
    run, calls, dual, failure, root = sandbox
    failure['direct_angle'] = .03
    result = run([.02, .03])
    assert calls == [.02, .03] and dual == [.03]
    assert result['maximum_selected_start'] == 1
    assert result['maximum']['status'] == 'precision_failure'
    assert result['maximum']['raw_primary_gap'] == .03
    assert not any(result['tier_counts'].values())
    assert result['rows'][0]['certified']
    assert result['outcome'] == 'fixed_maximum_failed_verification_no_substitution'
    assert (root/'out'/result['maximum']['coefficients']).exists()
    assert (root/'out'/result['maximum']['arrays']).exists()


def test_second_actual_gpu_and_all_times_are_required_for_cell_witness(sandbox):
    run, _, _, failure, _ = sandbox
    failure['gpu_angle'] = .03
    result = run([.02, .03])
    assert result['maximum']['actual_both_gpu_errors']['gpu1'] > cert.TOLERANCE
    assert result['maximum']['status'] == 'precision_failure'
    assert not result['reported_cell_maximum_no_factor_witness']
    assert result['tier_counts']['0.010'] == 0


def test_future_quantized_separation_requires_safe_changed_coordinate(sandbox):
    run, _, _, _, _ = sandbox
    result = run([.0105])
    assert result['maximum']['certified']
    assert result['maximum']['tiers']['0.010']
    assert not result['maximum']['safe_primary_future_bin_difference']
    assert result['tier_counts']['0.010'] == 0
    assert not result['reported_cell_maximum_no_factor_witness']


def test_raw_thresholds_do_not_scale_with_robust_bin_width():
    assert cert.raw_tiers(.025, True) == {'0.010': True, '0.015': True, '0.025': True, '0.050': False}
    assert not any(cert.raw_tiers(.001, True).values())
    assert not any(cert.raw_tiers(.06, False).values())
    value = np.zeros(52)
    value[0] = .000489
    assert cert.input_bins(value, np.zeros(52), np.ones(52), 'baseline')['matched']
    value[0] = .000491
    assert not cert.input_bins(value, np.zeros(52), np.ones(52), 'baseline')['matched']


def test_budget_interruption_is_explicit_and_resume_keeps_completed_rows(sandbox):
    run, calls, _, _, root = sandbox
    row = root/'out/certificates_v2/tile_24_fresh_baseline/rows/start_0000.json'

    def budget():
        if row.exists():
            raise TimeoutError('synthetic cap')

    first = run([.02, .03], budget)
    assert first['status'] == 'incomplete'
    assert first['starts_checked'] == 1
    assert first['unverified_start_ids'] == [1]
    digest = cert.sha(row)
    second = run([.02, .03])
    assert second['status'] == 'complete'
    assert calls == [.02, .03]
    assert cert.sha(row) == digest
    assert second['starts_checked'] == 2


def test_budget_after_direct_sampling_does_not_promote_an_unfinished_row(sandbox):
    run, calls, _, _, root = sandbox

    def budget():
        if calls:
            raise TimeoutError('cap after sampling')

    first = run([.03], budget)
    assert first['status'] == 'incomplete'
    assert first['starts_checked'] == 0
    assert first['unverified_start_ids'] == [0]
    assert not first['reported_cell_maximum_no_factor_witness']
    path = root/'out/certificates_v2/tile_24_fresh_baseline/rows/start_0000.json'
    assert json.loads(path.read_text())['completed'] is False
    second = run([.03])
    assert second['status'] == 'complete' and second['starts_checked'] == 1


def test_infeasible_rows_are_replayed_without_becoming_tier_witnesses(sandbox):
    run, calls, dual, _, _ = sandbox
    result = run([.02, .8])
    assert calls == [.02, .8]
    # An out-of-family angle is still submitted to replay; production's
    # unchanged replay rejects it and preserves a row failure. The synthetic
    # replay permits it, so the independent angle gate remains essential.
    assert dual == [.02]
    assert result['maximum_selected_start'] == 0
    assert result['rows'][1]['status'] == 'ineligible_endpoint'
    assert not result['rows'][1]['certified']


def test_resume_refuses_changed_inputs_and_corrupt_row_evidence(sandbox):
    run, _, _, _, root = sandbox
    result = run([.02, .03])
    with pytest.raises(ValueError, match='binding changed'):
        run([.02, .031])
    path = root/'out'/result['rows'][0]['arrays']
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='arrays changed'):
        run([.02, .03])


def test_archived_gpu_failure_preserves_selected_maximum(sandbox):
    run, _, _, _, _ = sandbox

    def mutate(archive):
        archive['gpu_target'][1] += .001

    result = run([.02, .03], mutate=mutate)
    assert result['maximum_selected_start'] == 1
    assert result['maximum']['errors']['archived_gpu_future_raw_k4'] > cert.TOLERANCE
    assert result['maximum']['status'] == 'precision_failure'
    assert not any(result['tier_counts'].values())
