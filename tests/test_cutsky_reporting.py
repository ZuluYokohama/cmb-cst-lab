"""Saved-metadata counting and status controls; no scientific backend is used."""
import importlib
import json
import hashlib
from pathlib import Path

import pytest
import numpy as np


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    return (importlib.import_module('report_cutsky_feasibility'),
            importlib.import_module('verify_cutsky_feasibility'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def plan(tmp_path, status='complete'):
    value = dict(fractions=[.1, .3, .6], apodization_degrees=[8., 12.],
                 tolerances=[1e-10, 1e-12], starts_per_case=4,
                 planned_profiles=6, planned_cases=12, planned_starts=48)
    write(tmp_path/'plan.json', value)
    write(tmp_path/'stage_audit.json', dict(status=status, elapsed_seconds=10.,
        utc_elapsed_seconds=10., budget_elapsed_seconds=10.))
    return value


def row_file(tmp_path, case, start, status, *, metrics=None):
    row = dict(case=case['name'], profile=case['profile'], tolerance=case['tolerance'],
               start_id=start, global_start_id=case['profile_index']*4+start, status=status)
    if status != 'in_progress':
        row['solver'] = dict(reason='finite_search_exhausted', rms_ratio=.0, max_scaled_residual=0.)
    if metrics is not None:
        row['certificate'] = dict(metrics=metrics)
    write(tmp_path/'cases'/case['name']/f'start_{start:04d}.json', row)


def test_manifest_counts_all_twelve_cases_with_paired_direction_identities(modules, tmp_path):
    report, _ = modules
    p = plan(tmp_path)
    cases = report.manifest(p)
    assert len(cases) == 12 and len({c['profile'] for c in cases}) == 6
    for index in range(6):
        pair = [c for c in cases if c['profile_index'] == index]
        assert len(pair) == 2 and {c['tolerance'] for c in pair} == {1e-10, 1e-12}


def test_completed_negative_search_is_inconclusive_not_a_closure_pass(modules, tmp_path):
    report, _ = modules
    p = plan(tmp_path)
    for case in report.manifest(p):
        for start in range(4):
            row_file(tmp_path, case, start, 'no_nontrivial_found')
        write(tmp_path/'cases'/case['name']/'case.json', dict(name=case['name']))
    summary = report.collect_run(tmp_path)['summary']
    assert summary['starts_started'] == summary['starts_terminal'] == summary['no_nontrivial_found'] == 48
    assert summary['verified_candidates'] == 0 and summary['closed_cases'] == 12
    assert summary['scientific_outcome'] == 'none_found_in_budget'
    assert not summary['universal_closure_verified'] and summary['closure_counterexamples'] == 0
    assert all(x['started'] == x['no_nontrivial_found'] == 24 for x in summary['tolerance_outcomes'].values())


def test_durable_started_row_survives_hard_cap_without_result_or_solver(modules, tmp_path):
    report, _ = modules
    p = plan(tmp_path, 'incomplete_hard_cap')
    case = report.manifest(p)[7]
    row_file(tmp_path, case, 2, 'in_progress')
    summary = report.collect_run(tmp_path)['summary']
    assert summary['starts_started'] == summary['incomplete_started'] == 1
    assert summary['starts_terminal'] == summary['verified_candidates'] == 0
    assert summary['unstarted'] == 47
    assert summary['scientific_outcome'] == 'incomplete_or_failed'


def test_preserved_verification_failure_is_not_counted_as_qualified(modules, tmp_path):
    report, _ = modules
    p = plan(tmp_path, 'complete_with_failures')
    row_file(tmp_path, report.manifest(p)[0], 0, 'verification_failed')
    summary = report.collect_run(tmp_path)['summary']
    assert summary['numerical_or_verification_failures'] == 1 and summary['verified_candidates'] == 0
    assert summary['scientific_outcome'] == 'incomplete_or_failed'


def test_verified_finite_candidate_does_not_create_future_or_physics_claim(modules, tmp_path):
    report, _ = modules
    p = plan(tmp_path, 'incomplete')
    metrics = dict(harmonic_coadd_perturbation_ratio=.012, realized_feature_lock_error=1e-13,
                   normalized_full_gram_error=1e-15, full_split_difference_scaled_error=1e-16)
    row_file(tmp_path, report.manifest(p)[0], 0, report.VERIFIED, metrics=metrics)
    summary = report.collect_run(tmp_path)['summary']
    assert summary['verified_candidates'] == 1 and summary['minimum_verified_harmonic_rms'] == .012
    assert summary['scientific_outcome'] == 'finite_feasibility_demonstrated_for_reported_profiles'
    assert summary['closure_counterexamples'] == 0 and not summary['physical_noise_model_validated']
    assert summary['unstarted'] == 47


def test_flat_root_input_manifest_uses_one_inputs_prefix_and_actual_bytes(modules, tmp_path):
    report, verifier = modules
    path = tmp_path/'inputs/splits.fits'
    path.parent.mkdir()
    path.write_bytes(b'fake fixture bytes, not a scientific FITS file')
    expected = {'inputs/splits.fits': report.sha(path)}
    write(tmp_path/'inputs/manifest.json', expected)
    assert verifier.input_hashes(tmp_path) == expected
    assert verifier.input_path(tmp_path, 'inputs/splits.fits') == path
    assert verifier.input_path(tmp_path, 'splits.fits') == path


def test_tiny_recorded_error_arithmetic_is_not_hidden_by_the_scientific_gate(modules):
    _, verifier = modules
    assert verifier.same_scalar(1.2e-14, 1.2e-14)
    assert not verifier.same_scalar(0., 1.2e-14)
    assert not verifier.same_scalar(None, 1.2e-14)
    assert not verifier.same_scalar(1.2e-14, None)


def test_array_hash_formats_are_field_specific_and_models_is_endian_canonical(modules):
    report, _ = modules
    # Fixed expected bytes, independent of either production helper.
    value = np.array([[1., 2.], [3., 4.]], dtype='<f8')
    payload = value.tobytes(order='C')
    model_header = b'{"dtype":"<f8","shape":[2,2]}'
    model_expected = hashlib.sha256(b'cmbcst.array.v1\0'+model_header+b'\0'+payload).hexdigest()
    legacy_expected = hashlib.sha256(b'<f8(2, 2)'+payload).hexdigest()
    assert report.models_array_sha(value) == model_expected
    assert report.array_sha(value) == legacy_expected
    assert model_expected != legacy_expected
    assert report.models_array_sha(value.astype('>f8')) == model_expected
    assert report.models_array_sha(value[:, ::-1][:, ::-1]) == model_expected
    assert report.models_array_sha(value.reshape(4)) != model_expected
    with pytest.raises(ValueError):
        report.models_array_sha(np.array([object()]))


def test_rank_amplitude_ranges_and_direction_pairing_use_saved_metadata_only(modules, tmp_path):
    report, _ = modules
    p = plan(tmp_path)
    cases = report.manifest(p)
    for case in cases[:2]:
        for start in range(2):
            metrics = dict(harmonic_coadd_perturbation_ratio=.0101+start*.001,
                observed_coadd_perturbation_ratio=.008+start*.002)
            row_file(tmp_path, case, start, report.VERIFIED, metrics=metrics)
            path = tmp_path/'cases'/case['name']/f'start_{start:04d}.json'
            row = json.loads(path.read_text())
            row['solver']['rank_history'] = [28, 28, 28]
            write(path, row)
    summary = report.collect_run(tmp_path)['summary']
    assert summary['planned_direction_streams'] == 24
    assert summary['starts_started'] == 4 and summary['direction_streams_started'] == 2
    assert summary['verified_tolerance_pairs'] == 2
    assert summary['minimum_solver_rank'] == summary['maximum_solver_rank'] == 28
    assert summary['minimum_paired_harmonic_rms_ratio'] == summary['maximum_paired_harmonic_rms_ratio'] == 1.
    assert summary['minimum_verified_weighted_observed_rms'] == .008
    assert summary['maximum_verified_weighted_observed_rms'] == .01
    assert summary['minimum_verified_harmonic_rms'] == .0101
    assert summary['maximum_verified_harmonic_rms'] == pytest.approx(.0111)
