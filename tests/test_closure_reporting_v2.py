"""Interrupted checkpoint-pair recovery must preserve global start identity."""
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    return (importlib.import_module('report_closure_fibers_v2'),
            importlib.import_module('verify_closure_fibers_v2'))


def fixture(tmp_path, *, first=64, phase='periodic', equal_iterations=False):
    name = 'tile_34_fresh_baseline'
    (tmp_path/'cases').mkdir()
    (tmp_path/'checkpoints').mkdir()
    saved = dict(started=np.r_[np.ones(64, dtype=int), np.zeros(64, dtype=int)],
                 theta=np.full((128, 4), .1), best_theta=np.full((128, 4), .1),
                 iterations=np.r_[np.full(64, 7 if equal_iterations else 1000), np.zeros(64, dtype=int)],
                 status=np.array(['iteration_cap']*64+['not_started']*64),
                 gpu_features=np.ones((128, 52)), gpu_target=np.ones(128),
                 gpu_ratios=np.ones((128, 2)), gpu_all_features=np.ones((128, 4, 52)))
    np.savez_compressed(tmp_path/'cases'/f'{name}.npz', **saved)
    checkpoint = dict(theta=np.full((64, 4), .2), iterations=np.full(64, 7),
                      status=np.full(64, 'active'), checkpoint_start_ids=np.arange(first, first+64),
                      checkpoint_first_start=np.array(first), checkpoint_phase=np.array(phase))
    path = tmp_path/'checkpoints'/f'{name}.npz'
    np.savez_compressed(path, **checkpoint)
    return name, saved, checkpoint, path


def audit(modules, tmp_path, name):
    report, verifier = modules
    arrays, saved, checkpoint, metadata, overlay, info = report._arrays(tmp_path, name, 128, 4)
    entry = dict(name=name, arrays=arrays, checkpoint=checkpoint, checkpoint_meta=metadata,
                 checkpoint_overlay=overlay, checkpoint_info=info)
    checks = verifier.Checks()
    verifier.validate_checkpoint(entry, checks)
    return arrays, info, overlay, checks


def test_second_gtx_batch_uses_npz_ids_when_json_still_describes_first_batch(modules, tmp_path):
    name, saved, _, path = fixture(tmp_path)
    path.with_suffix('.json').write_text(json.dumps(dict(first_start=0, phase='initialized', checkpoint_sha256='0'*64)))
    arrays, info, overlay, checks = audit(modules, tmp_path, name)
    assert overlay == list(range(64, 128))
    assert np.array_equal(arrays['theta'][:64], saved['theta'][:64])
    assert np.array_equal(arrays['status'][:64], saved['status'][:64])
    assert np.array_equal(arrays['gpu_features'][:64], saved['gpu_features'][:64])
    assert np.all(arrays['theta'][64:] == .2)
    assert np.isnan(arrays['gpu_features'][64:]).all()
    assert int(np.sum(arrays['started'])) == 128
    assert info['id_source'] == info['phase_source'] == 'npz'
    assert info['phase'] == 'periodic'
    assert info['pair_recovered'] and not info['contradictory_json']
    assert {'json_npz_hash_mismatch', 'json_batch_mismatch', 'json_phase_mismatch'} <= set(info['issues'])
    assert all(checks.values.values()), checks.values


def test_missing_companion_json_recovers_second_batch_and_phase_from_npz(modules, tmp_path):
    name, _, _, _ = fixture(tmp_path, phase='interrupted')
    arrays, info, overlay, checks = audit(modules, tmp_path, name)
    assert overlay == list(range(64, 128))
    assert info['start_ids'] == list(range(64, 128))
    assert info['phase'] == 'interrupted' and info['phase_valid']
    assert info['issues'] == ['json_missing']
    assert info['pair_recovered']
    assert all(checks.values.values()), checks.values


def test_unreadable_companion_json_is_explicitly_recovered(modules, tmp_path):
    name, _, _, path = fixture(tmp_path)
    path.with_suffix('.json').write_text('{an interrupted JSON document')
    _, info, overlay, checks = audit(modules, tmp_path, name)
    assert overlay == list(range(64, 128))
    assert 'json_unreadable' in info['issues'] and info['json_read_error']
    assert info['pair_recovered'] and all(checks.values.values())


def test_matching_companion_hash_and_identity_need_no_recovery(modules, tmp_path):
    name, _, _, path = fixture(tmp_path)
    path.with_suffix('.json').write_text(json.dumps(dict(first_start=64, phase='periodic',
        checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
    _, info, _, checks = audit(modules, tmp_path, name)
    assert info['json_matches_npz'] and not info['pair_recovered'] and not info['issues']
    assert all(checks.values.values())


def test_invalid_npz_ids_are_not_replaced_with_plausible_json_range(modules, tmp_path):
    name, saved, checkpoint, path = fixture(tmp_path)
    checkpoint['checkpoint_start_ids'][0] = 65
    np.savez_compressed(path, **checkpoint)
    path.with_suffix('.json').write_text(json.dumps(dict(first_start=0, phase='periodic', checkpoint_sha256='old')))
    arrays, info, overlay, checks = audit(modules, tmp_path, name)
    assert not info['valid'] and not overlay
    assert np.array_equal(arrays['theta'], saved['theta'])
    assert not checks.values[name+':checkpoint:self_identifiers']


def test_hash_bound_but_contradictory_json_is_an_actual_inconsistency(modules, tmp_path):
    name, saved, _, path = fixture(tmp_path)
    path.with_suffix('.json').write_text(json.dumps(dict(first_start=0, phase='periodic',
        checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
    arrays, info, overlay, checks = audit(modules, tmp_path, name)
    assert info['contradictory_json'] and not info['valid'] and not info['pair_recovered']
    assert not overlay and np.array_equal(arrays['theta'], saved['theta'])
    assert not checks.values[name+':checkpoint:companion_consistent_or_recovered']


def test_equal_iteration_new_angle_invalidates_stale_gpu_measurements(modules, tmp_path):
    name, _, _, _ = fixture(tmp_path, first=0, equal_iterations=True)
    arrays, _, overlay, checks = audit(modules, tmp_path, name)
    assert overlay == list(range(64))
    assert np.all(arrays['theta'][:64] == .2)
    assert np.isnan(arrays['gpu_features'][:64]).all()
    assert np.isfinite(arrays['gpu_features'][64:]).all()
    assert all(checks.values.values())
