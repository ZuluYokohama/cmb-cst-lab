"""Regression checks for fixture admission and real-split evidence validation."""
import csv
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import verify_real_splits as verify


@pytest.fixture
def preparation(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        cuda=SimpleNamespace(get_device_name=lambda index: f'test device {index}'),
        version=SimpleNamespace(cuda='test'), __version__='test'))
    spec = importlib.util.spec_from_file_location('prepare_real_splits', ROOT / 'scripts/prepare_real_splits.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = tmp_path
    test = tmp_path / 'tests/sandbox/jovin_tests_real_splits.py'
    test.parent.mkdir(parents=True)
    test.write_text('GPU_ATTESTATION = None\n')
    evidence = tmp_path / 'data/evidence'
    evidence.mkdir(parents=True)
    for name in ('src/cmbcst/real_splits.py', 'scripts/prepare_real_splits.py'):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    return module, test, evidence / 'real_splits_cuda_v1.json'


@pytest.mark.parametrize('metric', ['cross_statistics', 'bootstrap_tiles'])
@pytest.mark.parametrize('error', [np.nan, np.inf, -np.inf, 2e-10])
def test_fixture_rejects_invalid_precision_before_writing(preparation, monkeypatch, metric, error):
    module, test, evidence = preparation
    for name in ('cross_statistics', 'bootstrap_tiles'):
        value = error if name == metric else 0.
        monkeypatch.setattr(module, name, lambda *args, value=value: np.array([value if len(args) == 3 else 0.]))
    with pytest.raises(ValueError, match='actual GPU fixture precision failure'):
        module.main()
    assert test.read_text() == 'GPU_ATTESTATION = None\n'
    assert not evidence.exists()


@pytest.mark.parametrize('error', [0., 5e-11, 1e-10])
def test_fixture_accepts_finite_errors_through_tolerance(preparation, monkeypatch, error):
    module, test, evidence = preparation
    for name in ('cross_statistics', 'bootstrap_tiles'):
        monkeypatch.setattr(module, name, lambda *args: np.array([error if len(args) == 3 else 0.]))
    module.main()
    fixture = verify.read(evidence)['fixture']
    assert len(fixture['devices']) == 2
    assert all(d['maximum_scaled_error'] == error for d in fixture['devices'])
    assert 'GPU_ATTESTATION = None' not in test.read_text()


def test_input_identity_rehashes_every_current_map(tmp_path, monkeypatch):
    monkeypatch.setattr(verify, 'ROOT', tmp_path)
    raw = tmp_path / 'data/raw'
    raw.mkdir(parents=True)
    keys = ['100A', '100B']
    inventory = {}
    maps = []
    for key in keys:
        payload = key.encode()
        (raw / f'{key}.fits').write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        inventory[key] = dict(filename=f'{key}.fits', sha256=digest, bytes=len(payload))
        maps.append(dict(key=key, sha256=digest, bytes=len(payload), zero_filled_invalid_pixels=0))
    inputs, plan = dict(maps=maps), dict(maps=keys)
    assert verify.input_identities_match(inputs, plan, inventory)
    for key in keys:
        path = raw / inventory[key]['filename']
        original = path.read_bytes()
        path.write_bytes(b'bad!')  # Same length; recorded hash and byte count remain unchanged.
        assert not verify.input_identities_match(inputs, plan, inventory)
        path.write_bytes(original)
    for field, value in [('sha256', 'wrong'), ('bytes', 0), ('zero_filled_invalid_pixels', -1)]:
        original = maps[0][field]
        maps[0][field] = value
        assert not verify.input_identities_match(inputs, plan, inventory)
        maps[0][field] = original
    assert not verify.input_identities_match(dict(maps=maps[::-1]), plan, inventory)


@pytest.fixture(scope='module')
def saved_samples():
    out = ROOT / 'results/real_splits_v1'
    with (out / 'labeled_real_samples.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    plan = verify.read(out / 'plan.json')
    with np.load(out / 'geometry.npz', allow_pickle=False) as archive:
        g = {key: archive[key] for key in ('loc', 'anchor_pixel', 'valid', 'phase0', 'phase1')}
    with np.load(out / 'observations.npz', allow_pickle=False) as archive:
        anchor, endpoint = archive['anchor'], archive['endpoint']
    views = (endpoint * g['phase1'] - anchor[:, None, None] * g['phase0'])[None]
    return rows, plan, g, anchor, endpoint, views


def test_saved_labeled_rows_pass(saved_samples):
    assert verify.labeled_rows_match(*saved_samples)


@pytest.mark.parametrize('bearing', ['0.1', '44.99', '45.1', '-1', '180', 'nan', 'inf', '-inf'])
def test_labeled_rows_reject_inexact_or_invalid_bearings(saved_samples, bearing):
    rows, *context = saved_samples
    rows = [dict(row) for row in rows]
    rows[0]['bearing_deg'] = bearing
    assert not verify.labeled_rows_match(rows, *context)


@pytest.mark.parametrize('duplicate', ['sample_index', 'map_direction'])
def test_labeled_rows_reject_duplicate_labels(saved_samples, duplicate):
    rows, *context = saved_samples
    rows = [dict(row) for row in rows]
    if duplicate == 'sample_index':
        rows[1]['sample_index'] = '00'
    else:
        rows[1] = dict(rows[0], sample_index=rows[1]['sample_index'])
    assert not verify.labeled_rows_match(rows, *context)


@pytest.mark.parametrize('field,value', [
    ('Q_end_uK', '999'), ('galactic_latitude_deg', '0'),
    ('galactic_longitude_deg', '0'), ('lag_deg', '1'), ('valid_pair', 'False'),
])
def test_labeled_rows_preserve_existing_value_checks(saved_samples, field, value):
    rows, *context = saved_samples
    rows = [dict(row) for row in rows]
    rows[0][field] = value
    assert not verify.labeled_rows_match(rows, *context)


def test_labeled_rows_preserve_row_count_check(saved_samples):
    rows, *context = saved_samples
    assert not verify.labeled_rows_match(rows[:-1], *context)
