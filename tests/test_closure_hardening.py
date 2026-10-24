"""Synthetic checks for thread provenance, stalled compilers and interrupted archives."""
from contextlib import nullcontext
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from cmbcst import closure_certification as cert


ROOT = Path(__file__).resolve().parents[1]
BLAS_VARIABLES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')


@pytest.fixture
def scripts(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    # The runner sets process environment variables on import; restore them after tests.
    for name in BLAS_VARIABLES:
        monkeypatch.setenv(name, os.environ.get(name, '7'))
    return SimpleNamespace(run=importlib.import_module('run_closure_fibers'),
                           verify=importlib.import_module('verify_closure_fibers'),
                           lean=importlib.import_module('verify_closure_lean'))


def test_blas_threads_override_inherited_values_before_numpy_import():
    code = '''
import builtins, os, runpy
original = builtins.__import__
def checked(name, *args, **kwargs):
    if name == 'numpy':
        assert all(os.environ[key] == '1' for key in
                   ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'))
    return original(name, *args, **kwargs)
builtins.__import__ = checked
runpy.run_path('scripts/run_closure_fibers.py')
'''
    env = dict(os.environ, **dict.fromkeys(BLAS_VARIABLES, '7'))
    subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env, check=True, timeout=30)


def test_freeze_records_blas_settings_and_keeps_eight_torch_threads(scripts, monkeypatch, tmp_path):
    run = scripts.run
    threads = []
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        set_num_threads=threads.append, __version__='synthetic',
        cuda=SimpleNamespace(get_device_name=lambda i: f'synthetic:{i}'),
        version=SimpleNamespace(cuda='synthetic')))
    evidence = tmp_path/'data/evidence'
    evidence.mkdir(parents=True)
    profile = evidence/'preflight.json'
    profile.write_text(json.dumps(dict(status='passed', source_sha256={})))
    (evidence/'closure_lean.json').write_text(json.dumps(dict(status='passed', source_sha256='digest')))
    jovin = importlib.import_module('jovin_verify')
    monkeypatch.setattr(jovin, 'build_bundle', lambda name: ('synthetic', {}))
    (evidence/'jovin_receipts.json').write_text(json.dumps([dict(
        suite='closure_fibers', passed=True, receipt_verified=True, chain_valid=True,
        bundle_sha256=hashlib.sha256(b'synthetic').hexdigest(), tests_sha256='digest', receipt_id='synthetic')]))
    for name, value in dict(ROOT=tmp_path, OUT=tmp_path/'out', WORK=tmp_path/'work', PROFILE=profile).items():
        monkeypatch.setattr(run, name, value)
    monkeypatch.setattr(run, 'sha', lambda path: 'digest')
    monkeypatch.setattr(run, 'input_hashes', lambda root: {})
    monkeypatch.setattr(run.subprocess, 'check_output', lambda *args, **kwargs: 'synthetic-revision')
    for name in BLAS_VARIABLES:
        monkeypatch.setenv(name, '1')
    freeze = run.freeze

    class Frozen(Exception):
        pass

    def freeze_and_stop():
        freeze()
        raise Frozen

    monkeypatch.setattr(run, 'freeze', freeze_and_stop)
    with pytest.raises(Frozen):
        run.main()
    plan = json.loads((run.OUT/'plan.json').read_text())
    assert plan['blas_threads'] == dict.fromkeys(BLAS_VARIABLES, 1)
    assert plan['threads'] == 8 and threads == [8]


@pytest.mark.parametrize('bad_name', [None, *BLAS_VARIABLES, 'all_missing'])
@pytest.mark.parametrize('bad_value', [None, 2])
def test_verifier_requires_each_recorded_blas_setting(scripts, monkeypatch, bad_name, bad_value):
    verify = scripts.verify
    plan = json.loads((ROOT/'results/closure_fibers_v1/plan.json').read_text())
    plan['blas_threads'] = dict.fromkeys(BLAS_VARIABLES, 1)
    if bad_name == 'all_missing':
        del plan['blas_threads']
    elif bad_name:
        if bad_value is None:
            del plan['blas_threads'][bad_name]
        else:
            plan['blas_threads'][bad_name] = bad_value
    monkeypatch.setattr(sys, 'argv', ['verify_closure_fibers.py'])
    monkeypatch.setattr(verify, 'collect_run', lambda out: dict(plan=plan, stage={}, summary={}))

    class Checked(Exception):
        pass

    def checked(self, name, value):
        assert name == 'frozen_protocol'
        assert bool(value) == (bad_name is None)
        raise Checked

    monkeypatch.setattr(verify.Checks, 'add', checked)
    with pytest.raises(Checked):
        verify.main()


@pytest.mark.parametrize('stall', ['version', 'proof', None])
def test_lean_calls_are_bounded_and_timeout_never_writes_evidence(scripts, monkeypatch, tmp_path, stall):
    lean = scripts.lean
    output = tmp_path/'proof.json'
    monkeypatch.setattr(sys, 'argv', ['verify_closure_lean.py', '--lean', 'synthetic-lean',
                                     '--archive', 'synthetic-archive', '--output', str(output)])
    monkeypatch.setattr(lean, 'sha', lambda path: (
        lean.PINNED_ARCHIVE_SHA256 if str(path) == 'synthetic-archive' else lean.PINNED_COMPILER_SHA256))
    calls = []

    def version(command, *, text, timeout):
        assert timeout == 120 and text
        calls.append('version')
        if stall == 'version':
            raise subprocess.TimeoutExpired(command, timeout)
        return f'Lean version 4.35.0 {lean.PINNED_COMMIT}\n'

    def proof(command, **kwargs):
        assert kwargs['timeout'] == 120
        calls.append('proof')
        if stall == 'proof':
            raise subprocess.TimeoutExpired(command, kwargs['timeout'])
        stdout = '\n'.join(f"'CmbCst.{name}' does not depend on any axioms" for name in lean.THEOREMS)
        return SimpleNamespace(returncode=0, stdout=stdout, stderr='')

    monkeypatch.setattr(lean.subprocess, 'check_output', version)
    monkeypatch.setattr(lean.subprocess, 'run', proof)
    if stall:
        with pytest.raises(subprocess.TimeoutExpired):
            lean.main()
        assert not output.exists()
    else:
        lean.main()
        assert json.loads(output.read_text())['status'] == 'passed'
    assert calls == (['version'] if stall == 'version' else ['version', 'proof'])


@pytest.mark.parametrize('suffix', ['_endpoints.npz', '.npz', '_candidate_alms.npz'])
@pytest.mark.parametrize('existing', [False, True])
def test_certification_archive_interruption_preserves_final_path(monkeypatch, tmp_path, suffix, existing):
    def values(family, theta, u=0):
        result = np.zeros((*np.shape(theta)[:-1], 52))
        result[..., 22] = u*.1 if family.get('direct') or np.any(theta) else 0.
        return result

    monkeypatch.setattr(cert, 'features_numpy', values)
    monkeypatch.setattr(cert, 'oracle_features', values)
    monkeypatch.setattr(cert, 'ratios_numpy', lambda family, theta: np.full((*np.shape(theta)[:-1], 2), .1))
    monkeypatch.setattr(cert, 'replay_candidate', lambda *args: dict(
        direct_family={'direct': True}, candidate_alms=np.zeros(1, dtype=complex), perturbation_ratios=np.full(2, .1)))
    monkeypatch.setattr(cert, 'harmonic_layout', lambda: (np.zeros(1), np.zeros(1)))
    monkeypatch.setattr(cert, 'invariants', lambda *args: {'passed': True})
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        cuda=SimpleNamespace(device=lambda index: nullcontext()), no_grad=nullcontext,
        as_tensor=lambda value, **kwargs: value, float64=np.float64))
    fibers = importlib.import_module('cmbcst.closure_fibers')
    monkeypatch.setattr(fibers, 'ClosureCUDA', lambda family, device: SimpleNamespace(
        features=lambda theta, u: SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: values(family, theta, u)))))
    name = 'tile_24_fresh_baseline'
    directory = tmp_path/('certificates' if suffix == '_endpoints.npz' else 'witnesses')
    directory.mkdir()
    final = directory/(name+suffix)
    if existing:
        np.savez_compressed(final, original=np.array([42]))
    previous = final.read_bytes() if existing else None
    original = np.savez_compressed
    completed = []

    def interrupted(stream, **arrays):
        if Path(stream.name).name == final.name+'.tmp':
            stream.write(b'partial compressed archive')
            raise OSError('synthetic interrupted write')
        original(stream, **arrays)
        completed.append(Path(stream.name))

    monkeypatch.setattr(cert.np, 'savez_compressed', interrupted)
    case = dict(tile=24, stage='fresh', representation='baseline')
    archive = dict(best_theta=np.array([[.1]]), initial_theta=np.zeros((1, 1)), started=np.ones(1),
                   gpu_features=np.zeros((1, 52)), gpu_target=np.array([.1]), gpu_ratios=np.full((1, 2), .1))
    args = (tmp_path, tmp_path, tmp_path, case, {}, {'reference_alms': np.zeros(1)}, np.ones(52), archive)
    with pytest.raises(OSError, match='synthetic interrupted write'):
        cert.certify_case(*args)
    assert final.exists() == existing
    if existing:
        assert final.read_bytes() == previous
    for temporary in completed:
        assert not temporary.exists()
        with np.load(temporary.with_suffix('')) as saved:
            assert saved.files
    monkeypatch.setattr(cert.np, 'savez_compressed', original)
    result = cert.certify_case(*args)
    assert result['status'] == 'verified_witness'
    with np.load(final) as saved:
        assert 'original' not in saved.files
    assert not final.with_name(final.name+'.tmp').exists()
