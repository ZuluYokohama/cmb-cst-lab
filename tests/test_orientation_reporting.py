"""Orientation evidence must not turn failed checks or commands into passes."""
import hashlib
import importlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    return (importlib.import_module('report_orientation'),
            importlib.import_module('verify_orientation_evidence'))


@pytest.mark.parametrize('status,failures,expected_status', [
    ('passed', 0, 'passed'),
    ('failed', 2, 'failed'),
    ('failed', 0, 'failed'),
    ('passed', 2, 'failed'),
])
def test_report_and_figure_use_actual_pass_count(modules, tmp_path, monkeypatch,
                                                status, failures, expected_status):
    report, _ = modules
    out = tmp_path/'results/orientation_v1'
    out.mkdir(parents=True)
    (tmp_path/'docs').mkdir()
    source = tmp_path/'data/work/spherical_topology_v1/observed_patches.npz'
    source.parent.mkdir(parents=True)
    np.savez(source, NGP=np.arange(16.).reshape(4, 4))
    (out/'plan.json').write_text(json.dumps({
        'patch_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'jovin_receipt_id':'synthetic-receipt',
    }))
    (out/'result.json').write_text(json.dumps({
        'status':status, 'failures':failures, 'transform_checks':312,
        'maximum_scaled_errors':{'cpu_gpu_curves':0.}, 'seconds':1.,
    }))
    monkeypatch.setattr(report, 'ROOT', tmp_path)
    monkeypatch.setattr(report, 'OUT', out)
    figure_text = []
    monkeypatch.setattr(report.plt.Figure, 'savefig',
        lambda fig, *args, **kwargs: figure_text.extend(t.get_text() for t in fig.texts))

    report.main()

    text = (tmp_path/'docs/ORIENTATION_MERIT_RUN.md').read_text()
    count = f'{312-failures}/312'
    assert f'{count} exact-grid checks passed; {failures} failures. Audit {expected_status}.' in text
    assert any(f'{count} checks passed · {failures} failures · audit {expected_status}' in t
               for t in figure_text)
    if expected_status == 'failed':
        assert 'The audit did not pass.' in text
        assert 'Every CPU/CUDA comparison was finite and below' not in text
    else:
        assert 'Every CPU/CUDA comparison was finite and below' in text


@pytest.mark.parametrize('summary,exit_code,expected', [
    ('182 passed, 3 skipped in 1.23s\n', 0, True),
    ('=== 182 passed, 3 skipped, 1 warning in 1.23s ===\n', 0, True),
    ('182 passed, 3 skipped in 65.12s (0:01:05)\n', 0, True),
    ('182 passed, 3 skipped in 1.23s\n', 1, False),
    ('182 passed, 3 skipped in 1.23s\n1 failed in 2.34s\n', 0, False),
    ('1 failed, 182 passed, 3 skipped in 1.23s\n', 0, False),
    ('182 passed, 3 skipped, 1 error in 1.23s\n', 0, False),
    ('183 passed, 3 skipped in 1.23s\n', 0, False),
    ('182 passed, 2 skipped in 1.23s\n', 0, False),
    ('example: 182 passed, 3 skipped\n', 0, False),
    ('', 0, False),
])
def test_test_evidence_requires_successful_exit_and_final_summary(modules, summary, exit_code, expected):
    _, verifier = modules
    tests_run = subprocess.CompletedProcess(['pytest'], exit_code, stdout=summary)
    sandbox_run = subprocess.CompletedProcess(['sandbox'], 0, stdout='[PASS] suite\n'*19)
    assert verifier.test_command_checks(tests_run, sandbox_run) == (expected, True)


@pytest.mark.parametrize('output,exit_code,expected', [
    ('[PASS] suite\n'*19, 0, True),
    ('[PASS] suite\n'*19, 1, False),
    ('[PASS] suite\n'*18, 0, False),
    ('[PASS] suite\n'*20, 0, False),
    ('[PASS] suite\n'*19+'[FAIL] suite\n', 0, False),
])
def test_sandbox_evidence_preserves_pass_count_and_failure_checks(modules, output, exit_code, expected):
    _, verifier = modules
    tests_run = subprocess.CompletedProcess(['pytest'], 0, stdout='182 passed, 3 skipped in 1.23s\n')
    sandbox_run = subprocess.CompletedProcess(['sandbox'], exit_code, stdout=output)
    assert verifier.test_command_checks(tests_run, sandbox_run) == (True, expected)
