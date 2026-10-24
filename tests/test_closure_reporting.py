"""Saved evidence reports must retain diagnostics and limit absence claims."""
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    report = importlib.import_module('report_closure_fibers')
    plan = dict(selected_tiles=[24], starts_per_case=6, planes=2, planned_starts=24,
                compute_cap_seconds=14400, verification_reserve_seconds=900,
                rms_ratio_limit=.25, angle_limit=float(np.pi/4), witness_gap=.05,
                numerical_gate=1e-10,
                cases=[dict(tile=24, stage=stage, representation=representation)
                       for stage, representation in report.CASE_LABELS])
    (tmp_path/'plan.json').write_text(json.dumps(plan))
    (tmp_path/'stage_audit.json').write_text(json.dumps(dict(status='interrupted')))
    (tmp_path/'checkpoints').mkdir()
    name = report.case_name(plan['cases'][0])
    np.savez(tmp_path/'checkpoints'/f'{name}.npz', theta=np.zeros((3, 2)),
             iterations=np.zeros(3, dtype=int),
             status=np.array(['iteration_cap', 'stalled', 'active']),
             initial_retraction_succeeded=np.array([True, False, True]),
             signs=np.array([1., -1., 1.]), reference_target=np.zeros(3))
    (tmp_path/'checkpoints'/f'{name}.json').write_text(json.dumps(dict(first_start=2)))
    return report, tmp_path, name


def test_checkpoint_only_diagnostics_reach_runtime_totals(saved_run):
    report, out, _ = saved_run
    run = report.collect_run(out)
    entry = run['entries'][0]
    assert not entry['array_file_present']
    assert entry['unverified_started'] == 3
    arrays = entry['arrays']
    assert arrays['initial_retraction_succeeded'].shape == (6,)
    assert arrays['initial_retraction_succeeded'].dtype == np.dtype(bool)
    assert arrays['signs'].shape == (6,)
    assert np.issubdtype(arrays['signs'].dtype, np.number)
    context = report.saved_runtime_context(run, out)
    assert context['initial_retraction_failures'] == 1
    assert context['directions'] == dict(maximize=dict(started=2, completed=1),
                                       minimize=dict(started=1, completed=1))


def test_report_does_not_infer_threshold_failure_from_missing_certificates(saved_run, monkeypatch):
    report, out, name = saved_run
    (out/'cases').mkdir()
    np.savez(out/'cases'/f'{name}.npz', theta=np.zeros((6, 2)),
             best_theta=np.zeros((6, 2)), started=np.ones(6, dtype=int),
             iterations=np.zeros(6, dtype=int), status=np.full(6, 'iteration_cap'),
             nonfinite_trials=np.zeros(6, dtype=int), gpu_features=np.zeros((6, 52)),
             gpu_target=np.full(6, .1), gpu_ratios=np.zeros((6, 2)))
    np.savez(out/'reference_features.npz', tile_ids=np.array([24]),
             features=np.zeros((1, 52)), scales=np.ones(52))
    run = report.collect_run(out)
    assert run['entries'][0]['maximum_gap'] > run['plan']['witness_gap']
    context = report.saved_runtime_context(run, out)
    assert context['provisional_candidates'] == context['direct_replay_candidates'] == 0
    path = out/'report.md'
    monkeypatch.setattr(sys, 'argv', ['report_closure_fibers.py', '--run-dir', str(out),
                                    '--report', str(path)])
    report.main()
    text = path.read_text()
    assert 'No saved certificate records a provisional candidate.' in text
    assert 'No provisional candidate met the declared witness threshold' not in text
