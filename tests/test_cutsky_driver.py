"""The new pilot must enforce clock, evidence and duplicate-run boundaries."""
from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path

import pytest


def driver():
    path = Path(__file__).resolve().parents[1]/'scripts/run_cutsky_feasibility.py'
    spec = importlib.util.spec_from_file_location('cutsky_driver_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('monotonic,utc', [(100., 1.), (1., 100.)])
def test_whichever_clock_exhausts_the_budget_stops_work(monkeypatch, monotonic, utc):
    module = driver()
    clock = [0.]
    instant = [datetime(2026, 10, 24, tzinfo=timezone.utc)]
    monkeypatch.setattr(module.time, 'perf_counter', lambda: clock[0])
    monkeypatch.setattr(module, 'now', lambda: instant[0])
    budget = module.Budget(100., 10.)
    clock[0] = monotonic
    instant[0] += timedelta(seconds=utc)
    with pytest.raises(TimeoutError):
        budget.check()


def test_search_reserve_does_not_grant_an_extended_verification_cap(monkeypatch):
    module = driver()
    clock = [0.]
    monkeypatch.setattr(module.time, 'perf_counter', lambda: clock[0])
    budget = module.Budget(100., 10.)
    clock[0] = 91.
    with pytest.raises(TimeoutError):
        budget.check(search=True)
    budget.check()
    clock[0] = 101.
    with pytest.raises(TimeoutError):
        budget.check()


@pytest.mark.parametrize('cap,reserve', [(float('nan'), 1), (1801, 1), (100, 100), (100, -1), (True, 0)])
def test_invalid_or_extended_budgets_rejected(cap, reserve):
    with pytest.raises(ValueError):
        driver().Budget(cap, reserve)


def test_existing_outputs_and_started_runs_are_never_reset(tmp_path):
    module = driver()
    with pytest.raises(FileExistsError):
        module.prepare(tmp_path)
    module.save_json(tmp_path/'plan.json', {})
    with pytest.raises(RuntimeError, match='SHA256'):
        module.execute(tmp_path)
    module.save_json(tmp_path/'stage_audit.json', {'status': 'incomplete'})
    with pytest.raises(FileExistsError):
        module.execute(tmp_path)


def test_nonfinite_json_cannot_be_published_as_a_pass(tmp_path):
    module = driver()
    target = tmp_path/'receipt.json'
    with pytest.raises(ValueError):
        module.save_json(target, {'status': 'passed', 'error': float('nan')})
    assert not target.exists()


def test_watchdog_and_worker_stage_writes_are_serialized(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import json
    module = driver()
    target = tmp_path/'stage_audit.json'
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: module.save_json(target, {'writer': i}), range(64)))
    assert json.loads(target.read_text())['writer'] in range(64)
    assert not target.with_name(target.name+'.tmp').exists()
