"""Queued CPU work must respect the scientific search deadline."""
import importlib.util
from pathlib import Path
import pytest


def driver():
    path=Path(__file__).resolve().parents[1]/'scripts/run_closure_fibers_v2.py'
    spec=importlib.util.spec_from_file_location('closure_driver_v2_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_cutoff_crossed_while_waiting_releases_lock_without_work():
    clock=[0.];events=[]
    class Lock:
        def acquire(self,timeout):
            events.append('acquired');clock[0]=2.;return True
        def release(self):events.append('released')
    def check():
        if clock[0]>=1.:raise TimeoutError('reserve')
    with pytest.raises(TimeoutError):
        with driver().BudgetCPULock(Lock(),check):events.append('scientific_work')
    assert events==['acquired','released']


def test_unavailable_lock_checks_budget_before_starting_work():
    clock=[0.];events=[]
    class Lock:
        def acquire(self,timeout):clock[0]=2.;return False
        def release(self):events.append('unexpected_release')
    def check():
        if clock[0]>=1.:raise TimeoutError('reserve')
    with pytest.raises(TimeoutError):
        with driver().BudgetCPULock(Lock(),check):events.append('scientific_work')
    assert events==[]
