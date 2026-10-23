"""Synthetic non-Gaussian negative control for finite-F calibration."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("audit_non_gaussian", ROOT / "scripts" / "audit_non_gaussian_closure.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
audit = MODULE.audit


def test_stationary_gaussian_reference_stays_near_nominal():
    out = audit(trials=1200, seed=2026)
    assert out["cycle_dof"] == 3
    assert .025 < out["gaussian_false_alarm_fraction"] < .075


def test_variance_matched_heavy_tails_break_f_reference():
    out = audit(trials=1200, seed=2026)
    assert out["student_t_false_alarm_fraction"] > .07
    assert out["student_t_false_alarm_fraction"] > 1.5 * out["gaussian_false_alarm_fraction"]


def test_fixed_seed_exact_replay():
    assert audit(trials=220, seed=97) == audit(trials=220, seed=97)


@pytest.mark.parametrize("kwargs", [
    {"trials": 50},
    {"trials": True},
    {"calibration_count": 5},
    {"calibration_count": 1001},
    {"tail_df": 2},
    {"tail_df": 101},
])
def test_invalid_scenario_rejected(kwargs):
    with pytest.raises(ValueError):
        audit(trials=200, **kwargs)
