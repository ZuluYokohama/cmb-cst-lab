"""Time-dependent calibration negative controls; synthetic only."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_temporal_calibration", ROOT / "scripts" / "audit_temporal_calibration.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
audit = MODULE.audit


def test_independent_control_is_calibrated_to_nominal_range():
    out = audit(trials=1400, seed=2026, lag_correlation=0.8)
    assert out["cycle_dof"] == 3
    assert .025 < out["iid_calibration_false_alarm_fraction"] < .075


def test_correlated_gaussian_calibration_invalidates_f_reference():
    out = audit(trials=1400, seed=2026, lag_correlation=0.8)
    assert out["temporally_correlated_calibration_false_alarm_fraction"] > 0.18
    assert (out["temporally_correlated_calibration_false_alarm_fraction"]
            > 3 * out["iid_calibration_false_alarm_fraction"])


def test_zero_autocorrelation_is_identical_to_iid_control():
    out = audit(trials=250, seed=71, lag_correlation=0)
    assert (out["temporally_correlated_calibration_false_alarm_fraction"]
            == out["iid_calibration_false_alarm_fraction"])


def test_seeded_replay_is_identical():
    assert audit(trials=225, seed=10) == audit(trials=225, seed=10)


@pytest.mark.parametrize("kwargs", [
    {"trials": 199}, {"trials": True}, {"calibration_count": 5},
    {"calibration_count": 1001}, {"lag_correlation": -0.2},
    {"lag_correlation": 1.0}, {"lag_correlation": float("nan")},
    {"lag_correlation": True},
])
def test_invalid_scenarios_rejected(kwargs):
    with pytest.raises(ValueError):
        audit(**({"trials": 200} | kwargs))
