"""Statistical calibration controls for the experimental sparse sensor testbed."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("audit_sparse_noise", ROOT / "scripts" / "audit_sparse_noise.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_well_specified_null_roughly_matches_nominal_false_positive_rate():
    r = module.audit(trials=1200, seed=2026)
    assert r["cycle_dof"] == 3
    assert .03 < r["correct_covariance_false_positive_rate"] < .07


def test_wrong_covariance_breaks_calibrated_p_values():
    r = module.audit(trials=1200, seed=2026)
    assert r["underreported_variance_false_positive_rate"] > .35
    assert r["underreported_variance_false_positive_rate"] > 5 * r["correct_covariance_false_positive_rate"]


def test_deterministic_replay_for_fixed_seed():
    assert module.audit(trials=150, seed=9) == module.audit(trials=150, seed=9)


@pytest.mark.parametrize("trials", [0, 1, 99, 100001, True, 100.5])
def test_rejects_invalid_trial_count(trials):
    with pytest.raises(ValueError, match="trials"):
        module.audit(trials=trials)
