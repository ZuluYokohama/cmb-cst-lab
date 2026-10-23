"""Finite-calibration graph inference tests; synthetic and model-limited."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from scipy.stats import chi2, f
from cmbcst.estimated_covariance import calibrated_cycle_statistic
from cmbcst.sparse_network import graph_design

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("audit_finite_covariance", ROOT / "scripts" / "audit_finite_covariance.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
audit = MODULE.audit
EDGES = MODULE.EDGES


def test_reference_f_transform_and_independence_of_node_offsets():
    rng = np.random.default_rng(27)
    a, z = graph_design(5, EDGES, root=3)
    calibration = rng.normal(0, .2, (16, len(EDGES)))
    calibration += rng.normal(size=(16, 5)) @ a.T
    measurement = a @ rng.normal(size=5) + rng.normal(0, .2, len(EDGES))
    out = calibrated_cycle_statistic(measurement, calibration, 5, EDGES, gauge=3)
    closure = z @ measurement
    train = calibration @ z.T
    cov = np.cov(train, rowvar=False, ddof=1)
    score = float(closure @ np.linalg.solve(cov, closure))
    np.testing.assert_allclose(out["closure"], closure)
    np.testing.assert_allclose(out["cycle_covariance"], cov)
    assert out["chi2_statistic"] == pytest.approx(score, rel=1e-10)
    assert out["naive_chi2_pte"] == pytest.approx(chi2.sf(score, 3), rel=1e-10)
    assert out["f_statistic"] == pytest.approx(score * 13 / 45, rel=1e-10)
    assert (out["f_numerator_dof"], out["f_denominator_dof"]) == (3, 13)
    assert 0 <= f.sf(out["f_statistic"], 3, 13) <= 1


def test_tree_has_no_inferential_statistic_even_with_corruption():
    edges = [(0,1), (1,2), (2,3)]
    a, _ = graph_design(4, edges)
    y = a @ np.arange(4, dtype=float)
    y[0] += 5
    out = calibrated_cycle_statistic(y, np.ones((2,3)), 4, edges)
    assert out["cycle_dof"] == 0 and out["f_statistic"] is None
    assert out["naive_chi2_pte"] is None


def test_sample_count_and_singularity_guards():
    y = np.zeros(len(EDGES)); q = 3
    with pytest.raises(ValueError, match="too few"):
        calibrated_cycle_statistic(y, np.zeros((q+2,len(EDGES))), 5, EDGES)
    with pytest.raises(ValueError, match="positive definite"):
        calibrated_cycle_statistic(y, np.zeros((q+3,len(EDGES))), 5, EDGES)
    with pytest.raises(ValueError, match="shapes"):
        calibrated_cycle_statistic(y, np.zeros((12,4)), 5, EDGES)
    broken = y.copy(); broken[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        calibrated_cycle_statistic(broken, np.zeros((12,len(EDGES))), 5, EDGES)


def test_f_reference_corrects_naive_false_positives():
    result = audit(trials=1800, seed=2026)
    by_k = {x["calibration_count"]: x for x in result["results"]}
    assert by_k[8]["naive_chi2_rejection_fraction"] > .19
    assert by_k[8]["naive_chi2_rejection_fraction"] > 3*by_k[8]["independent_F_rejection_fraction"]
    for row in result["results"]:
        assert .025 < row["independent_F_rejection_fraction"] < .08
    assert by_k[64]["naive_chi2_rejection_fraction"] < by_k[8]["naive_chi2_rejection_fraction"]


def test_seed_replay():
    assert audit(trials=200, seed=97) == audit(trials=200, seed=97)


def test_input_limits():
    with pytest.raises(ValueError, match="trials"):
        audit(trials=100)
    with pytest.raises(ValueError, match="calibration sizes"):
        audit(trials=200, calibration_sizes=(5,))
