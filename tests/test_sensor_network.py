"""Experimental scalar sensor transfer tests (synthetic only)."""
import numpy as np
import pytest

from cmbcst.network import complete_complex
from cmbcst.sensor_network import fit_scalar_offsets


def test_noiseless_offsets_and_gauge_invariance():
    _, _, d0, _ = complete_complex(5)
    truth = np.array([0.21, -0.07, 0.10, 0.44, -0.12])
    y = d0 @ truth
    cov = np.eye(len(y)) * 0.0025
    for gauge in range(5):
        out = fit_scalar_offsets(y, cov, 5, gauge)
        np.testing.assert_allclose(out["offsets"], truth - truth[gauge], atol=1e-12)
        assert out["closure_dof"] == out["residual_dof"] == 6
        assert abs(out["residual_chi2"] - out["closure_chi2"]) < 1e-10
        assert out["residual_chi2"] < 1e-20
        np.testing.assert_allclose(out["closure"], 0, atol=1e-12)


def test_single_corrupt_edge_detects_inconsistency_but_does_not_attribute_fault():
    edges, _, d0, _ = complete_complex(5)
    y = d0 @ np.array([0.0, 0.2, -0.1, 0.4, -0.3])
    bad = y.copy()
    bad[edges.index((1, 3))] += 0.5
    cov = np.eye(len(y)) * 0.01**2
    out = fit_scalar_offsets(bad, cov, 5, gauge=0)
    assert out["residual_chi2"] > 100
    assert out["pte"] < 1e-12
    assert abs(out["residual_chi2"] - out["closure_chi2"]) < 1e-7
    assert np.linalg.norm(out["closure"]) > 0.5


def test_correlated_covariance_respects_weighted_closure():
    rng = np.random.default_rng(2026)
    _, _, d0, _ = complete_complex(4)
    m = d0.shape[0]
    p = rng.normal(size=(m, m))
    cov = 0.01 * (p @ p.T + np.eye(m))
    y = d0 @ rng.normal(size=4) + rng.multivariate_normal(np.zeros(m), cov)
    out = fit_scalar_offsets(y, cov, 4, gauge=2)
    assert out["residual_dof"] == out["closure_dof"] == 3
    assert np.isclose(out["residual_chi2"], out["closure_chi2"], rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("bad_cov", [np.zeros((6, 6)), np.eye(6) * -1])
def test_rejects_non_positive_definite_covariance(bad_cov):
    with pytest.raises(ValueError, match="positive definite"):
        fit_scalar_offsets(np.zeros(6), bad_cov, 4)


def test_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="expected 6 offsets"):
        fit_scalar_offsets(np.zeros(3), np.eye(6), 4)
    with pytest.raises(ValueError, match="symmetric"):
        cov = np.eye(6)
        cov[0, 1] = 0.2
        fit_scalar_offsets(np.zeros(6), cov, 4)
    with pytest.raises(ValueError, match="gauge"):
        fit_scalar_offsets(np.zeros(6), np.eye(6), 4, gauge=4)
    with pytest.raises(ValueError, match="finite"):
        bad = np.zeros(6)
        bad[0] = np.nan
        fit_scalar_offsets(bad, np.eye(6), 4)


def test_small_scale_covariance_asymmetry_is_not_hidden_by_absolute_tolerance():
    cov = np.eye(6) * 1e-12
    cov[0, 1] = 5e-11
    with pytest.raises(ValueError, match="symmetric"):
        fit_scalar_offsets(np.zeros(6), cov, 4)
