"""Sparse scalar sensor transfer tests, synthetic only."""
from itertools import combinations
import numpy as np
import pytest
from scipy.optimize import least_squares
from cmbcst.sparse_network import graph_design, fit_sparse_offsets
from cmbcst.sensor_network import fit_scalar_offsets


def test_tree_no_redundancy_and_gauge():
    edges = [(0,1), (3,1), (2,3), (3,4)]
    a, z = graph_design(5, edges)
    assert z.shape == (0,4)
    truth = np.array([.3,-.2,.5,.8,.1])
    y = a @ truth
    out = fit_sparse_offsets(y, np.eye(4)*.03, 5, edges, gauge=1)
    np.testing.assert_allclose(out["offsets"], truth-truth[1], atol=1e-11)
    assert out["dof"] == 0 and out["pte"] is None
    y[0] += .5
    assert fit_sparse_offsets(y, np.eye(4), 5, edges)["residual_chi2"] < 1e-20


def test_cycle_basis_and_orientation():
    edges = [(0,1), (2,1), (2,3), (3,4), (0,4), (1,4)]
    a, z = graph_design(5, edges, root=2)
    assert z.shape == (2,6)
    np.testing.assert_allclose(z @ a, 0, atol=0)
    assert np.linalg.matrix_rank(z) == 2


def test_complete_graph_matches_existing_kernel():
    rng = np.random.default_rng(73)
    edges = list(combinations(range(5), 2))
    a, z = graph_design(5, edges)
    k = rng.normal(size=(10,10))
    cov = k @ k.T + np.eye(10)
    y = a @ rng.normal(size=5) + rng.multivariate_normal(np.zeros(10), cov)
    old = fit_scalar_offsets(y, cov, 5, gauge=2)
    new = fit_sparse_offsets(y, cov, 5, edges, gauge=2)
    np.testing.assert_allclose(old["offsets"], new["offsets"], atol=2e-11)
    assert abs(old["residual_chi2"] - new["residual_chi2"]) < 1e-9
    assert abs(new["residual_chi2"] - new["closure_chi2"]) < 1e-9
    assert z.shape[0] == 6


def test_independent_scipy_least_squares_baseline():
    rng = np.random.default_rng(228)
    edges = [(0,1), (1,2), (2,3), (3,4), (4,0), (1,3)]
    a, _ = graph_design(5, edges)
    b = rng.normal(size=(6,6))
    cov = b @ b.T + np.eye(6)
    y = a @ rng.normal(size=5) + rng.multivariate_normal(np.zeros(6), cov)
    l = np.linalg.cholesky(cov)
    wa = np.linalg.solve(l, a[:,1:])
    wy = np.linalg.solve(l, y)
    fit = least_squares(lambda x: wa@x - wy, np.zeros(4), jac=lambda x: wa,
                        xtol=1e-13, ftol=1e-13, gtol=1e-13)
    out = fit_sparse_offsets(y, cov, 5, edges)
    np.testing.assert_allclose(out["offsets"][1:], fit.x, atol=1e-9)
    assert np.isclose(out["closure_chi2"], sum(fit.fun**2), atol=1e-9)


def test_bad_edge_detects_but_cannot_assign_fault():
    edges = [(0,1), (1,2), (2,3), (0,3), (1,3)]
    a, z = graph_design(4, edges)
    y = a @ np.array([0.,.1,-.2,.4])
    y[-1] += 1
    out = fit_sparse_offsets(y, np.eye(5)*.001, 4, edges)
    assert out["dof"] == 2
    assert out["residual_chi2"] > 100 and out["pte"] < 1e-9


def test_gaussian_calibration_moments():
    rng = np.random.default_rng(319)
    edges = [(0,1), (1,2), (2,3), (3,0), (0,2)]
    a, _ = graph_design(4, edges)
    cov = np.eye(5)*.04
    vals=[]
    for _ in range(150):
        y = a @ np.array([0,1,2,3]) + rng.multivariate_normal(np.zeros(5), cov)
        vals.append(fit_sparse_offsets(y, cov, 4, edges)["residual_chi2"])
    assert 1.5 < np.mean(vals) < 2.5
    assert 2.5 < np.var(vals) < 6.0


@pytest.mark.parametrize("edges", [
    [(0,1), (2,3), (0,1)],
    [(0,1), (1,2), (2,3), (3,3)],
    [(0,1), (1,2), (2,0), (3,4)],
])
def test_rejects_invalid_topology(edges):
    with pytest.raises(ValueError):
        graph_design(5, edges)


def test_small_scale_bad_covariance_rejected():
    edges = [(0,1), (1,2), (0,2)]
    cov = np.eye(3)*1e-12
    cov[0,2] = 5e-11
    with pytest.raises(ValueError, match="symmetric"):
        fit_sparse_offsets(np.zeros(3), cov, 3, edges)
    with pytest.raises(ValueError, match="positive definite"):
        fit_sparse_offsets(np.zeros(3), np.zeros((3,3)), 3, edges)
