"""AST-bundled verification of connected sparse scalar sensor graphs."""
import numpy as np
from solution import graph_design, fit_sparse_offsets


def test_tree_has_no_cycle_redundancy():
    edges = [(0,1),(1,2),(2,3)]
    a, z = graph_design(4, edges)
    assert z.shape == (0,3)
    result = fit_sparse_offsets(a @ np.array([1.,2.,3.,4.]), np.eye(3), 4, edges)
    assert result["dof"] == 0 and result["pte"] is None


def test_cycle_gls_identity_correlated():
    rng = np.random.default_rng(3301)
    edges = [(0,1),(2,1),(2,3),(3,0),(1,3)]
    a, z = graph_design(4, edges)
    assert z.shape == (2,5)
    assert np.max(np.abs(z @ a)) == 0
    p = rng.normal(size=(5,5))
    cov = p @ p.T + np.eye(5)
    y = a @ np.array([.3,-.2,.1,-.5]) + rng.multivariate_normal(np.zeros(5),cov)
    out = fit_sparse_offsets(y,cov,4,edges,gauge=2)
    assert abs(out["closure_chi2"]-out["residual_chi2"]) < 1e-9


def test_singular_covariance_rejected():
    try:
        fit_sparse_offsets(np.zeros(3), np.zeros((3,3)), 3, [(0,1),(1,2),(0,2)])
    except ValueError:
        pass
    else:
        raise AssertionError("singular covariance must be rejected")
