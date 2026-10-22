"""Post-hoc covariance-aware regional A/B stationarity controls."""
import numpy as np
import pytest
from scipy.stats import chi2

from cmbcst.spatial_ab_audit import same_frequency_joint_gls
from cmbcst.spatial import edge_constancy


def _expand(r, sub, positions, total=28):
    cov = np.zeros((r, total, total))
    for k in range(r):
        np.fill_diagonal(cov[k], 1.0)
        cov[k][np.ix_(positions, positions)] = sub[k]
    return cov


def test_joint_equals_diagonal_method_when_selected_edges_independent():
    rng = np.random.default_rng(14)
    r, n = 6, 28
    select = [0, 13, 22, 27]
    delta = rng.normal(size=(r, n))
    variance = rng.uniform(0.2, 2.5, size=(r, len(select)))
    covs = np.repeat(np.eye(n)[None], r, axis=0)
    for k in range(r):
        covs[k, select, select] = variance[k]
    diagonal = edge_constancy(delta, np.sqrt(np.diagonal(covs, axis1=1, axis2=2)), select)
    joint = same_frequency_joint_gls(delta, covs, select)
    assert abs(joint["chi2"] - diagonal["chi2"]) < 1e-10
    assert joint["dof"] == diagonal["dof"] == len(select) * (r - 1)
    assert joint["max_abs_within_region_edge_correlation"] < 1e-12


def test_joint_matches_independent_whitened_stacked_fit():
    rng = np.random.default_rng(18)
    select = [0, 13, 22, 27]
    r, m = 5, 4
    delta = rng.normal(size=(r, 28))
    blocks = []
    for _ in range(r):
        z = rng.normal(size=(m, m))
        blocks.append(z @ z.T + 0.2 * np.eye(m))
    blocks = np.asarray(blocks)
    cov = _expand(r, blocks, select)
    out = same_frequency_joint_gls(delta, cov, select)
    L = np.array([np.linalg.cholesky(x) for x in blocks])
    design = np.concatenate([np.linalg.solve(L[i], np.eye(m)) for i in range(r)], axis=0)
    target = np.concatenate([np.linalg.solve(L[i], delta[i, select]) for i in range(r)])
    fit = np.linalg.lstsq(design, target, rcond=None)[0]
    chi = np.sum((target - design @ fit)**2)
    assert np.allclose(out["mean_per_edge"], fit, rtol=1e-11, atol=1e-11)
    assert abs(out["chi2"] - chi) < 1e-10


def test_positive_cross_edge_correlation_breaks_diagonal_chi2_tail_calibration():
    """A controlled counterexample; not proof that Planck regional edges are correlated."""
    rng = np.random.default_rng(781)
    n_mc, r, m = 6000, 6, 4
    rho = 0.8
    block = (1 - rho) * np.eye(m) + rho * np.ones((m, m))
    l = np.linalg.cholesky(block)
    data = rng.standard_normal((n_mc, r, m)) @ l.T
    residual = data - data.mean(axis=1, keepdims=True)
    diagonal_score = np.sum(residual**2, axis=(1, 2))
    correct_score = np.einsum("sri,ij,srj->s", residual, np.linalg.inv(block), residual)
    cutoff = chi2.isf(0.05, (r-1)*m)
    # With sufficiently correlated selected edges, the diagonal approximation
    # understates false alarms at a nominal 5% chi-square threshold.
    assert 0.11 < np.mean(diagonal_score > cutoff) < 0.17
    assert 0.035 < np.mean(correct_score > cutoff) < 0.065


def test_rejects_invalid_covariance_and_duplicate_selection():
    data = np.zeros((3, 8))
    cov = np.tile(np.eye(8), (3, 1, 1))
    with pytest.raises(ValueError):
        same_frequency_joint_gls(data, cov, [1, 1])
    cov[:, 2, 2] = -1
    with pytest.raises(ValueError):
        same_frequency_joint_gls(data, cov, [2, 3])


def test_rejects_asymmetry_at_small_covariance_scale():
    cov = np.array([[[1e-12, 5e-11], [0.0, 1e-12]]] * 2)
    with pytest.raises(ValueError):
        same_frequency_joint_gls(np.array([[1.0, 1.0], [-1.0, -1.0]]), cov, [0, 1])
