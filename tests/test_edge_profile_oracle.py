"""Independent profile-fit oracle for the synthetic single-edge GLR.

Established Gaussian GLS identity, not evidence of a new topological estimator.
"""
from itertools import combinations

import numpy as np

from cmbcst.edge_diagnostics import edge_glrt


def _incidence(n=8):
    edges = list(combinations(range(n), 2))
    a = np.zeros((len(edges), n))
    for k, (i, j) in enumerate(edges):
        a[k, i], a[k, j] = 1.0, -1.0
    return a


def _gls_chi2(y, c, design):
    """Fit directly in whitened coordinates, independent of GLR's precision projector."""
    l = np.linalg.cholesky(c)
    z = np.linalg.solve(l, design)
    t = np.linalg.solve(l, y)
    beta = np.linalg.lstsq(z, t, rcond=None)[0]
    return float(np.sum((t - z @ beta)**2))


def test_edge_scan_equals_full_profile_likelihood_ratio_on_correlated_covariances():
    rng = np.random.default_rng(2255)
    d0 = _incidence(8)
    gauge = 2
    a = np.delete(d0, gauge, axis=1)
    for _ in range(12):
        x = rng.standard_normal((28, 28))
        c = x @ x.T / 28 + np.diag(np.exp(rng.uniform(-3, 1, 28)))
        y = rng.standard_normal(28) + d0 @ rng.standard_normal(8)
        base = _gls_chi2(y, c, a)
        scan = edge_glrt(y, c, d0, gauge=gauge)
        assert len(scan["delta_chi2"]) == 28
        for edge in (0, 4, 11, 24, 27):
            unit = np.eye(28)[:, edge]
            alt = _gls_chi2(y, c, np.column_stack([a, unit]))
            # A known GLS identity: 1-edge profile GLR equals delta chi-square.
            assert abs((base - alt) - scan["delta_chi2"][edge]) < 1e-8


def test_edge_scan_cannot_detect_fully_consistent_node_angles():
    rng = np.random.default_rng(778)
    d0 = _incidence(8)
    c = np.eye(28) + np.ones((28, 28)) * 0.06
    y = d0 @ rng.standard_normal(8)
    result = edge_glrt(y, c, d0, gauge=4)
    assert np.max(np.abs(result["delta_chi2"])) < 1e-20
