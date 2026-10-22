"""JoVin sandbox tests for cmbcst.network (numpy only)."""
from __future__ import annotations

import math

import numpy as np

from solution import betti_numbers, centering, chi2_diff, chi2_sf, closure_chi2, closures, complete_complex, solve_network


def _cov(rng, m, scale=0.05):
    a = rng.standard_normal((m, m)) * scale
    return a @ a.T + np.eye(m) * scale**2


def test_complex_shape_and_nilpotency():
    edges, tris, d0, d1 = complete_complex(8)
    assert (len(edges), len(tris)) == (28, 56)
    assert d0.shape == (28, 8) and d1.shape == (56, 28)
    assert np.max(np.abs(d1 @ d0)) == 0.0
    assert np.max(np.abs(d0 @ np.ones(8))) == 0.0  # A.1 = 0: common rotation is the gauge


def test_h1_vanishes_on_filled_k8():
    _, _, d0, d1 = complete_complex(8)
    b = betti_numbers(d0, d1)
    assert b == {"rank_d0": 7, "rank_d1": 21, "b0": 1, "b1": 0, "b2_skeleton": 35}


def test_graph_cycle_space_is_21_without_triangles():
    # Without 2-cells the cycle space has dim 28 - 8 + 1 = 21 (H^1 of the bare graph);
    # it disappears once the triangles are filled.
    _, _, d0, _ = complete_complex(8)
    assert 28 - np.linalg.matrix_rank(d0) == 21


def test_noiseless_recovery_any_gauge():
    rng = np.random.default_rng(1)
    _, _, d0, _ = complete_complex(8)
    alpha = rng.normal(0, 0.2, 8)
    cov = _cov(rng, 28)
    for g in range(8):
        res = solve_network(d0 @ alpha, cov, d0, gauge=g)
        assert np.max(np.abs(res["alpha"] - (alpha - alpha[g]))) < 1e-10
        assert res["chi2"] < 1e-18 and res["dof"] == 21
        assert np.max(np.abs(res["cov"][g])) == 0.0


def test_differences_and_chi2_gauge_invariant():
    rng = np.random.default_rng(2)
    _, _, d0, _ = complete_complex(8)
    cov = _cov(rng, 28)
    d = d0 @ rng.normal(0, 0.2, 8) + rng.multivariate_normal(np.zeros(28), cov)
    r0, r5 = solve_network(d, cov, d0, 0), solve_network(d, cov, d0, 5)
    assert abs(r0["chi2"] - r5["chi2"]) < 1e-9
    assert np.max(np.abs(centering(8) @ (r0["alpha"] - r5["alpha"]))) < 1e-10


def test_closure_chi2_equals_network_residual():
    rng = np.random.default_rng(3)
    _, _, d0, d1 = complete_complex(8)
    cov = _cov(rng, 28)
    d = d0 @ rng.normal(0, 0.2, 8) + rng.multivariate_normal(np.zeros(28), cov)
    net = solve_network(d, cov, d0, 2)
    c2, rank = closure_chi2(d, cov, d1)
    assert rank == 21
    assert abs(c2 - net["chi2"]) < 1e-8 * max(1.0, net["chi2"])
    # gradient part closes exactly, so closures only see the curl part
    assert np.max(np.abs(d1 @ net["gradient"])) < 1e-12
    assert np.allclose(closures(d, cov, d1)["closure"], d1 @ net["curl"], atol=1e-12)


def test_curl_covariance_is_the_residual_covariance():
    rng = np.random.default_rng(6)
    _, _, d0, _ = complete_complex(8)
    cov = _cov(rng, 28)
    res = solve_network(rng.multivariate_normal(np.zeros(28), cov), cov, d0, 3)
    cc = res["cov_curl"]
    assert np.linalg.matrix_rank(cc, 1e-10 * np.abs(cc).max()) == 21
    # Monte Carlo: the empirical covariance of the curl matches cov_curl, and it is smaller than C
    curls = np.array([solve_network(rng.multivariate_normal(np.zeros(28), cov), cov, d0, 3)["curl"]
                      for _ in range(3000)])
    emp = np.cov(curls.T)
    assert np.max(np.abs(emp - cc)) < 0.12 * np.max(np.abs(cc))
    assert np.all(np.diag(cc) < np.diag(cov) + 1e-15)


def test_residual_chi2_is_chi2_21_distributed():
    rng = np.random.default_rng(4)
    _, _, d0, _ = complete_complex(8)
    cov = _cov(rng, 28)
    chis = []
    for _ in range(400):
        d = rng.multivariate_normal(np.zeros(28), cov)
        chis.append(solve_network(d, cov, d0, 0)["chi2"])
    chis = np.array(chis)
    assert abs(chis.mean() - 21) < 1.0
    assert abs(chis.var() - 42) < 10.0


def test_chi2_sf_reference_values():
    # Paper: chi2 = 8.18 for 7 dof -> PTE 0.32
    assert abs(chi2_sf(8.18, 7) - 0.3167) < 5e-4
    assert abs(chi2_sf(1.0, 1) - 0.317310507862914) < 1e-10
    assert abs(chi2_sf(21.0, 21) - 0.4589) < 5e-4
    assert 0.0 <= chi2_sf(100.0, 10) < 1e-15
    assert abs(chi2_sf(2.0, 2) - math.exp(-1.0)) < 1e-12


def test_chi2_diff_ignores_common_mode():
    rng = np.random.default_rng(5)
    c = _cov(rng, 8, 0.1)
    q = rng.normal(0, 0.2, 8)
    a = chi2_diff(q, q + 0.37, c, c)  # pure common offset (e.g. beta) is invisible
    assert a["chi2"] < 1e-18 and a["dof"] == 7
    b = chi2_diff(q, q + np.r_[0.3, np.zeros(7)], c, c)
    assert b["chi2"] > 0
