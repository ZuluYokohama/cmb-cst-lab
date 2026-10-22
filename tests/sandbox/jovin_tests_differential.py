"""JoVin sandbox tests for cmbcst.differential (numpy only).

Spectra are simulated as full-sky Wishart draws: at each ell, 2l+1 real samples of the 16
fields (E_i, B_i of 8 maps) with covariance R(alpha + beta) Sigma_sky R^T + N.
"""
from __future__ import annotations

import numpy as np

from solution import (complete_complex, fit_pair, gaussian_cov, pair_angles, residual_cov, residual_weights,
                      rotation_matrix, solve_network)

EDGES, _, D0, _ = complete_complex(8)
DEG = np.pi / 180
ALPHA = np.array([+0.19, -0.31, 0.0, 0.04, -0.06, -0.06, -0.08, -0.10]) * DEG * 3  # a few tenths of a degree
NOISE = np.array([6.0, 6.0, 3.0, 3.0, 4.0, 4.0, 12.0, 12.0]) * 1e-2


def _sigma(ell, alpha, beta=0.0, eb=0.0):
    ee = 2.0 * (ell / 80.0) ** -0.6
    bb = 0.15 * ee
    sky = np.array([[ee, eb * ee], [eb * ee, bb]])
    s = np.kron(np.ones((8, 8)), sky)
    r = rotation_matrix(alpha + beta)
    return r @ s @ r.T + np.diag(np.repeat(NOISE, 2))


def _binned(alpha, beta=0.0, eb=0.0, rng=None, lmin=51, lmax=331, width=20):
    edges = np.arange(lmin, lmax + 1, width)
    cl = np.zeros((len(edges) - 1, 16, 16))
    nu = np.zeros(len(edges) - 1)
    for b, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        for ell in range(lo, hi):
            sig = _sigma(ell, alpha, beta, eb)
            if rng is None:
                cl[b] += (2 * ell + 1) * sig
            else:
                x = np.linalg.cholesky(sig) @ rng.standard_normal((16, 2 * ell + 1))
                cl[b] += x @ x.T
            nu[b] += 2 * ell + 1
        cl[b] /= nu[b]
    return cl, nu


def test_sparse_residual_cov_matches_dense_formula():
    cl, nu = _binned(ALPHA)
    t = np.array([0.01, -0.02, 0.03])
    ed = [(0, 1), (0, 3), (2, 7)]
    rc = residual_cov(cl, nu, ed, t)
    for b in (0, 5):
        for e in range(3):
            for f in range(3):
                we = residual_weights(*ed[e], t[e], 16)
                wf = residual_weights(*ed[f], t[f], 16)
                assert abs(rc[b, e, f] - gaussian_cov(we, wf, cl[b], nu[b])) < 1e-12 * abs(rc[b, e, e])


def test_exact_spectra_give_exact_pair_angles():
    cl, nu = _binned(ALPHA)
    res = pair_angles(cl, nu, EDGES)
    assert np.max(np.abs(res["delta"] - D0 @ ALPHA)) < 1e-12


def test_blind_to_common_rotation_and_sky_eb():
    cl0, nu = _binned(ALPHA)
    cl1, _ = _binned(ALPHA, beta=0.35 * DEG, eb=0.05)
    for i, j in [(0, 1), (2, 5), (6, 7)]:
        assert abs(fit_pair(cl0, nu, i, j)["delta"] - fit_pair(cl1, nu, i, j)["delta"]) < 1e-12


def test_noisy_recovery_and_covariance_calibration():
    rng = np.random.default_rng(7)
    cl, nu = _binned(ALPHA, beta=0.3 * DEG, eb=0.03, rng=rng)
    res = pair_angles(cl, nu, EDGES)
    cov = res["cov"]
    assert np.all(np.linalg.eigvalsh(cov) > 0)
    resid = res["delta"] - D0 @ ALPHA
    chi2 = float(resid @ np.linalg.solve(cov, resid))
    assert 28 - 4 * np.sqrt(56) < chi2 < 28 + 4 * np.sqrt(56)
    net = solve_network(res["delta"], cov, D0, gauge=2)
    pulls = (net["alpha"] - (ALPHA - ALPHA[2]))[[0, 1, 3, 4, 5, 6, 7]] / np.sqrt(np.diag(net["cov"]))[[0, 1, 3, 4, 5, 6, 7]]
    assert np.max(np.abs(pulls)) < 4.0
    assert 21 - 4 * np.sqrt(42) < net["chi2"] < 21 + 4 * np.sqrt(42)


def test_covariance_is_calibrated_over_an_ensemble():
    # Mean chi^2 of the 28 pair angles about the truth should be ~28, and of the network
    # residual ~21, if the propagated covariance is right (not just the point estimates).
    rng = np.random.default_rng(11)
    c_pairs, c_net = [], []
    for _ in range(40):
        cl, nu = _binned(ALPHA, rng=rng, lmax=231)
        res = pair_angles(cl, nu, EDGES)
        r = res["delta"] - D0 @ ALPHA
        c_pairs.append(float(r @ np.linalg.solve(res["cov"], r)))
        c_net.append(solve_network(res["delta"], res["cov"], D0, gauge=0)["chi2"])
    # 40 draws: standard error of the mean is sqrt(2k/40)
    assert abs(np.mean(c_pairs) - 28) < 3.5 * np.sqrt(56 / 40)
    assert abs(np.mean(c_net) - 21) < 3.5 * np.sqrt(42 / 40)


def test_pairs_sharing_a_map_are_correlated():
    cl, nu = _binned(ALPHA)
    cov = pair_angles(cl, nu, EDGES)["cov"]
    corr = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
    e01, e02, e34 = EDGES.index((0, 1)), EDGES.index((0, 2)), EDGES.index((3, 4))
    assert abs(corr[e01, e02]) > 0.05
    assert abs(corr[e01, e34]) < abs(corr[e01, e02])
