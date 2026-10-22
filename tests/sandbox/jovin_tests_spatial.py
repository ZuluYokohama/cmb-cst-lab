"""JoVin sandbox tests for cmbcst.spatial (numpy only)."""
from __future__ import annotations

import numpy as np

from solution import (chi2_sf, complete_complex, edge_constancy, inject, null_spectra, pair_angles,
                      same_frequency_edges, solve_network, stationarity_split, tracer_regression)

DET = ["100A", "100B", "143A", "143B", "217A", "217B", "353A", "353B"]
EDGES, _, D0, D1 = complete_complex(8)
GAUGE = 2


def _covs(rng, r, scale=0.1):
    out = []
    for _ in range(r):
        a = rng.standard_normal((28, 28)) * scale
        out.append(a @ a.T / 28 + np.eye(28) * scale**2)
    return np.array(out)


def _draw(rng, covs, alpha, gammas=None):
    r = len(covs)
    g = np.zeros((r, 8)) if gammas is None else gammas
    return np.array([D0 @ (alpha + g[k]) + rng.multivariate_normal(np.zeros(28), covs[k]) for k in range(r)])


def test_split_is_exact_and_matches_single_region_fits():
    rng = np.random.default_rng(0)
    covs = _covs(rng, 6)
    d = _draw(rng, covs, rng.normal(0, 0.2, 8), rng.normal(0, 0.1, (6, 8)))
    s = stationarity_split(d, covs, D0, GAUGE)
    assert abs(s["chi2_total"] - s["chi2_closure"] - s["chi2_spatial"]) < 1e-8 * s["chi2_total"]
    assert (s["dof_total"], s["dof_closure"], s["dof_spatial"]) == (6 * 28 - 7, 6 * 21, 5 * 7)
    for k in range(6):  # per-region closure chi2 equals the one-region network residual
        net = solve_network(d[k], covs[k], D0, GAUGE)
        assert abs(net["chi2"] - s["chi2_closure_region"][k]) < 1e-8 * max(1.0, net["chi2"])
        assert np.allclose(net["alpha"], s["alpha_region"][k], atol=1e-10)


def test_noiseless_stationary_pattern_gives_zero_and_gauge_is_invisible():
    rng = np.random.default_rng(1)
    covs = _covs(rng, 4)
    alpha = rng.normal(0, 0.2, 8)
    common = rng.normal(0, 0.3, (4, 1)) * np.ones((4, 8))  # region-dependent COMMON rotation
    d = np.array([D0 @ (alpha + common[k]) for k in range(4)])
    s = stationarity_split(d, covs, D0, GAUGE)
    assert s["chi2_total"] < 1e-18 and s["chi2_spatial"] < 1e-18  # anisotropic birefringence is gauge


def test_null_calibration_over_many_skies():
    rng = np.random.default_rng(2)
    covs = _covs(rng, 5)
    alpha = rng.normal(0, 0.2, 8)
    sp, cl = [], []
    for _ in range(300):
        s = stationarity_split(_draw(rng, covs, alpha), covs, D0, GAUGE)
        sp.append(s["chi2_spatial"])
        cl.append(s["chi2_closure"])
    assert abs(np.mean(sp) - 28) < 3 * np.sqrt(56 / 300) * 1.5
    assert abs(np.mean(cl) - 105) < 3 * np.sqrt(210 / 300) * 1.5
    ptes = np.array([chi2_sf(x, 28) for x in sp])
    assert abs(np.mean(ptes < 0.1) - 0.1) < 0.06


def test_spatial_signal_is_detected_and_lands_in_spatial_term():
    rng = np.random.default_rng(3)
    covs = _covs(rng, 5, scale=0.05)
    gam = np.zeros((5, 8))
    gam[:, 6] = np.linspace(-0.3, 0.3, 5)  # 353A angle varies across the sky
    s = stationarity_split(_draw(rng, covs, np.zeros(8), gam), covs, D0, GAUGE)
    assert chi2_sf(s["chi2_spatial"], s["dof_spatial"]) < 1e-6
    assert chi2_sf(s["chi2_closure"], s["dof_closure"]) > 1e-3  # a gradient change still closes


def test_same_frequency_edges_and_constancy():
    sf = same_frequency_edges(DET, EDGES)
    assert [EDGES[k] for k in sf] == [(0, 1), (2, 3), (4, 5), (6, 7)]
    rng = np.random.default_rng(4)
    sig = np.full((6, 28), 0.1)
    flat = edge_constancy(rng.normal(0.2, 0.1, (6, 28)), sig, sf)
    assert flat["dof"] == 4 * 5
    d = rng.normal(0.2, 0.1, (6, 28))
    d[:, sf[1]] += np.linspace(-1, 1, 6)
    assert edge_constancy(d, sig, sf)["chi2"] > flat["chi2"] + 50


def test_tracer_regression_recovers_slope():
    rng = np.random.default_rng(5)
    covs = _covs(rng, 8, scale=0.03)
    tracer = np.linspace(0, 1, 8)
    k_true = np.zeros(8)
    k_true[7] = 0.4  # 353B tracks the tracer
    gam = np.outer(tracer - tracer.mean(), k_true)
    s = stationarity_split(_draw(rng, covs, np.zeros(8), gam), covs, D0, GAUGE)
    reg = tracer_regression(s["gamma"], s["fisher_region"], tracer, GAUGE)
    se = np.sqrt(np.diag(reg["slope_cov"]))
    keep = [k for k in range(8) if k != GAUGE]
    z = (reg["slope"][keep] - k_true[keep]) / se
    assert np.max(np.abs(z)) < 4 and reg["delta_chi2"] > 20


def test_blinded_null_spectra_impose_the_global_pattern_exactly():
    # The validation nulls never form a real regional pair angle: they remove the antisymmetric
    # part after derotating by the public global angles. Their pair angles must equal d0 alpha.
    rng = np.random.default_rng(6)
    a = rng.standard_normal((8, 16, 16))
    cl = a @ a.transpose(0, 2, 1) + 16 * np.eye(16)
    alpha = np.array([0.3, -0.5, 0.0, 0.1, -0.1, -0.12, -0.25, -0.27])
    n = null_spectra(cl, alpha)
    assert np.max(np.abs(n - n.transpose(0, 2, 1))) < 1e-10 and np.linalg.eigvalsh(n).min() > 0
    deg = np.pi / 180
    pa = pair_angles(n, np.full(8, 1e4), EDGES)
    assert np.max(np.abs(pa["delta"] / deg - D0 @ alpha)) < 1e-10
    gam = np.array([0, 0, 0, 0, 0, 0, 0.2, 0.2])  # injection adds exactly d0 gamma
    pa2 = pair_angles(inject(n, gam), np.full(8, 1e4), EDGES)
    assert np.max(np.abs(pa2["delta"] / deg - D0 @ (alpha + gam))) < 1e-10


def _unequal_covs(rng, r):
    # regions with very different noise levels (as for different sky fractions)
    return np.array([c * s for c, s in zip(_covs(rng, r, scale=0.04), np.linspace(0.3, 3.0, r))])


def test_gamma_covariance_accounts_for_the_global_fit():
    rng = np.random.default_rng(7)
    covs = _unequal_covs(rng, 5)
    alpha = rng.normal(0, 0.2, 8)
    gams = np.array([stationarity_split(_draw(rng, covs, alpha), covs, D0, GAUGE)["gamma"] for _ in range(800)])
    s = stationarity_split(_draw(rng, covs, alpha), covs, D0, GAUGE)
    keep = [k for k in range(8) if k != GAUGE]
    for r in range(5):
        emp = np.var(gams[:, r, keep], axis=0)
        pred = np.diag(s["gamma_cov"][r])
        assert np.max(np.abs(emp / pred - 1)) < 0.22           # F_r^-1 - F_tot^-1, not F_r^-1
        assert np.all(pred < np.diag(np.linalg.inv(s["fisher_region"][r])))


def test_tracer_regression_profiles_the_vector_intercept():
    # Unequal regional Fisher matrices: null slope calibration (delta chi2 ~ chi2_7) and agreement
    # with an explicit stacked GLS over (a, k).
    rng = np.random.default_rng(8)
    covs = _unequal_covs(rng, 7)
    tracer = np.linspace(0.2, 5.0, 7) ** 1.5
    alpha = rng.normal(0, 0.2, 8)
    dchi = []
    for _ in range(200):
        s = stationarity_split(_draw(rng, covs, alpha), covs, D0, GAUGE)
        dchi.append(tracer_regression(s["gamma"], s["fisher_region"], tracer, GAUGE)["delta_chi2"])
    assert abs(np.mean(dchi) - 7) < 3.5 * np.sqrt(14 / 200)
    s = stationarity_split(_draw(rng, covs, alpha), covs, D0, GAUGE)
    keep = [k for k in range(8) if k != GAUGE]
    g, fis = s["gamma"][:, keep], s["fisher_region"]
    rows, rhs = np.zeros((14, 14)), np.zeros(14)
    for r in range(7):
        x = np.hstack([np.eye(7), tracer[r] * np.eye(7)])
        rows += x.T @ fis[r] @ x
        rhs += x.T @ fis[r] @ g[r]
    k_explicit = np.linalg.solve(rows, rhs)[7:]
    reg = tracer_regression(s["gamma"], fis, tracer, GAUGE)
    assert np.allclose(reg["slope"][keep], k_explicit, atol=1e-10)
    assert np.allclose(reg["slope_cov"], np.linalg.inv(rows)[7:, 7:], atol=1e-10)
