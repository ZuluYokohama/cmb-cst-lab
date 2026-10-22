"""JoVin sandbox tests for cmbcst.simulator (numpy only)."""
from __future__ import annotations

import numpy as np

from solution import (ScoreCompressor, bartlett_draws, complete_complex, dust_angles, gaussian_cov_upper, mbb_sed,
                      model_covariance, pair_angles, upper)

NB = 12
ELL = np.linspace(61, 281, NB)
EDGES, _, D0, _ = complete_complex(8)


def _fixed(rng=np.random.default_rng(0)):
    ee = 20.0 * (ELL / 80.0) ** -0.6
    th_ee = np.repeat(ee[:, None, None], 8, 1).repeat(8, 2) * 0.01
    return {"th_ee": th_ee, "th_bb": th_ee * 0.01, "dust_ee": 2.0 * (ELL / 80.0) ** -2.4,
            "dust_bb": 1.0 * (ELL / 80.0) ** -2.4, "wratio": np.ones((NB, 8, 8)),
            "noise_ee": np.full((NB, 8), 1e-3), "noise_bb": np.full((NB, 8), 1e-3), "ell": ELL}


THETA0 = np.array([0.2, -0.3, 0.0, 0.1, -0.1, -0.1, -0.2, -0.3, 0.3, 1.5, -0.3, 0.5, 1.0, 1.53])


def test_sed_is_one_at_pivot_and_rises_with_frequency():
    s = mbb_sed(np.array([100.0, 143.0, 217.0, 353.0]), 1.53)
    assert abs(s[-1] - 1.0) < 1e-12 and np.all(np.diff(s) > 0)


def test_covariance_is_symmetric_positive_definite():
    sig = model_covariance(THETA0, _fixed())
    assert np.max(np.abs(sig - sig.transpose(0, 2, 1))) < 1e-14
    assert np.linalg.eigvalsh(sig).min() > 0


def test_pair_angles_of_model_equal_alpha_differences_without_dust_eb():
    th = THETA0.copy()
    th[9:12] = 0.0  # M0: no dust EB; beta is common and must cancel
    sig = model_covariance(th, _fixed())
    pa = pair_angles(sig, np.full(NB, 1e5), EDGES)
    assert np.max(np.abs(np.degrees(pa["delta"]) - D0 @ th[:8])) < 1e-9


def test_dust_eb_breaks_blindness_only_through_frequency_decorrelation():
    # A frequency-independent dust EB is common to all maps up to amplitude -> pair angles shift
    # only where dust differs between maps; same-frequency A/B pairs stay exact.
    th = THETA0.copy()
    th[10:12] = [0.0, 0.0]
    sig = model_covariance(th, _fixed())
    pa = np.degrees(pair_angles(sig, np.full(NB, 1e5), EDGES)["delta"])
    for k, (i, j) in enumerate(EDGES):
        if i // 2 == j // 2:
            assert abs(pa[k] - (th[i] - th[j])) < 1e-9


def test_dust_angle_model_subspaces():
    assert np.all(dust_angles(0.0, 0.0, 0.0, ELL) == 0)
    const = dust_angles(2.0, 0.0, 0.0, ELL)
    assert np.allclose(const, np.radians(2.0))
    assert np.allclose(dust_angles(2.0, -0.5, 0.0, np.array([200.0]))[0], np.radians(2.0))


def test_bartlett_draws_are_unbiased_with_wishart_variance():
    rng = np.random.default_rng(1)
    sig = model_covariance(THETA0, _fixed())[:3]
    dof = np.array([60, 200, 900])
    draws = bartlett_draws(sig, dof, 4000, rng)
    mean = draws.mean(0)
    rel = np.abs(mean - sig) / np.sqrt(np.diagonal(sig, axis1=1, axis2=2)[:, :, None] * np.diagonal(sig, axis1=1, axis2=2)[:, None, :])
    assert rel.max() < 6 * np.sqrt(2 / 60 / 4000) * 3
    var = draws.var(0)
    pred = gaussian_cov_upper(sig, dof)
    m = np.diagonal(pred, axis1=1, axis2=2)
    emp = upper(var)
    assert np.median(np.abs(emp / m - 1)) < 0.06


def test_score_compression_recovers_parameters_linearly():
    rng = np.random.default_rng(2)
    fx = _fixed()
    dof = np.full(NB, 3000)
    steps = np.full(14, 1e-3)
    comp = ScoreCompressor(THETA0, fx, dof, steps)
    shift = np.zeros(14)
    shift[8] = 0.05  # small beta shift
    t = comp(upper(model_covariance(THETA0 + shift, fx)))
    est = np.linalg.solve(comp.fisher, t)  # quasi-ML estimate from the scores
    assert abs(est[8] - 0.05) < 5e-3
    # noise-free fiducial compresses to zero
    assert np.max(np.abs(comp(comp.mu))) < 1e-8 * np.max(np.abs(comp.fisher))


def test_frequency_dependent_misalignment_acts_like_a_calibration_difference():
    # Dust only (no CMB, no noise), scale-independent misalignment that depends on frequency:
    # band i looks rotated by alpha_i + psi_i, so pair angles must be (alpha_i+psi_i)-(alpha_j+psi_j)
    # exactly, and E_iB_j != B_iE_j when psi_i != psi_j (an averaged EB block cannot do this).
    fx = _fixed()
    fx["th_ee"] = fx["th_ee"] * 0
    fx["th_bb"] = fx["th_bb"] * 0
    fx["noise_ee"] = fx["noise_ee"] * 0 + 1e-12
    fx["noise_bb"] = fx["noise_bb"] * 0 + 1e-12
    th = THETA0.copy()
    th[9:12] = [1.0, 0.0, 2.0]  # psi0 = 1 deg, n = 0, psi1 = 2 deg per unit ln(nu)
    sig = model_covariance(th, fx)
    psi = np.degrees(dust_angles(1.0, 0.0, 2.0, ELL))[0]
    pa = np.degrees(pair_angles(sig, np.full(NB, 1e5), EDGES)["delta"])
    total = th[:8] + psi
    assert np.max(np.abs(pa - D0 @ total)) < 1e-8
    i, j = 0, 7  # 100 GHz vs 353 GHz: different psi
    assert abs(sig[0, 2 * i, 2 * j + 1] - sig[0, 2 * i + 1, 2 * j]) > 1e-6 * abs(sig[0, 2 * i, 2 * j])
