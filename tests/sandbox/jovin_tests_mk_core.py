"""JoVin sandbox tests for cmbcst.mk_likelihood (numpy only).

`_ref_A_and_B*` are scalar transcriptions of the reference implementation
(antolonappan/cosmic_birefringence, src/cosmic_bire/tools_fast.py, MIT) used to check the port.
"""
from __future__ import annotations

import numpy as np
from numpy import cos, sin

from solution import (MKLikelihood, complete_complex, model_rows, ordered_pairs, pair_angles, rotation_matrix,
                      solve_network)

DEG = np.pi / 180
N = 8
ALPHA = np.array([0.19, -0.31, 0.05, 0.04, -0.06, -0.06, -0.08, -0.10]) * DEG
BETA = 0.35 * DEG
DUST = np.array([0.05, 0.05, 0.12, 0.12, 0.5, 0.5, 4.0, 4.0])  # per-map dust amplitude
NOISE = np.array([6.0, 6.0, 3.0, 3.0, 4.0, 4.0, 12.0, 12.0]) * 1e-3


# ---- reference (scalar) ---------------------------------------------------------------------
def _R_mat(ti2, tj2, inv=False):
    diag, a_diag = cos(ti2) * cos(tj2), sin(ti2) * sin(tj2)
    if inv:
        return 2 / (cos(2 * ti2) + cos(2 * tj2)) * np.array([[diag, -a_diag], [-a_diag, diag]])
    return np.array([[diag, a_diag], [a_diag, diag]])


def _R_vec(ti2, tj2):
    return np.array([cos(ti2) * sin(tj2), -sin(ti2) * cos(tj2)])


def _D_vec(ai2, aj2):
    return np.array([cos(ai2) * cos(aj2), -sin(ai2) * sin(aj2)])


def _D_mat(ai2, aj2):
    diag, a_diag = cos(ai2) * sin(aj2), cos(aj2) * sin(ai2)
    return np.array([[-diag, -a_diag], [a_diag, diag]])


def _ref_A_and_B(ai2, aj2, b2):
    r_inv = _R_mat(ai2, aj2, inv=True)
    block_a = -np.dot(_R_vec(ai2, aj2), r_inv)
    b_ = _R_vec(ai2 + b2, aj2 + b2) - np.dot(_R_vec(ai2, aj2), np.dot(r_inv, _R_mat(ai2 + b2, aj2 + b2)))
    return block_a, b_


def _ref_A_and_B_primed(ai2, aj2, b2, psi_l, a):
    f = a * sin(4 * psi_l) * np.array([[1, 0], [1, 0]])
    lam_vec = _R_vec(ai2, aj2) + np.dot(_D_vec(ai2, aj2), f)
    lam_mat = _R_mat(ai2, aj2) + np.dot(_D_mat(ai2, aj2), f)
    m = -np.dot(lam_vec, np.linalg.inv(lam_mat))
    return m, _R_vec(ai2 + b2, aj2 + b2) + np.dot(m, _R_mat(ai2 + b2, aj2 + b2))


# ---- simulation -------------------------------------------------------------------------------
def _sigma(ell, alpha=ALPHA, beta=BETA):
    ee_c = 2.0 * (ell / 80.0) ** -0.6 * (1 + 0.5 * np.sin(ell / 40.0))
    bb_c = 0.01 * ee_c
    ee_d = (ell / 80.0) ** -2.4
    cmb = np.kron(np.ones((N, N)), np.diag([ee_c, bb_c]))
    rb = rotation_matrix(np.full(N, beta))
    dust = np.kron(np.outer(DUST, DUST), np.diag([ee_d, 0.5 * ee_d]))
    r = rotation_matrix(alpha)
    sky = rb @ cmb @ rb.T + dust
    return r @ sky @ r.T + np.diag(np.repeat(NOISE, 2)), ee_c, bb_c


def _binned(rng=None, lmin=51, lmax=291, width=20):
    edges = np.arange(lmin, lmax + 1, width)
    nb = len(edges) - 1
    cl, nu = np.zeros((nb, 2 * N, 2 * N)), np.zeros(nb)
    th_ee, th_bb = np.zeros((nb, N, N)), np.zeros((nb, N, N))
    for b in range(nb):
        for ell in range(edges[b], edges[b + 1]):
            sig, ee_c, bb_c = _sigma(ell)
            if rng is None:
                cl[b] += (2 * ell + 1) * sig
            else:
                x = np.linalg.cholesky(sig) @ rng.standard_normal((2 * N, 2 * ell + 1))
                cl[b] += x @ x.T
            th_ee[b] += (2 * ell + 1) * ee_c
            th_bb[b] += (2 * ell + 1) * bb_c
            nu[b] += 2 * ell + 1
        cl[b] /= nu[b]
        th_ee[b] /= nu[b]
        th_bb[b] /= nu[b]
    return cl, nu, th_ee, th_bb


def test_port_matches_reference_rows():
    rng = np.random.default_rng(0)
    pairs = ordered_pairs(N)
    for _ in range(2):
        alpha = rng.normal(0, 0.5 * DEG, N)
        m, b = model_rows(alpha, BETA, 0.0, pairs)
        for k, (i, j) in enumerate(pairs):
            rm, rb = _ref_A_and_B(2 * alpha[i], 2 * alpha[j], 2 * BETA)
            assert np.allclose(m[k], rm, atol=1e-14) and np.allclose(b[k], rb, atol=1e-14)
    for amp, psi in ((0.3, 0.2), (0.05, -0.6)):
        alpha = rng.normal(0, 0.5 * DEG, N)
        m, b = model_rows(alpha, BETA, amp * np.sin(4 * psi), pairs)
        for k, (i, j) in enumerate(pairs):
            rm, rb = _ref_A_and_B_primed(2 * alpha[i], 2 * alpha[j], 2 * BETA, psi, amp)
            assert np.allclose(m[k], rm, atol=1e-14) and np.allclose(b[k], rb, atol=1e-14)


def test_small_angle_limit():
    # EB_o = 2 a_j EE - 2 a_i BB + 2 beta (EE - BB)_th  to first order
    alpha = np.array([1e-4, -2e-4, 0, 0, 0, 0, 0, 0])
    m, b = model_rows(alpha, 3e-4, 0.0, [(0, 1)])
    assert np.allclose(m[0], [-2 * alpha[1], 2 * alpha[0]], atol=1e-10)
    assert np.allclose(b[0], [2 * 3e-4, -2 * 3e-4], atol=1e-10)


def test_exact_spectra_recover_alpha_and_beta():
    cl, nu, th_ee, th_bb = _binned()
    fit = MKLikelihood(cl, nu, th_ee, th_bb).fit()
    assert np.max(np.abs(fit["alpha"] - ALPHA)) < 1e-9
    assert abs(fit["beta"] - BETA) < 1e-9


def test_noisy_recovery_within_errors():
    rng = np.random.default_rng(3)
    cl, nu, th_ee, th_bb = _binned(rng)
    fit = MKLikelihood(cl, nu, th_ee, th_bb).fit()
    sig = np.sqrt(np.diag(fit["cov"]))
    z = (fit["theta"] - np.r_[ALPHA, BETA]) / sig
    assert np.max(np.abs(z)) < 4.0


def test_dust_eb_amplitudes_vanish_without_dust_eb():
    cl, nu, th_ee, th_bb = _binned(lmax=571)  # 26 bins: every amplitude range has data
    psi = 0.15 * np.cos(np.arange(cl.shape[0]) / 3.0)
    fit = MKLikelihood(cl, nu, th_ee, th_bb, psi=psi, n_a=4).fit()
    assert np.max(np.abs(fit["A"])) < 1e-8
    assert np.max(np.abs(fit["alpha"] - ALPHA)) < 1e-9 and abs(fit["beta"] - BETA) < 1e-9


def test_mk_and_network_agree_on_differences():
    cl, nu, th_ee, th_bb = _binned()
    edges, _, d0, _ = complete_complex(N)
    pa = pair_angles(cl, nu, edges)
    net = solve_network(pa["delta"], pa["cov"], d0, gauge=2)
    mk = MKLikelihood(cl, nu, th_ee, th_bb).fit()
    assert np.max(np.abs(net["alpha"] - (mk["alpha"] - mk["alpha"][2]))) < 1e-9
