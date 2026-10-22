"""End-to-end injection test on synthetic HEALPix maps (local, not sandboxable: needs ducc0).

Eight Q/U maps share one sky (with intrinsic EB), are rotated by known per-map angles plus a
common beta in *map space*, get independent white noise and a Galactic-style cut, then go
through the same spectra -> pair-angle -> network chain as the real data.
"""
from __future__ import annotations

import numpy as np
import pytest

from cmbcst import network, spectra
from cmbcst.differential import pair_angles

DEG = np.pi / 180
ALPHA = np.array([0.188, -0.305, 0.0, 0.039, -0.063, -0.058, -0.082, -0.096]) * DEG
NSIDE, LMAX = 256, 600


@pytest.fixture(scope="module")
def recovered():
    rng = np.random.default_rng(2026)
    ell = np.arange(LMAX + 1)
    ee = np.where(ell >= 2, 50.0 / (ell + 10.0) ** 1.6, 0.0)
    bb = 0.2 * ee
    e_alm = spectra.random_alm(ee, LMAX, rng)
    b_alm = 0.3 * np.sqrt(0.2) * e_alm + np.sqrt(1 - 0.09) * spectra.random_alm(bb, LMAX, rng)  # sky EB != 0
    q0, u0 = spectra.alm2map_pol(np.stack([e_alm, b_alm]), NSIDE, LMAX)
    theta = ducc_theta(NSIDE)
    mask = (np.abs(np.pi / 2 - theta) > 20 * DEG).astype(float)
    w2, fsky_eff = spectra.mask_moments(mask)
    noise = np.array([2.0, 2.0, 1.0, 1.0, 1.5, 1.5, 4.0, 4.0]) * np.sqrt(ee[100]) * 0.5
    alms = []
    for i, a in enumerate(ALPHA):
        q, u = spectra.rotate_qu(q0, u0, a + 0.35 * DEG)
        sig = noise[i] * np.sqrt(12 * NSIDE**2 / (4 * np.pi))  # white noise with C_l^N = noise^2
        q = q + sig * rng.standard_normal(q.size)
        u = u + sig * rng.standard_normal(u.size)
        alms.append(spectra.map2alm_pol(q, u, mask, LMAX))
    cl = spectra.spectra_matrix(alms, LMAX)
    binned = spectra.bin_spectra(cl, 51, 551, 20, w2, fsky_eff)
    edges, _, d0, d1 = network.complete_complex(8)
    pa = pair_angles(binned["cl"], binned["nu"], edges)
    return pa, network.solve_network(pa["delta"], pa["cov"], d0, gauge=2), d0, d1


def ducc_theta(nside):
    import ducc0
    return ducc0.healpix.Healpix_Base(nside, "RING").pix2ang(np.arange(12 * nside * nside))[:, 0]


def test_pair_angles_recovered(recovered):
    pa, _, d0, _ = recovered
    z = (pa["delta"] - d0 @ ALPHA) / np.sqrt(np.diag(pa["cov"]))
    assert np.max(np.abs(z)) < 4.0
    # sign convention: map-space Q+iU -> (Q+iU)e^{2i alpha} must come back with the same sign
    e01 = 0
    assert np.sign(pa["delta"][e01]) == np.sign(ALPHA[0] - ALPHA[1])


def test_network_recovers_injected_pattern(recovered):
    _, net, _, _ = recovered
    truth = ALPHA - ALPHA[2]
    keep = [0, 1, 3, 4, 5, 6, 7]
    z = (net["alpha"] - truth)[keep] / np.sqrt(np.diag(net["cov"]))[keep]
    assert np.max(np.abs(z)) < 3.5
    assert net["pte"] > 1e-3


def test_common_beta_invisible(recovered):
    # 0.35 deg common rotation was injected; the differential pattern has no common mode
    _, net, _, _ = recovered
    assert abs(net["alpha"][2]) == 0.0
