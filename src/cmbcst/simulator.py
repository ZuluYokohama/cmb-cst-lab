"""Bandpower-level forward model for simulation-based inference of (beta, dust EB) (numpy only).

Fields are f = 2*i + (E=0, B=1) for maps i. For parameters

    theta = (alpha_1..8, beta, psi0, n_psi, psi1, A_d, beta_d)        angles in degrees

the expected spectra in bin b are

    Sigma_b = R(alpha) [ R(beta) CMB_b R(beta)^T  +  DUST_b(psi, A_d, beta_d) ] R(alpha)^T + N_b,

where CMB_b holds the beam/pixel-window/mask-transfer smoothed LCDM EE, BB per map pair, DUST_b
holds s_i s_j A_d D_b^{EE,BB} w_ij (SED s, measured 353 GHz dust spectra D, window ratio w), with
the dust of band i intrinsically rotated by its misalignment angle:

    DUST_b,ij = R(psi_i,b) diag(EE_ij, BB_ij) R(psi_j,b)^T,
    psi_i,b   = psi0 (l_b / 200)^n_psi + psi1 ln(nu_i / 353 GHz).

For equal angles this is the familiar EB = 1/2 sin(4 psi) (EE - BB); for psi_i != psi_j the
E_iB_j and B_iE_j entries differ, so a frequency-dependent misalignment enters the antisymmetric
(calibration-sensitive) spectra exactly like a calibration difference.

M0 (no dust EB) is psi0 = psi1 = 0; M1 constant misalignment is n_psi = psi1 = 0; M2 scale-dependent
is psi1 = 0; M3 frequency-decorrelated is n_psi = 0. Bandpowers are exact Wishart draws via the
Bartlett decomposition (136 random numbers per bin, independent of the mode count nu_b).
"""
from __future__ import annotations

import numpy as np

from .differential import rotation_matrix

H_OVER_K = 0.0479924  # h / k_B in K / GHz
T_CMB = 2.7255
FREQ_GHZ = np.array([100.0, 100.0, 143.0, 143.0, 217.0, 217.0, 353.0, 353.0])
PARAMS = ("a100A", "a100B", "a143A", "a143B", "a217A", "a217B", "a353A", "a353B",
          "beta", "psi0", "n_psi", "psi1", "A_d", "beta_d")
L_PIVOT = 200.0


def mbb_sed(nu_ghz: np.ndarray, beta_d: float, t_d: float = 19.6, nu0: float = 353.0) -> np.ndarray:
    """Modified blackbody in K_CMB units, normalised to 1 at nu0."""
    def k_cmb(nu):
        x = H_OVER_K * nu / T_CMB
        rj_to_cmb = (np.expm1(x) ** 2) / (x**2 * np.exp(x))
        return nu ** (beta_d + 1.0) / np.expm1(H_OVER_K * nu / t_d) * rj_to_cmb
    return k_cmb(np.asarray(nu_ghz, dtype=float)) / k_cmb(nu0)


def dust_angles(psi0: float, n_psi: float, psi1: float, ell: np.ndarray) -> np.ndarray:
    """Intrinsic dust misalignment psi_i,b of each map's band, in radians, shape (nb, 8)."""
    lnu = np.log(FREQ_GHZ / 353.0)
    scale = psi0 * (ell / L_PIVOT) ** n_psi
    return np.radians(scale[:, None] + psi1 * lnu[None, :])


def model_covariance(theta: np.ndarray, fixed: dict) -> np.ndarray:
    """Sigma_b for all bins, shape (nb, 16, 16). `fixed` holds th_ee, th_bb (nb,8,8), dust_ee,
    dust_bb (nb,) at 353 A x B, wratio (nb,8,8), noise_ee, noise_bb (nb,8) and ell (nb,)."""
    alpha, beta = np.radians(theta[:8]), np.radians(theta[8])
    psi0, n_psi, psi1, a_d, beta_d = theta[9:14]
    nb = fixed["ell"].size
    s = mbb_sed(FREQ_GHZ, beta_d)
    c, sn = np.cos(2 * beta), np.sin(2 * beta)
    ee_c, bb_c = fixed["th_ee"], fixed["th_bb"]
    sky = np.zeros((nb, 16, 16))
    sky[:, 0::2, 0::2] = c * c * ee_c + sn * sn * bb_c                       # common rotation by beta
    sky[:, 1::2, 1::2] = sn * sn * ee_c + c * c * bb_c
    sky[:, 0::2, 1::2] = sky[:, 1::2, 0::2] = c * sn * (ee_c - bb_c)
    ss = np.outer(s, s)[None] * fixed["wratio"] * a_d
    ee_d = ss * fixed["dust_ee"][:, None, None]
    bb_d = ss * fixed["dust_bb"][:, None, None]
    # Dust in band i is intrinsically rotated by psi_i: block_ij = R(psi_i) diag(EE, BB) R(psi_j)^T.
    # For psi_i != psi_j the E_iB_j and B_iE_j entries differ, so frequency-dependent misalignment
    # enters the antisymmetric spectra exactly like a calibration difference.
    psi = dust_angles(psi0, n_psi, psi1, fixed["ell"])
    c, sn2 = np.cos(2 * psi), np.sin(2 * psi)
    ci, cj, si, sj = c[:, :, None], c[:, None, :], sn2[:, :, None], sn2[:, None, :]
    sky[:, 0::2, 0::2] += ci * cj * ee_d + si * sj * bb_d
    sky[:, 1::2, 1::2] += si * sj * ee_d + ci * cj * bb_d
    sky[:, 0::2, 1::2] += ci * sj * ee_d - si * cj * bb_d   # E_i B_j
    sky[:, 1::2, 0::2] += si * cj * ee_d - ci * sj * bb_d   # B_i E_j
    r = rotation_matrix(alpha)
    out = np.einsum("ik,bkl,jl->bij", r, sky, r)
    idx = np.arange(8)
    out[:, 2 * idx, 2 * idx] += fixed["noise_ee"]
    out[:, 2 * idx + 1, 2 * idx + 1] += fixed["noise_bb"]
    return out


def bartlett_draws(sigma: np.ndarray, dof: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """n exact Wishart(Sigma_b, nu_b)/nu_b draws per bin: (n, nb, d, d)."""
    nb, d, _ = sigma.shape
    w, v = np.linalg.eigh(sigma)
    root = v * np.sqrt(np.clip(w, 1e-12 * w.max(axis=1, keepdims=True), None))[:, None, :]
    a = np.zeros((n, nb, d, d))
    k = np.arange(d)
    a[..., k, k] = np.sqrt(rng.chisquare(dof[None, :, None] - k[None, None, :], size=(n, nb, d)))
    il = np.tril_indices(d, -1)
    a[..., il[0], il[1]] = rng.standard_normal((n, nb, il[0].size))
    la = np.einsum("bij,nbjk->nbik", root, a)
    return np.einsum("nbik,nbjk->nbij", la, la) / dof[None, :, None, None]


def upper(c: np.ndarray) -> np.ndarray:
    """Unique entries (upper triangle incl. diagonal) of (..., d, d) -> (..., d(d+1)/2)."""
    iu = np.triu_indices(c.shape[-1])
    return c[..., iu[0], iu[1]]


def gaussian_cov_upper(sigma: np.ndarray, dof: np.ndarray) -> np.ndarray:
    """Gaussian covariance of the unique bandpowers per bin: (nb, m, m),
    Cov(C_ab, C_cd) = (S_ac S_bd + S_ad S_bc) / nu."""
    iu = np.triu_indices(sigma.shape[-1])
    a, b = iu
    s = sigma
    cov = s[:, a[:, None], a[None, :]] * s[:, b[:, None], b[None, :]] + s[:, a[:, None], b[None, :]] * s[:, b[:, None], a[None, :]]
    return cov / dof[:, None, None]


class ScoreCompressor:
    """MOPED-style score compression at a fiducial theta: t = dmu^T C^{-1} (d - mu), one number
    per parameter, with dmu by central differences and C the Gaussian bandpower covariance."""

    def __init__(self, theta0: np.ndarray, fixed: dict, dof: np.ndarray, steps: np.ndarray):
        self.theta0 = np.asarray(theta0, dtype=float)
        sig0 = model_covariance(self.theta0, fixed)
        self.mu = upper(sig0)                                   # (nb, m)
        self.cinv = np.linalg.inv(gaussian_cov_upper(sig0, dof))
        grads = []
        for p, h in enumerate(steps):
            tp, tm = self.theta0.copy(), self.theta0.copy()
            tp[p] += h
            tm[p] -= h
            grads.append((upper(model_covariance(tp, fixed)) - upper(model_covariance(tm, fixed))) / (2 * h))
        self.dmu = np.stack(grads, axis=-1)                     # (nb, m, P)
        self.cinv_dmu = np.einsum("bmn,bnp->bmp", self.cinv, self.dmu)
        self.fisher = np.einsum("bmp,bmq->pq", self.dmu, self.cinv_dmu)

    def __call__(self, d_upper: np.ndarray) -> np.ndarray:
        """Compress (..., nb, m) bandpowers to (..., P) scores."""
        return np.einsum("bmp,...bm->...p", self.cinv_dmu, d_upper - self.mu)
