"""Birefringence-blind pair angles from binned cross-spectra (numpy only).

Fields are indexed f = 2*i + X with X = 0 for E and 1 for B of map i. If map i is rotated by
alpha_i (E+iB -> (E+iB) e^{2i alpha_i}), then for i != j

    D_ij = C^{B_i E_j} - C^{E_i B_j} = tan(2 (alpha_i - alpha_j)) * S_ij,
    S_ij = C^{E_i E_j} + C^{B_i B_j},

for any sky EE, BB and intrinsic EB, so a common rotation (beta) and the sky's own EB drop out.
Beams and pixel windows multiply D and S alike and cancel in the ratio. Each pair angle is a
weighted least-squares fit over bins with Gaussian (Knox) covariance from the measured spectra,
and the 28x28 covariance of the pair angles is propagated linearly through those fits.
"""
from __future__ import annotations

import numpy as np


def field(i: int, x: str) -> int:
    return 2 * i + (0 if x == "E" else 1)


def residual_weights(i: int, j: int, t: float, nf: int) -> np.ndarray:
    """W with r = sum_ab W_ab C_ab = D_ij - t S_ij."""
    w = np.zeros((nf, nf))
    w[field(i, "B"), field(j, "E")] += 1.0
    w[field(i, "E"), field(j, "B")] -= 1.0
    w[field(i, "E"), field(j, "E")] -= t
    w[field(i, "B"), field(j, "B")] -= t
    return w


def gaussian_cov(wa: np.ndarray, wb: np.ndarray, c: np.ndarray, nu: float) -> float:
    """Cov(sum W^a C, sum W^b C) for Gaussian fields: Cov(C_ab, C_cd) = (C_ac C_bd + C_ad C_bc)/nu."""
    return float(np.sum(wa * (c @ (wb + wb.T) @ c)) / nu)


def pair_spectra(cl: np.ndarray, i: int, j: int) -> tuple[np.ndarray, np.ndarray]:
    """D_ij and S_ij per bin from cl[bin, field, field]."""
    d = cl[:, field(i, "B"), field(j, "E")] - cl[:, field(i, "E"), field(j, "B")]
    s = cl[:, field(i, "E"), field(j, "E")] + cl[:, field(i, "B"), field(j, "B")]
    return d, s


def fit_pair(cl: np.ndarray, nu: np.ndarray, i: int, j: int, n_iter: int = 6) -> dict[str, object]:
    """Iteratively reweighted fit of D = t S over bins; delta = arctan(t)/2 in radians."""
    d, s = pair_spectra(cl, i, j)
    t = 0.0
    for _ in range(n_iter):
        var = residual_cov(cl, nu, [(i, j)], np.array([t]))[:, 0, 0]
        w = 1.0 / var
        t = float(np.sum(w * d * s) / np.sum(w * s * s))
    g = w * s / np.sum(w * s * s)  # d t_hat / d r_b
    return {"delta": 0.5 * np.arctan(t), "t": t, "g": g, "var_r": var, "kappa": 0.5 / (1.0 + t * t),
            "chi2": float(np.sum(w * (d - t * s) ** 2)), "dof": len(d) - 1}


def residual_cov(cl: np.ndarray, nu: np.ndarray, edges: list[tuple[int, int]], t: np.ndarray) -> np.ndarray:
    """Cov(r_e,b, r_f,b) for all pairs e, f and bins b, shape (nb, ne, ne).

    Each residual r_e = D_e - t_e S_e touches four spectra C[a_k, b_k] with coefficients c_k,
    so the Gaussian formula is evaluated on those entries directly (same result as
    `gaussian_cov` on the dense W matrices, without the 16x16 products).
    """
    a = np.array([[field(i, "B"), field(i, "E"), field(i, "E"), field(i, "B")] for i, _ in edges])
    b = np.array([[field(j, "E"), field(j, "B"), field(j, "E"), field(j, "B")] for _, j in edges])
    c = np.stack([np.ones(len(edges)), -np.ones(len(edges)), -t, -t], axis=1)
    ak, bk = a[:, :, None, None], b[:, :, None, None]
    al, bl = a[None, None], b[None, None]
    g = cl[:, ak, al] * cl[:, bk, bl] + cl[:, ak, bl] * cl[:, bk, al]  # (nb, ne, 4, ne, 4)
    cov = np.einsum("ek,fl,xekfl->xef", c, c, g)
    return cov / nu[:, None, None]


def pair_angles(cl: np.ndarray, nu: np.ndarray, edges: list[tuple[int, int]]) -> dict[str, object]:
    """All pair angles (radians) and their joint covariance, including pairs sharing a map."""
    fits = [fit_pair(cl, nu, i, j) for (i, j) in edges]
    t = np.array([f["t"] for f in fits])
    g = np.array([f["g"] for f in fits])  # (ne, nb)
    rc = residual_cov(cl, nu, edges, t)
    cov = np.einsum("eb,fb,bef->ef", g, g, rc)
    kap = np.array([f["kappa"] for f in fits])
    cov = cov * np.outer(kap, kap)
    return {"delta": np.array([f["delta"] for f in fits]), "cov": cov,
            "pair_chi2": np.array([f["chi2"] for f in fits]), "pair_dof": fits[0]["dof"]}


def rotation_matrix(alpha: np.ndarray) -> np.ndarray:
    """Block-diagonal (2n x 2n) map rotation acting on (E_i, B_i): E' = cE - sB, B' = sE + cB."""
    n = len(alpha)
    r = np.zeros((2 * n, 2 * n))
    for i, a in enumerate(alpha):
        c, s = np.cos(2 * a), np.sin(2 * a)
        r[2 * i:2 * i + 2, 2 * i:2 * i + 2] = [[c, -s], [s, c]]
    return r
