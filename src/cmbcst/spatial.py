"""Region-resolved calibration network: is the per-map angle pattern the same across the sky?
(numpy only)

For R disjoint sky regions with pair angles dalpha_r (28) and covariance C_r (block-diagonal across
regions to first order), consider

    global model    dalpha_r = d0 alpha + n_r                 (one calibration pattern)
    regional model  dalpha_r = d0 (alpha + gamma_r) + n_r     (pattern varies by region)

With F_r = A^T C_r^{-1} A (A = d0 with the gauge column removed) the total residual of the global
fit splits exactly, by Pythagoras in the C^{-1} metric, into

    chi2_total   (28R - 7)  =  sum_r chi2_closure,r  (21R)  +  chi2_spatial  (7(R - 1)),
    chi2_spatial = sum_r (alpha_r - alpha_glob)^T F_r (alpha_r - alpha_glob).

chi2_closure tests triangle closure inside each region; chi2_spatial tests stationarity of the
calibration pattern across regions. A common sky rotation (isotropic or anisotropic
birefringence) is pure gauge and invisible to both. Same-frequency A/B edges compare detectors
in one band, so foregrounds cancel there: spatial variation on those edges is instrumental.
"""
from __future__ import annotations

import numpy as np

from .differential import rotation_matrix

_DEG = np.pi / 180


def _reduced(d0: np.ndarray, gauge: int) -> tuple[np.ndarray, list[int]]:
    keep = [k for k in range(d0.shape[1]) if k != gauge]
    return d0[:, keep], keep


def stationarity_split(deltas: np.ndarray, covs: np.ndarray, d0: np.ndarray, gauge: int) -> dict[str, object]:
    """Global and per-region GLS fits and the exact chi^2 split.

    deltas: (R, E) pair angles; covs: (R, E, E). Returns alphas (gauge entry 0), chi^2 values,
    their dofs and the per-region contributions to chi2_spatial.
    """
    a, keep = _reduced(d0, gauge)
    n = d0.shape[1]
    r_count, e_count = deltas.shape
    cinv = np.linalg.inv(covs)
    fis = np.einsum("ek,ref,fl->rkl", a, cinv, a)          # F_r
    rhs = np.einsum("ek,ref,rf->rk", a, cinv, deltas)
    alpha_r = np.linalg.solve(fis, rhs[..., None])[..., 0]  # per-region fits
    f_tot = fis.sum(0)
    alpha_g = np.linalg.solve(f_tot, rhs.sum(0))            # global fit
    res_r = deltas - alpha_r @ a.T
    res_g = deltas - alpha_g @ a.T
    chi_closure_r = np.einsum("re,ref,rf->r", res_r, cinv, res_r)
    chi_total = float(np.einsum("re,ref,rf->", res_g, cinv, res_g))
    diff = alpha_r - alpha_g
    chi_spatial_r = np.einsum("rk,rkl,rl->r", diff, fis, diff)
    full = lambda x: np.insert(x, gauge, 0.0, axis=-1)
    return {
        "alpha_global": full(alpha_g), "alpha_region": full(alpha_r), "gamma": full(diff),
        "cov_global": np.linalg.inv(f_tot), "fisher_region": fis,
        "chi2_total": chi_total, "dof_total": r_count * e_count - (n - 1),
        "chi2_closure": float(chi_closure_r.sum()), "dof_closure": r_count * (e_count - (n - 1)),
        "chi2_closure_region": chi_closure_r,
        "chi2_spatial": float(chi_spatial_r.sum()), "dof_spatial": (r_count - 1) * (n - 1),
        "chi2_spatial_region": chi_spatial_r,
    }


def edge_constancy(deltas: np.ndarray, sigmas: np.ndarray, edges: list[int]) -> dict[str, object]:
    """Per-edge test that a raw pair angle is the same in every region (inverse-variance mean),
    summed over the given edges, assumed mutually independent (true for disjoint-map edges such as
    the four same-frequency A/B pairs). dof = len(edges) * (R - 1)."""
    out_chi, means = [], []
    for e in edges:
        w = 1.0 / sigmas[:, e] ** 2
        m = float(np.sum(w * deltas[:, e]) / np.sum(w))
        means.append(m)
        out_chi.append(float(np.sum(w * (deltas[:, e] - m) ** 2)))
    return {"chi2": float(np.sum(out_chi)), "dof": len(edges) * (deltas.shape[0] - 1),
            "chi2_per_edge": out_chi, "mean_per_edge": means}


def same_frequency_edges(detectors: list[str], edges: list[tuple[int, int]]) -> list[int]:
    """Indices of edges joining the A and B maps of one frequency."""
    return [k for k, (i, j) in enumerate(edges) if detectors[i][:-1] == detectors[j][:-1]]


def tracer_regression(gamma: np.ndarray, fisher: np.ndarray, tracer: np.ndarray, gauge: int) -> dict[str, object]:
    """Fit gamma_r = k * t_r (one slope per map, centred tracer) by GLS with the regional Fisher
    matrices as weights; chi^2 improvement tests whether the spatial pattern follows the tracer.

    gamma: (R, n) with the gauge entry 0; fisher: (R, n-1, n-1); tracer: (R,).
    """
    keep = [k for k in range(gamma.shape[1]) if k != gauge]
    g = gamma[:, keep]
    t = tracer - np.average(tracer, weights=np.trace(fisher, axis1=1, axis2=2))
    lhs = np.einsum("r,rkl->kl", t * t, fisher)
    rhs = np.einsum("r,rkl,rl->k", t, fisher, g)
    k_hat = np.linalg.solve(lhs, rhs)
    chi_before = float(np.einsum("rk,rkl,rl->", g, fisher, g))
    resid = g - np.outer(t, k_hat)
    chi_after = float(np.einsum("rk,rkl,rl->", resid, fisher, resid))
    return {"slope": np.insert(k_hat, gauge, 0.0), "slope_cov": np.linalg.inv(lhs),
            "delta_chi2": chi_before - chi_after, "dof": len(keep)}


def null_spectra(cl: np.ndarray, alpha_deg: np.ndarray) -> np.ndarray:
    """Regional spectra with the antisymmetric part removed after derotating by the public global
    angles, then re-rotated: every pair angle equals the global pattern exactly. No regional pair
    angle is formed."""
    r_m = rotation_matrix(-alpha_deg * _DEG)
    r_p = rotation_matrix(alpha_deg * _DEG)
    s = np.einsum("ik,bkl,jl->bij", r_m, cl, r_m)
    n = len(alpha_deg)
    for i in range(n):
        for j in range(n):
            eb, be = s[:, 2 * i, 2 * j + 1], s[:, 2 * i + 1, 2 * j]
            avg = 0.5 * (eb + be)
            s[:, 2 * i, 2 * j + 1] = avg
            s[:, 2 * i + 1, 2 * j] = avg
    s = 0.5 * (s + s.transpose(0, 2, 1))
    return np.einsum("ik,bkl,jl->bij", r_p, s, r_p)


def inject(cl: np.ndarray, gamma_deg: np.ndarray) -> np.ndarray:
    r = rotation_matrix(gamma_deg * _DEG)
    return np.einsum("ik,bkl,jl->bij", r, cl, r)
