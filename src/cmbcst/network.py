"""The pairwise calibration network as a cochain complex (numpy only).

Maps are the 8 vertices of the complete graph K_8, pairs are its 28 edges and the
56 triangles fill in the 2-skeleton of the 7-simplex. Convention (matches the paper):

    Delta alpha_ij = alpha_i - alpha_j      for i < j,
    (d0 x)_e = x_i - x_j,                   so the paper's incidence matrix A is d0,
    (d1 w)_t = w_ij + w_jk - w_ik           for triangle t = (i, j, k), i < j < k.

d1 d0 = 0, ker d0 = span{1} (the common rotation: beta plus any shared offset), and because
every triangle is filled, ker d1 = im d0. So H^1 = 0: there is no harmonic component, and
anything outside im d0 is a failure of triangle closure.

The fit uses the edge covariance C, so the decomposition is orthogonal in the C^{-1} inner
product. The GLS residual r obeys d0^T C^{-1} r = 0, i.e. C^{-1} r lies in ker d0^T = im d1^T:
r is a *covariance-weighted* coexact component (in im d1^T only if C is proportional to the
identity). The triangle closures d1 dalpha are the metric-free statement of inconsistency, and
their weighted chi^2 equals the residual chi^2 of the gauge-fixed fit (see `closure_chi2`).
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np


def complete_complex(n: int = 8) -> tuple[list[tuple[int, int]], list[tuple[int, int, int]], np.ndarray, np.ndarray]:
    """Edges, triangles, d0 (n_edges x n) and d1 (n_tri x n_edges) of the filled complete graph."""
    edges = list(combinations(range(n), 2))
    tris = list(combinations(range(n), 3))
    e_idx = {e: k for k, e in enumerate(edges)}
    d0 = np.zeros((len(edges), n))
    for k, (i, j) in enumerate(edges):
        d0[k, i], d0[k, j] = 1.0, -1.0
    d1 = np.zeros((len(tris), len(edges)))
    for t, (i, j, k) in enumerate(tris):
        d1[t, e_idx[(i, j)]] = 1.0
        d1[t, e_idx[(j, k)]] = 1.0
        d1[t, e_idx[(i, k)]] = -1.0
    return edges, tris, d0, d1


def betti_numbers(d0: np.ndarray, d1: np.ndarray, tol: float = 1e-10) -> dict[str, int]:
    """Ranks and Betti numbers of C^0 -> C^1 -> C^2 (b2 counts only the 2-skeleton)."""
    r0 = int(np.linalg.matrix_rank(d0, tol))
    r1 = int(np.linalg.matrix_rank(d1, tol))
    n0, n1, n2 = d0.shape[1], d0.shape[0], d1.shape[0]
    return {"rank_d0": r0, "rank_d1": r1, "b0": n0 - r0, "b1": n1 - r0 - r1, "b2_skeleton": n2 - r1}


def solve_network(dalpha: np.ndarray, cov: np.ndarray, d0: np.ndarray, gauge: int) -> dict[str, object]:
    """Generalised least squares for d = d0 x + n with x[gauge] = 0.

    Returns the per-map angles, their covariance (zero row/column at the gauge map),
    the residual chi^2 with n_edges - (n - 1) degrees of freedom, and the gradient
    (im d0) and curl (im d1^T) parts of the data in the C^{-1} inner product.
    """
    n = d0.shape[1]
    keep = [k for k in range(n) if k != gauge]
    a = d0[:, keep]
    cinv = np.linalg.inv(cov)
    fisher = a.T @ cinv @ a
    cov_red = np.linalg.inv(fisher)
    x_red = cov_red @ (a.T @ cinv @ dalpha)
    x = np.zeros(n)
    x[keep] = x_red
    cov_x = np.zeros((n, n))
    cov_x[np.ix_(keep, keep)] = cov_red
    grad = d0 @ x
    curl = dalpha - grad
    chi2 = float(curl @ cinv @ curl)
    dof = d0.shape[0] - len(keep)
    # The curl is a residual of a fit to the same data: Cov = C - A Cov(x) A^T, not C
    cov_curl = cov - a @ cov_red @ a.T
    return {"alpha": x, "cov": cov_x, "chi2": chi2, "dof": dof, "pte": chi2_sf(chi2, dof),
            "gradient": grad, "curl": curl, "cov_curl": cov_curl}


def closures(dalpha: np.ndarray, cov: np.ndarray, d1: np.ndarray) -> dict[str, np.ndarray]:
    """Triangle closures c = d1 dalpha with propagated errors and per-triangle pulls."""
    c = d1 @ dalpha
    cov_c = d1 @ cov @ d1.T
    sig = np.sqrt(np.diag(cov_c))
    return {"closure": c, "sigma": sig, "z": c / sig, "cov": cov_c}


def closure_chi2(dalpha: np.ndarray, cov: np.ndarray, d1: np.ndarray, tol: float = 1e-10) -> tuple[float, int]:
    """chi^2 of all triangle closures jointly, c^T (d1 C d1^T)^+ c, and its rank.

    For a filled complete graph this equals `solve_network(...)["chi2"]`: both measure the
    component of the data outside im d0 = ker d1.
    """
    c = d1 @ dalpha
    cov_c = d1 @ cov @ d1.T
    w, v = np.linalg.eigh(cov_c)
    keep = w > tol * w.max()
    proj = v[:, keep].T @ c
    return float(np.sum(proj**2 / w[keep])), int(keep.sum())


def centering(n: int) -> np.ndarray:
    """Projector onto the (n-1)-dimensional differential subspace (removes the common mode)."""
    return np.eye(n) - np.full((n, n), 1.0 / n)


def chi2_diff(q_rel: np.ndarray, q_mk: np.ndarray, c_rel: np.ndarray, c_mk: np.ndarray,
              c_cross: np.ndarray | None = None, tol: float = 1e-10) -> dict[str, object]:
    """Paper eq. for comparing two per-map angle patterns on the differential subspace.

    dq = P (q_rel - q_mk), C_dq = P (C_rel + C_mk - C_x - C_x^T) P, chi^2 = dq^T C_dq^+ dq
    with dof = rank(C_dq) (7 for 8 maps). `c_cross` = Cov(q_rel, q_mk), zero if omitted.
    """
    n = len(q_rel)
    p = centering(n)
    cx = np.zeros((n, n)) if c_cross is None else c_cross
    dq = p @ (q_rel - q_mk)
    cdq = p @ (c_rel + c_mk - cx - cx.T) @ p
    w, v = np.linalg.eigh(cdq)
    keep = w > tol * w.max()
    pinv = (v[:, keep] / w[keep]) @ v[:, keep].T
    chi2 = float(dq @ pinv @ dq)
    dof = int(keep.sum())
    return {"dq": dq, "cov": cdq, "chi2": chi2, "dof": dof, "pte": chi2_sf(chi2, dof),
            "pulls": dq / np.sqrt(np.diag(cdq))}


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x, y = x - x.mean(), y - y.mean()
    return float(x @ y / math.sqrt((x @ x) * (y @ y)))


def chi2_sf(x: float, k: int) -> float:
    """Survival function of chi^2_k, via the regularised upper incomplete gamma Q(k/2, x/2)."""
    if x <= 0:
        return 1.0
    a, z = 0.5 * k, 0.5 * x
    if z < a + 1.0:  # series for P, then Q = 1 - P
        term = total = 1.0 / a
        ap = a
        for _ in range(1000):
            ap += 1.0
            term *= z / ap
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return max(0.0, 1.0 - total * math.exp(-z + a * math.log(z) - math.lgamma(a)))
    # Lentz continued fraction for Q
    tiny = 1e-300
    b = z + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h * math.exp(-z + a * math.log(z) - math.lgamma(a))
