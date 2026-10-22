"""Post-hoc covariance-aware test of the four same-frequency A/B pair angles.

This compares the existing diagonal-edge approximation with a generalized
least-squares fit retaining covariance BETWEEN the four same-frequency edges
WITHIN each region. It continues to assume independence BETWEEN regions and
therefore is not a complete map-level covariance correction.

This is conventional GLS, not a new CMB estimator or a preregistered test.
"""
from __future__ import annotations

import numpy as np


def same_frequency_joint_gls(deltas: np.ndarray, covs: np.ndarray,
                             edge_indices: list[int]) -> dict[str, object]:
    """Test constancy of m selected pair angles across R regions.

    Region r has measurement vector y_r and m-by-m covariance C_r. Under H0
    all regions share one unknown vector mu. Independence between regions
    is assumed. The statistic is sum_r (y_r-mu)^T C_r^-1 (y_r-mu),
    with nominal m*(R-1) degrees of freedom for fixed known covariances.

    Covariances must include correlations between selected edges; this
    function DOES NOT model between-region correlations or estimation noise
    in the covariance itself.
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    covs = np.asarray(covs, dtype=np.float64)
    if deltas.ndim != 2:
        raise ValueError("deltas must be a region-by-edge matrix")
    r, n = deltas.shape
    if r < 2 or n < 1 or covs.shape != (r, n, n):
        raise ValueError("deltas and covariance dimensions disagree")
    edges = list(edge_indices)
    if not edges or len(set(edges)) != len(edges) or any(not 0 <= e < n for e in edges):
        raise ValueError("Invalid or duplicate selected edge")
    sub = covs[:, edges][:, :, edges]
    if not np.all(np.isfinite(sub)) or not np.all(np.isfinite(deltas[:, edges])):
        raise ValueError("Non-finite values")
    if not np.allclose(sub, sub.transpose(0, 2, 1), rtol=1e-10, atol=1e-10):
        raise ValueError("Covariance must be symmetric")
    try:
        inv = np.linalg.solve(sub, np.broadcast_to(np.eye(len(edges)), sub.shape))
    except np.linalg.LinAlgError as exc:
        raise ValueError("Covariance must be invertible") from exc
    try:
        np.linalg.cholesky(sub)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Covariance must be positive definite") from exc
    y = deltas[:, edges]
    information = np.sum(inv, axis=0)
    rhs = np.einsum("rij,rj->i", inv, y)
    mu = np.linalg.solve(information, rhs)
    residual = y - mu
    by_region = np.einsum("ri,rij,rj->r", residual, inv, residual)
    diag = np.sqrt(np.diagonal(sub, axis1=1, axis2=2))
    correlation = sub / diag[:, :, None] / diag[:, None, :]
    off = correlation - np.eye(len(edges))[None, :, :]
    return {
        "chi2": float(np.sum(by_region)),
        "dof": len(edges) * (r - 1),
        "mean_per_edge": mu,
        "cov_mean": np.linalg.inv(information),
        "chi2_per_region": by_region,
        "max_abs_within_region_edge_correlation": float(np.max(np.abs(off))),
        "n_regions": r,
        "n_edges": len(edges),
        "assumptions": "block diagonal across regions; full covariance among selected edges within a region",
        "analysis_status": "post-hoc diagnostic: cannot retroactively be preregistered",
    }
