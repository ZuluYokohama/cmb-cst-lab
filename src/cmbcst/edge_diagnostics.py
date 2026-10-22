"""Covariance-aware single-edge generalized likelihood-ratio (GLR) diagnostic.

Standard GLS statistics, not a new topological estimator. Synthetic study only.
"""
from __future__ import annotations

import numpy as np


def edge_glrt(y: np.ndarray, covariance: np.ndarray, incidence: np.ndarray,
              gauge: int = 0) -> dict:
    """Profile node angles and scan for a bias on exactly one edge.

    Reported edgewise chi-square improvements are correlated; do not interpret
    the maximum without familywise calibration (see null_max_scores).
    """
    y = np.asarray(y, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    incidence = np.asarray(incidence, dtype=float)
    if incidence.shape[0] != len(y) or covariance.shape != (len(y), len(y)):
        raise ValueError("Inconsistent observation/covariance/incidence dimensions")
    if not 0 <= gauge < incidence.shape[1]:
        raise ValueError("Invalid gauge")
    if not np.allclose(covariance, covariance.T, atol=1e-10, rtol=1e-10):
        raise ValueError("Covariance must be symmetric")
    np.linalg.cholesky(covariance)
    a = np.delete(incidence, gauge, axis=1)
    if np.linalg.matrix_rank(a) != a.shape[1]:
        raise ValueError("Incidence is not full rank after fixing the gauge")
    inverse = np.linalg.solve(covariance, np.eye(len(y)))
    ia = inverse @ a
    p = inverse - ia @ np.linalg.solve(a.T @ ia, ia.T)
    p = (p + p.T) / 2
    diagonal = np.diag(p)
    if np.any(diagonal <= 0):
        raise ValueError("One or more edge-specific biases are not identifiable")
    score = p @ y
    return {
        "delta_chi2": score**2 / diagonal,
        "amplitude": score / diagonal,
        "sigma": 1 / np.sqrt(diagonal),
        "index": int(np.argmax(score**2 / diagonal)),
        "precision_residual": p,
    }


def null_max_scores(covariance: np.ndarray, incidence: np.ndarray, n_sims: int = 2000,
                    seed: int = 2317, gauge: int = 0) -> np.ndarray:
    """Gaussian max-score null; use independent data for calibration checks."""
    if n_sims < 1:
        raise ValueError("n_sims must be >= 1")
    p = edge_glrt(np.zeros(len(covariance)), covariance, incidence, gauge)["precision_residual"]
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n_sims, len(covariance))) @ np.linalg.cholesky(covariance).T
    scores = noise @ p.T
    return np.max(scores**2 / np.diag(p), axis=1)


def monte_carlo_pvalue(observed_max: float, calibration_maxima: np.ndarray) -> float:
    """Conservative nonzero simulated upper-tail p value."""
    if not np.isfinite(observed_max) or len(calibration_maxima) < 1:
        raise ValueError("Invalid statistic or empty calibration distribution")
    return (1 + int(np.count_nonzero(calibration_maxima >= observed_max))) / (len(calibration_maxima) + 1)
