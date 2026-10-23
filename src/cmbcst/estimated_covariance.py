"""Finite-calibration closure statistics for connected scalar sensor graphs.

A held-out cycle vector c=Z@y is N(0,Sigma_cycle) under the additive-offset
Gaussian null. Estimate its covariance from K INDEPENDENT calibration vectors
using the usual centered sample covariance. For q independent cycles:

    Q = c.T @ S_hat^-1 @ c
    ((K-q)/(q*(K-1))) * Q ~ F(q, K-q)

The test observation must NOT also occur among calibration observations.
This numpy-only kernel returns F parameters; scipy.stats.f.sf computes p.
"""
from __future__ import annotations
import numpy as np
from .network import chi2_sf
from .sparse_network import graph_design


def calibrated_cycle_statistic(measurement: np.ndarray, calibration: np.ndarray,
                               n_sensors: int, edges: list[tuple[int, int]],
                               gauge: int = 0) -> dict[str, object]:
    """Cycle statistic corrected for independent finite Gaussian calibration.

    measurement: (m,) held-out edge pair observations.
    calibration: (K,m) independent observations with the same stationary
      zero-mean Gaussian noise law (node offsets may differ by observation).

    The projected closure has theoretical mean zero because Z@A=0.
    We do NOT subtract the training sample mean from the held-out vector.
    """
    _, z = graph_design(n_sensors, edges, root=gauge)
    m = len(edges)
    obs = np.asarray(measurement, dtype=float)
    cal = np.asarray(calibration, dtype=float)
    if obs.shape != (m,) or cal.ndim != 2 or cal.shape[1] != m:
        raise ValueError("measurement and calibration shapes must be (m,) and (K,m)")
    if not (np.all(np.isfinite(obs)) and np.all(np.isfinite(cal))):
        raise ValueError("all observations must be finite")
    k = len(cal)
    q = len(z)
    if q == 0:
        return {
            "cycles_available": False, "cycle_dof": 0,
            "calibration_count": k, "closure": np.zeros(0),
            "cycle_covariance": np.zeros((0, 0)),
            "chi2_statistic": None, "naive_chi2_pte": None,
            "f_statistic": None, "f_numerator_dof": None,
            "f_denominator_dof": None,
        }
    if k < q + 3:
        raise ValueError("too few independent calibration observations for stable F correction")
    closure = z @ obs
    train = cal @ z.T
    centered = train - train.mean(axis=0)
    empirical = centered.T @ centered / (k - 1)
    try:
        chol = np.linalg.cholesky(empirical)
    except np.linalg.LinAlgError as exc:
        raise ValueError("calibration cycle covariance is not positive definite") from exc
    projected = np.linalg.solve(chol, closure)
    score = float(projected @ projected)
    scale = (k - q) / (q * (k - 1))
    return {
        "cycles_available": True,
        "cycle_dof": q, "calibration_count": k,
        "closure": closure, "cycle_covariance": empirical,
        "chi2_statistic": score, "naive_chi2_pte": chi2_sf(score, q),
        "f_statistic": scale * score,
        "f_numerator_dof": q, "f_denominator_dof": k - q,
    }
