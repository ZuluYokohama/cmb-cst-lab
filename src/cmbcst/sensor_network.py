"""Experimental scalar relative-offset calibration for complete sensor networks.

This is a transfer demonstration of the CMB pair-angle cochain/GLS pattern.
Measurements must be additive scalar differences: y_ij = a_i - a_j + noise.
It does NOT implement SO(3) orientation, clocks with wrapping, nonlinear
sensor physics, fault attribution, or operational spoofing detection.
"""
from __future__ import annotations

import numpy as np

from .network import complete_complex, solve_network, closure_chi2


def fit_scalar_offsets(pair_offsets: np.ndarray, covariance: np.ndarray,
                       n_sensors: int, gauge: int = 0) -> dict[str, object]:
    """Fit scalar differences with a fixed gauge and covariance-aware closures.

    Edges are ordered as itertools.combinations(range(n_sensors), 2).
    Assumes a connected *complete* graph, Gaussian errors, and a known,
    symmetric positive-definite covariance (including cross-edge covariance).
    The result reports inconsistency but cannot uniquely identify faulty sensors.
    Returns offset estimates relative to the gauge, their covariance,
    the GLS residual statistic, and the equivalent triangle-closure statistic.
    """
    if isinstance(n_sensors, bool) or not isinstance(n_sensors, int) or n_sensors < 3:
        raise ValueError("n_sensors must be an integer >= 3")
    if isinstance(gauge, bool) or not isinstance(gauge, int) or not 0 <= gauge < n_sensors:
        raise ValueError("gauge must be a valid sensor index")
    edges, triangles, d0, d1 = complete_complex(n_sensors)
    n_edges = len(edges)
    values = np.asarray(pair_offsets, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    if values.shape != (n_edges,):
        raise ValueError(f"expected {n_edges} offsets in complete-graph edge order")
    if cov.shape != (n_edges, n_edges):
        raise ValueError(f"expected covariance shape ({n_edges}, {n_edges})")
    if not (np.isfinite(values).all() and np.isfinite(cov).all()):
        raise ValueError("measurements and covariance must be finite")
    if not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-12 * max(1.0, np.max(np.abs(cov)))):
        raise ValueError("covariance must be symmetric")
    if not np.array_equal(cov, cov.T):
        cov = (cov + cov.T) / 2.0
    try:
        np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be positive definite") from exc

    fit = solve_network(values, cov, d0, gauge=gauge)
    closure_value, closure_dof = closure_chi2(values, cov, d1)
    expected_dof = n_edges - n_sensors + 1
    if closure_dof != expected_dof:
        raise ValueError("covariance is too ill-conditioned for reliable closure rank")
    return {
        "edges": edges,
        "triangles": triangles,
        "gauge": gauge,
        "offsets": fit["alpha"],
        "offset_covariance": fit["cov"],
        "residual": fit["curl"],
        "closure": d1 @ values,
        "residual_chi2": fit["chi2"],
        "residual_dof": fit["dof"],
        "pte": fit["pte"],
        "closure_chi2": closure_value,
        "closure_dof": closure_dof,
    }
