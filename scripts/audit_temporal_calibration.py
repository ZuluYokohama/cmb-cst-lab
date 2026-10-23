"""Synthetic temporal-dependence negative control for finite-F cycle closure.

Usage: python scripts/audit_temporal_calibration.py --trials 2400 --seed 2026

A fixed edge graph has q=3 cycles. Calibration is either independent standard
Gaussian edge noise or stationary AR(1) edge noise with exactly the SAME
per-time marginal covariance, identity. The held-out test measurement stays
independent in both arms to isolate the effects of training-row dependence.
A scaled-F reference is exact in the independent arm, not in the AR(1) arm.

Not a correction, a detector, or an empirical field-sensor benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import f

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cmbcst.estimated_covariance import calibrated_cycle_statistic
from cmbcst.sparse_network import graph_design

EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (1, 4), (0, 2)]


def _scores(calibration: np.ndarray, observed: np.ndarray,
            cycles: np.ndarray) -> np.ndarray:
    """Batched scaled F scores, with centered sample cycle covariance."""
    trials, k, _ = calibration.shape
    q = cycles.shape[0]
    if observed.shape[0] != trials:
        raise ValueError("test/calibration trial count differs")
    projected = calibration @ cycles.T
    centered = projected - projected.mean(axis=1, keepdims=True)
    covs = np.einsum("tki,tkj->tij", centered, centered) / (k - 1)
    heldout = observed @ cycles.T
    white = np.linalg.solve(np.linalg.cholesky(covs), heldout[..., None])[..., 0]
    qstat = np.sum(white ** 2, axis=1)
    return (k - q) * qstat / (q * (k - 1))


def audit(trials: int = 2400, seed: int = 2026, calibration_count: int = 16,
          lag_correlation: float = 0.8) -> dict[str, object]:
    """Compare paired independent and AR(1) calibration at fixed marginal noise.

    AR(1) recursion is x_t = rho*x_(t-1)+sqrt(1-rho^2)*e_t, with
    x_0=e_0. Every t has N(0,I) marginal, but samples are dependent.
    Independent held-out test draws are shared across both arms.
    """
    if isinstance(trials, bool) or not isinstance(trials, int) or not 200 <= trials <= 30000:
        raise ValueError("trials must be an integer in [200, 30000]")
    if (isinstance(calibration_count, bool) or not isinstance(calibration_count, int)
            or not 6 <= calibration_count <= 1000):
        raise ValueError("calibration_count must be an integer in [6, 1000]")
    if isinstance(lag_correlation, bool) or not np.isfinite(lag_correlation) or not 0 <= lag_correlation < 1:
        raise ValueError("lag_correlation must be a finite number in [0, 1)")
    rho = float(lag_correlation)
    a, z = graph_design(5, EDGES)
    q, m = z.shape
    if q != 3 or not np.allclose(z @ a, 0):
        raise AssertionError("unexpected cycle space for fixed reference graph")

    rng = np.random.default_rng(seed)
    innovations = rng.standard_normal((trials, calibration_count, m))
    correlated = np.empty_like(innovations)
    correlated[:, 0, :] = innovations[:, 0, :]
    fresh_scale = np.sqrt(1 - rho * rho)
    for index in range(1, calibration_count):
        correlated[:, index, :] = (rho * correlated[:, index - 1, :]
                                 + fresh_scale * innovations[:, index, :])
    heldout = rng.standard_normal((trials, m))
    iid_score = _scores(innovations, heldout, z)
    correlated_score = _scores(correlated, heldout, z)
    for training, scores in ((innovations, iid_score), (correlated, correlated_score)):
        for t in range(min(3, trials)):
            reference = calibrated_cycle_statistic(heldout[t], training[t], 5, EDGES)
            if not np.isclose(reference["f_statistic"], scores[t], rtol=1e-9, atol=1e-9):
                raise AssertionError("batched F statistic differs from shipped kernel")

    denominator_df = calibration_count - q
    iid_pte = f.sf(iid_score, q, denominator_df)
    correlated_pte = f.sf(correlated_score, q, denominator_df)
    # An empirical correlation descriptor is not an effective sample size
    # estimator. No effective-K correction is applied or implied here.
    return {
        "model": "paired Gaussian edge-noise calibration, same N(0,I) marginal",
        "seed": seed,
        "trials": trials,
        "calibration_count": calibration_count,
        "cycle_dof": q,
        "lag_one_ar1_rho": rho,
        "heldout_test_independent_of_calibration": True,
        "nominal_false_alarm_fraction": 0.05,
        "iid_calibration_false_alarm_fraction": float(np.mean(iid_pte < 0.05)),
        "temporally_correlated_calibration_false_alarm_fraction": float(np.mean(correlated_pte < 0.05)),
        "correction_applied": "none: F reference is knowingly invalid for correlated calibration",
        "limits": "synthetic Gaussian AR(1) dependence; no real sensor, independent attestation, or corrected correlated-series test",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--calibration-count", type=int, default=16)
    parser.add_argument("--lag-correlation", type=float, default=0.8)
    opts = parser.parse_args()
    print(json.dumps(audit(opts.trials, opts.seed, opts.calibration_count, opts.lag_correlation), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
