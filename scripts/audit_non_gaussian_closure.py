"""Synthetic negative control: finite-F closure scores under heavy-tailed noise.

Run: python scripts/audit_non_gaussian_closure.py --trials 2400 --seed 2026
No real-world fault-detection, robustness, or security claims.
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


def _scores(training: np.ndarray, heldout: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Vectorized F scores, with one independent training sample per trial."""
    _, k, _ = training.shape
    q = len(z)
    c_train = training @ z.T
    c_test = heldout @ z.T
    centered = c_train - c_train.mean(axis=1, keepdims=True)
    sample_cov = np.einsum("tki,tkj->tij", centered, centered) / (k - 1)
    whitened = np.linalg.solve(np.linalg.cholesky(sample_cov), c_test[..., None])[..., 0]
    qscore = np.einsum("ti,ti->t", whitened, whitened)
    return (k - q) / (q * (k - 1)) * qscore


def audit(trials: int = 2400, seed: int = 2026,
          calibration_count: int = 16, tail_df: int = 5) -> dict[str, object]:
    """Paired Gaussian and covariance-matched elliptical Student-t controls.

    Under the Student-t arm, every full edge vector has an independent
    chi-square scale. Multiplication by sqrt((df-2)/chi2_df) ensures its
    population covariance equals the Gaussian reference (I), for df>2.
    """
    if isinstance(trials, bool) or not isinstance(trials, int) or not 200 <= trials <= 30000:
        raise ValueError("trials must be an integer in [200, 30000]")
    if (isinstance(calibration_count, bool) or not isinstance(calibration_count, int)
            or not 6 <= calibration_count <= 1000):
        raise ValueError("calibration_count must be an integer in [6, 1000]")
    if isinstance(tail_df, bool) or not isinstance(tail_df, int) or not 3 <= tail_df <= 100:
        raise ValueError("tail_df must be an integer in [3, 100]")
    a, z = graph_design(5, EDGES)
    if z.shape[0] != 3 or not np.allclose(z @ a, 0):
        raise AssertionError("unexpected reference graph topology")
    rng = np.random.default_rng(seed)
    m = len(EDGES)
    train = rng.normal(size=(trials, calibration_count, m))
    heldout = rng.normal(size=(trials, m))
    scale_train = np.sqrt((tail_df - 2) / rng.chisquare(tail_df, size=(trials, calibration_count, 1)))
    scale_test = np.sqrt((tail_df - 2) / rng.chisquare(tail_df, size=(trials, 1)))
    heavy_train, heavy_test = train * scale_train, heldout * scale_test
    gaussian_scores = _scores(train, heldout, z)
    heavy_scores = _scores(heavy_train, heavy_test, z)

    # The vectorized experiment must implement precisely the shipped reference.
    for t, h, scores in ((train, heldout, gaussian_scores), (heavy_train, heavy_test, heavy_scores)):
        for i in range(min(3, trials)):
            check = calibrated_cycle_statistic(h[i], t[i], 5, EDGES)
            if not np.isclose(check["f_statistic"], scores[i], rtol=1e-9, atol=1e-9):
                raise AssertionError("vectorized F score differs from the shipped kernel")
    q = len(z)
    denom = calibration_count - q
    return {
        "type": "synthetic_negative_control",
        "seed": seed, "trials": trials, "calibration_count": calibration_count,
        "student_degrees_of_freedom": tail_df, "cycle_dof": q,
        "nominal_false_alarm_fraction": 0.05,
        "gaussian_false_alarm_fraction": float(np.mean(f.sf(gaussian_scores, q, denom) < 0.05)),
        "student_t_false_alarm_fraction": float(np.mean(f.sf(heavy_scores, q, denom) < 0.05)),
        "model": "independent full-edge vectors; Student-t elliptical scale mixture; equal population covariance",
        "scope": "synthetic failure-mode demonstration, not a robust estimator or field guarantee",
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials", type=int, default=2400)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--calibration-count", type=int, default=16)
    p.add_argument("--tail-df", type=int, default=5)
    a = p.parse_args()
    print(json.dumps(audit(a.trials, a.seed, a.calibration_count, a.tail_df), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
