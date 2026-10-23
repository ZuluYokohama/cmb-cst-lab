"""Synthetic rank calibration across independent whole AR(1) sessions.

Usage: python scripts/audit_session_rank.py --reference-count 1800 --test-count 1700
Unlike an independent-row F reference, a held-out score can be compared against
a bank of *independent whole-session* scores, provided all sessions are
exchangeable under the null and the scoring algorithm is fixed in advance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import f

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cmbcst.sparse_network import graph_design
from cmbcst.estimated_covariance import calibrated_cycle_statistic
from cmbcst.empirical_calibration import upper_tail_rank_pvalues

EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (1, 4), (0, 2)]


def _session_scores(session_count: int, k: int, rho: float, z: np.ndarray,
                    rng: np.random.Generator) -> np.ndarray:
    """Independent episodes; each episode has a correlated training sequence."""
    m = z.shape[1]
    q = len(z)
    eps = rng.standard_normal((session_count, k, m))
    training = np.empty_like(eps)
    training[:, 0, :] = eps[:, 0, :]
    for t in range(1, k):
        training[:, t, :] = rho * training[:, t - 1, :] + np.sqrt(1.0-rho*rho)*eps[:, t, :]
    heldout = rng.standard_normal((session_count, m))
    cycle_train = training @ z.T
    centered = cycle_train - cycle_train.mean(axis=1, keepdims=True)
    cov = np.einsum("tki,tkj->tij", centered, centered) / (k - 1)
    projected_test = heldout @ z.T
    w = np.linalg.solve(np.linalg.cholesky(cov), projected_test[..., None])[..., 0]
    score = np.einsum("ti,ti->t", w, w) * (k - q) / (q * (k - 1))
    for i in range(min(2, session_count)):
        direct = calibrated_cycle_statistic(heldout[i], training[i], 5, EDGES)
        if not np.isclose(direct["f_statistic"], score[i], rtol=1e-9, atol=1e-9):
            raise AssertionError("batched session statistic differs from shipped reference")
    return score


def audit(reference_count: int = 1800, test_count: int = 1700,
          calibration_count: int = 16, rho: float = 0.8,
          shifted_rho: float = 0.95, seed: int = 2026) -> dict[str, object]:
    """Compare parametric F and independent-session ranks under match and shift.

    The *same* rho is used across the reference-bank and matching test
    sessions. A separate shift arm deliberately violates score exchangeability
    by changing within-session autocorrelation to shifted_rho.
    """
    for name, n in (("reference_count", reference_count), ("test_count", test_count)):
        if isinstance(n, bool) or not isinstance(n, int) or not 100 <= n <= 10000:
            raise ValueError(f"{name} must be an integer in [100, 10000]")
    if (isinstance(calibration_count, bool) or not isinstance(calibration_count, int)
            or not 6 <= calibration_count <= 200):
        raise ValueError("calibration_count must be an integer in [6, 200]")
    if any(isinstance(v, bool) or not np.isfinite(v) or not 0 <= v < 1 for v in (rho, shifted_rho)):
        raise ValueError("rho and shifted_rho must be finite numbers in [0,1)")
    a, z = graph_design(5, EDGES)
    q = len(z)
    if not np.allclose(z @ a, 0):
        raise AssertionError("invalid cycle topology")
    rng = np.random.default_rng(seed)
    ref = _session_scores(reference_count, calibration_count, rho, z, rng)
    matched = _session_scores(test_count, calibration_count, rho, z, rng)
    shifted = _session_scores(test_count, calibration_count, shifted_rho, z, rng)
    rank_p_matched = upper_tail_rank_pvalues(matched, ref)
    rank_p_shifted = upper_tail_rank_pvalues(shifted, ref)
    f_p_matched = f.sf(matched, q, calibration_count-q)
    f_p_shifted = f.sf(shifted, q, calibration_count-q)
    return {
        "type": "synthetic_whole_session_rank_calibration",
        "seed": seed, "reference_sessions": reference_count, "heldout_sessions": test_count,
        "within_session_calibration_count": calibration_count, "cycle_dof": q,
        "reference_ar1_rho": float(rho), "shifted_ar1_rho": float(shifted_rho),
        "nominal_false_alarm_rate": 0.05,
        "matched_session_rank_rejection_fraction": float(np.mean(rank_p_matched < .05)),
        "matched_session_wrong_F_rejection_fraction": float(np.mean(f_p_matched < .05)),
        "shifted_session_rank_rejection_fraction": float(np.mean(rank_p_shifted < .05)),
        "shifted_session_wrong_F_rejection_fraction": float(np.mean(f_p_shifted < .05)),
        "rank_resolution": 1 / (reference_count+1),
        "limit": "empirical rank validity requires independent exchangeable entire sessions; shift arm intentionally violates it",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-count", type=int, default=1800)
    parser.add_argument("--test-count", type=int, default=1700)
    parser.add_argument("--calibration-count", type=int, default=16)
    parser.add_argument("--rho", type=float, default=0.8)
    parser.add_argument("--shifted-rho", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=2026)
    a = parser.parse_args()
    print(json.dumps(audit(a.reference_count, a.test_count,
                           a.calibration_count, a.rho, a.shifted_rho, a.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
