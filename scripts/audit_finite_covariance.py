"""Finite-calibration cycle covariance Monte Carlo stress test (synthetic).

Run: python scripts/audit_finite_covariance.py [--trials 2400] [--seed 2026]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import chi2, f
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cmbcst.sparse_network import graph_design
from cmbcst.estimated_covariance import calibrated_cycle_statistic

EDGES = [(0,1), (1,2), (2,3), (3,4), (0,4), (1,4), (0,2)]


def audit(trials: int = 2400, seed: int = 2026,
          calibration_sizes: tuple[int, ...] = (8, 16, 64)) -> dict[str, object]:
    """Check false-alarm calibration with separate test and training draws."""
    if isinstance(trials, bool) or not isinstance(trials, int) or not 200 <= trials <= 30000:
        raise ValueError("trials must be an integer between 200 and 30000")
    _, z = graph_design(5, EDGES)
    q, m = z.shape
    if any(isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < q + 3 or k > 1000
           for k in calibration_sizes):
        raise ValueError("calibration sizes must be integers from q+3 through 1000")
    rng = np.random.default_rng(seed)
    nominal = .05
    out = []
    for k in calibration_sizes:
        training_edges = rng.normal(size=(trials, k, m))
        training_cycles = training_edges @ z.T
        centered = training_cycles - training_cycles.mean(axis=1, keepdims=True)
        estimated = np.einsum("tki,tkj->tij", centered, centered) / (k - 1)
        heldout = rng.normal(size=(trials, m)) @ z.T
        white = np.linalg.solve(np.linalg.cholesky(estimated), heldout[..., None])[..., 0]
        statistic = np.einsum("ti,ti->t", white, white)
        f_score = statistic * (k - q) / (q * (k - 1))
        naive_p = chi2.sf(statistic, q)
        corrected_p = f.sf(f_score, q, k - q)
        for i in range(min(3, trials)):
            # Choose any edge vector with this closure. The closure statistic
            # is invariant to components in the gradient/im(A) subspace.
            edge_vector = heldout[i] @ np.linalg.pinv(z.T)
            check = calibrated_cycle_statistic(edge_vector, training_edges[i], 5, EDGES)
            if not np.isclose(check["chi2_statistic"], statistic[i], atol=1e-8, rtol=1e-8):
                raise AssertionError("batched and shipped sample-covariance scores disagree")
        out.append({
            "calibration_count": int(k),
            "cycle_dof": int(q), "f_denominator_dof": int(k-q),
            "naive_chi2_rejection_fraction": float(np.mean(naive_p < nominal)),
            "independent_F_rejection_fraction": float(np.mean(corrected_p < nominal)),
        })
    return {"model": "independent stationary Gaussian scalar graph edge noise",
            "seed": seed, "trials_per_setting": trials,
            "nominal_rejection_fraction": nominal, "results": out,
            "scope": "synthetic conditional calibration only; not field or security certification"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials", type=int, default=2400)
    p.add_argument("--seed", type=int, default=2026)
    a = p.parse_args()
    print(json.dumps(audit(a.trials, a.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
