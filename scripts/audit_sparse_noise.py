"""Small synthetic audit of sparse graph closure under noise misspecification.

Usage: python scripts/audit_sparse_noise.py [--trials 2000] [--seed 2026]
This diagnostic does NOT assert field-hardware performance, operational fault
probabilities, or real-world sensor-spoofing detection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cmbcst.sparse_network import fit_sparse_offsets, graph_design


def audit(trials: int = 2000, seed: int = 2026) -> dict:
    """Compare correctly reported covariance with an underreported variance.

    A fixed connected 5-sensor network with 3 independent cycles is used.
    The Gaussian null is true in both cases; the *reported* covariance is
    wrong in the second case. Threshold at a nominal 5% false-alarm rate.
    """
    if isinstance(trials, bool) or not isinstance(trials, int) or not 100 <= trials <= 100000:
        raise ValueError("trials must be an integer between 100 and 100000")
    edges = [(0,1),(1,2),(2,3),(3,4),(0,4),(1,4),(0,2)]
    a, z = graph_design(5, edges)
    rng = np.random.default_rng(seed)
    true_sigma = 0.04
    true_cov = np.eye(len(edges)) * true_sigma**2
    reported_cov = true_cov * 0.25  # variance underreported by 4x; std by 2x
    offset = np.array([0.0, 0.1, -0.3, 0.15, 0.28])
    simulated = a @ offset + rng.normal(0, true_sigma, (trials, len(edges)))
    closed = simulated @ z.T  # Z A = 0: true offsets cancel
    true_cycle_cov = z @ true_cov @ z.T
    wrong_cycle_cov = z @ reported_cov @ z.T
    w_true = np.linalg.solve(np.linalg.cholesky(true_cycle_cov), closed.T)
    w_wrong = np.linalg.solve(np.linalg.cholesky(wrong_cycle_cov), closed.T)
    chi_true = np.sum(w_true*w_true, axis=0)
    chi_wrong = np.sum(w_wrong*w_wrong, axis=0)
    df = z.shape[0]
    cutoff = float(chi2.isf(.05, df))
    # Ensure the batched cycle statistic agrees with the public solver.
    for k in range(min(5, trials)):
        t = fit_sparse_offsets(simulated[k], true_cov, 5, edges)
        u = fit_sparse_offsets(simulated[k], reported_cov, 5, edges)
        if not np.isclose(chi_true[k], t["residual_chi2"], atol=1e-9):
            raise AssertionError("true-covariance batched statistic disagrees with GLS")
        if not np.isclose(chi_wrong[k], u["residual_chi2"], atol=1e-9):
            raise AssertionError("reported-covariance batched statistic disagrees with GLS")
    return {
        "model": "synthetic connected scalar graph; Gaussian iid edge noise",
        "seed": seed, "trials": trials, "n_edges": len(edges),
        "n_sensors": 5, "cycle_dof": df,
        "nominal_false_positive_rate": 0.05,
        "chi2_cutoff": cutoff,
        "correct_covariance_false_positive_rate": float(np.mean(chi_true > cutoff)),
        "underreported_variance_false_positive_rate": float(np.mean(chi_wrong > cutoff)),
        "reported_variance_fraction_of_true": 0.25,
        "limits": "Synthetic model only; no field calibration, fault attribution, real-time or security assurance.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    print(json.dumps(audit(args.trials, args.seed), indent=2))


if __name__ == "__main__":
    main()
