"""Reproducible scalar multi-sensor closure experiment; NOT a safety system.

Run: python scripts/demo_sensor_network.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cmbcst.network import complete_complex  # noqa: E402
from cmbcst.sensor_network import fit_scalar_offsets  # noqa: E402


def main() -> int:
    rng = np.random.default_rng(2026)
    true_offsets = np.array([0.0, 0.12, -0.08, 0.22, -0.14])
    edges, _, d0, _ = complete_complex(len(true_offsets))
    sigma = 0.02
    cov = np.eye(len(edges)) * sigma**2
    clean = d0 @ true_offsets + rng.normal(0, sigma, size=len(edges))
    inconsistent = clean.copy()
    # Corrupt one *pair measurement*. Closure flags inconsistency without
    # proving which endpoint sensor, channel, or upstream process caused it.
    inconsistent[edges.index((1, 3))] += 0.4
    output = {}
    for label, y in (("clean", clean), ("one_edge_corrupted", inconsistent)):
        result = fit_scalar_offsets(y, cov, len(true_offsets), gauge=0)
        output[label] = {
            "offsets_relative_to_sensor_0": np.round(result["offsets"], 6).tolist(),
            "residual_chi2": round(float(result["residual_chi2"]), 6),
            "closure_chi2": round(float(result["closure_chi2"]), 6),
            "dof": int(result["residual_dof"]),
            "pte_assuming_correct_gaussian_covariance": float(result["pte"]),
        }
    print(json.dumps({"model": "additive scalar complete-graph differences",
                      "simulation_only": True, "results": output}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
