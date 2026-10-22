"""Independent whitened-projection check of the cached *synthetic* pilot.

This validates a deterministic, pre-existing exploratory benchmark receipt.
It does not test the Planck sky, statistical novelty, or method superiority.
"""
from itertools import combinations
import json
from pathlib import Path
import numpy as np


def test_synthetic_pilot_receipt_matches_independent_projection():
    root = Path(__file__).resolve().parents[1]
    reported = json.loads((root / "results" / "edge_fault_pilot.json").read_text())
    assert reported["status"].startswith("exploratory synthetic")
    n = 8
    edges = list(combinations(range(n), 2))
    incidence = np.zeros((len(edges), n))
    for k, (i, j) in enumerate(edges):
        incidence[k, i] = 1.0
        incidence[k, j] = -1.0

    rng = np.random.default_rng(20261022)
    z = rng.normal(size=(28, 28))
    scales = np.diag(np.linspace(0.65, 1.45, 28))
    covariance = scales @ (0.35 * z @ z.T / 28 + 0.65 * np.eye(28)) @ scales

    # Independent formulation: residual projector in whitened coordinates.
    # Whitening yields W A; Q spans its column space. This avoids the
    # source's inverse-covariance Schur-complement construction.
    chol = np.linalg.cholesky(covariance)
    whitened = np.linalg.solve(chol, incidence[:, 1:])
    q = np.linalg.qr(whitened, mode="reduced")[0]
    inv_chol = np.linalg.solve(chol, np.eye(28))
    residual_precision = inv_chol.T @ (np.eye(28) - q @ q.T) @ inv_chol
    residual_precision = (residual_precision + residual_precision.T) / 2
    diagonal = np.diag(residual_precision)

    def maximum_scores(seed):
        samples = np.random.default_rng(seed).standard_normal((4000, 28)) @ chol.T
        scores = (samples @ residual_precision.T)**2 / diagonal
        return scores.max(axis=1)

    cal = maximum_scores(23)
    cutoff = float(np.quantile(cal, 0.95, method="higher"))
    holdout = maximum_scores(24)
    edge = 9
    noise = rng.standard_normal((4000, 28)) @ chol.T
    noise[:, edge] += 4.0 / np.sqrt(diagonal[edge])
    likelihood_scores = (noise @ residual_precision.T)**2 / diagonal
    maxima = likelihood_scores.max(axis=1)
    located = likelihood_scores.argmax(axis=1) == edge
    generated = {
        "calibrated_max_glrt_threshold": cutoff,
        "heldout_null_fwer": float((holdout > cutoff).mean()),
        "injected_detection_power": float((maxima > cutoff).mean()),
        "correct_localization_unconditional": float(located.mean()),
        "joint_detection_correct_localization": float((located & (maxima > cutoff)).mean()),
        "first_injection_empirical_p": float(
            (1 + np.count_nonzero(cal >= maxima[0])) / (len(cal) + 1)),
    }
    for key, actual in generated.items():
        assert np.isclose(actual, reported[key], rtol=1e-11, atol=1e-11), (key, actual, reported[key])
    assert reported["injected_edge_vertices"] == list(edges[edge])
