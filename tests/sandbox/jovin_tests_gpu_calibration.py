"""Numerical experiment checks: independent draws, correlated maps, known sky injection."""
import math

import numpy as np

from solution import _estimate, summarize_chi2, summarize_pulls, synthetic_sigma


def test_repeating_correlated_maps_does_not_invent_precision():
    draws = np.array([-2.5, -1.2, -0.5, 0.2, 0.6, 1.4, 2.1, -0.1])[:, None]
    single = summarize_pulls(draws)
    duplicated = summarize_pulls(np.repeat(draws, 7, axis=1))
    for name in single:
        assert np.isclose(single[name]["value"], duplicated[name]["value"], rtol=1e-14, atol=1e-14)
        assert np.isclose(single[name]["mcse"], duplicated[name]["mcse"], rtol=1e-14, atol=1e-14)


def test_coverage_mcse_uses_draw_fractions():
    pulls = np.array([[0.0, 0.5], [0.7, 1.5], [2.5, 3.0], [-1.5, -0.3]])
    result = summarize_pulls(pulls)["coverage_1sigma"]
    fractions = np.array([1.0, 0.5, 0.0, 0.5])
    assert result["value"] == 0.5
    assert abs(result["mcse"] - fractions.std(ddof=1) / 2) < 1e-15
    assert abs(result["target"] - math.erf(1 / math.sqrt(2))) < 1e-15


def test_five_mcse_gate_and_degenerate_mcse_are_explicit():
    contributions = np.array([-1.0, 1.0])
    assert _estimate(5.0, 0.0, contributions)["passed"]
    assert not _estimate(5.01, 0.0, contributions)["passed"]
    degenerate = _estimate(0.0, 0.0, np.zeros(10))
    assert degenerate["mcse"] == 0
    assert degenerate["deviation_mcse"] is None
    assert not degenerate["passed"]


def test_chi2_dof_and_observed_mcse_are_distinct():
    pair = summarize_chi2(np.array([20.0, 24.0, 32.0, 36.0]), 28)
    assert pair["value"] == pair["target"] == 28
    assert pair["passed"]
    assert abs(pair["ideal_chi2_mcse"] - math.sqrt(56 / 4)) < 1e-15
    assert abs(pair["mcse"] - np.std([20.0, 24.0, 32.0, 36.0], ddof=1) / 2) < 1e-15


def test_synthetic_injection_preserves_differential_angle_identity():
    alpha = np.array([0.19, -0.31, 0, 0.04, -0.06, -0.06, -0.08, -0.10]) * np.pi / 60
    covariance = synthetic_sigma(100, alpha, beta=0.3 * np.pi / 180, eb=0.03)
    assert np.min(np.linalg.eigvalsh(covariance)) > 0
    for i in range(8):
        for j in range(i + 1, 8):
            difference = covariance[2 * i + 1, 2 * j] - covariance[2 * i, 2 * j + 1]
            total = covariance[2 * i, 2 * j] + covariance[2 * i + 1, 2 * j + 1]
            assert abs(0.5 * np.arctan(difference / total) - (alpha[i] - alpha[j])) < 1e-14


def test_invalid_summary_inputs_fail_closed():
    invalid = [np.ones((1, 7)), np.empty((10, 0)), np.array([[0.0], [np.nan]]), np.ones(10)]
    for value in invalid:
        try:
            summarize_pulls(value)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid pull samples were accepted")
