"""Finite-simulation and multiple-testing checks for the current persistence API."""
import numpy as np
import pytest
from cmbcst.persistence import mc_pvalue, holm


def test_two_sided_tail_nonzero_and_finite_sample_correction():
    values = np.arange(200, dtype=float)
    assert mc_pvalue(values, -1.0) == 2 / 201
    assert mc_pvalue(values, 201.0) == 2 / 201
    assert mc_pvalue(values, 99.5) == 1.0


def test_mc_pvalue_rejects_invalid_samples():
    with pytest.raises(ValueError):
        mc_pvalue(np.array([]), 1.0)
    with pytest.raises(ValueError):
        mc_pvalue(np.array([1.0, np.nan]), 1.0)
    with pytest.raises(ValueError):
        mc_pvalue(np.arange(5), np.nan)
    with pytest.raises(ValueError):
        mc_pvalue(np.ones((2,2)), 1.0)


def test_holm_stepdown_is_monotonic_and_conservative():
    p = np.array([0.01, 0.04, 0.30, 0.02])
    h = holm(p)
    assert np.allclose(h, np.array([0.04, 0.08, 0.3, 0.06]))
    assert np.all(h >= p)
    assert np.all((0 <= h) & (h <= 1))
