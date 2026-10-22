"""Bootstrap draws: float32 GPU path against float64, and draw statistics on both backends."""
from __future__ import annotations

import numpy as np
import pytest

from cmbcst import accel


def _spectra(nb=12, nf=8, seed=0):
    rng = np.random.default_rng(seed)
    cl = np.empty((nb, nf, nf))
    for b in range(nb):
        a = rng.standard_normal((nf, nf))
        cl[b] = a @ a.T + nf * np.eye(nf) * (1 + b)
    nu = np.linspace(2000, 40000, nb)
    return cl, nu


def _has_xpu():
    try:
        import torch

        return torch.xpu.is_available()
    except ImportError:
        return False


@pytest.mark.skipif(not _has_xpu(), reason="no Intel XPU device")
def test_float32_gram_matches_float64():
    cl, nu = _spectra()
    assert accel.check_float32(cl, nu, seed=1)["max_rel_err"] < accel.REL_TOL


@pytest.mark.parametrize("force_cpu", [True, False])
def test_draws_are_unbiased_with_gaussian_spread(monkeypatch, force_cpu):
    if force_cpu:
        monkeypatch.setattr(accel, "_device", lambda: (None, None))
    elif not _has_xpu():
        pytest.skip("no Intel XPU device")
    cl, nu = _spectra()
    draws = np.array(list(accel.wishart_draws(cl, nu, 300, seed=4)))
    mean, sd = draws.mean(0), draws.std(0)
    diag = lambda x: np.diagonal(x, axis1=1, axis2=2)
    z = diag((mean - cl) / (sd / np.sqrt(len(draws))))
    assert abs(z.mean()) < 0.3 and 0.7 < z.std() < 1.3
    ratio = diag(sd) / (np.sqrt(2 / np.round(nu))[:, None] * diag(cl))  # Var(C_ii) = 2 C_ii^2 / nu
    assert 0.9 < np.median(ratio) < 1.1
