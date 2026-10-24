"""Bootstrap draws: float32 GPU path against float64, and draw statistics on both backends."""
from __future__ import annotations

import sys
from types import SimpleNamespace

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


def _select_backend(monkeypatch, selected):
    monkeypatch.setenv("CMB_CST_DEVICE", selected)
    if selected == "cpu":
        return
    torch = pytest.importorskip("torch")
    kind, index = selected.split(":")
    backend = getattr(torch, kind, None)
    if backend is None or not backend.is_available() or int(index) >= backend.device_count():
        pytest.skip(f"no available {selected} device")
    # Allocation, computation, or precision errors on an available GPU must fail the test.


@pytest.mark.parametrize("selected", ["cuda:0", "xpu:0"])
def test_float32_gram_matches_float64(monkeypatch, selected):
    _select_backend(monkeypatch, selected)
    cl, nu = _spectra()
    assert accel.check_float32(cl, nu, seed=1)["max_rel_err"] < accel.REL_TOL


@pytest.mark.parametrize("selected", ["cpu", "cuda:0", "xpu:0"])
def test_draws_are_unbiased_with_gaussian_spread(monkeypatch, selected):
    _select_backend(monkeypatch, selected)
    cl, nu = _spectra()
    draws = np.array(list(accel.wishart_draws(cl, nu, 300, seed=4)))
    mean, sd = draws.mean(0), draws.std(0)
    diag = lambda x: np.diagonal(x, axis1=1, axis2=2)
    z = diag((mean - cl) / (sd / np.sqrt(len(draws))))
    assert abs(z.mean()) < 0.3 and 0.7 < z.std() < 1.3
    ratio = diag(sd) / (np.sqrt(2 / np.round(nu))[:, None] * diag(cl))  # Var(C_ii) = 2 C_ii^2 / nu
    assert 0.9 < np.median(ratio) < 1.1


@pytest.mark.parametrize("selected", ["cpu", "cuda:0", "xpu:0"])
def test_float32_input_gives_float64_draws(monkeypatch, selected):
    _select_backend(monkeypatch, selected)
    cl, nu = _spectra(nb=4)
    draws = list(accel.wishart_draws(cl.astype(np.float32), nu, 3, seed=1, batch=2))
    assert len(draws) == 3  # also exercises the final partial GPU batch
    assert all(d.dtype == np.float64 for d in draws)
    assert accel.square_roots(cl.astype(np.float32)).dtype == np.float64


def test_precision_check_not_reported_without_a_float32_step(monkeypatch):
    monkeypatch.setenv("CMB_CST_DEVICE", "cpu")
    chk = accel.check_float32(*_spectra(nb=3))
    assert chk["applicable"] is False and chk["max_rel_err"] is None


def _fake_torch(cuda_count=2, xpu_count=1):
    def backend(count):
        return SimpleNamespace(is_available=lambda: count > 0, device_count=lambda: count)
    return SimpleNamespace(cuda=backend(cuda_count), xpu=backend(xpu_count), device=lambda name: name)


@pytest.mark.parametrize("requested, cuda_count, xpu_count, expected", [
    ("auto", 2, 1, "cuda:0"),
    ("auto", 0, 1, "xpu:0"),
    ("auto", 0, 0, None),
    ("cuda:1", 2, 1, "cuda:1"),
    ("xpu:0", 2, 1, "xpu:0"),
])
def test_device_selection(monkeypatch, requested, cuda_count, xpu_count, expected):
    fake = _fake_torch(cuda_count, xpu_count)
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setenv("CMB_CST_DEVICE", requested)
    torch, dev = accel._device()
    assert dev == expected
    assert torch is (None if expected is None else fake)


@pytest.mark.parametrize("requested", ["cpu", "auto"])
def test_cpu_operation_without_torch(monkeypatch, requested):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setenv("CMB_CST_DEVICE", requested)
    assert accel._device() == (None, None)


@pytest.mark.parametrize("requested", ["cuda", "cuda:-1", "cuda:1.5", "xpu", "gpu:0", ""])
def test_invalid_device_request_is_rejected(monkeypatch, requested):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setenv("CMB_CST_DEVICE", requested)
    with pytest.raises(ValueError, match="CMB_CST_DEVICE"):
        accel._device()


@pytest.mark.parametrize("requested", ["cuda:2", "xpu:1"])
def test_explicit_missing_device_cannot_fall_back(monkeypatch, requested):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    monkeypatch.setenv("CMB_CST_DEVICE", requested)
    with pytest.raises(RuntimeError, match="not available"):
        accel._device()


def test_explicit_device_requires_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setenv("CMB_CST_DEVICE", "cuda:0")
    with pytest.raises(RuntimeError, match="requires PyTorch"):
        accel._device()


def test_device_runtime_failure_cannot_fall_back(monkeypatch):
    fake = _fake_torch()
    def failed_probe():
        raise RuntimeError("driver failure")
    fake.cuda.is_available = failed_probe
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setenv("CMB_CST_DEVICE", "auto")
    with pytest.raises(RuntimeError, match="driver failure"):
        accel._device()


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_cuda_precision_is_scoped_and_restored(legacy, fail):
    setting, original, required = ("allow_tf32", True, False) if legacy else ("fp32_precision", "tf32", "ieee")
    matmul = SimpleNamespace(**{setting: original})
    fake = SimpleNamespace(backends=SimpleNamespace(cuda=SimpleNamespace(matmul=matmul)))
    def product():
        with accel._ieee_matmul(fake, SimpleNamespace(type="cuda")):
            assert getattr(matmul, setting) == required
            if fail:
                raise RuntimeError("product failed")
    if fail:
        with pytest.raises(RuntimeError, match="product failed"):
            product()
    else:
        product()
    assert getattr(matmul, setting) == original
    assert vars(matmul) == {setting: original}  # never mix old and new precision controls


def test_xpu_does_not_touch_cuda_precision():
    with accel._ieee_matmul(SimpleNamespace(), SimpleNamespace(type="xpu")):
        pass


@pytest.mark.parametrize("batch", [0, -1, 1.5, True])
def test_invalid_batch_cannot_hang_generator(monkeypatch, batch):
    monkeypatch.setenv("CMB_CST_DEVICE", "cpu")
    cl, nu = _spectra(nb=1)
    with pytest.raises(ValueError, match="batch"):
        list(accel.wishart_draws(cl, nu, 1, seed=0, batch=batch))
    with pytest.raises(ValueError, match="batch"):
        accel.check_float32(cl, nu, batch=batch)


@pytest.mark.parametrize("n_sims", [-1, 1.5, True])
def test_invalid_draw_count_is_rejected(monkeypatch, n_sims):
    monkeypatch.setenv("CMB_CST_DEVICE", "cpu")
    cl, nu = _spectra(nb=1)
    with pytest.raises(ValueError, match="n_sims"):
        list(accel.wishart_draws(cl, nu, n_sims, seed=0))


@pytest.mark.parametrize("error", [float("nan"), float("inf"), 2 * accel.REL_TOL])
def test_bad_precision_gate_stops_before_drawing(monkeypatch, error):
    monkeypatch.setattr(accel, "_device", lambda: (SimpleNamespace(), SimpleNamespace(type="cuda")))
    monkeypatch.setattr(accel, "check_float32", lambda *a, **kw: {"max_rel_err": error})
    cl, nu = _spectra(nb=1)
    with pytest.raises(RuntimeError, match="float32 GPU Gram"):
        next(accel.wishart_draws(cl, nu, 1, seed=0))


def test_nonfinite_production_gram_stops_even_after_a_passing_gate(monkeypatch):
    class Generator:
        def __init__(self, **kwargs):
            pass
        def manual_seed(self, seed):
            return self
    class Tensor:
        def cpu(self):
            return self
        def numpy(self):
            return np.full((1, 8, 8), np.nan)
    monkeypatch.setattr(accel, "_device", lambda: (SimpleNamespace(Generator=Generator), SimpleNamespace(type="cuda")))
    monkeypatch.setattr(accel, "check_float32", lambda *a, **kw: {"max_rel_err": 0.0})
    monkeypatch.setattr(accel, "_gram_batch_gpu", lambda *a: (Tensor(), None))
    cl, nu = _spectra(nb=1)
    with pytest.raises(RuntimeError, match="non-finite GPU Gram"):
        next(accel.wishart_draws(cl, nu, 1, seed=0))
