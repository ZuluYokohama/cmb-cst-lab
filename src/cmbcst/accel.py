"""Bandpower bootstrap draws on PyTorch CUDA/XPU when available, else numpy.

A draw is a bin-level Wishart sample around measured bandpowers C_b:

    C_hat_b = L_b (Z Z^T / nu_b) L_b^T,   Z ~ N(0, 1) of shape (n_fields, nu_b),   L_b L_b^T = C_b.

The heavy part, generating Z and forming the Gram matrix Z Z^T / nu_b (16 x ~5e4 per bin), runs on
the GPU in float32, with CUDA TF32 disabled. The 16 x 16 transform with L_b stays in float64 on
the CPU. `check_float32` recomputes one batch from the same normals in float64; the accelerated
path must agree to REL_TOL relative or the run stops.

The numpy-only kernels (`square_roots`, `assemble`, `gram64`, `max_rel_err`, `counter_normals`,
`numpy_draw`) are verified in the JoVin sandbox, together with Gram matrices recorded from the
real devices by scripts/gpu_attest.py (tests/sandbox/jovin_tests_accel.py).

CMB_CST_DEVICE selects auto (CUDA first, then XPU), cpu, cuda:N, or xpu:N. An explicitly
requested accelerator must be available; device failures never trigger a silent CPU fallback.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
import re
from typing import Iterator

import numpy as np

REL_TOL = 1e-5


# ---- numpy kernels (sandbox-verified) ---------------------------------------------------------
def square_roots(cl: np.ndarray) -> np.ndarray:
    """L_b with L_b L_b^T = C_b (eigen square root, tiny negative eigenvalues clipped), float64."""
    cl = np.asarray(cl, dtype=np.float64)
    out = np.empty_like(cl)
    for b in range(cl.shape[0]):
        w, v = np.linalg.eigh(cl[b])
        out[b] = v * np.sqrt(np.clip(w, 1e-12 * w.max(), None))
    return out


def assemble(roots: np.ndarray, grams: np.ndarray) -> np.ndarray:
    """C_hat_b = L_b G_b L_b^T for all bins, float64: (nb, nf, nf) x (nb, nf, nf)."""
    return np.einsum("bik,bkl,bjl->bij", roots, grams.astype(np.float64), roots)


def gram64(z: np.ndarray) -> np.ndarray:
    """Float64 reference Gram Z Z^T / dof over the last axis (batched)."""
    z = np.asarray(z, dtype=np.float64)
    return np.einsum("...ik,...jk->...ij", z, z) / z.shape[-1]


def max_rel_err(g: np.ndarray, ref: np.ndarray) -> float:
    """max |g - ref| / max |ref|, the accelerated-path acceptance metric.

    Non-finite device output (NaN/inf) returns inf, so it can never pass a REL_TOL gate (a NaN
    would otherwise vanish inside max()).
    """
    g = np.asarray(g, dtype=np.float64)
    if not np.all(np.isfinite(g)):
        return float("inf")
    err = float(np.max(np.abs(g - ref)) / np.max(np.abs(ref)))
    return err if np.isfinite(err) else float("inf")


def counter_normals(seed: int, nf: int, dof: int) -> np.ndarray:
    """Deterministic float32 N(0,1) array (nf, dof) from splitmix64 + Box-Muller.

    Pure uint64 arithmetic, so any numpy on any platform regenerates the same values; used to
    feed identical inputs to a device and to the sandbox's float64 recomputation.
    """
    n = nf * dof
    m = (n + 1) // 2
    with np.errstate(over="ignore"):
        x = np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15) + np.arange(1, 2 * m + 1, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    u = ((x >> np.uint64(11)).astype(np.float64) + 0.5) / float(1 << 53)  # in (0, 1)
    u1, u2 = u[:m], u[m:]
    r = np.sqrt(-2.0 * np.log(u1))
    z = np.concatenate([r * np.cos(2 * np.pi * u2), r * np.sin(2 * np.pi * u2)])[:n]
    return z.astype(np.float32).reshape(nf, dof)


def numpy_draw(roots: np.ndarray, dof: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One bootstrap draw entirely in float64 numpy (the CPU fallback)."""
    nb, nf, _ = roots.shape
    out = np.empty(roots.shape, dtype=np.float64)
    for b in range(nb):
        x = roots[b] @ rng.standard_normal((nf, int(dof[b])))
        out[b] = x @ x.T / dof[b]
    return out


# ---- devices ----------------------------------------------------------------------------------
def _device():
    """Resolve the requested accelerator without catching selected-device failures."""
    request = os.environ.get("CMB_CST_DEVICE", "auto").strip().lower()
    selected = re.fullmatch(r"(cuda|xpu):([0-9]+)", request)
    if request not in ("auto", "cpu") and selected is None:
        raise ValueError("CMB_CST_DEVICE must be auto, cpu, cuda:N, or xpu:N (N >= 0)")
    if request == "cpu":
        return None, None
    try:
        import torch
    except ImportError as exc:
        if request == "auto":
            return None, None
        raise RuntimeError(f"CMB_CST_DEVICE={request} requires PyTorch") from exc
    candidates = ((selected.group(1), int(selected.group(2))),) if selected else (("cuda", 0), ("xpu", 0))
    for kind, index in candidates:
        backend = getattr(torch, kind, None)
        if backend is not None and backend.is_available() and index < backend.device_count():
            return torch, torch.device(f"{kind}:{index}")
    if selected:
        raise RuntimeError(f"CMB_CST_DEVICE={request} is not available")
    return None, None


def backend_name() -> str:
    torch, dev = _device()
    if dev is None:
        return "numpy float64 (CPU)"
    name = getattr(torch, dev.type).get_device_name(dev.index)
    return f"PyTorch {torch.__version__} on {name} [{dev}] (float32 Gram, float64 transform)"


@contextmanager
def _ieee_matmul(torch, dev):
    """Scope CUDA precision changes to our product and restore them even on failure.

    PyTorch >=2.9 uses fp32_precision. The legacy branch supports older installations;
    the two precision APIs are deliberately never mixed in a single execution.
    """
    if dev.type != "cuda":
        yield
        return
    matmul = torch.backends.cuda.matmul
    setting = "fp32_precision" if hasattr(matmul, "fp32_precision") else "allow_tf32"
    previous = getattr(matmul, setting)
    try:
        setattr(matmul, setting, "ieee" if setting == "fp32_precision" else False)
        yield
    finally:
        setattr(matmul, setting, previous)


def gpu_gram(z32: np.ndarray) -> np.ndarray:
    """Float32 Gram on the selected GPU (for attestation and same-normal checks)."""
    torch, dev = _device()
    if dev is None:
        raise RuntimeError("no CUDA/XPU device selected")
    z = torch.from_numpy(np.ascontiguousarray(z32, dtype=np.float32)).to(dev)
    if z.ndim == 2:
        z = z[None]
    with _ieee_matmul(torch, dev):
        gram = torch.bmm(z, z.transpose(1, 2)) / z.shape[-1]
    return gram.cpu().numpy()


def _gram_batch_gpu(torch, dev, nf: int, dof: int, batch: int, gen):
    """(batch, nf, nf) float32 Z Z^T / dof on the GPU, plus the Z it used (for checking)."""
    z = torch.randn(batch, nf, dof, device=dev, dtype=torch.float32, generator=gen)
    with _ieee_matmul(torch, dev):
        gram = torch.bmm(z, z.transpose(1, 2)) / dof
    return gram, z


def _validate_batch(batch: int) -> None:
    if isinstance(batch, (bool, np.bool_)) or not isinstance(batch, (int, np.integer)) or batch <= 0:
        raise ValueError("batch must be a positive integer")


def check_float32(cl: np.ndarray, nu: np.ndarray, seed: int = 0, batch: int = 4) -> dict[str, object]:
    """Max relative error of the float32 GPU Gram matrices against float64 from the same normals.

    On the numpy backend nothing runs in float32, so the check does not apply: `applicable` is
    False and `max_rel_err` is None rather than an unmeasured zero.
    """
    _validate_batch(batch)
    torch, dev = _device()
    if dev is None:
        return {"max_rel_err": None, "applicable": False, "backend": backend_name()}
    gen = torch.Generator(device=dev).manual_seed(seed)
    nf = cl.shape[1]
    worst = 0.0
    for b in sorted({0, len(nu) // 2, len(nu) - 1}):
        dof = max(int(round(nu[b])), nf + 1)
        g32, z = _gram_batch_gpu(torch, dev, nf, dof, batch, gen)
        worst = max(worst, max_rel_err(g32.cpu().numpy(), gram64(z.cpu().numpy())))
    return {"max_rel_err": worst, "applicable": True, "backend": backend_name()}


def wishart_draws(cl: np.ndarray, nu: np.ndarray, n_sims: int, seed: int, batch: int = 50) -> Iterator[np.ndarray]:
    """Yield n_sims bootstrap bandpower sets (nb, nf, nf), float64 whatever the input dtype."""
    _validate_batch(batch)
    if isinstance(n_sims, (bool, np.bool_)) or not isinstance(n_sims, (int, np.integer)) or n_sims < 0:
        raise ValueError("n_sims must be a nonnegative integer")
    cl = np.asarray(cl, dtype=np.float64)
    nb, nf, _ = cl.shape
    roots = square_roots(cl)
    dof = np.maximum(np.round(nu).astype(int), nf + 1)
    torch, dev = _device()
    if dev is None:
        rng = np.random.default_rng(seed)
        for _ in range(n_sims):
            yield numpy_draw(roots, dof, rng)
        return
    chk = check_float32(cl, nu, seed=seed + 1)
    if not np.isfinite(chk["max_rel_err"]) or chk["max_rel_err"] > REL_TOL:
        raise RuntimeError(f"float32 GPU Gram differs from float64 by {chk['max_rel_err']:.2e} > {REL_TOL:g}")
    gen = torch.Generator(device=dev).manual_seed(seed)
    done = 0
    while done < n_sims:
        k = min(batch, n_sims - done)
        grams = np.empty((k, nb, nf, nf))
        for b in range(nb):
            g, _ = _gram_batch_gpu(torch, dev, nf, int(dof[b]), k, gen)
            grams[:, b] = g.cpu().numpy()
            if not np.all(np.isfinite(grams[:, b])):
                raise RuntimeError("non-finite GPU Gram matrix in Wishart draws")
        for s in range(k):
            yield assemble(roots, grams[s])
        done += k
