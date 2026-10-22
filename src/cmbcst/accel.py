"""Bandpower bootstrap draws, on the Intel Arc GPU (PyTorch XPU) when available, else numpy.

A draw is a bin-level Wishart sample around measured bandpowers C_b:

    C_hat_b = L_b (Z Z^T / nu_b) L_b^T,   Z ~ N(0, 1) of shape (n_fields, nu_b),   L_b L_b^T = C_b.

The heavy part, generating Z and forming Z Z^T / nu_b (16 x ~5e4 per bin), runs on the GPU in
float32: Xe2 has no fast float64. The 16 x 16 transform with L_b stays in float64 on the CPU.
`check_float32` recomputes one batch from the same normals in float64; the accelerated path must
agree to 1e-5 relative or the run stops.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

REL_TOL = 1e-5


def square_roots(cl: np.ndarray) -> np.ndarray:
    """L_b with L_b L_b^T = C_b (eigen square root, tiny negative eigenvalues clipped)."""
    out = np.empty_like(cl)
    for b in range(cl.shape[0]):
        w, v = np.linalg.eigh(cl[b])
        out[b] = v * np.sqrt(np.clip(w, 1e-12 * w.max(), None))
    return out


def _device():
    try:
        import torch

        if torch.xpu.is_available():
            return torch, torch.device("xpu")
    except ImportError:
        pass
    return None, None


def backend_name() -> str:
    torch, dev = _device()
    return f"PyTorch {torch.__version__} on {torch.xpu.get_device_name(0)} (float32 Gram, float64 transform)" if dev else "numpy float64 (CPU)"


def _gram_batch_gpu(torch, dev, nf: int, dof: int, batch: int, gen):
    """(batch, nf, nf) float32 Z Z^T / dof on the GPU, plus the Z it used (for checking)."""
    z = torch.randn(batch, nf, dof, device=dev, dtype=torch.float32, generator=gen)
    return torch.bmm(z, z.transpose(1, 2)) / dof, z


def check_float32(cl: np.ndarray, nu: np.ndarray, seed: int = 0, batch: int = 4) -> dict[str, float]:
    """Max relative error of the float32 GPU Gram matrices against float64 from the same normals."""
    torch, dev = _device()
    if dev is None:
        return {"max_rel_err": 0.0, "backend": "numpy"}
    gen = torch.Generator(device=dev).manual_seed(seed)
    nf = cl.shape[1]
    worst = 0.0
    for b in sorted({0, len(nu) // 2, len(nu) - 1}):
        dof = max(int(round(nu[b])), nf + 1)
        g32, z = _gram_batch_gpu(torch, dev, nf, dof, batch, gen)
        z64 = z.cpu().numpy().astype(np.float64)
        g64 = np.einsum("bik,bjk->bij", z64, z64) / dof
        worst = max(worst, float(np.max(np.abs(g32.cpu().numpy() - g64)) / np.max(np.abs(g64))))
    return {"max_rel_err": worst, "backend": backend_name()}


def wishart_draws(cl: np.ndarray, nu: np.ndarray, n_sims: int, seed: int, batch: int = 50) -> Iterator[np.ndarray]:
    """Yield n_sims bootstrap bandpower sets (nb, nf, nf), float64."""
    nb, nf, _ = cl.shape
    roots = square_roots(cl)
    dof = np.maximum(np.round(nu).astype(int), nf + 1)
    torch, dev = _device()
    if dev is None:
        rng = np.random.default_rng(seed)
        for _ in range(n_sims):
            out = np.empty_like(cl)
            for b in range(nb):
                x = roots[b] @ rng.standard_normal((nf, dof[b]))
                out[b] = x @ x.T / dof[b]
            yield out
        return
    chk = check_float32(cl, nu, seed=seed + 1)
    if chk["max_rel_err"] > REL_TOL:
        raise RuntimeError(f"float32 GPU Gram differs from float64 by {chk['max_rel_err']:.2e} > {REL_TOL:g}")
    gen = torch.Generator(device=dev).manual_seed(seed)
    done = 0
    while done < n_sims:
        k = min(batch, n_sims - done)
        grams = np.empty((k, nb, nf, nf))
        for b in range(nb):
            g, _ = _gram_batch_gpu(torch, dev, nf, int(dof[b]), k, gen)
            grams[:, b] = g.cpu().numpy().astype(np.float64)
        for s in range(k):
            yield np.einsum("bik,bkl,bjl->bij", roots, grams[s], roots)
        done += k
