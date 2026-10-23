"""Exploratory, spectrum/histogram-constrained patch surrogates and exact PD0 summaries.

IAAFT is an established surrogate method, extended here to 2D patches. Its
algorithmic ensemble is not an exact conditional randomization distribution.
"""
from __future__ import annotations

import numpy as np

from .persistence import UnionFindPersistence


def checked_field(field: np.ndarray) -> np.ndarray:
    x = np.asarray(field, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 2 or not np.all(np.isfinite(x)) or x.std() == 0:
        raise ValueError("need a finite, nonconstant 2D field")
    return x


def rank_match(candidate: np.ndarray, target: np.ndarray) -> np.ndarray:
    candidate, target = checked_field(candidate), checked_field(target)
    if candidate.shape != target.shape:
        raise ValueError("candidate and target shapes differ")
    out = np.empty(candidate.size, dtype=np.float64)
    out[np.argsort(candidate.ravel(), kind="stable")] = np.sort(target.ravel())
    return out.reshape(target.shape)


def spectral_projection(candidate: np.ndarray, amplitude: np.ndarray) -> np.ndarray:
    z = np.fft.fft2(checked_field(candidate))
    phase = np.divide(z, np.abs(z), out=np.ones_like(z), where=np.abs(z) > 0)
    return np.fft.ifft2(amplitude * phase).real


def iaaft_reference(target: np.ndarray, initial: np.ndarray, iterations: int) -> tuple[np.ndarray, np.ndarray]:
    target, current = checked_field(target), checked_field(initial).copy()
    if current.shape != target.shape or iterations < 1:
        raise ValueError("need matching shapes and positive iterations")
    amplitude = np.abs(np.fft.fft2(target))
    for _ in range(iterations):
        exact_spectrum = spectral_projection(current, amplitude)
        current = rank_match(exact_spectrum, target)
    return exact_spectrum, current


def histogram_ks(candidate: np.ndarray, target: np.ndarray) -> float:
    a, b = np.sort(candidate.ravel()), np.sort(target.ravel())
    grid = np.sort(np.concatenate([a, b]))
    return float(np.max(np.abs(np.searchsorted(a, grid, side="right") / len(a)
                               - np.searchsorted(b, grid, side="right") / len(b))))


def surrogate_quality(exact_spectrum: np.ndarray, exact_histogram: np.ndarray, target: np.ndarray) -> dict:
    amplitude = np.abs(np.fft.fft2(target))
    scale = np.linalg.norm(amplitude)
    return {"spectrum_stage_relative_error": float(np.linalg.norm(np.abs(np.fft.fft2(exact_spectrum)) - amplitude) / scale),
            "histogram_stage_exact": bool(np.array_equal(np.sort(exact_histogram.ravel()), np.sort(target.ravel()))),
            "histogram_stage_spectrum_error": float(np.linalg.norm(np.abs(np.fft.fft2(exact_histogram)) - amplitude) / scale),
            "spectrum_stage_histogram_ks": histogram_ks(exact_spectrum, target)}


def pd0_summary(field: np.ndarray) -> np.ndarray:
    """Finite open-boundary PD0 component count and total lifetime, elder rule.

    Reuses the shipped UnionFindPersistence; omits the essential global component.
    No claim is made about PD1, torus topology, or boundaries outside this patch.
    """
    x = checked_field(field)
    h, w = x.shape
    uf = UnionFindPersistence(x.ravel())
    pairs = []
    for value_index in np.argsort(x.ravel(), kind="stable"):
        idx = int(value_index)
        uf.active[idx] = True
        row, col = divmod(idx, w)
        for nb in (idx - w if row else -1, idx + w if row + 1 < h else -1,
                   idx - 1 if col else -1, idx + 1 if col + 1 < w else -1):
            if nb >= 0 and uf.active[nb]:
                uf.union(idx, nb, float(x.ravel()[idx]), pairs)
    return np.array([len(pairs), sum(d - b for b, d in pairs)], dtype=np.float64)


def iaaft_gpu(target: np.ndarray, initials: np.ndarray, iterations: int, device: str = "cuda:0"):
    """Double-precision CUDA reference-equivalent projection/rank steps, batched."""
    import torch

    target = checked_field(target)
    initials = np.asarray(initials, dtype=np.float64)
    if initials.ndim != 3 or initials.shape[1:] != target.shape or not len(initials) or not np.all(np.isfinite(initials)) or iterations < 1:
        raise ValueError("need finite batched initials with target shape and positive iterations")
    z = torch.as_tensor(initials, dtype=torch.float64, device=device).clone()
    t = torch.as_tensor(target, dtype=torch.float64, device=device)
    amp = torch.abs(torch.fft.fft2(t))
    values = torch.sort(t.ravel(), stable=True).values.expand(len(z), -1)
    for _ in range(iterations):
        f = torch.fft.fft2(z)
        phase = torch.where(torch.abs(f) > 0, f / torch.abs(f).clamp_min(1e-300), torch.ones_like(f))
        exact_spectrum = torch.fft.ifft2(amp * phase).real
        order = torch.argsort(exact_spectrum.flatten(1), dim=1, stable=True)
        z = torch.empty_like(order, dtype=torch.float64).scatter_(1, order, values).reshape_as(z)
    return exact_spectrum.cpu().numpy(), z.cpu().numpy()
