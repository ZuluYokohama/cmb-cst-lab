"""Whole-sphere, fixed-C_ell Gaussian conditional directions and patch observables.

For an isotropic Gaussian scalar field, each real harmonic shell direction is
uniform on its sphere conditional on its norm. This is an explicit null model,
not a foreground-cleaned CMB simulation or a new CST theorem.
"""
from __future__ import annotations

import numpy as np

from .spectra import alm_ell_m, cross_cl
from .surrogate_topology import pd0_summary, checked_field
from .copula_topology import normal_scores


def checked_harmonic_inputs(cl, normals):
    cl = np.asarray(cl, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    if cl.ndim != 1 or len(cl) < 2 or not np.all(np.isfinite(cl)) or np.any(cl < 0):
        raise ValueError("need a finite nonnegative C_ell vector")
    nalm = len(cl) * (len(cl) + 1) // 2
    if normals.ndim != 3 or normals.shape[1:] != (2, nalm) or not len(normals) or not np.all(np.isfinite(normals)):
        raise ValueError("need finite batched real/imaginary normal arrays")
    return cl, normals


def fixed_power_reference(cl, normals):
    """Normalize independent real Gaussian harmonic shells to specified C_ell."""
    cl, normals = checked_harmonic_inputs(cl, normals)
    lmax = len(cl) - 1
    ell, m = alm_ell_m(lmax)
    raw = (normals[:, 0] + 1j * normals[:, 1] * (m > 0)) / np.sqrt(np.where(m == 0, 1., 2.))
    power = np.stack([cross_cl(a, a, lmax) for a in raw])
    if np.any((power == 0) & (cl > 0)):
        raise ValueError("a positive-power shell has a zero direction")
    scale = np.sqrt(np.divide(cl, power, out=np.zeros_like(power), where=power > 0))
    return raw * scale[:, ell]


def fixed_power_cuda(cl, normals, device="cuda:0"):
    """Same shell normalization on CUDA in float64; recorded outputs are audited.

    CUDA indexed reductions can vary in their final rounding bits. Every actual
    draw is independently checked in NumPy against the fixed power tolerance.
    """
    import torch

    cl, normals = checked_harmonic_inputs(cl, normals)
    ell, m = alm_ell_m(len(cl)-1)
    z = torch.as_tensor(normals, dtype=torch.float64, device=device)
    multipole = torch.as_tensor(ell, dtype=torch.int64, device=device)
    nonzero_m = torch.as_tensor(m > 0, device=device)
    weight = torch.where(nonzero_m, 2., 1.).to(torch.float64)
    real = z[:, 0] / torch.sqrt(weight)
    imag = z[:, 1] * nonzero_m / torch.sqrt(weight)
    shell = torch.zeros((len(z), len(cl)), dtype=torch.float64, device=device)
    shell.scatter_add_(1, multipole.expand(len(z), -1), (real*real + imag*imag)*weight)
    shell /= 2*torch.arange(len(cl), dtype=torch.float64, device=device) + 1
    target = torch.as_tensor(cl, dtype=torch.float64, device=device)
    if bool(torch.any((shell == 0) & (target > 0)).item()):
        raise ValueError("a positive-power shell has a zero direction")
    scale = torch.sqrt(target / torch.where(shell > 0, shell, torch.ones_like(shell)))
    return torch.complex(real*scale[:, multipole], imag*scale[:, multipole]).cpu().numpy()


def patch_observables(field):
    """Rank PD0 count/lifetime plus conventional temperature std/skew/kurtosis."""
    x = checked_field(field)
    std = float(x.std())
    z = (x-x.mean())/std
    count, total = pd0_summary(normal_scores(x))
    return np.array([count, total, std, np.mean(z**3), np.mean(z**4)-3], dtype=np.float64)
