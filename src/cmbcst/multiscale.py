"""Finite-scale structure functions, not an estimator of asymptotic fractal dimension."""
from __future__ import annotations
import numpy as np

LAGS = (1, 2, 4, 8, 16)
FEATURE_NAMES = ("S2_log_slope", "mean_flatness_excess", "mean_sixth_excess",
                 "flatness_log_slope", "sixth_log_slope", "mean_directional_anisotropy")


def checked_patches(patches):
    x = np.asarray(patches, dtype=np.float64)
    if x.ndim == 2:
        x = x[None]
    if x.ndim != 3 or not len(x) or min(x.shape[1:]) <= max(LAGS) or not np.all(np.isfinite(x)):
        raise ValueError("need finite batched patches larger than the largest lag")
    return x


def features_from_curves(curves):
    """Columns: S2, S4, S6, S4/S2²-3, S6/S2³-15, directional contrast."""
    c = np.asarray(curves, dtype=np.float64)
    if c.ndim != 3 or c.shape[1:] != (5, 6) or not np.all(np.isfinite(c)) or np.any(c[:, :, 0] <= 0):
        raise ValueError("invalid structure curves")
    fourth, sixth = c[:, :, 3]+3, c[:, :, 4]+15
    if np.any(fourth <= 0) or np.any(sixth <= 0):
        raise ValueError("moment ratios must be positive")
    t = np.log(np.asarray(LAGS, dtype=np.float64))
    t -= t.mean()
    slope = lambda y: np.sum(np.log(y)*t, axis=1)/np.sum(t*t)
    return np.column_stack([slope(c[:, :, 0]), c[:, :, 3].mean(1), c[:, :, 4].mean(1),
                            slope(fourth), slope(sixth), c[:, :, 5].mean(1)])


def multiscale_reference(patches):
    x = checked_patches(patches)
    curves = []
    for lag in LAGS:
        dx = (x[:, :, lag:] - x[:, :, :-lag]).reshape(len(x), -1)
        dy = (x[:, lag:, :] - x[:, :-lag, :]).reshape(len(x), -1)
        increments = np.concatenate([dx, dy], axis=1)
        square = increments*increments
        m2, m4, m6 = square.mean(1), (square*square).mean(1), (square*square*square).mean(1)
        if np.any(m2 <= 0):
            raise ValueError("zero increment power")
        mx, my = (dx*dx).mean(1), (dy*dy).mean(1)
        curves.append(np.column_stack([m2, m4, m6, m4/m2**2-3, m6/m2**3-15, (mx-my)/(mx+my)]))
    result = np.stack(curves, axis=1)
    return result, features_from_curves(result)


def multiscale_cuda(patches, device="cuda:0"):
    """Compute increment moments on the actual CUDA device, float64 throughout."""
    import torch
    x = torch.as_tensor(checked_patches(patches), dtype=torch.float64, device=device)
    curves = []
    for lag in LAGS:
        dx = (x[:, :, lag:] - x[:, :, :-lag]).reshape(len(x), -1)
        dy = (x[:, lag:, :] - x[:, :-lag, :]).reshape(len(x), -1)
        d = torch.cat([dx, dy], dim=1)
        square = d*d
        m2, m4, m6 = square.mean(1), (square*square).mean(1), (square*square*square).mean(1)
        if bool(torch.any(m2 <= 0).item()):
            raise ValueError("zero increment power")
        mx, my = (dx*dx).mean(1), (dy*dy).mean(1)
        curves.append(torch.stack([m2, m4, m6, m4/m2**2-3, m6/m2**3-15, (mx-my)/(mx+my)], dim=1))
    result = torch.stack(curves, dim=1).cpu().numpy()
    return result, features_from_curves(result)
