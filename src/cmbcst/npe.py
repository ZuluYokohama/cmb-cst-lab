"""Conditional masked autoregressive flow (MAF) for neural posterior estimation: numpy reference.

The density of standardised targets x (dim D) given standardised summaries c (dim K) is a stack
of L autoregressive affine layers. Layer l (with a fixed variable order) computes, for each i,

    h = tanh(W1m x + V1 c + b1),  h = tanh(W2m h + b2),  [mu, s] = W3m h + b3     (masked: MADE)
    u_i = (x_i - mu_i) * exp(-s_i),   s clipped to [-S_MAX, S_MAX] with a tanh,

then reverses the variable order for the next layer. log p(x | c) = log N(u_L; 0, I) - sum s.
The PyTorch trainer (npe_torch.py) uses the same parameterisation; `log_prob` here reproduces
it from exported weights so the sandbox can recompute device outputs.
"""
from __future__ import annotations

import numpy as np

S_MAX = 5.0


def made_masks(d: int, hidden: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MADE masks for inputs 1..d (degree order), two hidden layers of width `hidden`.
    Output unit for variable i may depend only on inputs < i."""
    deg_in = np.arange(1, d + 1)
    deg_h = (np.arange(hidden) % max(d - 1, 1)) + 1
    m1 = (deg_h[:, None] >= deg_in[None, :]).astype(np.float64)      # (H, D)
    m2 = (deg_h[:, None] >= deg_h[None, :]).astype(np.float64)       # (H, H)
    deg_out = np.concatenate([deg_in, deg_in])
    m3 = (deg_out[:, None] > deg_h[None, :]).astype(np.float64)      # (2D, H)
    return m1, m2, m3


def _layer_forward(x: np.ndarray, c: np.ndarray, p: dict, masks) -> tuple[np.ndarray, np.ndarray]:
    m1, m2, m3 = masks
    h = np.tanh(x @ (p["W1"] * m1).T + c @ p["V1"].T + p["b1"])
    h = np.tanh(h @ (p["W2"] * m2).T + p["b2"])
    out = h @ (p["W3"] * m3).T + p["b3"]
    d = x.shape[-1]
    mu, s = out[..., :d], S_MAX * np.tanh(out[..., d:] / S_MAX)
    return (x - mu) * np.exp(-s), s


def log_prob(x: np.ndarray, c: np.ndarray, layers: list[dict], hidden: int) -> np.ndarray:
    """log p(x | c) for standardised x (N, D) and c (N, K), float64."""
    d = x.shape[-1]
    masks = made_masks(d, hidden)
    total = np.zeros(x.shape[0])
    u = np.asarray(x, dtype=np.float64)
    for k, p in enumerate(layers):
        u, s = _layer_forward(u, c, p, masks)
        total -= s.sum(-1)
        if k < len(layers) - 1:
            u = u[..., ::-1]
    return total - 0.5 * np.sum(u**2, -1) - 0.5 * d * np.log(2 * np.pi)


def sbc_ranks(posterior_samples: np.ndarray, truths: np.ndarray) -> np.ndarray:
    """Simulation-based calibration ranks: #{posterior samples < truth} per simulation and
    parameter. posterior_samples (N, S, D), truths (N, D) -> (N, D) integers in [0, S]."""
    return np.sum(posterior_samples < truths[:, None, :], axis=1)


def rank_uniformity(ranks: np.ndarray, n_samples: int, n_bins: int = 10) -> dict[str, np.ndarray]:
    """Chi^2 of rank histograms against uniform, per parameter, and the coverage of central
    68% / 95% intervals implied by the ranks."""
    n, d = ranks.shape
    edges = np.linspace(0, n_samples + 1, n_bins + 1)
    chi2 = np.zeros(d)
    for j in range(d):
        h, _ = np.histogram(ranks[:, j], bins=edges)
        e = n / n_bins
        chi2[j] = np.sum((h - e) ** 2 / e)
    q = ranks / n_samples
    cov68 = np.mean((q > 0.16) & (q < 0.84), axis=0)
    cov95 = np.mean((q > 0.025) & (q < 0.975), axis=0)
    return {"chi2": chi2, "dof": n_bins - 1, "coverage68": cov68, "coverage95": cov95}


def weighted_q(v: np.ndarray, w: np.ndarray) -> dict:
    """Importance-weighted median and 68/95% intervals of samples v with weights w, plus the
    effective sample size (sum w)^2 / sum w^2. Used to re-express the posterior under a different
    prior on one parameter when the training prior on it was uniform."""
    order = np.argsort(v)
    cw = np.cumsum(w[order]) / w.sum()
    pick = lambda p: float(v[order][min(np.searchsorted(cw, p), len(v) - 1)])
    return {"median": pick(0.5), "lo68": pick(0.16), "hi68": pick(0.84), "lo95": pick(0.025), "hi95": pick(0.975),
            "ess": float(w.sum() ** 2 / np.sum(w**2))}


def interval_scale(med: np.ndarray, lo68: np.ndarray, hi68: np.ndarray, lo95: np.ndarray, hi95: np.ndarray,
                   truth: np.ndarray) -> dict:
    """Factors k by which the central 68/95% half-widths must be scaled (about the median) to cover
    the truth at the nominal rate on simulations. Pre-registered: the unblind widens real-data
    intervals by max(1, k) and never narrows them."""
    err = np.abs(truth - med)
    return {"k68": float(np.quantile(err / ((hi68 - lo68) / 2), 0.68)),
            "k95": float(np.quantile(err / ((hi95 - lo95) / 2), 0.95))}
