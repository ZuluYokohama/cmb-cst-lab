"""Sublevel persistent homology of sky patches (numpy-only kernels) and patch extraction.

PD_0 comes from union-find with the elder rule on the 4-connected sublevel filtration (exact).
PD_1 comes from the 8-connected superlevel components by Alexander duality. That is exact in the
plane, but the dual construction keeps the global maximum as the essential class, so a hot spot
enclosed by a colder ring has its loop lifetime cut at the outside level. PD_1 lifetimes are
therefore lower bounds, and the analysis uses PD_0.

Patches are gnomonic projections; pixel lookup goes through ducc0's HEALPix geometry.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

class UnionFindPersistence:
    """Exact 0-dimensional persistent homology via Disjoint-Set with the Elder Rule."""

    def __init__(self, values: np.ndarray):
        self.values = values.ravel()
        n = len(self.values)
        self.parent = np.arange(n, dtype=np.int32)
        self.birth = self.values.copy()
        self.active = np.zeros(n, dtype=bool)

    def find(self, i: int) -> int:
        path = []
        root = i
        while self.parent[root] != root:
            path.append(root)
            root = self.parent[root]
        for node in path:
            self.parent[node] = root
        return root

    def union(self, i: int, j: int, current_val: float, pairs: List[Tuple[float, float]]) -> None:
        ri = self.find(i)
        rj = self.find(j)
        if ri == rj:
            return
        if self.birth[ri] <= self.birth[rj]:
            elder, younger = ri, rj
        else:
            elder, younger = rj, ri
        b_young = float(self.birth[younger])
        d_young = float(current_val)
        if d_young > b_young:
            pairs.append((b_young, d_young))
        self.parent[younger] = elder


def compute_cubical_persistent_homology_2d(
    field: np.ndarray, periodic: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Return exact PD_0 for the chosen four-neighbor grid graph, plus a PD_1 proxy.

    The dual eight-neighbor superlevel construction does NOT implement a full
    exact cubical-complex PD_1 calculation. Its lifetimes are lower-bound
    proxies, and the public sky-patch statistical analysis uses PD_0 only.
    """
    h, w = field.shape
    flat = field.ravel()
    order_sub = np.argsort(flat, kind="mergesort")

    uf0 = UnionFindPersistence(flat)
    pd0_pairs: List[Tuple[float, float]] = []
    for idx in order_sub:
        uf0.active[idx] = True
        val = float(flat[idx])
        r, c = divmod(int(idx), w)
        neighbors = []
        if r > 0:
            neighbors.append((r - 1) * w + c)
        elif periodic:
            neighbors.append((h - 1) * w + c)
        if r < h - 1:
            neighbors.append((r + 1) * w + c)
        elif periodic:
            neighbors.append(0 * w + c)
        if c > 0:
            neighbors.append(r * w + (c - 1))
        elif periodic:
            neighbors.append(r * w + (w - 1))
        if c < w - 1:
            neighbors.append(r * w + (c + 1))
        elif periodic:
            neighbors.append(r * w + 0)

        for nb in neighbors:
            if uf0.active[nb]:
                uf0.union(int(idx), nb, val, pd0_pairs)

    pd0_pairs.append((float(np.min(flat)), float(np.max(flat))))
    pd0 = np.array(pd0_pairs, dtype=np.float64)

    neg_flat = -flat
    order_sup = np.argsort(neg_flat, kind="mergesort")
    uf1 = UnionFindPersistence(neg_flat)
    pd1_dual_pairs: List[Tuple[float, float]] = []
    for idx in order_sup:
        uf1.active[idx] = True
        val = float(neg_flat[idx])
        r, c = divmod(int(idx), w)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if periodic:
                    nr %= h
                    nc %= w
                elif not (0 <= nr < h and 0 <= nc < w):
                    continue
                nb = nr * w + nc
                if uf1.active[nb]:
                    uf1.union(int(idx), nb, val, pd1_dual_pairs)

    pd1_pairs = [(-d_neg, -b_neg) for (b_neg, d_neg) in pd1_dual_pairs if -b_neg > -d_neg]
    pd1 = np.array(pd1_pairs, dtype=np.float64) if pd1_pairs else np.zeros((0, 2))
    return pd0, pd1


def evaluate_betti_and_euler(
    pd0: np.ndarray, pd1: np.ndarray, thresholds: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Implements Eq. (2.8) of Cole & Shiu (2018):
       beta_k(nu) = sum_{b <= nu < d} O_k(b, d),   chi(nu) = beta_0(nu) - beta_1(nu).
    """
    beta0 = np.array([np.sum((pd0[:, 0] <= nu) & (nu < pd0[:, 1])) for nu in thresholds], dtype=np.float64)
    beta1 = np.array([np.sum((pd1[:, 0] <= nu) & (nu < pd1[:, 1])) for nu in thresholds], dtype=np.float64)
    chi = beta0 - beta1
    return beta0, beta1, chi


def lifetimes(pd: np.ndarray, drop_essential: bool = False) -> np.ndarray:
    p = pd[:-1] if drop_essential else pd
    return p[:, 1] - p[:, 0] if len(p) else np.zeros(0)


def patch_summary(field: np.ndarray) -> dict[str, float]:
    """Statistics compared between real patches and Gaussian simulations.

    Patches are not periodic (gnomonic projections), so periodic=False; PD_1 lifetimes are
    lower bounds (see jovin_tests_persistence.test_pd1_duality_truncates_an_enclosed_peak)."""
    pd0, pd1 = compute_cubical_persistent_homology_2d(field, periodic=False)
    l0, l1 = lifetimes(pd0, drop_essential=True), lifetimes(pd1)
    return {"n0": float(len(l0)), "n1": float(len(l1)), "mean_life0": float(l0.mean()),
            "mean_life1": float(l1.mean()) if len(l1) else 0.0, "max_life0": float(l0.max()),
            "min_value": float(field.min())}


def gnomonic_angles(center_theta: float, center_phi: float, size: int = 128, patch_deg: float = 20.0):
    """Colatitude/longitude of a size x size tangent-plane grid (inverse gnomonic projection)."""
    half = 0.5 * np.deg2rad(patch_deg)
    u = np.linspace(-np.tan(half), np.tan(half), size)
    xx, yy = np.meshgrid(u, u, indexing="ij")
    rho = np.sqrt(xx**2 + yy**2)
    c = np.arctan(rho)
    cos_c, sin_c = np.cos(c), np.sin(c)
    st0, ct0 = np.sin(center_theta), np.cos(center_theta)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.clip(cos_c * ct0 + np.where(rho > 1e-12, yy * sin_c * st0 / rho, 0.0), -1.0, 1.0)
        phi = np.mod(center_phi + np.arctan2(xx * sin_c, rho * st0 * cos_c - yy * ct0 * sin_c + 1e-15), 2 * np.pi)
    return np.arccos(z), phi


def standardise(patch: np.ndarray) -> np.ndarray:
    return (patch - patch.mean()) / (patch.std() + 1e-15)


def project_patch(hpx_map: np.ndarray, center_theta: float, center_phi: float, size: int = 128,
                  patch_deg: float = 20.0) -> np.ndarray:
    """Nearest-pixel gnomonic patch of a RING-ordered HEALPix map (not standardised)."""
    import ducc0

    nside = int(round(np.sqrt(hpx_map.size / 12)))
    th, ph = gnomonic_angles(center_theta, center_phi, size, patch_deg)
    pix = ducc0.healpix.Healpix_Base(nside, "RING").ang2pix(np.stack([th.ravel(), ph.ravel()], axis=1))
    return hpx_map[pix].reshape(th.shape)


def mc_pvalue(sims: np.ndarray, observed: float) -> float:
    """Two-sided Monte Carlo p-value with the +1 correction (never zero):
    p = min(1, 2 min((1 + #{T_sim <= T_obs}) / (1 + N), (1 + #{T_sim >= T_obs}) / (1 + N)))."""
    sims = np.asarray(sims, dtype=float)
    if sims.ndim != 1 or sims.size < 1 or not np.isfinite(observed) or not np.all(np.isfinite(sims)):
        raise ValueError("Observed value and a non-empty finite 1-D null sample are required")
    n = sims.size
    lo = (1 + np.sum(sims <= observed)) / (1 + n)
    hi = (1 + np.sum(sims >= observed)) / (1 + n)
    return float(min(1.0, 2 * min(lo, hi)))


def holm(pvals) -> np.ndarray:
    """Holm step-down adjusted p-values (family-wise error control), in the input order."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj
