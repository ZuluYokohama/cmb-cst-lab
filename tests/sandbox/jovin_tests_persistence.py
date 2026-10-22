"""JoVin sandbox tests for cmbcst.persistence (numpy only)."""
from __future__ import annotations

import numpy as np

from solution import compute_cubical_persistent_homology_2d, evaluate_betti_and_euler, gnomonic_angles, lifetimes


def _components(mask: np.ndarray, conn8: bool = False) -> int:
    """Independent flood-fill count of periodic components."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)] + ([(-1, -1), (-1, 1), (1, -1), (1, 1)] if conn8 else [])
    n = 0
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not seen[r, c]:
                n += 1
                stack = [(r, c)]
                seen[r, c] = True
                while stack:
                    y, x = stack.pop()
                    for dy, dx in steps:
                        yy, xx = (y + dy) % h, (x + dx) % w
                        if mask[yy, xx] and not seen[yy, xx]:
                            seen[yy, xx] = True
                            stack.append((yy, xx))
    return n


def test_beta0_from_pd0_matches_flood_fill():
    rng = np.random.default_rng(0)
    f = rng.standard_normal((20, 20))
    pd0, pd1 = compute_cubical_persistent_homology_2d(f)
    for nu in np.quantile(f, [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]):
        b0, _, _ = evaluate_betti_and_euler(pd0, pd1, np.array([nu]))
        # pairs are half-open [b, d): a component merging exactly at nu is already gone at nu
        assert b0[0] == _components(f <= nu)


def test_pd1_counts_superlevel_components():
    # PD_1 is built from 8-connected superlevel components of f: each finite pair alive at nu
    # is one superlevel component minus the essential one
    rng = np.random.default_rng(1)
    f = rng.standard_normal((18, 18))
    pd0, pd1 = compute_cubical_persistent_homology_2d(f)
    for nu in np.quantile(f, [0.3, 0.6, 0.9]):
        alive = np.sum((pd1[:, 0] < nu) & (nu <= pd1[:, 1]))
        assert alive == _components(f >= nu, conn8=True) - 1


def test_three_dips_give_two_long_lived_components():
    y, x = np.mgrid[0:40, 0:40]
    f = np.zeros((40, 40))
    for cy, cx, depth in ((8, 8, 3.0), (30, 12, 2.0), (20, 32, 1.5)):
        f -= depth * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / 8.0)
    pd0, _ = compute_cubical_persistent_homology_2d(f)
    life = lifetimes(pd0, drop_essential=True)
    long = np.sort(life[life > 0.5])
    assert len(long) == 2
    assert np.allclose(long, [1.5, 2.0], atol=0.05)


def test_pd1_duality_truncates_an_enclosed_peak():
    # Known limitation of the dual PD_1 construction, pinned so it stays documented: a cold ring (min -1) around a hot
    # peak (+2) on a 0 background. True sublevel H_1 has one loop born when the ring closes
    # (-0.905) that dies when the hole fills at the peak (+2.0), lifetime 2.9. The dual
    # construction keeps the global maximum as the essential class and so kills the loop at the
    # outside level (0.0), lifetime 0.9. PD_1 lifetimes are only lower bounds here; the analysis
    # uses PD_0 (exact) as its primary statistic.
    y, x = np.mgrid[0:41, 0:41] - 20.0
    r = np.hypot(x, y)
    f = 2.0 * np.exp(-r**2 / 10.0) - 1.0 * np.exp(-(r - 10.0) ** 2 / 4.0)
    _, pd1 = compute_cubical_persistent_homology_2d(f, periodic=False)
    life = lifetimes(pd1)
    k = int(np.argmax(life))
    assert np.sum(life > 0.5) == 1
    assert abs(pd1[k, 0] + 0.905) < 0.01 and abs(pd1[k, 1] - 0.0) < 1e-3  # true death would be 2.0


def test_gnomonic_centre_and_extent():
    th, ph = gnomonic_angles(np.radians(147.0), np.radians(209.0), size=129, patch_deg=20.0)
    assert abs(th[64, 64] - np.radians(147.0)) < 1e-12 and abs(ph[64, 64] - np.radians(209.0)) < 1e-12
    # corner-to-centre distance of a 20 deg gnomonic square: atan(sqrt(2) tan 10 deg)
    v = lambda t, p: np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])
    d = np.arccos(np.clip(v(th[0, 0], ph[0, 0]) @ v(th[64, 64], ph[64, 64]), -1, 1))
    assert abs(d - np.arctan(np.sqrt(2) * np.tan(np.radians(10.0)))) < 1e-9
