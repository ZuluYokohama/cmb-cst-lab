"""Maps -> masked spin-2 harmonic coefficients -> binned cross-spectra of all 16 fields.

Pseudo-C_l with an f_sky correction (NaMaster has no Windows build; the paper used it). For the
differential estimator the correction is irrelevant (it is a ratio), and E/B mode coupling by
the mask leaves the antisymmetric D_ij with zero expectation from EE or BB.
"""
from __future__ import annotations

import os

import ducc0
import numpy as np

NTHREADS = os.cpu_count() or 4


def nest_to_ring(m: np.ndarray, nside: int) -> np.ndarray:
    idx = ducc0.healpix.Healpix_Base(nside, "NESTED").nest2ring(np.arange(12 * nside * nside, dtype=np.int64))
    out = np.empty_like(m)
    out[idx] = m
    return out


def load_qu(path: str, nside: int = 2048) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Q, U in uK_CMB (RING order) and a validity mask from a Planck PR4 map file."""
    from astropy.io import fits

    with fits.open(path, memmap=True) as h:
        hd = h[1].header
        assert int(hd["NSIDE"]) == nside and str(hd["POLCCONV"]).strip() == "COSMO"
        q = np.asarray(h[1].data["Q_STOKES"], dtype=np.float64).ravel()
        u = np.asarray(h[1].data["U_STOKES"], dtype=np.float64).ravel()
        nested = str(hd["ORDERING"]).strip().upper().startswith("NEST")
    good = (np.abs(q) < 1e20) & (np.abs(u) < 1e20)
    q, u = np.where(good, q * 1e6, 0.0), np.where(good, u * 1e6, 0.0)
    if nested:
        q, u, good = nest_to_ring(q, nside), nest_to_ring(u, nside), nest_to_ring(good, nside)
    return q, u, good


def _geom(nside: int) -> dict[str, np.ndarray]:
    return ducc0.healpix.Healpix_Base(nside, "RING").sht_info()


def map2alm_pol(q: np.ndarray, u: np.ndarray, weight: np.ndarray, lmax: int) -> np.ndarray:
    """(E, B) alms of the weighted Q/U map, healpy ordering, area-weighted adjoint synthesis."""
    nside = int(round(np.sqrt(q.size / 12)))
    alm = ducc0.sht.experimental.adjoint_synthesis(
        map=np.stack([q * weight, u * weight]), spin=2, lmax=lmax, mmax=lmax, nthreads=NTHREADS, **_geom(nside))
    return alm * (4 * np.pi / q.size)


def alm2map_pol(alm: np.ndarray, nside: int, lmax: int) -> np.ndarray:
    """Q, U maps (2, npix) from (E, B) alms."""
    return ducc0.sht.experimental.synthesis(alm=alm, spin=2, lmax=lmax, mmax=lmax, nthreads=NTHREADS, **_geom(nside))


def alm_ell_m(lmax: int) -> tuple[np.ndarray, np.ndarray]:
    """ell and m of each coefficient in healpy (m-major) ordering."""
    m = np.concatenate([np.full(lmax + 1 - mm, mm) for mm in range(lmax + 1)])
    ell = np.concatenate([np.arange(mm, lmax + 1) for mm in range(lmax + 1)])
    return ell, m


def cross_cl(a: np.ndarray, b: np.ndarray, lmax: int) -> np.ndarray:
    """C_l = sum_m Re(a_lm b_lm^*) / (2l+1), counting m > 0 twice (real fields)."""
    ell, m = alm_ell_m(lmax)
    w = np.where(m == 0, 1.0, 2.0)
    return np.bincount(ell, weights=w * (a * np.conj(b)).real, minlength=lmax + 1) / (2 * np.arange(lmax + 1) + 1)


def spectra_matrix(alms: list[np.ndarray], lmax: int) -> np.ndarray:
    """cl[ell, f, g] over fields f = 2*i + (E=0, B=1) for maps i."""
    fields = [x for a in alms for x in (a[0], a[1])]
    nf = len(fields)
    cl = np.zeros((lmax + 1, nf, nf))
    for f in range(nf):
        for g in range(f, nf):
            cl[:, f, g] = cl[:, g, f] = cross_cl(fields[f], fields[g], lmax)
    return cl


def bin_spectra(cl: np.ndarray, lmin: int = 51, lmax: int = 1491, width: int = 20,
                fsky_w2: float = 1.0, fsky_eff: float = 1.0) -> dict[str, np.ndarray]:
    """Uniform bins [lmin, lmin+width), ... up to lmax (exclusive). Divides by w2 = <mask^2>;
    nu_b = fsky_eff * sum(2l+1) is the effective number of modes for Knox variances."""
    edges = np.arange(lmin, lmax + 1, width)
    ells = np.arange(cl.shape[0])
    nb = len(edges) - 1
    out = np.zeros((nb,) + cl.shape[1:])
    nu = np.zeros(nb)
    for b in range(nb):
        sel = slice(edges[b], edges[b + 1])
        out[b] = cl[sel].mean(axis=0) / fsky_w2
        nu[b] = fsky_eff * np.sum(2 * ells[sel] + 1)
    return {"cl": out, "nu": nu, "edges": edges, "ell": 0.5 * (edges[:-1] + edges[1:] - 1)}


def mask_moments(w: np.ndarray) -> tuple[float, float]:
    """(w2, fsky_eff) with w2 = <w^2> and fsky_eff = w2^2 / <w^4>."""
    w2, w4 = float(np.mean(w**2)), float(np.mean(w**4))
    return w2, w2 * w2 / w4


def random_alm(cl: np.ndarray, lmax: int, rng: np.random.Generator) -> np.ndarray:
    """Gaussian alm with power cl (m=0 real)."""
    ell, m = alm_ell_m(lmax)
    amp = np.sqrt(cl[ell] / np.where(m == 0, 1.0, 2.0))
    z = rng.standard_normal(ell.size) + 1j * rng.standard_normal(ell.size) * (m > 0)
    return amp * z


def rotate_qu(q: np.ndarray, u: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Q + iU -> (Q + iU) e^{2i alpha}: the polarisation rotation the estimator measures."""
    c, s = np.cos(2 * alpha), np.sin(2 * alpha)
    return c * q - s * u, s * q + c * u
