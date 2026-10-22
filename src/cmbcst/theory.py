"""LCDM theory spectra (CAMB, same parameters as the reference config), pixel window and beams."""
from __future__ import annotations

import numpy as np

# Reference config (configs/planck_hfi.yml, `theory` block)
PLANCK18 = {"tau": 0.0544, "ns": 0.9649, "H0": 67.36, "ombh2": 0.02237, "omch2": 0.12, "As": 2.1e-9}

# NPIPE effective Gaussian FWHM (arcmin), used only if no transfer function is available
FWHM_ARCMIN = {"100": 9.66, "143": 7.22, "217": 4.90, "353": 4.92}


def lcdm_ee_bb(lmax: int = 1600) -> tuple[np.ndarray, np.ndarray]:
    """Lensed C_l^EE, C_l^BB in uK^2 (not D_l), index = ell."""
    import camb

    pars = camb.set_params(lmax=lmax, **PLANCK18)
    cl = camb.get_results(pars).get_cmb_power_spectra(lmax=lmax, raw_cl=True, CMB_unit="muK")["total"]
    return cl[:, 1].copy(), cl[:, 2].copy()


def legendre_mean(cosg: np.ndarray, lmax: int) -> np.ndarray:
    """mean over samples of P_l(cos g) for l = 0..lmax (three-term recurrence)."""
    out = np.zeros(lmax + 1)
    p0, p1 = np.ones_like(cosg), cosg.copy()
    out[0], out[1] = 1.0, p1.mean()
    for ell in range(2, lmax + 1):
        p0, p1 = p1, ((2 * ell - 1) * cosg * p1 - (ell - 1) * p0) / ell
        out[ell] = p1.mean()
    return out


def pixel_window(nside: int, lmax: int, n_samples: int = 2_000_000, seed: int = 0) -> np.ndarray:
    """Monte Carlo HEALPix pixel window w_l = < P_l(cos gamma) >, gamma = angle from a uniformly
    drawn point to the centre of its pixel. Approximates healpy's pixwin (which averages the same
    kernel over pixel shapes); used only for the theory term of the MK likelihood."""
    import ducc0

    rng = np.random.default_rng(seed)
    base = ducc0.healpix.Healpix_Base(nside, "RING")
    out = np.zeros(lmax + 1)
    chunk = 250_000
    for start in range(0, n_samples, chunk):
        m = min(chunk, n_samples - start)
        z = rng.uniform(-1, 1, m)
        phi = rng.uniform(0, 2 * np.pi, m)
        s = np.sqrt(1 - z * z)
        v = np.stack([s * np.cos(phi), s * np.sin(phi), z], axis=1)
        c = base.pix2vec(base.vec2pix(v))
        out += legendre_mean(np.clip(np.sum(v * c, axis=1), -1, 1), lmax) * m
    return out / n_samples


def gaussian_beam(fwhm_arcmin: float, lmax: int) -> np.ndarray:
    sigma = np.radians(fwhm_arcmin / 60.0) / np.sqrt(8 * np.log(2))
    ell = np.arange(lmax + 1)
    return np.exp(-0.5 * ell * (ell + 1) * sigma**2)
