"""Analysis masks built from public PLA products.

The paper's masks (`combined_ps_co_{0,30}_2048`, f_sky 0.92 and 0.62) are not published. Following
the construction it cites (Diego-Palazuelos et al. 2022: CO mask + point-source mask, optionally a
Galactic cut), we build:

    mask "0"  = polarized point sources (union of 100-353 GHz) x CO
    mask "30" = mask "0" x the PLA Galactic plane mask (GAL060 or GAL070) nearest f_sky 0.62

The CO threshold (applied to a 1-degree smoothed CO(2-1) map) is chosen so that mask "0" has the
paper's stated f_sky = 0.92; that number is matched, nothing downstream is tuned.
"""
from __future__ import annotations

import numpy as np

from .data import path_of
from .spectra import NTHREADS, nest_to_ring

NSIDE = 2048


def _read_field(key: str, column: str, extname: str | None = None) -> np.ndarray:
    from astropy.io import fits

    with fits.open(path_of(key), memmap=True) as h:
        hdu = h[extname] if extname else next(x for x in h if x.data is not None and column in x.columns.names)
        data = np.asarray(hdu.data[column], dtype=np.float64).ravel()
        nested = str(hdu.header.get("ORDERING", "RING")).strip().upper().startswith("NEST")
        nside = int(hdu.header.get("NSIDE", NSIDE))
    assert nside == NSIDE, f"{key}:{column} has nside {nside}"
    return nest_to_ring(data, nside) if nested else data


def columns(key: str) -> list[str]:
    from astropy.io import fits

    with fits.open(path_of(key), memmap=True) as h:
        return [c for x in h if getattr(x, "columns", None) is not None for c in x.columns.names]


def smooth_scalar(m: np.ndarray, fwhm_deg: float, lmax: int = 512) -> np.ndarray:
    import ducc0

    from .theory import gaussian_beam

    geom = ducc0.healpix.Healpix_Base(NSIDE, "RING").sht_info()
    alm = ducc0.sht.experimental.adjoint_synthesis(map=m[None], spin=0, lmax=lmax, mmax=lmax,
                                                   nthreads=NTHREADS, **geom) * (4 * np.pi / m.size)
    from .spectra import alm_ell_m

    ell, _ = alm_ell_m(lmax)
    alm = alm * gaussian_beam(fwhm_deg * 60, lmax)[ell]
    return ducc0.sht.experimental.synthesis(alm=alm, spin=0, lmax=lmax, mmax=lmax, nthreads=NTHREADS, **geom)[0]


def build_masks(target_fsky: float = 0.92) -> dict[str, object]:
    ps = np.ones(12 * NSIDE**2)
    for f in ("F100", "F143", "F217", "F353"):  # polarized point-source masks (HDU SRC-POL)
        ps *= (_read_field("pointsrc", f, "SRC-POL") > 0.5)
    co_col = "INTEN21"
    co_raw = _read_field("co", co_col)
    co = smooth_scalar(np.where(np.isfinite(co_raw) & (np.abs(co_raw) < 1e20), co_raw, 0.0), 1.0)  # drop BAD_DATA sentinels
    # fraction of sky to remove with CO so that ps * co reaches target_fsky
    keep_ps = ps > 0.5
    order = np.sort(co[keep_ps])[::-1]
    n_drop = int(round(keep_ps.sum() - target_fsky * ps.size))
    thresh = float(order[max(n_drop, 0)])
    co_mask = co < thresh
    m0 = (keep_ps & co_mask).astype(np.float64)
    # Galactic cut for mask "30": the PLA plane mask whose combination with m0 lands nearest the paper's 0.62
    cands = {g: m0 * (_read_field("galplane", g) > 0.5) for g in ("GAL060", "GAL070")}
    gal = min(cands, key=lambda g: abs(cands[g].mean() - 0.62))
    m30 = cands[gal]
    return {"0": m0, "30": m30, "co_column": co_col, "co_threshold": thresh, "galactic_mask_30": gal,
            "fsky": {"0": float(m0.mean()), "30": float(m30.mean()), "ps_only": float(ps.mean()),
                     **{f"30_with_{g}": float(c.mean()) for g, c in cands.items()}}}
