"""NPIPE QuickPol beam window functions for the 8 A/B maps.

The 64 files `Wl_npipe6v20_{f}{s}x{f}{s}.fits` (EE_2_EE, BB_2_BB) are the ones the reference
pipeline reads; they ship in the public PLANCK_RIMO_TF_R4.00.tar.gz (NERSC planck2020/misc).
Extract them with `extract()`; `split_windows` falls back to Gaussian beams if they are missing.
"""
from __future__ import annotations

import tarfile

import numpy as np

from . import DETECTORS
from .data import REPO, path_of

BEAM_DIR = REPO / "data" / "work" / "beams"


def _name(a: str, b: str) -> str:
    return f"Wl_npipe6v20_{a}x{b}.fits"


def extract() -> None:
    want = {_name(a, b) for a in DETECTORS for b in DETECTORS}
    BEAM_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path_of("rimo"), "r:gz") as tf:
        for m in tf.getmembers():
            base = m.name.rsplit("/", 1)[-1]
            if "/AB/" in m.name and base in want and not (BEAM_DIR / base).exists():
                with tf.extractfile(m) as src, open(BEAM_DIR / base, "wb") as dst:
                    dst.write(src.read())


def split_windows(lmax: int) -> tuple[np.ndarray, np.ndarray, str]:
    """W^EE[i, j, l], W^BB[i, j, l] (beam products, no pixel window) and a provenance string."""
    from astropy.io import fits

    n = len(DETECTORS)
    wee, wbb = np.zeros((n, n, lmax + 1)), np.zeros((n, n, lmax + 1))
    if all((BEAM_DIR / _name(a, b)).exists() for a in DETECTORS for b in DETECTORS):
        for i, a in enumerate(DETECTORS):
            for j, b in enumerate(DETECTORS):
                with fits.open(BEAM_DIR / _name(a, b)) as h:
                    wee[i, j] = np.asarray(h["EE"].data["EE_2_EE"])[0, : lmax + 1]
                    wbb[i, j] = np.asarray(h["BB"].data["BB_2_BB"])[0, : lmax + 1]
        return wee, wbb, "NPIPE QuickPol Wl_npipe6v20 A/B (PLANCK_RIMO_TF_R4.00)"
    from .theory import FWHM_ARCMIN, gaussian_beam

    for i, a in enumerate(DETECTORS):
        for j, b in enumerate(DETECTORS):
            wee[i, j] = wbb[i, j] = gaussian_beam(FWHM_ARCMIN[a[:3]], lmax) * gaussian_beam(FWHM_ARCMIN[b[:3]], lmax)
    return wee, wbb, "Gaussian FWHM fallback"
