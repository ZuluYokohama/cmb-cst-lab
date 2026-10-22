"""Sky regions for the region-resolved calibration network.

Two partitions of the analysis mask "0":
    base12  the 12 HEALPix N_side = 1 base pixels (equal-area, Galaxy-agnostic)
    lat8    4 Galactic-latitude bands x 2 hemispheres (ordered by dust column)

Each region's weight is its binary footprint times a 1-degree Gaussian taper of itself, so the
weight vanishes outside the region and rolls off at its edges (limits mode coupling between
neighbouring regions).
"""
from __future__ import annotations

import numpy as np

NSIDE = 2048
LAT_EDGES = (0.0, 20.0, 40.0, 60.0, 90.0)


def region_ids(scheme: str) -> tuple[np.ndarray, list[str]]:
    """Region index of every N_side 2048 RING pixel, and region labels."""
    import ducc0

    ang = ducc0.healpix.Healpix_Base(NSIDE, "RING").pix2ang(np.arange(12 * NSIDE**2, dtype=np.int64))
    if scheme == "base12":
        return ducc0.healpix.Healpix_Base(1, "RING").ang2pix(ang).astype(np.int16), [f"P{k:02d}" for k in range(12)]
    if scheme == "lat8":
        b = 90.0 - np.degrees(ang[:, 0])
        band = np.clip(np.searchsorted(np.array(LAT_EDGES[1:-1]), np.abs(b), side="right"), 0, 3)
        ids = (band + 4 * (b < 0)).astype(np.int16)
        labels = [f"{'N' if h == 0 else 'S'}{int(LAT_EDGES[k])}-{int(LAT_EDGES[k + 1])}" for h in (0, 1) for k in range(4)]
        return ids, labels
    raise ValueError(scheme)


def region_weight(ids: np.ndarray, k: int, mask: np.ndarray, taper_deg: float = 1.0) -> np.ndarray:
    """Footprint of region k inside the analysis mask, with a Gaussian edge taper."""
    from .masks import smooth_scalar

    binary = ((ids == k) & (mask > 0.5)).astype(np.float64)
    taper = np.clip(smooth_scalar(binary, taper_deg), 0.0, 1.0)
    return binary * taper
