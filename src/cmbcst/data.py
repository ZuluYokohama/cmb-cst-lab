"""Public Planck inputs: manifest and a resumable, size-checked downloader.

The eight maps are the PR4 (NPIPE) HFI detector-subset splits served by NERSC. NPIPE's
A/B split for HFI is a split by detector subset; which file is "A" is checked against the
FITS header in `split_header_summary` rather than assumed.

Usage:
    python -m cmbcst.data            # download everything missing into data/raw
    python -m cmbcst.data --list     # print the manifest
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw"

_NERSC = "https://portal.nersc.gov/project/cmb/planck2020"
_PLA = "http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID="


@dataclass(frozen=True)
class Product:
    key: str
    filename: str
    url: str
    role: str


MAP_FILES = {
    "100A": "HFI_SkyMap_100-1-4_2048_R4.00_full.fits",
    "100B": "HFI_SkyMap_100-2-3_2048_R4.00_full.fits",
    "143A": "HFI_SkyMap_143-1-3-5-7_2048_R4.00_full.fits",
    "143B": "HFI_SkyMap_143-2-4-6_2048_R4.00_full.fits",
    "217A": "HFI_SkyMap_217-1-3-5-7_2048_R4.00_full.fits",
    "217B": "HFI_SkyMap_217-2-4-6-8_2048_R4.00_full.fits",
    "353A": "HFI_SkyMap_353-1-3-5-7_2048_R4.00_full.fits",
    "353B": "HFI_SkyMap_353-2-4-6-8_2048_R4.00_full.fits",
}

MASK_FILES = {
    "galplane": "HFI_Mask_GalPlane-apo0_2048_R2.00.fits",
    "pointsrc": "HFI_Mask_PointSrc_2048_R2.00.fits",
    "co": "HFI_CompMap_CO-Type1_2048_R2.00.fits",
}

MANIFEST: tuple[Product, ...] = tuple(
    [Product(k, f, f"{_NERSC}/frequency_maps/Multi-detector/{f}", "map") for k, f in MAP_FILES.items()]
    + [Product(k, f, _PLA + f, "mask") for k, f in MASK_FILES.items()]
    + [Product("rimo", "PLANCK_RIMO_TF_R4.00.tar.gz", f"{_NERSC}/misc/PLANCK_RIMO_TF_R4.00.tar.gz", "beams")]
)


def path_of(key: str) -> Path:
    return RAW / next(p.filename for p in MANIFEST if p.key == key)


def _remote_size(url: str) -> int | None:
    r = requests.head(url, allow_redirects=True, timeout=60)
    r.raise_for_status()
    n = r.headers.get("Content-Length")
    return int(n) if n else None


def fetch(p: Product, chunk: int = 1 << 22) -> Path:
    """Download one product, resuming a partial `.part` file. Complete files are skipped."""
    dest = RAW / p.filename
    part = dest.with_suffix(dest.suffix + ".part")
    total = _remote_size(p.url)
    if dest.exists() and (total is None or dest.stat().st_size == total):
        return dest
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(p.url, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        if have and r.status_code != 206:  # server ignored the range: start over
            have = 0
        t0, done = time.monotonic(), have
        with open(part, "ab" if have else "wb") as f:
            for block in r.iter_content(chunk):
                f.write(block)
                done += len(block)
        rate = (done - have) / max(time.monotonic() - t0, 1e-9) / 1e6
    if total is not None and part.stat().st_size != total:
        raise IOError(f"{p.filename}: got {part.stat().st_size} bytes, expected {total}")
    part.replace(dest)
    print(f"[ok] {p.filename} {done / 1e9:.2f} GB at {rate:.1f} MB/s", flush=True)
    return dest


def split_header_summary(key: str) -> dict[str, str]:
    """Header cards that identify which detectors went into a split map."""
    from astropy.io import fits

    with fits.open(path_of(key), memmap=True) as hdul:
        h = hdul[1].header
        return {k: str(h[k]) for k in h if any(s in k for s in ("DET", "SPLIT", "FREQ", "NSIDE", "ORDER", "COORD", "TUNIT"))}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("keys", nargs="*", help="subset of manifest keys (default: all)")
    args = ap.parse_args(argv)
    todo = [p for p in MANIFEST if not args.keys or p.key in args.keys]
    if args.list:
        for p in todo:
            print(f"{p.key:9s} {p.role:6s} {p.url}")
        return 0
    RAW.mkdir(parents=True, exist_ok=True)
    failed = []
    for p in todo:
        for attempt in range(5):
            try:
                fetch(p)
                break
            except (requests.RequestException, IOError) as e:
                print(f"[retry {attempt + 1}] {p.filename}: {e}", flush=True)
                time.sleep(10 * (attempt + 1))
        else:
            failed.append(p.filename)
    print("FAILED: " + ", ".join(failed) if failed else "all products present", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
