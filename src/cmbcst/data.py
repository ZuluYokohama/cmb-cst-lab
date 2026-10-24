"""Public Planck inputs: resumable HTTPS downloads, structural checks and local SHA256 provenance.

The eight maps are the PR4 (NPIPE) HFI detector-subset splits served by NERSC. NPIPE's
A/B split for HFI is a split by detector subset; which file is "A" is checked against the
FITS header in `split_header_summary` rather than assumed.

Usage:
    python -m cmbcst.data            # download everything missing into data/raw
    python -m cmbcst.data --list     # print the manifest
    python -m cmbcst.data --raw-dir /path/to/planck
    python -m cmbcst.data --verify-only --provenance data/evidence/planck_inputs.json

CMB_CST_RAW_DIR also redirects path_of() for analysis programs. Existing partial files without
an entity-validator sidecar restart safely. Locally computed hashes are not publisher checksums.
Structural checks do not establish the expected scientific columns/HDUs or validate sky values.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import tarfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw"

_NERSC = "https://portal.nersc.gov/cfs/cmb/planck2020"
_PLA = "https://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID="


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


def raw_directory(raw_dir: Path | str | None = None) -> Path:
    return Path(raw_dir or os.environ.get("CMB_CST_RAW_DIR") or RAW).expanduser().resolve()


def path_of(key: str) -> Path:
    return raw_directory() / next(p.filename for p in MANIFEST if p.key == key)


def transfer_layout(status: int, headers: dict, offset: int, known_total: int | None = None) -> tuple[int, int | None, int | None]:
    """Validate HTTP framing before writing. Returns write offset, body length, total."""
    import re

    h = {k.lower(): str(v).strip() for k, v in headers.items()}
    if h.get("content-encoding", "identity").lower() not in ("", "identity"):
        raise ValueError("encoded response cannot be checked against byte ranges")
    length = h.get("content-length")
    if length is not None and not re.fullmatch(r"[0-9]+", length):
        raise ValueError("invalid Content-Length")
    length = int(length) if length is not None else None
    if status == 206:
        match = re.fullmatch(r"bytes ([0-9]+)-([0-9]+)/([0-9]+)", h.get("content-range", ""))
        if not match:
            raise ValueError("partial response needs a complete Content-Range")
        start, end, total = map(int, match.groups())
        if start != offset or not start <= end < total:
            raise ValueError("Content-Range does not match the requested offset")
        if known_total is not None and known_total != total:
            raise ValueError("remote size changed during a ranged transfer")
        if length is not None and length != end - start + 1:
            raise ValueError("Content-Length differs from Content-Range")
        return start, end - start + 1, total
    if status != 200 or "content-range" in h:
        raise ValueError(f"unexpected download response status/framing: {status}")
    # A server may ignore Range or change its entity and answer 200; start from zero.
    return 0, length, length if length is not None else known_total


def fits_data_bytes(cards: dict[str, str]) -> int:
    """FITS HDU payload size, excluding its 2880-byte padding; no data allocation."""
    import math

    bitpix, naxis = int(cards["BITPIX"]), int(cards["NAXIS"])
    if bitpix not in (8, 16, 32, 64, -32, -64) or not 0 <= naxis <= 999:
        raise ValueError("unsupported FITS BITPIX/NAXIS")
    if cards.get("GROUPS") == "T":
        raise ValueError("random-group FITS is not a supported Planck product")
    axes = [int(cards[f"NAXIS{i}"]) for i in range(1, naxis + 1)]
    pcount, gcount = int(cards.get("PCOUNT", "0")), int(cards.get("GCOUNT", "1"))
    if any(n < 0 for n in axes) or pcount < 0 or gcount < 1:
        raise ValueError("invalid FITS dimensions/group counts")
    return abs(bitpix) // 8 * gcount * (pcount + (math.prod(axes) if axes else 0))


def validate_payload(path: Path) -> dict:
    """Reject truncated/wrong-format responses even when ESA omits Content-Length.

    This establishes structural completeness, not scientific validity or publisher authenticity.
    """
    size = path.stat().st_size
    if path.name.endswith((".fits", ".fits.part")):
        if size == 0 or size % 2880:
            raise ValueError("FITS size must be a nonzero multiple of 2880")
        hdus = 0
        with path.open("rb") as stream:
            while stream.tell() < size:
                cards, ended, first = {}, False, True
                while not ended:
                    block = stream.read(2880)
                    if len(block) != 2880:
                        raise ValueError("truncated FITS header")
                    if first and not block.startswith(b"SIMPLE  =" if hdus == 0 else b"XTENSION="):
                        raise ValueError("response is not the expected FITS HDU layout")
                    first = False
                    for i in range(0, 2880, 80):
                        card = block[i:i + 80].decode("ascii")
                        key = card[:8].strip()
                        if key == "END":
                            ended = True
                            break
                        if card[8:10] == "= ":
                            cards[key] = card[10:].split("/", 1)[0].strip().strip("'").strip()
                if hdus == 0 and cards.get("SIMPLE") != "T":
                    raise ValueError("primary FITS HDU must declare SIMPLE = T")
                length = fits_data_bytes(cards)
                end = stream.tell() + ((length + 2879) // 2880) * 2880
                if end > size:
                    raise ValueError("truncated FITS payload")
                stream.seek(end)
                hdus += 1
        return {"format": "FITS", "hdus": hdus, "complete_block_layout": True}
    if path.name.endswith((".tar.gz", ".tar.gz.part")):
        with tarfile.open(path, "r:gz") as archive:
            count = sum(1 for _ in archive)
        # Consume through the gzip trailer so CRC and uncompressed size are checked too.
        with gzip.open(path, "rb") as stream:
            while stream.read(4 << 20):
                pass
        if not count:
            raise ValueError("empty beam archive")
        return {"format": "tar.gz", "members": count, "gzip_crc_checked": True}
    raise ValueError(f"unsupported product format: {path.name}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _https(response) -> None:
    if any(urlparse(r.url).scheme.lower() != "https" for r in [*response.history, response]):
        raise ValueError("product download redirected outside HTTPS")


def _remote_metadata(url: str) -> dict:
    try:
        with requests.head(url, headers={"Accept-Encoding": "identity"}, allow_redirects=True, timeout=(30, 60)) as response:
            response.raise_for_status()
            _https(response)
            _, _, total = transfer_layout(response.status_code, response.headers, 0)
            return {"total": total, "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"),
                    "ranges": response.headers.get("Accept-Ranges", "").lower() == "bytes"}
    except requests.RequestException:
        # Some archive endpoints implement GET only. GET framing and format still must validate.
        return {}


def _validator(metadata: dict) -> str | None:
    etag = metadata.get("etag")
    return etag if etag and not etag.startswith("W/") else metadata.get("last_modified")


def _record(p: Product, path: Path, metadata: dict, validation: dict, origin: str) -> dict:
    record = {"schema": "cmbcst.product-provenance.v1", "key": p.key, "filename": p.filename,
              "role": p.role, "requested_url": p.url, "observed_utc": datetime.now(timezone.utc).isoformat(),
              "origin": origin, "bytes": path.stat().st_size, "sha256": _sha256(path),
              "publisher_checksum": None,
              "checksum_scope": "Locally computed SHA256 identifies these bytes; no publisher checksum was supplied.",
              "http": metadata, "structure": validation}
    _write_json(path.with_name(path.name + ".provenance.json"), record)
    return record


def fetch(p: Product, chunk: int = 1 << 22, raw_dir: Path | str | None = None,
          range_bytes: int = 32 << 20) -> Path:
    """Download with bounded ranges, entity-aware resume, structural validation and local hashes."""
    if chunk < 1 or range_bytes < 1 or Path(p.filename).name != p.filename or urlparse(p.url).scheme != "https":
        raise ValueError("downloads require a basename, HTTPS, and positive chunk/range sizes")
    dest = raw_directory(raw_dir) / p.filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    sidecar = part.with_name(part.name + ".json")
    remote = _remote_metadata(p.url)
    total = remote.get("total")
    if dest.exists() and (total is None or dest.stat().st_size == total):
        verify_existing(p, raw_dir, remote)
        return dest
    saved = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    have = part.stat().st_size if part.exists() else 0
    validator = _validator(saved)
    if (not validator or saved.get("requested_url") != p.url
            or (_validator(remote) and _validator(remote) != validator)
            or (total is not None and have >= total)):
        have = 0  # Unknown or changed partial files are replaced only after a successful GET.
    t0, initial = time.monotonic(), have
    while True:
        headers = {"Accept-Encoding": "identity"}
        if remote.get("ranges") or have:
            end = have + range_bytes - 1
            if total is not None:
                end = min(end, total - 1)
            headers["Range"] = f"bytes={have}-{end}"
            if have and validator:
                headers["If-Range"] = validator
        with requests.get(p.url, headers=headers, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            _https(response)
            start, body_size, expected = transfer_layout(response.status_code, response.headers, have, total)
            metadata = {"requested_url": p.url, "resolved_url": response.url, "total": expected,
                        "etag": response.headers.get("ETag"), "last_modified": response.headers.get("Last-Modified")}
            current_validator = _validator(metadata)
            if start and validator and current_validator and current_validator != validator:
                raise ValueError("remote entity changed despite a partial response")
            validator = current_validator or (validator if start else None)
            if start and not current_validator:
                metadata.update(etag=saved.get("etag") or remote.get("etag"),
                                last_modified=saved.get("last_modified") or remote.get("last_modified"))
            saved = metadata
            _write_json(sidecar, metadata)
            done, received = start, 0
            with part.open("ab" if start else "wb") as stream:
                for block in response.iter_content(chunk):
                    if not block:
                        continue
                    received += len(block)
                    done += len(block)
                    if (body_size is not None and received > body_size) or (expected is not None and done > expected):
                        raise IOError("response exceeds its declared size")
                    stream.write(block)
            if body_size is not None and received != body_size:
                raise IOError(f"truncated response: {received} of {body_size} bytes")
            if expected is None or done == expected:
                break
            if response.status_code != 206 or done <= have or not validator:
                raise IOError(f"incomplete response: {done} of {expected} bytes; no safe continuation")
            have, total = done, expected
            if done // (256 << 20) != (done - received) // (256 << 20):
                print(f"[download] {p.filename} {done / 1e9:.2f}/{total / 1e9:.2f} GB", flush=True)
    validation = validate_payload(part)
    rate = max(done - initial, 0) / max(time.monotonic() - t0, 1e-9) / 1e6
    part.replace(dest)
    _record(p, dest, metadata, validation, "downloaded")
    sidecar.unlink(missing_ok=True)
    print(f"[ok] {p.filename} {done / 1e9:.2f} GB at {rate:.1f} MB/s", flush=True)
    return dest


def verify_existing(p: Product, raw_dir: Path | str | None = None, remote: dict | None = None) -> dict:
    """Hash and structurally validate local bytes; a preexisting file's origin is not proven."""
    dest = raw_directory(raw_dir) / p.filename
    sidecar = dest.with_name(dest.name + ".provenance.json")
    previous = json.loads(sidecar.read_text()) if sidecar.exists() else None
    if remote and remote.get("total") is not None and dest.stat().st_size != remote["total"]:
        raise ValueError("local product size differs from current HTTP metadata")
    validation = validate_payload(dest)
    if previous:
        if previous.get("requested_url") != p.url or previous.get("filename") != p.filename or previous.get("sha256") != _sha256(dest):
            raise ValueError("local product differs from its saved provenance")
        return previous
    return _record(p, dest, remote or {}, validation, "existing_file_validated_origin_unverified")


def split_header_summary(key: str) -> dict[str, str]:
    """Header cards that identify which detectors went into a split map."""
    from astropy.io import fits

    with fits.open(path_of(key), memmap=True) as hdul:
        h = hdul[1].header
        return {k: str(h[k]) for k in h if any(s in k for s in ("DET", "SPLIT", "FREQ", "NSIDE", "ORDER", "COORD", "TUNIT"))}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--raw-dir", type=Path, help="download directory (default CMB_CST_RAW_DIR or data/raw)")
    ap.add_argument("--verify-only", action="store_true", help="validate and hash existing files without downloading")
    ap.add_argument("--provenance", type=Path, help="write a JSON manifest of the selected validated products")
    ap.add_argument("keys", nargs="*", help="subset of manifest keys (default: all)")
    args = ap.parse_args(argv)
    unknown = set(args.keys) - {p.key for p in MANIFEST}
    if unknown:
        ap.error(f"unknown product keys: {', '.join(sorted(unknown))}")
    todo = [p for p in MANIFEST if not args.keys or p.key in args.keys]
    if args.list:
        for p in todo:
            print(f"{p.key:9s} {p.role:6s} {p.url}")
        return 0
    raw_directory(args.raw_dir).mkdir(parents=True, exist_ok=True)
    failed = []
    records = []
    for p in todo:
        for attempt in range(5):
            try:
                if not args.verify_only:
                    fetch(p, raw_dir=args.raw_dir)
                records.append(verify_existing(p, args.raw_dir))
                if args.verify_only:
                    print(f"[verified] {p.filename}", flush=True)
                break
            except (requests.RequestException, OSError, ValueError, KeyError, tarfile.TarError) as e:
                print(f"[retry {attempt + 1}] {p.filename}: {e}", flush=True)
                if args.verify_only:
                    failed.append(p.filename)
                    break
                if attempt < 4:
                    time.sleep(10 * (attempt + 1))
        else:
            failed.append(p.filename)
    if args.provenance:
        args.provenance.parent.mkdir(parents=True, exist_ok=True)
        _write_json(args.provenance, {"schema": "cmbcst.input-manifest.v1",
                    "generated_utc": datetime.now(timezone.utc).isoformat(),
                    "raw_directory": str(raw_directory(args.raw_dir)), "products": records, "failed": failed})
    print("FAILED: " + ", ".join(failed) if failed else "all products present", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
