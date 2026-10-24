"""Audit the twelve local Planck inputs without downloading or modifying them.

Checks each downloader provenance record against the current manifest and actual
bytes, FITS structure/headers and advertised checksums, detector A/B source
identities, and the 64 required QuickPol A/B archive members. Missing FITS
checksums are reported separately from failed checksums. No map arrays are
copied or extracted; SHA256 is streamed and Astropy uses read-only memory maps.

Usage: python scripts/audit_planck_inputs.py [--raw-dir PATH] [--output PATH]
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile
import time
import warnings

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cmbcst import DETECTORS, data  # noqa: E402

NPIX = 12 * 2048**2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def table_metadata(hdu, required_columns: tuple[str, ...]) -> dict:
    """Read only column/header metadata, including vector-valued HEALPix rows."""
    header = hdu.header
    names = list(hdu.columns.names or ())
    require(set(required_columns) <= set(names), f"{hdu.name}: missing columns {sorted(set(required_columns) - set(names))}")
    require(int(header.get("NSIDE", 0)) == 2048, f"{hdu.name}: expected NSIDE=2048")
    require(str(header.get("ORDERING", "")).strip().upper() == "NESTED", f"{hdu.name}: expected NESTED ordering")
    pixels = {name: int(header["NAXIS2"]) * int(hdu.columns[name].format.repeat) for name in required_columns}
    require(all(n == NPIX for n in pixels.values()), f"{hdu.name}: required columns must each contain {NPIX} pixels")
    return {"hdu": hdu.name, "nside": 2048, "ordering": "NESTED", "columns": names,
            "required_column_pixels": pixels}


def map_identity(hdus, product: data.Product) -> dict:
    require(len(hdus) == 2 and hdus[1].name == "FREQ-MAP", "map must contain PRIMARY and FREQ-MAP HDUs")
    hdu = hdus[1]
    result = table_metadata(hdu, ("I_STOKES", "Q_STOKES", "U_STOKES", "II_COV"))
    h = hdu.header
    frequency, split = product.key[:3], product.key[3:]
    require(str(h.get("POLCCONV", "")).strip() == "COSMO", "map polarization convention must be COSMO")
    require(str(h.get("COORDSYS", "")).strip().upper() in ("G", "GALACTIC"), "map coordinates must be Galactic")
    require(int(h.get("FREQ", 0)) == int(frequency), "map FREQ does not match the manifest")
    require(str(h.get("PROCVER", "")).strip() == "npipe6v20", "unexpected NPIPE processing version")
    require(str(h.get("FILENAME", "")).strip() in (product.filename, Path(product.filename).stem), "FITS FILENAME does not match the manifest")
    units = {name: str(hdu.columns[name].unit).strip() for name in ("I_STOKES", "Q_STOKES", "U_STOKES")}
    require(all(unit == "K_CMB" for unit in units.values()), "I/Q/U units must be K_CMB")
    # The delivered FITS COMMENT/HISTORY cards identify the underlying detector
    # split. A filename alone is not evidence of which input map was processed.
    cards = [str(card.value) for item in hdus for card in item.header.cards]
    sources = sorted(set(re.findall(r"\bnpipe6v20([AB])_([0-9]{3})_map\.fits\b", "\n".join(cards))))
    require(sources == [(split, frequency)], f"map source split/frequency is missing or inconsistent: {sources}")
    result.update(frequency_ghz=int(frequency), detector_split=split, polarization_convention="COSMO",
                  stokes_units=units, input_map=f"npipe6v20{split}_{frequency}_map.fits",
                  identity_evidence=[card for card in cards if re.search(r"npipe6v20[AB]_[0-9]{3}_(?:map|hmap|wcov_mcscaled)\.fits", card)])
    return result


def fits_audit(path: Path, product: data.Product) -> dict:
    from astropy.io import fits

    output: dict = {"hdus": [], "failures": []}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with fits.open(path, mode="readonly", memmap=True, lazy_load_hdus=False, checksum=False) as hdus:
            hdus.verify("exception")
            for index, hdu in enumerate(hdus):
                checks = {}
                for keyword, verify in (("CHECKSUM", hdu.verify_checksum), ("DATASUM", hdu.verify_datasum)):
                    if keyword not in hdu.header:
                        checks[keyword] = {"status": "absent"}
                        continue
                    value = str(hdu.header[keyword])
                    try:
                        code = int(verify())
                        checks[keyword] = {"status": "valid" if code == 1 else "failed", "value": value, "verification_code": code}
                    except Exception as exc:
                        checks[keyword] = {"status": "failed", "value": value, "error": f"{type(exc).__name__}: {exc}"}
                    if checks[keyword]["status"] != "valid":
                        output["failures"].append(f"HDU {index} ({hdu.name}) {keyword} verification failed")
                output["hdus"].append({"index": index, "name": hdu.name, "checksums": checks})
            try:
                if product.role == "map":
                    output["metadata"] = map_identity(hdus, product)
                else:
                    columns = {"pointsrc": ("F100", "F143", "F217", "F353"),
                               "galplane": ("GAL060", "GAL070"), "co": ("INTEN21",)}[product.key]
                    matches = [h for h in hdus if isinstance(h, fits.BinTableHDU)
                               and set(columns) <= set(h.columns.names or ())
                               and (product.key != "pointsrc" or h.name == "SRC-POL")]
                    require(len(matches) == 1, f"expected exactly one {'SRC-POL ' if product.key == 'pointsrc' else ''}table with {columns}")
                    output["metadata"] = table_metadata(matches[0], columns)
            except (ValueError, KeyError, AttributeError) as exc:
                output["failures"].append(f"metadata: {exc}")
        output["warnings"] = [str(w.message) for w in caught]
    return output


def beam_audit(path: Path) -> dict:
    expected = {f"Wl_npipe6v20_{a}x{b}.fits" for a in DETECTORS for b in DETECTORS}
    found: Counter = Counter()
    failures = []
    # Streaming tar inspection never extracts archive paths or follows links.
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if "AB" not in name.parts or name.name not in expected:
                continue
            if not member.isfile() or member.size <= 0:
                failures.append(f"required beam is not a nonempty regular file: {member.name}")
            if name.is_absolute() or ".." in name.parts:
                failures.append(f"required beam has an unsafe archive path: {member.name}")
            found[name.name] += 1
    missing = sorted(expected - set(found))
    duplicate = sorted(name for name, count in found.items() if count != 1)
    if missing:
        failures.append(f"missing {len(missing)} required QuickPol A/B members")
    if duplicate:
        failures.append(f"duplicate required QuickPol A/B members: {duplicate}")
    return {"required_count": len(expected), "present_count": len(found), "missing": missing,
            "duplicates": duplicate, "required_members": sorted(expected), "failures": failures,
            "scope": "Archive names, type and completeness only; beam FITS numerical contents are not checked."}


def audit_product(product: data.Product, directory: Path) -> dict:
    path = directory / product.filename
    result = {"key": product.key, "filename": product.filename, "role": product.role,
              "expected_source_url": product.url, "passed": False, "failures": []}
    try:
        before = path.stat()
        result["bytes"] = before.st_size
        sidecar = path.with_name(path.name + ".provenance.json")
        provenance = json.loads(sidecar.read_text(encoding="utf-8"))
        require(provenance.get("schema") == "cmbcst.product-provenance.v1", "missing or unknown provenance schema")
        for name, expected in (("key", product.key), ("filename", product.filename),
                               ("role", product.role), ("requested_url", product.url), ("bytes", before.st_size)):
            require(provenance.get(name) == expected, f"provenance {name} does not match the manifest/local file")
        require(isinstance(provenance.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", provenance["sha256"]) is not None,
                "provenance must contain a lowercase SHA256")
        result["provenance"] = {"path": str(sidecar), "record_sha256": sha256_file(sidecar),
                                "origin": provenance.get("origin"), "origin_verified": provenance.get("origin") == "downloaded",
                                "requested_url": provenance["requested_url"],
                                "observed_utc": provenance.get("observed_utc")}
        if not result["provenance"]["origin_verified"]:
            warnings.warn(f"{product.key}: download origin is unverified ({provenance.get('origin')!r})",
                          stacklevel=2)
        result["sha256"] = sha256_file(path)
        require(result["sha256"] == provenance["sha256"], "local bytes do not match provenance SHA256")
        result["structure"] = data.validate_payload(path)
        details = beam_audit(path) if product.role == "beams" else fits_audit(path, product)
        result["content_audit"] = details
        result["failures"].extend(details["failures"])
        after = path.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "input changed during its audit")
    except Exception as exc:
        result["failures"].append(f"{type(exc).__name__}: {exc}")
    result["passed"] = not result["failures"]
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, help="default: CMB_CST_RAW_DIR or data/raw")
    parser.add_argument("--output", type=Path, default=REPO / "data/evidence/planck_input_audit.json")
    args = parser.parse_args(argv)
    directory = data.raw_directory(args.raw_dir)
    output = args.output.expanduser().resolve()
    if output == directory or directory in output.parents:
        parser.error("audit output must be outside the raw input directory")
    started = time.perf_counter()
    products = []
    for product in data.MANIFEST:
        record = audit_product(product, directory)
        products.append(record)
        print(f"[{'PASS' if record['passed'] else 'FAIL'}] {product.key}: "
              + ("input provenance and required content verified" if record["passed"] else "; ".join(record["failures"])), flush=True)
    import astropy
    passed = len(products) == 12 and all(p["passed"] for p in products)
    report = {"schema": "cmbcst.planck-input-audit.v1", "completed_utc": datetime.now(timezone.utc).isoformat(),
              "status": "passed" if passed else "failed",
              "raw_directory": str(directory), "passed": passed, "expected_products": 12,
              "products_passed": sum(p["passed"] for p in products), "products": products,
              "failures": [{"key": p["key"], "messages": p["failures"]} for p in products if not p["passed"]],
              "seconds": round(time.perf_counter() - started, 3), "astropy_version": astropy.__version__,
              "source_sha256": {"scripts/audit_planck_inputs.py": sha256_file(Path(__file__)),
                                "src/cmbcst/data.py": sha256_file(Path(data.__file__))},
              "limits": "Local hashes bind bytes to saved download provenance, not a publisher-signed digest. "
                        "Absent FITS checksums are reported, not treated as verified. This is an input audit, not validation of scientific inference."}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(f"Audit {'passed' if passed else 'failed'}: {report['products_passed']}/12 products; {output}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
