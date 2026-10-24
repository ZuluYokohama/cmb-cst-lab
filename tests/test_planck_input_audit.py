"""Input integrity failures on small real FITS/tar fixtures, without network access."""
from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import tarfile

from astropy.io import fits
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_planck_inputs as audit
from cmbcst import data


def _co_input(tmp_path, monkeypatch, checksum=True):
    # Four-pixel FITS fixture exercises the real parser and checksum routines;
    # full PR4 pixel counts remain enforced by the unmodified production script.
    monkeypatch.setattr(audit, "NPIX", 4)
    product = next(p for p in data.MANIFEST if p.key == "co")
    table = fits.BinTableHDU.from_columns([fits.Column(name="INTEN21", format="E", array=np.arange(4, dtype=np.float32))])
    table.header["NSIDE"], table.header["ORDERING"] = 2048, "NESTED"
    path = tmp_path / product.filename
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path, checksum=checksum)
    provenance = {"schema": "cmbcst.product-provenance.v1", "key": product.key, "filename": product.filename,
                  "role": product.role, "requested_url": product.url,
                  "bytes": path.stat().st_size, "sha256": audit.sha256_file(path)}
    sidecar = path.with_name(path.name + ".provenance.json")
    sidecar.write_text(json.dumps(provenance))
    return product, path, provenance, sidecar


@pytest.mark.parametrize("checksum", [True, False])
def test_real_fits_checksums_are_valid_or_explicitly_absent(tmp_path, monkeypatch, checksum):
    product, path, _, _ = _co_input(tmp_path, monkeypatch, checksum)
    before = audit.sha256_file(path)
    result = audit.audit_product(product, tmp_path)
    assert result["passed"], result["failures"]
    expected = "valid" if checksum else "absent"
    assert all(check["status"] == expected for hdu in result["content_audit"]["hdus"] for check in hdu["checksums"].values())
    assert audit.sha256_file(path) == before  # audit never repairs or changes input bytes


def test_corruption_fails_advertised_fits_checksum_even_with_matching_local_hash(tmp_path, monkeypatch):
    product, path, provenance, sidecar = _co_input(tmp_path, monkeypatch)
    with fits.open(path) as hdus:
        offset = hdus[1].fileinfo()["datLoc"]
    with path.open("r+b") as stream:
        stream.seek(offset)
        stream.write(b"\x01")
    provenance["sha256"] = audit.sha256_file(path)
    sidecar.write_text(json.dumps(provenance))
    result = audit.audit_product(product, tmp_path)
    assert not result["passed"]
    assert result["content_audit"]["hdus"][1]["checksums"]["DATASUM"]["status"] == "failed"


def test_wrong_provenance_source_is_rejected(tmp_path, monkeypatch):
    product, _, provenance, sidecar = _co_input(tmp_path, monkeypatch)
    provenance["requested_url"] = "https://example.invalid/wrong.fits"
    sidecar.write_text(json.dumps(provenance))
    result = audit.audit_product(product, tmp_path)
    assert not result["passed"] and "requested_url" in result["failures"][0]


@pytest.mark.parametrize("origin", ["downloaded", "existing_file_validated_origin_unverified", None])
def test_download_origin_is_reported_without_weakening_integrity_checks(tmp_path, monkeypatch, origin):
    product, _, provenance, sidecar = _co_input(tmp_path, monkeypatch)
    provenance["origin"] = origin
    sidecar.write_text(json.dumps(provenance))
    if origin == "downloaded":
        result = audit.audit_product(product, tmp_path)
    else:
        with pytest.warns(UserWarning, match="download origin is unverified"):
            result = audit.audit_product(product, tmp_path)
    assert result["passed"]
    assert result["provenance"]["origin_verified"] is (origin == "downloaded")
    assert result["sha256"] == provenance["sha256"]


def test_map_source_cards_detect_swapped_detector_split(monkeypatch):
    monkeypatch.setattr(audit, "NPIX", 4)
    product = next(p for p in data.MANIFEST if p.key == "100B")
    table = fits.BinTableHDU.from_columns([fits.Column(name=name, format="E", unit="K_CMB", array=np.zeros(4))
                                         for name in ("I_STOKES", "Q_STOKES", "U_STOKES", "II_COV")], name="FREQ-MAP")
    for key, value in {"NSIDE": 2048, "ORDERING": "NESTED", "POLCCONV": "COSMO", "COORDSYS": "G",
                       "FREQ": 100, "PROCVER": "npipe6v20", "FILENAME": Path(product.filename).stem}.items():
        table.header[key] = value
    table.header["COMMENT"] = "npipe6v20A_100_map.fits"
    with pytest.raises(ValueError, match="source split/frequency"):
        audit.map_identity(fits.HDUList([fits.PrimaryHDU(), table]), product)


@pytest.mark.parametrize("omit_last", [False, True])
def test_beam_archive_requires_all_64_members_without_extracting(tmp_path, omit_last):
    members = [f"Wl_npipe6v20_{a}x{b}.fits" for a in audit.DETECTORS for b in audit.DETECTORS]
    path = tmp_path / "beams.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name in members[:-1] if omit_last else members:
            info = tarfile.TarInfo(f"root/AB/{name}")
            info.size = 3
            archive.addfile(info, io.BytesIO(b"abc"))
    result = audit.beam_audit(path)
    assert result["required_count"] == 64
    assert bool(result["failures"]) is omit_last
    assert result["missing"] == ([members[-1]] if omit_last else [])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["beams.tar.gz"]
