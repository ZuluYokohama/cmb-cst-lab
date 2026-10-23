"""Network-free regressions for ranged transfers, incomplete payloads and provenance."""
from __future__ import annotations

import json

import pytest
import requests

from cmbcst import data


def fits_bytes():
    cards = ["SIMPLE  =                    T", "BITPIX  =                    8", "NAXIS   =                    1",
             "NAXIS1  =                    4", "END"]
    return "".join(card.ljust(80) for card in cards).ljust(2880).encode() + b"test" + b"\0" * 2876


class Response:
    def __init__(self, status=200, headers=None, payload=b"", error=None):
        self.status_code = status
        self.headers = headers or {}
        self.payload = payload
        self.error = error
        self.url = "https://example.org/data.fits"
        self.history = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def iter_content(self, chunk):
        yield self.payload
        if self.error:
            raise self.error


PRODUCT = data.Product("test", "data.fits", "https://example.org/data.fits", "map")


@pytest.mark.parametrize("headers,offset", [
    ({"Content-Range": "bytes 0-9/20", "Content-Length": "10"}, 10),
    ({"Content-Range": "bytes 10-20/20"}, 10),
    ({"Content-Range": "bytes 10-19/*"}, 10),
    ({"Content-Range": "bytes 10-19/20", "Content-Length": "9"}, 10),
    ({"Content-Range": "bytes 10-19/20", "Content-Encoding": "gzip"}, 10),
])
def test_bad_range_framing_is_rejected(headers, offset):
    with pytest.raises(ValueError):
        data.transfer_layout(206, headers, offset)


def test_ranged_download_checks_offsets_and_creates_local_hash(monkeypatch, tmp_path):
    payload = fits_bytes()
    monkeypatch.setattr(data.requests, "head", lambda *a, **kw: Response(headers={
        "Content-Length": str(len(payload)), "Accept-Ranges": "bytes", "ETag": '"fixed"'}))
    ranges = []
    def get(url, *, headers, **kwargs):
        ranges.append(headers["Range"])
        start, end = map(int, headers["Range"].removeprefix("bytes=").split("-"))
        if start:
            assert headers["If-Range"] == '"fixed"'
        return Response(206, {"Content-Range": f"bytes {start}-{end}/{len(payload)}",
                              "Content-Length": str(end - start + 1), "ETag": '"fixed"'}, payload[start:end + 1])
    monkeypatch.setattr(data.requests, "get", get)
    path = data.fetch(PRODUCT, raw_dir=tmp_path, range_bytes=3000)
    assert path.read_bytes() == payload
    assert ranges == ["bytes=0-2999", "bytes=3000-5759"]
    record = json.loads(path.with_name(path.name + ".provenance.json").read_text())
    assert record["origin"] == "downloaded"
    assert record["sha256"] == data._sha256(path)
    assert record["publisher_checksum"] is None


def test_head_unsupported_and_chunked_fits_can_complete(monkeypatch, tmp_path):
    monkeypatch.setattr(data.requests, "head", lambda *a, **kw: Response(405))
    monkeypatch.setattr(data.requests, "get", lambda *a, **kw: Response(payload=fits_bytes()))
    assert data.fetch(PRODUCT, raw_dir=tmp_path).read_bytes() == fits_bytes()


def test_unknown_length_truncated_fits_is_never_promoted(monkeypatch, tmp_path):
    monkeypatch.setattr(data.requests, "head", lambda *a, **kw: Response())
    monkeypatch.setattr(data.requests, "get", lambda *a, **kw: Response(payload=fits_bytes()[:2880]))
    with pytest.raises(ValueError, match="truncated FITS payload"):
        data.fetch(PRODUCT, raw_dir=tmp_path)
    assert not (tmp_path / PRODUCT.filename).exists()


@pytest.mark.parametrize("payload,declared", [(b"too short", "100"), (fits_bytes(), "10")])
def test_declared_body_underflow_and_overflow_are_rejected(monkeypatch, tmp_path, payload, declared):
    monkeypatch.setattr(data.requests, "head", lambda *a, **kw: Response())
    monkeypatch.setattr(data.requests, "get", lambda *a, **kw: Response(
        headers={"Content-Length": declared}, payload=payload))
    with pytest.raises(IOError):
        data.fetch(PRODUCT, raw_dir=tmp_path)
    assert not (tmp_path / PRODUCT.filename).exists()


def test_https_redirect_downgrade_is_rejected_before_writing(monkeypatch, tmp_path):
    monkeypatch.setattr(data.requests, "head", lambda *a, **kw: Response(405))
    response = Response(payload=fits_bytes())
    response.url = "http://example.org/data.fits"
    monkeypatch.setattr(data.requests, "get", lambda *a, **kw: response)
    with pytest.raises(ValueError, match="outside HTTPS"):
        data.fetch(PRODUCT, raw_dir=tmp_path)
    assert not (tmp_path / PRODUCT.filename).exists()


def test_interrupted_download_resumes_only_matching_entity(monkeypatch, tmp_path):
    payload = fits_bytes()
    headers = {"Content-Length": str(len(payload)), "ETag": '"fixed"', "Accept-Ranges": "bytes"}
    monkeypatch.setattr(data.requests, "head", lambda *a, **kw: Response(headers=headers))
    monkeypatch.setattr(data.requests, "get", lambda *a, **kw: Response(
        206, dict(headers, **{"Content-Range": "bytes 0-5759/5760"}), payload[:3000], requests.ConnectionError("reset")))
    with pytest.raises(requests.ConnectionError):
        data.fetch(PRODUCT, raw_dir=tmp_path)
    def resumed(url, *, headers, **kwargs):
        assert headers["Range"] == "bytes=3000-5759"
        assert headers["If-Range"] == '"fixed"'
        return Response(206, {"Content-Range": "bytes 3000-5759/5760", "ETag": '"fixed"'}, payload[3000:])
    monkeypatch.setattr(data.requests, "get", resumed)
    assert data.fetch(PRODUCT, raw_dir=tmp_path).read_bytes() == payload


def test_legacy_partial_restarts_and_ignored_range_overwrites(monkeypatch, tmp_path):
    (tmp_path / "data.fits.part").write_bytes(b"old-corrupt-prefix")
    monkeypatch.setattr(data.requests, "head", lambda *a, **kw: Response(headers={
        "Content-Length": "5760", "Accept-Ranges": "bytes", "ETag": '"fixed"'}))
    def full_response(url, *, headers, **kwargs):
        assert headers["Range"].startswith("bytes=0-")
        return Response(headers={"Content-Length": "5760"}, payload=fits_bytes())
    monkeypatch.setattr(data.requests, "get", full_response)
    assert data.fetch(PRODUCT, raw_dir=tmp_path).read_bytes() == fits_bytes()


def test_saved_hash_detects_same_size_local_corruption(tmp_path):
    path = tmp_path / PRODUCT.filename
    path.write_bytes(fits_bytes())
    record = data.verify_existing(PRODUCT, tmp_path)
    assert record["origin"] == "existing_file_validated_origin_unverified"
    blob = bytearray(path.read_bytes())
    blob[2880] ^= 1
    path.write_bytes(blob)
    with pytest.raises(ValueError, match="saved provenance"):
        data.verify_existing(PRODUCT, tmp_path)


def test_raw_directory_env_and_explicit_override(monkeypatch, tmp_path):
    monkeypatch.setenv("CMB_CST_RAW_DIR", str(tmp_path))
    assert data.path_of("100A").parent == tmp_path.resolve()
    assert data.raw_directory(tmp_path / "explicit") == (tmp_path / "explicit").resolve()


def test_unknown_keys_fail_before_downloading():
    with pytest.raises(SystemExit):
        data.main(["not-a-product"])
