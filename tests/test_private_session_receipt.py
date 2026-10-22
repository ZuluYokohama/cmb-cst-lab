"""Ensure private AI provenance receipts never publish raw logs or write into Git."""
import json
from pathlib import Path
import pytest

from scripts.private_session_receipt import make_receipt, outside_public_repo, ROOT


def test_private_session_receipt_contains_only_hash_and_metadata(tmp_path):
    raw = tmp_path / "session-private.log"
    raw.write_text("Sensitive session transcript. NEVER PUBLISH.", encoding="utf-8")
    meta = tmp_path / "session-metadata.json"
    meta.write_text(json.dumps({
        "provider": "Sample provider", "product": "Sample agent", "model": None,
        "affected_files": ["src/cmbcst/network.py"], "commits": ["abc123"],
        "human_decisions": ["Require gauge invariance"],
        "external_sources": ["Known published analysis"]
    }), encoding="utf-8")
    receipt = tmp_path / "session-receipt.json"
    out = make_receipt(raw, meta, receipt)
    data = json.loads(receipt.read_text(encoding="utf-8"))
    assert out["raw_log_was_copied"] is False
    assert len(data["source_sha256"]) == 64
    assert data["human_decisions_user_reported"] == ["Require gauge invariance"]
    assert "Sensitive session transcript" not in receipt.read_text(encoding="utf-8")
    assert data["legal_authorship"] == "UNDETERMINED"
    with pytest.raises(ValueError):
        make_receipt(raw, meta, receipt)


def test_refuse_private_output_inside_public_repository():
    with pytest.raises(ValueError):
        outside_public_repo(ROOT / "private" / "secret-receipt.json")


def test_requires_explicit_human_annotation_fields(tmp_path):
    log = tmp_path / "session.log"
    log.write_text("test", encoding="utf-8")
    meta = tmp_path / "meta.json"
    meta.write_text('{"provider":"test","product":"test"}', encoding="utf-8")
    with pytest.raises(ValueError):
        make_receipt(log, meta, tmp_path / "receipt.json")
