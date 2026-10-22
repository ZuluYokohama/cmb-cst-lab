#!/usr/bin/env python3
"""Create a LOCAL PRIVATE cryptographic receipt for an AI development session.

The raw session log and metadata MUST be outside the public repo; raw content
is never copied. The receipt documents what the operator reported, not who
legally authored an output or whether a stated decision was actually creative.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def outside_public_repo(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved == ROOT or ROOT in resolved.parents:
        raise ValueError("Refusing confidential evidence within the public repository")
    return resolved


def file_hash(file: Path) -> str:
    h = hashlib.sha256()
    with file.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def make_receipt(log: Path, metadata: Path, out: Path) -> dict:
    log = outside_public_repo(log)
    metadata = outside_public_repo(metadata)
    out = outside_public_repo(out)
    if not log.is_file() or not metadata.is_file():
        raise ValueError("Private log and metadata files are required")
    if out.exists():
        raise ValueError("Refusing to overwrite an existing receipt")
    meta = json.loads(metadata.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError("Metadata must be a JSON object")
    for key in ("provider", "product"):
        if not isinstance(meta.get(key), str) or not meta[key].strip():
            raise ValueError("Missing required metadata field: " + key)
    for key in ("affected_files", "commits", "human_decisions", "external_sources"):
        if not isinstance(meta.get(key), list) or not all(isinstance(x, str) for x in meta[key]):
            raise ValueError("Expected array of strings: " + key)
    record = {
        "schema_version": 1,
        "receipt_recorded_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": file_hash(log),
        "source_bytes": log.stat().st_size,
        "source_file_not_embedded": True,
        "provider": meta["provider"],
        "product": meta["product"],
        "model": meta.get("model") or "unspecified",
        "affected_files": meta["affected_files"],
        "commits": meta["commits"],
        "human_decisions_user_reported": meta["human_decisions"],
        "external_sources_user_reported": meta["external_sources"],
        "legal_authorship": "UNDETERMINED",
        "independent_verification": "NOT_PERFORMED",
        "not_a_trusted_timestamp": True,
    }
    out.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    text = json.dumps(record, indent=2, ensure_ascii=False) + "\n"
    with out.open("x", encoding="utf-8") as fh:
        fh.write(text)
    try:
        os.chmod(out, 0o600)
    except OSError:
        pass
    return {"receipt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "receipt_path": str(out), "raw_log_was_copied": False}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", type=Path, required=True, help="private raw session export outside repo")
    ap.add_argument("--metadata", type=Path, required=True, help="private JSON annotations outside repo")
    ap.add_argument("--out", type=Path, required=True, help="new private receipt path outside repo")
    args = ap.parse_args()
    try:
        info = make_receipt(args.log, args.metadata, args.out)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print("Receipt not produced:", type(exc).__name__, file=sys.stderr)
        return 2
    print("PASS: private session receipt written; no raw transcript copied.")
    print("Receipt SHA-256:", info["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
