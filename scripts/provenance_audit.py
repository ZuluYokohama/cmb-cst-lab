#!/usr/bin/env python3
"""Non-destructive provenance guardrails (stdlib only).

check: verify notices, registry shape, ignored private material, and run a limited
current-tree credential-pattern scan.
inventory: SHA-256 and unresolved classifications for Git-tracked files.
snapshot: make a PRIVATE local Git bundle + manifest outside this repository.

NOT a proof of authorship, license clearance, security, novelty, or scientific validity.
GitHub PR/CI logs are public. Never emit source content or secret values in this tool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "provenance" / "component_registry.json"
NOTICE = ROOT / "THIRD_PARTY_LICENSES" / "cosmic_birefringence-MIT.txt"
REQUIRED = (
    "RIGHTS_AND_REUSE.md", "THIRD_PARTY_NOTICES.md", "CONTRIBUTING.md",
    "docs/PROVENANCE_AND_LICENSING_AUDIT.md",
    "docs/VALUE_GENESIS_VV_PLAN.md",
    "THIRD_PARTY_LICENSES/cosmic_birefringence-MIT.txt",
    "provenance/component_registry.json",
)
SECRET_PATTERNS = {
    "github_token": re.compile(r"(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}"),
    "openai_token": re.compile(r"sk-[A-Za-z0-9]{20,}"),
    "aws_access_id": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "private_key_header": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}
TEXT_SUFFIXES = {".py", ".js", ".html", ".md", ".json", ".toml", ".txt",
                 ".yml", ".yaml", ".css", ".sh", ".xml", ".tex", ".ini"}
VALID_ORIGINS = {
    "UPSTREAM_ADAPTED", "PROJECT_AI_ASSISTED_UNRESOLVED", "EXTERNAL_WRAPPER",
    "GENERATED_RESULT", "GENERATED_EXPLORATORY_RESULT",
    "GENERATED_LOCAL_TEST_RECEIPTS", "GENERATED_PRESENTATION",
    "LOCAL_CONFIG", "UNRESOLVED",
}


def git(*args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True,
        check=True, timeout=120,
    )
    return result.stdout if binary else result.stdout.decode("utf-8", "replace").strip()


def tracked() -> list[str]:
    return sorted(os.fsdecode(x) for x in git("ls-files", "-z", binary=True).split(b"\0") if x)


def load_registry() -> dict:
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("entries"), dict):
        raise ValueError("Unrecognized component registry schema")
    return data["entries"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def current_tree_findings(paths: list[str]) -> list[tuple[str, int, str]]:
    findings = []
    for rel in paths:
        file = ROOT / rel
        if file.is_symlink() or not file.is_file() or file.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if file.stat().st_size > 2500000:
            continue
        try:
            data = file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for line, text in enumerate(data.splitlines(), 1):
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(text):
                    findings.append((rel, line, label))
    return findings


def run_check() -> int:
    errors, warnings = [], []
    paths = set(tracked())
    for rel in REQUIRED:
        if rel not in paths or not (ROOT / rel).is_file():
            errors.append("Missing required tracked notice/registry: " + rel)
    for name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "LICENSE.rst"):
        if name in paths:
            warnings.append("Root license detected; confirm intended project-wide grant: " + name)
    if not REGISTRY.is_file():
        print("FAIL: component registry missing")
        return 1
    entries = load_registry()
    for rel, rec in entries.items():
        p = PurePosixPath(rel)
        if p.is_absolute() or ".." in p.parts or "\\" in rel or str(p) != rel:
            errors.append("Invalid registry path: " + rel)
            continue
        if rel not in paths:
            errors.append("Registry references untracked file: " + rel)
        if not isinstance(rec, dict) or rec.get("origin") not in VALID_ORIGINS:
            errors.append("Invalid origin classification for: " + rel)
        if isinstance(rec, dict) and rec.get("license_file") and rec["license_file"] not in paths:
            errors.append("Missing linked license for: " + rel)
    if NOTICE.is_file():
        t = NOTICE.read_text(encoding="utf-8")
        for phrase in (
            "MIT License", "Copyright (c) 2026 Anto Idicherian Lonappan",
            "Permission is hereby granted", "The above copyright notice",
            'THE SOFTWARE IS PROVIDED "AS IS"',
        ):
            if phrase not in t:
                errors.append("Upstream MIT notice missing required provision")
                break
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for phrase in ("private/", ".env", ".env.*", "*.pem", "*.key"):
        if phrase not in ignore:
            errors.append("Missing privacy ignore pattern: " + phrase)
    suspicious = current_tree_findings(sorted(paths))
    if suspicious:
        for path, line, category in suspicious[:30]:
            errors.append("Potential secret signature: " + path + ":" + str(line) + " (" + category + ")")
    print("Current-tree audit: tracked=" + str(len(paths)) +
          ", classified=" + str(len(entries)) +
          ", unresolved=" + str(len(paths - set(entries))))
    print("Scope: selected file patterns in CURRENT tracked text only; no Git-history or provider credential audit.")
    for item in warnings:
        print("WARN:", item)
    for item in errors:
        print("FAIL:", item)
    if not errors:
        print("PASS: structural notices and current-tree pattern checks (NOT rights or security clearance)")
    return int(bool(errors))


def inventory(summary: bool) -> int:
    entries = load_registry()
    records = []
    for rel in tracked():
        f = ROOT / rel
        if not f.is_file() or f.is_symlink():
            records.append({"path": rel, "kind": "not-a-regular-file", "origin": "UNRESOLVED"})
            continue
        records.append({
            "path": rel, "sha256": digest(f), "size": f.stat().st_size,
            "origin": entries.get(rel, {}).get("origin", "UNRESOLVED"),
        })
    unresolved = sum(r["origin"] == "UNRESOLVED" for r in records)
    if summary:
        print(json.dumps({"files": len(records), "unresolved": unresolved,
                          "declared": len(records) - unresolved,
                          "scope": "tracked worktree, not human-authorship verification"}))
    else:
        print(json.dumps({"commit": git("rev-parse", "HEAD"),
                          "files": records,
                          "warning": "SHA hashes are integrity checks, not proofs of creator or date"},
                         indent=2))
    return 0


def snapshot(destination: str) -> int:
    repo = ROOT.resolve()
    target = Path(destination).expanduser().resolve()
    if target == repo or repo in target.parents:
        raise ValueError("Refusing to put confidential snapshot inside the public repository")
    if target.exists():
        raise ValueError("Snapshot directory already exists; refusing overwrite")
    target.mkdir(parents=True, mode=0o700)
    print("Creating LOCAL PRIVATE evidence snapshot. Treat all contents as potentially sensitive.")
    bundle = target / "all-refs.bundle"
    subprocess.run(["git", "-C", str(repo), "bundle", "create", str(bundle), "--all"],
                   check=True, timeout=180, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "bundle", "verify", str(bundle)],
                   check=True, timeout=180, capture_output=True)
    refs = git("show-ref")
    (target / "refs.txt").write_text(refs + "\n", encoding="utf-8")
    changed = git("status", "--porcelain=v1")
    (target / "worktree_status.txt").write_text(changed + "\n", encoding="utf-8")
    paths = tracked()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "head": git("rev-parse", "HEAD"), "bundle_sha256": digest(bundle),
        "scope": "Git commit history and current worktree tracked-file hashes",
        "note": "Untracked files are not copied. Dirty tracked modifications are NOT captured in the Git bundle.",
        "tracked": [{"path": rel, "sha256": digest(ROOT / rel)}
                    for rel in paths if (ROOT / rel).is_file()],
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                          encoding="utf-8")
    print("PASS: private local snapshot created outside public repository; no files uploaded.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("check")
    inv = sub.add_parser("inventory")
    inv.add_argument("--summary", action="store_true")
    snap = sub.add_parser("snapshot")
    snap.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        if args.action == "check":
            return run_check()
        if args.action == "inventory":
            return inventory(args.summary)
        return snapshot(args.out)
    except (OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print("Audit tool could not complete:", type(exc).__name__, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
