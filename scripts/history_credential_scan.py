#!/usr/bin/env python3
"""Read-only potential-credential scanner for reachable Git history.

Does NOT scan unreferenced/reflog objects, Git LFS payloads, package caches,
compressed archives, hosted secrets, or remote forks. Never prints secret text.
Exits nonzero for any high-confidence pattern; inspect hits locally.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "github_token": re.compile(rb"(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}"),
    "openai_token": re.compile(rb"sk-[A-Za-z0-9]{20,}"),
    "aws_access_id": re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "private_key_header": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def run_git(*args: str) -> bytes:
    p = subprocess.run(["git", "-C", str(ROOT), *args],
                       check=True, capture_output=True, timeout=120)
    return p.stdout


def audit(max_blob_bytes: int = 4000000) -> tuple[int, int]:
    objects = run_git("rev-list", "--objects", "--all").splitlines()
    visited: set[bytes] = set()
    hits, checked = 0, 0
    for item in objects:
        parts = item.split(b" ", 1)
        oid = parts[0]
        if oid in visited or not re.fullmatch(rb"[a-f0-9]{40,64}", oid):
            continue
        visited.add(oid)
        sha = oid.decode("ascii")
        if run_git("cat-file", "-t", sha) != b"blob\n":
            continue
        size = int(run_git("cat-file", "-s", sha))
        if size > max_blob_bytes:
            continue
        data = run_git("cat-file", "blob", sha)
        checked += 1
        if b"\0" in data:
            continue
        for label, expr in PATTERNS.items():
            if expr.search(data):
                hits += 1
                # No file content, matching string or token is printed.
                print("POTENTIAL_CREDENTIAL", sha[:12], label)
    print(f"History scan: {checked} text blobs inspected; {hits} possible signatures.")
    print("This is a limited pattern scan, NOT proof that Git history contains no secrets.")
    return hits, checked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-blob-bytes", type=int, default=4000000)
    args = parser.parse_args()
    if args.max_blob_bytes < 1:
        parser.error("max-blob-bytes must be positive")
    try:
        found, count = audit(args.max_blob_bytes)
    except (OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print("History scan could not complete:", type(exc).__name__, file=sys.stderr)
        return 2
    return int(found > 0)


if __name__ == "__main__":
    raise SystemExit(main())
