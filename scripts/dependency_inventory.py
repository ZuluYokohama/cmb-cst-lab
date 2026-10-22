#!/usr/bin/env python3
"""Create a private dependency/environment inventory from installed Python packages.

This is a reproducibility aid, NOT a verified SPDX/SBOM, rights clearance,
complete transitive-dependency graph, or license-compliance opinion.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import re
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def outside_repo(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path == ROOT or ROOT in path.parents:
        raise ValueError("Refusing to put local environment information in public Git")
    return path


def build_inventory(include_everything: bool = False) -> dict:
    file = ROOT / "pyproject.toml"
    raw = file.read_bytes()
    project = tomllib.loads(raw.decode("utf-8"))
    declared = list(project.get("project", {}).get("dependencies", []))
    for extra in project.get("project", {}).get("optional-dependencies", {}).values():
        declared.extend(extra)
    selected = {re.match(r"^[A-Za-z0-9_.-]+", req).group(0).lower().replace("_", "-")
                for req in declared if re.match(r"^[A-Za-z0-9_.-]+", req)}
    distributions = {}
    for dist in md.distributions():
        name = dist.metadata.get("Name")
        if name is None:
            continue
        key = name.lower().replace("_", "-")
        if key not in selected and not include_everything:
            continue
        distributions[key] = {
            "name": name,
            "version": dist.version,
            "declared_license_metadata_unverified":
                dist.metadata.get("License-Expression") or dist.metadata.get("License") or "UNKNOWN",
        }
    for key in selected:
        distributions.setdefault(key, {"name": key, "version": "NOT_INSTALLED",
                                       "declared_license_metadata_unverified": "UNKNOWN"})
    return {
        "schema_version": 1,
        "pyproject_sha256": hashlib.sha256(raw).hexdigest(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "scope": "declared and installed packages" if not include_everything else "all installed Python distributions",
        "declared_specifications": declared,
        "packages": [distributions[k] for k in sorted(distributions)],
        "warning": "Package metadata may be missing or inaccurate. Verify license texts, attribution, transitive packages and artifacts separately.",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--all-installed", action="store_true",
                    help="include EVERY distribution installed in current Python environment")
    args = ap.parse_args()
    try:
        out = outside_repo(args.out)
        if out.exists():
            raise ValueError("Refusing to overwrite")
        result = build_inventory(args.all_installed)
        out.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with out.open("x", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
            fh.write("\n")
        try:
            os.chmod(out, 0o600)
        except OSError:
            pass
    except (OSError, ValueError) as exc:
        print("Dependency inventory was not created:", type(exc).__name__, file=sys.stderr)
        return 2
    print("PASS: private dependency metadata inventory created (NOT license clearance)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
