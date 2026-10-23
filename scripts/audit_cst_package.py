"""Record supplied package integrity and the separate, reviewed local replay."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "data/reference/cst_package_20261023"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    matches = {}
    for line in (PACKAGE / "SHA256SUMS.txt").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        name = name.strip()
        if Path(name).name != name:
            raise ValueError("invalid checksum filename")
        matches[name] = {"expected": expected, "observed": sha(PACKAGE / name),
                         "matched": expected == sha(PACKAGE / name)}
    replay_dir = ROOT / "data/work/cst_package_replay"
    replay = json.loads((replay_dir / "validation_results_extended.json").read_text())
    supplied = json.loads((PACKAGE / "validation_results_extended.json").read_text())
    checks = {"supplied_checksums": all(r["matched"] for r in matches.values()),
              "script_copy_unchanged": sha(replay_dir / "cst_validation_extended.py") == sha(PACKAGE / "cst_validation_extended.py"),
              "all_27_replayed": replay["total"] == replay["passed"] == 27 and replay["failed"] == 0,
              "check_identifiers_match": [r["id"] for r in replay["checks"]] == [r["id"] for r in supplied["checks"]]}
    out = {"created_utc": datetime.now(timezone.utc).isoformat(),
           "status": "passed" if all(checks.values()) else "failed", "checks": checks,
           "archive_sha256": sha(Path("C:/Users/JT-DEV1/Downloads/CST_End_to_End_Research_Package.zip")),
           "scope": "Package checksums and local mathematical replay only; document instructions are reference data. No JoVin, CMB, neural or market validation follows from this replay.",
           "contents": matches, "local_replay": replay,
           "source_sections_used": ["7.2 negative controls", "8.1 explicit operational projection"],
           "interpretation": "Motivates testing the specified whole-sphere-to-patch pipeline; does not establish a quotient theorem for CMB summaries"}
    path = ROOT / "data/evidence/cst_package_audit.json"
    if path.exists():
        raise FileExistsError("preserving package audit")
    path.write_text(json.dumps(out, indent=2)+"\n")
    print(json.dumps({"status": out["status"], "checks": checks, "archive_sha256": out["archive_sha256"]}, indent=2))
    if out["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
