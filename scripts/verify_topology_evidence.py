"""Compare frozen experiments and current source with authentic JoVin receipts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from jovin_verify import REPO, SUITES, build_bundle, configuration, _harness
from evidence_test_summary import parse_test_summary


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    harness = parser.add_mutually_exclusive_group()
    harness.add_argument("--harness", default=None)
    harness.add_argument("--harness-python", default=None)
    parser.add_argument("--runtime", default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    command, env = configuration(args)
    ret, chain = _harness(command, env, ["receipts", "verify"])
    suites = {}
    history = json.loads((REPO / "data/evidence/jovin_receipts.json").read_text())
    for suite, (_, test_path) in SUITES.items():
        record = [r for r in history if r["suite"] == suite][-1]
        show_ret, authentic = _harness(command, env, ["receipts", "show", record["receipt_id"]])
        member_ret, member = _harness(command, env, ["receipts", "verify", record["receipt_id"]])
        matched = (record["passed"] and record["receipt_verified"] and record["chain_valid"]
                   and sha(REPO / test_path) == record["tests_sha256"]
                   and hashlib.sha256(build_bundle(suite)[0].encode()).hexdigest() == record["bundle_sha256"]
                   and show_ret == 0 and authentic == record["receipt"]
                   and member_ret == 0 and member.get("valid") is True and member.get("chain") == "linked")
        suites[suite] = {"receipt_id": record["receipt_id"], "matched": bool(matched),
                         "passed_count": record["passed_count"], "total": record["total"]}
    frozen = {}
    for name in ("surrogate_topology_v1", "copula_topology_v2", "copula_boundary_audit_v1"):
        plan = json.loads((REPO / "results" / name / "plan.json").read_text())
        frozen[name] = {path: sha(REPO / path) == digest for path, digest in plan["source_sha256"].items()}
    out = REPO / "results/copula_topology_v2"
    result = json.loads((out / "result.json").read_text())
    bindings = {"plan": result["plan_sha256"] == sha(out / "plan.json"),
                "validation": result["validation_sha256"] == sha(out / "validation.json"),
                "patches": result["patches_sha256"] == sha(REPO / "data/work/copula_topology_v2/patches.npz")}
    boundary_path = REPO / "results/copula_boundary_audit_v1"
    boundary = json.loads((boundary_path / "result.json").read_text())
    bindings["boundary_plan"] = boundary["plan_sha256"] == sha(boundary_path / "plan.json")
    assessment = json.loads((out / "assessment.json").read_text())
    flags = sum(record["flag"] for record in boundary["records"])
    expected_status = "calibration_failed" if flags > boundary["cutoff"] else "exploratory_only"
    bindings["scientific_assessment"] = (assessment["sky_result_sha256"] == sha(out / "result.json")
        and assessment["boundary_result_sha256"] == sha(boundary_path / "result.json")
        and flags == boundary["familywise_flags"] == assessment["boundary_false_flags"]
        and assessment["status"] == expected_status
        and not assessment["accepted_cosmological_claims"])
    test_log = REPO / "data/copula_topology_tests.log"
    summary = parse_test_summary(test_log.read_text())
    tests_ok = (summary is not None and summary["passed"] == 175 and summary["skipped"] == 3
                and summary["failed"] == 0 and summary["errors"] == 0)
    ok = (ret == 0 and chain.get("valid") is True and all(r["matched"] for r in suites.values())
          and all(all(checks.values()) for checks in frozen.values()) and all(bindings.values()) and tests_ok)
    evidence = {"verified_utc": datetime.now(timezone.utc).isoformat(), "status": "passed" if ok else "failed",
                "scope": "Implementation/provenance checks; does not validate physical surrogate assumptions",
                "jovin_chain": chain, "jovin_suites": suites, "frozen_sources": frozen,
                "local_tests": {"passed": summary["passed"] if summary else None,
                                "skipped": summary["skipped"] if summary else None, "summary_matched": tests_ok,
                                "log_sha256": sha(test_log)},
                "result_bindings": bindings,
                "boundary_result_sha256": sha(boundary_path / "result.json"),
                "artifact_sha256": {name: sha(out / name) for name in ("plan.json", "validation.json", "result.json", "assessment.json", "topology.png", "calibration_controls.png")},
                "report_sha256": sha(REPO / "docs/COPULA_TOPOLOGY_RUN.md")}
    (out / "verification.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"status": evidence["status"], "suites": len(suites),
                      "checks": sum(r["total"] for r in suites.values()), "result_bindings": bindings}, indent=2))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
