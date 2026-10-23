"""Audit authentic receipts, immutable inputs, all test decisions and old artifacts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from jovin_verify import REPO, SUITES, build_bundle, configuration, _harness

OUT = REPO / "results/spherical_topology_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def independent_pvalues(observed, references):
    raw = []
    for row in range(3):
        for col in range(5):
            values = [s[row][col] for s in references]
            value = observed[row][col]
            n = len(values)+1
            raw.append(min(1., 2*min((1+sum(v <= value for v in values))/n,
                                    (1+sum(v >= value for v in values))/n)))
    adjusted = [0.]*15
    running = 0.
    for rank, idx in enumerate(sorted(range(15), key=lambda i: raw[i])):
        running = max(running, (15-rank)*raw[idx])
        adjusted[idx] = min(1., running)
    return raw, adjusted


def flatten(values):
    return [v for row in values for v in row]


def close(values, expected):
    return len(values) == len(expected) and all(abs(a-b) <= 1e-14 for a,b in zip(values, expected))


def main():
    args = argparse.Namespace(harness=None, harness_python=None, runtime="wsl:JovinHarness", data_dir=None)
    command, env = configuration(args)
    ret, chain = _harness(command, env, ["receipts", "verify"])
    history = read(REPO / "data/evidence/jovin_receipts.json")
    suites = {}
    for suite, (_, test_path) in SUITES.items():
        record = [r for r in history if r["suite"] == suite][-1]
        show_ret, authentic = _harness(command, env, ["receipts", "show", record["receipt_id"]])
        member_ret, member = _harness(command, env, ["receipts", "verify", record["receipt_id"]])
        ok = (record["passed"] and record["receipt_verified"] and record["chain_valid"]
              and hashlib.sha256(build_bundle(suite)[0].encode()).hexdigest() == record["bundle_sha256"]
              and sha(REPO / test_path) == record["tests_sha256"]
              and show_ret == 0 and authentic == record["receipt"]
              and member_ret == 0 and member.get("valid") is True and member.get("chain") == "linked")
        suites[suite] = {"matched": bool(ok), "receipt_id": record["receipt_id"], "total": record["total"]}
    frozen = {}
    for name in ("surrogate_topology_v1", "copula_topology_v2", "copula_boundary_audit_v1", "spherical_topology_v1"):
        plan = read(REPO / "results" / name / "plan.json")
        frozen[name] = {path: sha(REPO / path) == digest for path,digest in plan["source_sha256"].items()}
    old = read(REPO / "results/copula_topology_v2/verification.json")
    preserved = {name: sha(REPO / "results/copula_topology_v2" / name) == digest
                 for name,digest in old["artifact_sha256"].items()}
    preserved["copula_report"] = sha(REPO / "docs/COPULA_TOPOLOGY_RUN.md") == old["report_sha256"]
    preserved["boundary_result"] = sha(REPO / "results/copula_boundary_audit_v1/result.json") == old["boundary_result_sha256"]
    full_planck = read(REPO / "data/evidence/planck_final_validation.json")
    preserved["full_planck_result"] = sha(REPO / "results/rtx5070_real/reproduction.json") == full_planck["result_sha256"]

    plan, result, stage = [read(OUT / name) for name in ("plan.json", "result.json", "stage_audit.json")]
    ensemble, validation, conditioning, positive = [read(OUT / name) for name in
        ("ensemble.json", "validation.json", "conditioning.json", "positive_control.json")]
    references = ensemble["statistics"][:999]
    heldout = ensemble["statistics"][999:]
    control_decisions = []
    for record, stats in zip(validation["records"], heldout):
        raw, adjusted = independent_pvalues(stats, references)
        control_decisions.append(record["statistics"] == stats
            and close(flatten(record["raw_p"]), raw) and close(flatten(record["p_holm"]), adjusted)
            and record["flag"] == (min(adjusted) <= .05))
    flags = sum(r["flag"] for r in validation["records"])
    scientific = {"ensemble_shape": len(references) == 999 and len(heldout) == 64
                    and all(len(s) == 3 and all(len(row) == 5 for row in s) for s in ensemble["statistics"]),
                  "finite_statistics": all(math.isfinite(x) for s in ensemble["statistics"] for row in s for x in row),
                  "all_1063_precision_checks": len(ensemble["quality"]) == 1063 and all(q["passed"]
                      and q["finite"] and q["zero_shells_exact"] and q["mzero_real"]
                      and q["maximum_relative_shell_error"] <= 1e-10 for q in ensemble["quality"]),
                  "all_control_tail_and_holm_values": len(control_decisions) == 64 and all(control_decisions),
                  "gate_decision": flags == validation["familywise_flags"]
                      and validation["status"] == ("passed" if flags <= 8 else "failed")}
    if validation["status"] == "passed":
        raw, adjusted = independent_pvalues(result["observed"], references)
        scientific["all_sky_tail_and_holm_values"] = close(flatten(result["raw_p"]), raw) and close(flatten(result["p_holm"]), adjusted)
        expected = [{"patch": result["patches"][idx//5], "statistic": result["statistics"][idx%5], "p_holm": value}
                    for idx,value in enumerate(adjusted) if value <= .05]
        scientific["all_candidate_flags"] = expected == result["candidate_flags"]
    else:
        scientific["real_sky_blocked"] = "observed" not in result and not result["candidate_flags"]
    raw, adjusted = independent_pvalues(positive["statistics"], references)
    scientific["positive_control"] = close(flatten(positive["raw_p"]), raw) and close(flatten(positive["p_holm"]), adjusted) and positive["quality"]["passed"]

    bindings = {"stage_result": stage["result_sha256"] == sha(OUT / "result.json"),
                "stage_plan": stage["plan_sha256"] == sha(OUT / "plan.json"),
                "input_audit": plan["input_audit_sha256"] == sha(REPO / "data/evidence/planck_input_audit.json"),
                "package_audit": plan["package_audit_sha256"] == sha(REPO / "data/evidence/cst_package_audit.json"),
                "input_inventory": plan["input_inventory_sha256"] == sha(REPO / "data/evidence/planck_dataset.json"),
                "observed_alm": conditioning["observed_alm_sha256"] == sha(REPO / "data/work/spherical_topology_v1/observed_alm.npy"),
                "conditional_cl": conditioning["cl_sha256"] == sha(REPO / "data/work/spherical_topology_v1/conditional_cl.npy")}
    for key, filename in (("plan_sha256", "plan.json"), ("conditioning_sha256", "conditioning.json"),
                          ("ensemble_sha256", "ensemble.json"), ("validation_sha256", "validation.json")):
        bindings[key] = result[key] == sha(OUT / filename)
    if "patches_sha256" in result:
        bindings["observed_patches"] = result["patches_sha256"] == sha(REPO / "data/work/spherical_topology_v1/observed_patches.npz")
    log = REPO / "data/spherical_topology_tests.log"
    tests_ok = "177 passed, 3 skipped" in log.read_text()
    ok = (ret == 0 and chain.get("valid") is True and all(s["matched"] for s in suites.values())
          and all(all(s.values()) for s in frozen.values()) and all(preserved.values())
          and all(scientific.values()) and all(bindings.values()) and tests_ok)
    artifacts = ["plan.json", "conditioning.json", "ensemble.json", "validation.json", "positive_control.json",
                 "result.json", "stage_audit.json", "assessment.json", "calibration.png"]
    if (OUT / "topology.png").exists():
        artifacts.append("topology.png")
    evidence = {"verified_utc": datetime.now(timezone.utc).isoformat(), "status": "passed" if ok else "failed",
        "scope": "Implementation and provenance verification; physical null assumptions are separate",
        "jovin_suites": suites, "jovin_chain": chain, "frozen_sources": frozen,
        "prior_artifacts_preserved": preserved, "scientific_calculation_checks": scientific,
        "bindings": bindings, "local_tests": {"passed": 177, "skipped": 3, "summary_matched": tests_ok, "log_sha256": sha(log)},
        "artifact_sha256": {name: sha(OUT / name) for name in artifacts},
        "report_sha256": sha(REPO / "docs/SPHERICAL_TOPOLOGY_RUN.md")}
    (OUT / "verification.json").write_text(json.dumps(evidence, indent=2)+"\n")
    print(json.dumps({"status": evidence["status"], "suites": len(suites), "checks": sum(s["total"] for s in suites.values()),
                      "scientific_calculation_checks": scientific, "prior_artifacts_preserved": all(preserved.values())}, indent=2))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
