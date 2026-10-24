"""Evidence decisions must reflect complete records and final test summaries."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_cst_package as package
from evidence_test_summary import parse_test_summary
import verify_spherical_evidence as spherical
import verify_topology_evidence as topology
import report_copula_topology as copula_report


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.mark.parametrize("text,expected", [
    ("=== 175 passed, 3 skipped in 2.10s ===", (175, 3, 0, 0)),
    ("1175 passed, 3 skipped in 2.10s", (1175, 3, 0, 0)),
    ("177 passed, 3 skipped in 1.00s\n1 failed, 177 passed, 3 skipped in 2.00s", (177, 3, 1, 0)),
    ("177 passed, 3 skipped, 1 error in 2.00s", (177, 3, 0, 1)),
    ("177 passed in 1.00s\n=== 1 error in 0.01s ===", (0, 0, 0, 1)),
    ("=== 177 passed, 3 skipped, 4 warnings in 61.00s (0:01:01) ===", (177, 3, 0, 0)),
    ("example: 175 passed, 3 skipped", None),
    ("175 passed, 3 skipped in 2.10s but failed later", None),
    ("no summary", None),
])
def test_complete_final_summary(text, expected):
    summary = parse_test_summary(text)
    assert (tuple(summary[k] for k in ("passed", "skipped", "failed", "errors"))
            if summary is not None else None) == expected


@pytest.mark.parametrize("change", ["valid", "empty_files", "missing_file", "extra_file",
    "empty_checks", "missing_check", "duplicate_check", "unknown_check", "failed_check", "truthy_pass"])
def test_package_requires_complete_checksum_and_replay_sets(monkeypatch, tmp_path, change):
    supplied = tmp_path / "package"
    supplied.mkdir()
    replay_dir = tmp_path / "data/work/cst_package_replay"
    replay_dir.mkdir(parents=True)
    (tmp_path / "data/evidence").mkdir()
    monkeypatch.setattr(package, "ROOT", tmp_path)
    monkeypatch.setattr(package, "PACKAGE", supplied)
    replay = {"total": 27, "passed": 27, "failed": 0,
              "checks": [{"id": f"V{i:02d}", "pass": True} for i in range(1, 28)]}
    original = copy.deepcopy(replay)
    if change == "empty_checks":
        replay["checks"] = []
    elif change == "missing_check":
        replay["checks"].pop()
    elif change == "duplicate_check":
        replay["checks"][-1] = replay["checks"][0]
    elif change == "unknown_check":
        replay["checks"][-1]["id"] = "V99"
    elif change == "failed_check":
        replay["checks"][0]["pass"] = False
    elif change == "truthy_pass":
        replay["checks"][0]["pass"] = 1
    for name in package.EXPECTED_FILES:
        (supplied / name).write_bytes(b"fixture")
    write_json(supplied / "validation_results_extended.json", original)
    (replay_dir / "cst_validation_extended.py").write_bytes(b"fixture")
    write_json(replay_dir / "validation_results_extended.json", replay)
    files = set(package.EXPECTED_FILES)
    if change == "empty_files":
        files.clear()
    elif change == "missing_file":
        files.remove("CST_End_to_End_Expanded.pdf")
    elif change == "extra_file":
        (supplied / "extra.txt").write_bytes(b"extra")
        files.add("extra.txt")
    (supplied / "SHA256SUMS.txt").write_text("\n".join(f"{package.sha(supplied / name)}  {name}" for name in sorted(files)))
    archive = tmp_path / "elsewhere/archive.zip"
    archive.parent.mkdir()
    archive.write_bytes(b"synthetic archive bytes")
    if change == "valid":
        package.main([str(archive)])
    else:
        with pytest.raises(SystemExit) as exit:
            package.main([str(archive)])
        assert exit.value.code == 1
    evidence = json.loads((tmp_path / "data/evidence/cst_package_audit.json").read_text())
    assert evidence["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert (evidence["status"] == "passed") is (change == "valid")
    if change in {"duplicate_check", "unknown_check"}:
        assert not evidence["checks"]["all_27_replayed"]


def verifier_setup(monkeypatch, tmp_path, module):
    monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(module, "SUITES", {})
    monkeypatch.setattr(module, "configuration", lambda args: (["unused"], {}))
    monkeypatch.setattr(module, "_harness", lambda *args: (0, {"valid": True}))
    monkeypatch.setattr(module, "sha", lambda path: "digest")
    write_json(tmp_path / "data/evidence/jovin_receipts.json", [])
    for name in ("surrogate_topology_v1", "copula_topology_v2", "copula_boundary_audit_v1", "spherical_topology_v1"):
        write_json(tmp_path / "results" / name / "plan.json", {"source_sha256": {}})


@pytest.mark.parametrize("flags", [0, 8, 9])
@pytest.mark.parametrize("tamper", [None, "count", "assessment_count", "status", "large_pass_count", "failure", "error", "no_summary"])
def test_topology_assessment_and_test_counts(monkeypatch, tmp_path, flags, tamper):
    verifier_setup(monkeypatch, tmp_path, topology)
    captured = []
    monkeypatch.setattr(topology, "configuration", lambda args: (captured.append(args) or ["unused"], {}))
    out = tmp_path / "results/copula_topology_v2"
    write_json(out / "result.json", {key: "digest" for key in ("plan_sha256", "validation_sha256", "patches_sha256")})
    boundary = {"records": [{"flag": i < flags} for i in range(64)], "familywise_flags": flags,
                "cutoff": 8, "plan_sha256": "digest"}
    assessment = {"sky_result_sha256": "digest", "boundary_result_sha256": "digest",
                  "boundary_false_flags": flags, "status": "calibration_failed" if flags > 8 else "exploratory_only",
                  "accepted_cosmological_claims": []}
    if tamper == "count":
        boundary["familywise_flags"] += 1
    if tamper == "assessment_count":
        assessment["boundary_false_flags"] += 1
    if tamper == "status":
        assessment["status"] = "wrong"
    write_json(tmp_path / "results/copula_boundary_audit_v1/result.json", boundary)
    write_json(out / "assessment.json", assessment)
    summary = {"large_pass_count": "1175 passed, 3 skipped", "failure": "1 failed, 175 passed, 3 skipped",
               "error": "175 passed, 3 skipped, 1 error", "no_summary": "unparseable"}.get(tamper, "175 passed, 3 skipped")
    (tmp_path / "data/copula_topology_tests.log").write_text(summary + " in 1.00s")
    if tamper:
        with pytest.raises(SystemExit):
            topology.main([])
    else:
        topology.main([])
    args = captured[0]
    assert all(getattr(args, key) is None for key in ("runtime", "harness", "harness_python", "data_dir"))
    evidence = json.loads((out / "verification.json").read_text())
    assert evidence["local_tests"]["passed"] == (1175 if tamper == "large_pass_count" else None if tamper == "no_summary" else 175)
    assert (evidence["status"] == "passed") is (tamper is None)


def test_topology_cli_overrides_reach_configuration(monkeypatch, tmp_path):
    class Configured(Exception):
        pass
    def configuration(args):
        assert args.runtime == "custom-runtime"
        assert args.harness_python == "custom-python"
        assert args.harness is None
        assert args.data_dir == tmp_path
        raise Configured
    monkeypatch.setattr(topology, "configuration", configuration)
    with pytest.raises(Configured):
        topology.main(["--runtime", "custom-runtime", "--harness-python", "custom-python", "--data-dir", str(tmp_path)])


@pytest.mark.parametrize("status,candidates,expected", [
    ("matching_failed", [1], "matching failures block candidate interpretation"),
    ("completed", [1], "passed the run's quality gate. These are exploratory leads."),
    ("completed", [], "No comparison crossed the adjusted .05 threshold."),
])
def test_boundary_caveat_keeps_status_specific_note(monkeypatch, tmp_path, status, candidates, expected):
    monkeypatch.setattr(copula_report, "REPO", tmp_path)
    out = tmp_path / "results/copula_topology_v2"
    write_json(out / "result.json", dict(status=status, candidate_flags=candidates,
        completed_utc="2026-10-23T12:00:01+00:00", records={"patch": dict(observed=[1],
        surrogate_mean=[1], surrogate_sd=[0], raw_p=[1], p_holm=[1], quality=[dict(passed=True,
        spectrum_stage_relative_error=0, histogram_stage_spectrum_error=0, spectrum_stage_histogram_ks=0)])}))
    write_json(out / "plan.json", dict(source_sha256={}, statistics=["stat"],
        created_utc="2026-10-23T12:00:00+00:00", jovin_receipt_id="fixture"))
    controls = dict(familywise_flags=0, records=[dict(flag=False, quality_failures=0)])
    write_json(out / "validation.json", controls)
    write_json(tmp_path / "results/surrogate_topology_v1/validation.json", controls)
    write_json(tmp_path / "results/copula_boundary_audit_v1/result.json", dict(controls, status="failed", fields=64))
    (tmp_path / "docs").mkdir()
    copula_report.main(["--report-only"])
    report = (tmp_path / "docs/COPULA_TOPOLOGY_RUN.md").read_text()
    assert expected in report
    assert "**The cropped-Gaussian boundary audit failed;" in report
    assert "(limit 1e-10)" in report


@pytest.mark.parametrize("cache_state", ["absent", "present", "corrupt"])
@pytest.mark.parametrize("test_summary", ["177 passed, 3 skipped", "1177 passed, 3 skipped", "1 failed, 177 passed, 3 skipped", "177 passed, 3 skipped, 1 error"])
def test_spherical_shipped_bindings_and_optional_cache(monkeypatch, tmp_path, cache_state, test_summary):
    verifier_setup(monkeypatch, tmp_path, spherical)
    out = tmp_path / "results/spherical_topology_v1"
    monkeypatch.setattr(spherical, "OUT", out)
    hashed = []
    def sha(path):
        path = Path(path)
        hashed.append(path)
        if "work" in path.parts:
            assert path.is_file()
            return path.read_text()
        return "digest"
    monkeypatch.setattr(spherical, "sha", sha)
    write_json(tmp_path / "results/copula_topology_v2/verification.json",
               {"artifact_sha256": {}, "report_sha256": "digest", "boundary_result_sha256": "digest"})
    write_json(tmp_path / "data/evidence/planck_final_validation.json", {"result_sha256": "digest"})
    stats = [[0.] * 5 for _ in range(3)]
    ps = [[1.] * 5 for _ in range(3)]
    quality = {"passed": True, "finite": True, "zero_shells_exact": True,
               "mzero_real": True, "maximum_relative_shell_error": 0.}
    write_json(out / "plan.json", dict(source_sha256={}, input_audit_sha256="digest",
        package_audit_sha256="digest", input_inventory_sha256="digest"))
    write_json(out / "result.json", dict(observed=stats, raw_p=ps, p_holm=ps, candidate_flags=[],
        patches=["a", "b", "c"], statistics=list("abcde"), **{key: "digest" for key in
        ("plan_sha256", "conditioning_sha256", "ensemble_sha256", "validation_sha256", "patches_sha256")}))
    write_json(out / "stage_audit.json", {"result_sha256": "digest", "plan_sha256": "digest"})
    write_json(out / "ensemble.json", {"statistics": [stats] * 1063, "quality": [quality] * 1063})
    write_json(out / "validation.json", dict(records=[dict(statistics=stats, raw_p=ps, p_holm=ps, flag=False)] * 64,
        familywise_flags=0, status="passed"))
    write_json(out / "conditioning.json", {"observed_alm_sha256": "digest", "cl_sha256": "digest"})
    write_json(out / "positive_control.json", dict(statistics=stats, raw_p=ps, p_holm=ps, quality=quality))
    cache = tmp_path / "data/work/spherical_topology_v1"
    if cache_state != "absent":
        cache.mkdir(parents=True)
        for name in ("observed_alm.npy", "conditional_cl.npy", "observed_patches.npz"):
            (cache / name).write_text("wrong" if cache_state == "corrupt" and name == "conditional_cl.npy" else "digest")
    (tmp_path / "data/spherical_topology_tests.log").write_text(test_summary + " in 1.00s")
    expected = cache_state != "corrupt" and test_summary == "177 passed, 3 skipped"
    if expected:
        spherical.main()
    else:
        with pytest.raises(SystemExit):
            spherical.main()
    evidence = json.loads((out / "verification.json").read_text())
    assert (evidence["status"] == "passed") is expected
    assert all(evidence["bindings"].values())
    assert "observed_alm" not in evidence["bindings"]
    assert set(evidence["local_cache_checks"]) == {"observed_alm", "conditional_cl", "observed_patches"}
    assert evidence["local_cache_checks"]["observed_alm"]["available"] is (cache_state != "absent")
    assert evidence["local_cache_checks"]["conditional_cl"]["matched"] == (
        None if cache_state == "absent" else cache_state == "present")
    assert (out / "conditioning.json") in hashed
