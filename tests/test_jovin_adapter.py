"""The JoVin adapter must never convert transport or provenance failures to passes."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "jovin_verify_adapter", Path(__file__).resolve().parents[1] / "scripts" / "jovin_verify.py"
)
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


def _passed_result():
    return {"status": "passed", "passed": True, "total": 2, "passed_count": 2,
            "timed_out": False, "integrity": "shared_interpreter"}


@pytest.mark.parametrize("change", [
    {"passed": "true"}, {"status": "timeout"}, {"timed_out": True},
    {"total": 0, "passed_count": 0}, {"total": True, "passed_count": True},
    {"passed_count": 1}, {"integrity": "unverified"},
])
def test_result_must_be_an_unambiguous_complete_pass(change):
    result = _passed_result()
    assert adapter._passed(result)
    result.update(change)
    assert not adapter._passed(result)


@pytest.mark.parametrize("stdout", ["[]", "null", "not-json", '"passed"'])
def test_non_object_harness_response_fails_closed(monkeypatch, stdout):
    monkeypatch.setattr(adapter.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=0, stdout=stdout, stderr=""))
    _, result = adapter._harness(["unused"], {}, ["doctor"])
    assert result["adapter_error"]
    assert not adapter._passed(result)


def test_explicit_command_and_profile_override_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("JOVIN_HARNESS_PYTHON", "old-python")
    monkeypatch.setenv("JOVIN_HARNESS_RUNTIME", "old-runtime")
    monkeypatch.setenv("JOVIN_HARNESS_DATA_DIR", "old-data")
    args = argparse.Namespace(harness="new-harness", harness_python=None,
                              runtime="new-runtime", data_dir=tmp_path)
    command, env = adapter.configuration(args)
    assert command == ["new-harness"]
    assert env["JOVIN_HARNESS_RUNTIME"] == "new-runtime"
    assert env["JOVIN_HARNESS_DATA_DIR"] == str(tmp_path.resolve())


def _receipt():
    result = dict(_passed_result(), receipt_id="20261023T120000Z-0123456789ab")
    doctor = {"image": "image-pinned", "packaged_runner_sha256": "runner-pinned"}
    receipt = {
        "format": "jovin-harness.receipt.v1", "id": result["receipt_id"],
        "tool": "verify_python", "integrity": "shared_interpreter",
        "inputs": {"code_sha256": "code", "tests_sha256": "tests",
                   "params": {"isolation": "shared", "timeout_s": 30}},
        "result": {k: result[k] for k in ("status", "total", "passed_count", "timed_out")},
        "sandbox": {"inspect_verified": True, "kernel_verified": True, "cleanup": "removed",
                    "image": {"id": doctor["image"], "runner_sha256": doctor["packaged_runner_sha256"]}},
    }
    digest = hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    receipt["sha256"] = result["receipt_sha256"] = digest
    return receipt, result, doctor


def test_receipt_is_bound_to_source_tests_profile_and_saved_content():
    receipt, result, doctor = _receipt()
    assert not adapter._receipt_problems(receipt, result, "code", "tests", 30, doctor)
    assert adapter._receipt_problems(receipt, result, "changed-code", "tests", 30, doctor)
    assert adapter._receipt_problems(receipt, result, "code", "changed-tests", 30, doctor)
    assert adapter._receipt_problems(receipt, result, "code", "tests", 10, doctor)
    receipt["sandbox"]["cleanup"] = "failed"
    problems = adapter._receipt_problems(receipt, result, "code", "tests", 30, doctor)
    assert any("content hash" in p for p in problems)
    assert any("removal" in p for p in problems)


def test_unlinked_receipt_cannot_pass_even_with_a_passing_payload(monkeypatch, tmp_path):
    receipt, result, doctor = _receipt()
    # Return a deliberately unlinked receipt even though the verifier reports success.
    def harness(command, env, args):
        if args[0] == "verify":
            return 0, result
        if args[1] == "show":
            return 0, receipt
        return 0, {"valid": True, "chain": "unlinked"}
    monkeypatch.setattr(adapter, "_harness", harness)
    report = adapter.run_suite("accel", tmp_path, ["unused"], {}, 30, doctor)
    assert report["passed"] is False
    assert "linked member" in report["failures"]["adapter_problems"][0]


@pytest.mark.parametrize("limit,explicit,expected", [
    (60, None, 30), (12, None, 12), (30, 7, 7),
    (0, None, None), ("30", None, None), (None, None, None),
    (True, None, None), (60, 31, None), (12, 13, None), (30, 0, None),
])
def test_timeout_respects_profile_and_default_cap(monkeypatch, tmp_path, limit, explicit, expected):
    monkeypatch.setattr(adapter, "configuration", lambda args: (["unused"], {}))
    monkeypatch.setattr(adapter, "_harness", lambda *args: (0, {"ready": True,
        "limits": {"timeout_s": limit}, "valid": True}))
    timeouts = []
    def run(*args):
        timeouts.append(args[4])
        return dict(passed=True, passed_count=1, total=1, receipt_id="fixture", failures=None)
    monkeypatch.setattr(adapter, "run_suite", run)
    args = ["accel", "--ledger", str(tmp_path / "ledger.json")]
    if explicit is not None:
        args += ["--timeout", str(explicit)]
    if expected is None:
        with pytest.raises(SystemExit):
            adapter.main(args)
        assert not timeouts
    else:
        assert adapter.main(args) == 0
        assert timeouts == [expected]


def test_failed_chain_records_details_and_preserves_prior_failures(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "configuration", lambda args: (["unused"], {}))
    chain = {"valid": False, "reason": "broken link"}
    monkeypatch.setattr(adapter, "_harness", lambda command, env, args:
        (0, {"ready": True, "limits": {"timeout_s": 30}}) if args == ["doctor"] else (2, chain))
    prior = {"adapter_problems": ["original failure"]}
    reports = iter([dict(passed=True, failures=None), dict(passed=False, failures=prior)])
    def run(*args):
        return dict(next(reports), passed_count=1, total=1, receipt_id="fixture")
    monkeypatch.setattr(adapter, "run_suite", run)
    ledger = tmp_path / "ledger.json"
    assert adapter.main(["accel", "network", "--ledger", str(ledger)]) == 1
    first, second = json.loads(ledger.read_text())
    assert not first["passed"] and not second["passed"]
    assert first["failures"] == {"adapter_problems": ["receipt-chain verification failed"],
                                 "chain_verification": chain, "harness_exit": 2}
    assert second["failures"] == prior
