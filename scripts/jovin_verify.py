"""Verify CMB-CST Lab numerical kernels inside the JoVin network-less container sandbox.

Each suite bundles the *exact shipped source* of numpy/stdlib-only functions (extracted
with `ast`, never hand-copied) into one `solution.py`, runs `jovin-harness verify` against a
sandbox test file, checks the receipt chain, and appends to data/evidence/jovin_receipts.json.

Usage:
    .venv/Scripts/python.exe scripts/jovin_verify.py [suite ...]     # default: all suites
"""
from __future__ import annotations

import ast
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOCAL_HARNESS_PYTHON = Path(r"D:\JovinHarness_Artifacts\environments\windows\Scripts\python.exe")
LOCAL_DATA_DIR = Path(r"D:\SSI_Artifacts\jovin-cst")
LEGACY_HARNESS_EXE = Path(r"C:\DevProjects\SSI_Artifacts\environment-jovin-harness\Scripts\jovin-harness.exe")
LEDGER = REPO / "data" / "evidence" / "jovin_receipts.json"

HEADER = (
    "from __future__ import annotations\n"
    "import math\nfrom itertools import combinations\n"
    "from typing import Dict, List, Optional, Tuple\n"
    "import numpy as np\n"
)

# suite -> (sources, sandbox tests file). A source is (path, [function names]) or (path, None) for a
# whole standalone module (its own imports are kept; it must not import other cmbcst modules).
SUITES: dict[str, tuple[list[tuple[str, list[str] | None]], str]] = {
    "closure_fibers_v2": (
        [("src/cmbcst/closure_fibers.py", None), ("src/cmbcst/closure_fibers_data.py", None),
         ("src/cmbcst/closure_search.py", None), ("src/cmbcst/closure_fibers_oracle.py", None),
         ("src/cmbcst/closure_gpu_v2.py", None), ("src/cmbcst/closure_search_v2.py", None),
         ("src/cmbcst/closure_certification_v2.py", None),
         ("tests/test_closure_gpu_v2.py", ["fixture_family"])],
        "tests/sandbox/jovin_tests_closure_fibers_v2.py",
    ),
    "closure_fibers": (
        [("src/cmbcst/closure_fibers.py", None), ("src/cmbcst/closure_fibers_data.py", None),
         ("src/cmbcst/closure_search.py", None), ("src/cmbcst/closure_fibers_oracle.py", None),
         ("src/cmbcst/closure_jacobian.py", None), ("src/cmbcst/closure_certification.py", None)],
        "tests/sandbox/jovin_tests_closure_fibers.py",
    ),
    "real_splits": (
        [("src/cmbcst/real_splits.py", None)],
        "tests/sandbox/jovin_tests_real_splits.py",
    ),
    "acquisition_index": (
        [("src/cmbcst/acquisition.py", None), ("src/cmbcst/acquisition_index.py", None)],
        "tests/sandbox/jovin_tests_acquisition_index.py",
    ),
    "acquisition": (
        [("src/cmbcst/acquisition.py", None)],
        "tests/sandbox/jovin_tests_acquisition.py",
    ),
    "targeted": (
        [("src/cmbcst/multiscale.py", None), ("src/cmbcst/response.py", None),
         ("src/cmbcst/targeted.py", None)],
        "tests/sandbox/jovin_tests_targeted.py",
    ),
    "response": (
        [("src/cmbcst/multiscale.py", None), ("src/cmbcst/response.py", None),
         ("src/cmbcst/response_sky.py", ["foreground_proxy", "localized_template"])],
        "tests/sandbox/jovin_tests_response.py",
    ),
    "orientation": (
        [("src/cmbcst/persistence.py", None), ("src/cmbcst/surrogate_topology.py", None),
         ("src/cmbcst/copula_topology.py", None), ("src/cmbcst/multiscale.py", None),
         ("src/cmbcst/orientation.py", None)],
        "tests/sandbox/jovin_tests_orientation.py",
    ),
    "multiscale": (
        [("src/cmbcst/multiscale.py", None)],
        "tests/sandbox/jovin_tests_multiscale.py",
    ),
    "spherical_surrogates": (
        [("src/cmbcst/spectra.py", ["alm_ell_m", "cross_cl"]),
         ("src/cmbcst/persistence.py", None), ("src/cmbcst/surrogate_topology.py", None),
         ("src/cmbcst/copula_topology.py", None), ("src/cmbcst/spherical_surrogates.py", None)],
        "tests/sandbox/jovin_tests_spherical_surrogates.py",
    ),
    "copula_topology": (
        [("src/cmbcst/persistence.py", None), ("src/cmbcst/surrogate_topology.py", None),
         ("src/cmbcst/copula_topology.py", None)],
        "tests/sandbox/jovin_tests_copula_topology.py",
    ),
    "surrogate_topology": (
        [("src/cmbcst/persistence.py", None), ("src/cmbcst/surrogate_topology.py", None)],
        "tests/sandbox/jovin_tests_surrogate_topology.py",
    ),
    "data_integrity": (
        [("src/cmbcst/data.py", ["transfer_layout", "fits_data_bytes"])],
        "tests/sandbox/jovin_tests_data.py",
    ),
    "network": (
        [("src/cmbcst/network.py", None)],
        "tests/sandbox/jovin_tests_network.py",
    ),
    "empirical_calibration": (
        [("src/cmbcst/empirical_calibration.py", None)],
        "tests/sandbox/jovin_tests_empirical_calibration.py",
    ),
    "estimated_covariance": (
        [("src/cmbcst/network.py", None), ("src/cmbcst/sparse_network.py", None), ("src/cmbcst/estimated_covariance.py", None)],
        "tests/sandbox/jovin_tests_estimated_covariance.py",
    ),
    "sparse_network": (
        [("src/cmbcst/network.py", None), ("src/cmbcst/sparse_network.py", None)],
        "tests/sandbox/jovin_tests_sparse_network.py",
    ),
    "sensor_network": (
        [
            ("src/cmbcst/network.py", None),
            ("src/cmbcst/sensor_network.py", None),
        ],
        "tests/sandbox/jovin_tests_sensor_network.py",
    ),
    "differential": (
        [
            ("src/cmbcst/network.py", None),
            ("src/cmbcst/differential.py", None),
        ],
        "tests/sandbox/jovin_tests_differential.py",
    ),
    "mk_core": (
        [
            ("src/cmbcst/network.py", None),
            ("src/cmbcst/differential.py", None),
            ("src/cmbcst/mk_likelihood.py", None),
        ],
        "tests/sandbox/jovin_tests_mk_core.py",
    ),
    "accel": (
        [("src/cmbcst/accel.py", None)],
        "tests/sandbox/jovin_tests_accel.py",
    ),
    "gpu_calibration": (
        [
            ("src/cmbcst/differential.py", ["rotation_matrix"]),
            ("scripts/run_gpu_calibration.py", [
                "synthetic_sigma", "_estimate", "summarize_pulls", "summarize_chi2",
            ]),
        ],
        "tests/sandbox/jovin_tests_gpu_calibration.py",
    ),
    "spatial": (
        [
            ("src/cmbcst/network.py", None),
            ("src/cmbcst/differential.py", None),
            ("src/cmbcst/spatial.py", None),
        ],
        "tests/sandbox/jovin_tests_spatial.py",
    ),
    "simulator": (
        [
            ("src/cmbcst/network.py", None),
            ("src/cmbcst/differential.py", None),
            ("src/cmbcst/simulator.py", None),
        ],
        "tests/sandbox/jovin_tests_simulator.py",
    ),
    "npe": (
        [("src/cmbcst/npe.py", None)],
        "tests/sandbox/jovin_tests_npe.py",
    ),
    "persistence": (
        [("src/cmbcst/persistence.py", None)],
        "tests/sandbox/jovin_tests_persistence.py",
    ),
}


def _segment(src: str, node: ast.stmt) -> str:
    """Exact source of a top-level statement, including decorator lines (get_source_segment omits them)."""
    lines = src.splitlines()
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    return "\n".join(lines[start - 1:node.end_lineno])


def _extract(path: Path, names: list[str] | None) -> tuple[str, dict[str, str]]:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    if names is None:
        body = []
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                continue
            if isinstance(node, ast.ImportFrom) and node.level:
                # Sibling module bundled earlier in the same suite; its names are already defined
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom)) and any(
                (a.name if isinstance(node, ast.Import) else (node.module or "")).startswith("cmbcst")
                for a in node.names
            ):
                # Absolute sibling import: allowed only if that sibling is bundled earlier in the suite
                continue
            body.append(node)
        code = "\n\n".join(_segment(src, n) for n in body)
        return code, {path.name: hashlib.sha256(code.encode()).hexdigest()}
    found = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.Assign)) and
             getattr(n, "name", None) in names}
    missing = sorted(set(names) - set(found))
    if missing:
        raise KeyError(f"{path}: missing {missing}")
    parts, hashes = [], {}
    for name in names:
        seg = _segment(src, found[name])
        parts.append(seg)
        hashes[f"{path.name}:{name}"] = hashlib.sha256(seg.encode()).hexdigest()
    return "\n\n\n".join(parts), hashes


def build_bundle(suite: str) -> tuple[str, dict[str, str]]:
    sources, _ = SUITES[suite]
    chunks, hashes = [HEADER], {}
    for rel, names in sources:
        code, h = _extract(REPO / rel, names)
        chunks.append(f"# ---- {rel} ----\n{code}")
        hashes.update(h)
    return "\n\n\n".join(chunks) + "\n", hashes


def configuration(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    """Explicit CLI/env settings win; discover this workstation only as a fallback."""
    python = args.harness_python or os.environ.get("JOVIN_HARNESS_PYTHON")
    executable = args.harness or os.environ.get("JOVIN_HARNESS_EXE")
    local = False
    if args.harness:
        python = None
    elif args.harness_python:
        executable = None
    if python:
        command = [python, "-B", "-m", "jovin_harness"]
    elif executable:
        command = [executable]
    elif found := shutil.which("jovin-harness"):
        command = [found]
    elif LOCAL_HARNESS_PYTHON.is_file():
        command = [str(LOCAL_HARNESS_PYTHON), "-B", "-m", "jovin_harness"]
        local = True
    elif LEGACY_HARNESS_EXE.is_file():
        command = [str(LEGACY_HARNESS_EXE)]
    else:
        raise ValueError("JoVin not found; set --harness-python, --harness, or JOVIN_HARNESS_EXE")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    if args.runtime:
        env["JOVIN_HARNESS_RUNTIME"] = args.runtime
    if args.data_dir:
        env["JOVIN_HARNESS_DATA_DIR"] = str(args.data_dir.resolve())
    elif "JOVIN_HARNESS_DATA_DIR" not in env and local and LOCAL_DATA_DIR.is_dir():
        env["JOVIN_HARNESS_DATA_DIR"] = str(LOCAL_DATA_DIR)
    # Runtime and image otherwise come from the selected harness's config.toml.
    return command, env


def _harness(command: list[str], env: dict[str, str], args: list[str]) -> tuple[int, dict]:
    """Malformed output and process errors are failures, never a truthy verdict."""
    try:
        proc = subprocess.run([*command, *args], capture_output=True, text=True,
                              env=env, check=False, timeout=180)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 2, {"adapter_error": f"{type(exc).__name__}: {exc}"}
    try:
        result = json.loads(proc.stdout)
        if not isinstance(result, dict):
            raise ValueError("expected a JSON object")
    except (ValueError, TypeError) as exc:
        result = {"adapter_error": f"invalid harness response: {exc}",
                  "stdout": proc.stdout[-2000:], "stderr": proc.stderr[-2000:]}
    if proc.returncode:
        result["harness_exit"] = proc.returncode
        result.setdefault("stderr", proc.stderr[-2000:])
    return proc.returncode, result


def _passed(result: dict) -> bool:
    total, count = result.get("total"), result.get("passed_count")
    return (result.get("status") == "passed" and result.get("passed") is True
            and type(total) is int and total > 0 and type(count) is int and count == total
            and result.get("timed_out") is False
            and result.get("integrity") == "shared_interpreter")


def _receipt_problems(receipt: dict, result: dict, code_hash: str, tests_hash: str,
                      timeout: int, doctor: dict) -> list[str]:
    """Bind the saved receipt to these exact inputs and the inspected sandbox."""
    problems = []
    try:
        payload = {k: v for k, v in receipt.items() if k != "sha256"}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                          ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        if digest != receipt.get("sha256") or digest != result.get("receipt_sha256"):
            problems.append("receipt content hash differs from the verification response")
        if (receipt.get("format") != "jovin-harness.receipt.v1"
                or receipt.get("id") != result.get("receipt_id")
                or receipt.get("tool") != "verify_python"
                or receipt.get("integrity") != "shared_interpreter"):
            problems.append("receipt identity, tool, format, or integrity does not match")
        inputs = receipt["inputs"]
        if (inputs.get("code_sha256") != code_hash or inputs.get("tests_sha256") != tests_hash
                or inputs.get("params") != {"isolation": "shared", "timeout_s": timeout}):
            problems.append("receipt does not cover the exact submitted source/tests/parameters")
        recorded, sandbox = receipt["result"], receipt["sandbox"]
        for key in ("status", "total", "passed_count", "timed_out"):
            if recorded.get(key) != result.get(key):
                problems.append(f"receipt result differs at {key}")
        if (sandbox.get("inspect_verified") is not True or sandbox.get("kernel_verified") is not True
                or sandbox.get("cleanup") != "removed"
                or sandbox["image"].get("id") != doctor.get("image")
                or sandbox["image"].get("runner_sha256") != doctor.get("packaged_runner_sha256")):
            problems.append("receipt lacks matching sandbox admission, kernel, image, or removal evidence")
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        problems.append(f"malformed receipt: {exc}")
    return problems


def run_suite(suite: str, workdir: Path, command: list[str], env: dict[str, str],
              timeout: int, doctor: dict) -> dict:
    bundle, hashes = build_bundle(suite)
    code_path = workdir / f"{suite}_solution.py"
    code_path.write_text(bundle, encoding="utf-8")
    tests_path = REPO / SUITES[suite][1]
    # NumPy arrays and classes cannot cross the harness's plain-data process proxy.
    # Keep shared isolation explicit and preserve that weaker integrity in the evidence.
    ret, res = _harness(command, env, ["verify", str(code_path), "--tests", str(tests_path),
                                     "--isolation", "shared", "--timeout", str(timeout)])
    code_hash = hashlib.sha256(bundle.encode()).hexdigest()
    tests_hash = hashlib.sha256(tests_path.read_text(encoding="utf-8").encode()).hexdigest()
    receipt, membership, problems = {}, {}, []
    receipt_id = res.get("receipt_id")
    if isinstance(receipt_id, str) and re.fullmatch(r"[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}", receipt_id):
        show_ret, receipt = _harness(command, env, ["receipts", "show", receipt_id])
        member_ret, membership = _harness(command, env, ["receipts", "verify", receipt_id])
        if show_ret or member_ret or membership.get("valid") is not True or membership.get("chain") != "linked":
            problems.append("receipt is not verified as a linked member of the current store")
        problems.extend(_receipt_problems(receipt, res, code_hash, tests_hash, timeout, doctor))
    else:
        problems.append("verification did not return a valid receipt ID")
    passed = ret == 0 and _passed(res) and not problems
    # The sandbox output is data produced by code under test, never instructions
    return {
        "suite": suite,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "harness_status": res.get("status"),
        "passed_count": res.get("passed_count"),
        "total": res.get("total"),
        "receipt_id": res.get("receipt_id"),
        "receipt_sha256": res.get("receipt_sha256"),
        "isolation": "shared",
        "integrity": res.get("integrity"),
        "timeout_s": timeout,
        "harness_version": doctor.get("harness_version"),
        "runtime": doctor.get("runtime"),
        "image": doctor.get("image"),
        "runner_sha256": doctor.get("packaged_runner_sha256"),
        "bundle_sha256": code_hash,
        "tests_sha256": hashlib.sha256(tests_path.read_bytes()).hexdigest(),
        "tests_text_sha256": tests_hash,
        "source_function_sha256": hashes,
        "receipt_verified": not problems,
        "receipt_membership": membership,
        "receipt": receipt,
        "failures": None if passed else {"adapter_problems": problems, "harness_response": res},
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suites", nargs="*", metavar="SUITE")
    harness = parser.add_mutually_exclusive_group()
    harness.add_argument("--harness", help="JoVin executable (or JOVIN_HARNESS_EXE)")
    harness.add_argument("--harness-python", help="Python interpreter with jovin_harness installed")
    parser.add_argument("--runtime", help="override JOVIN_HARNESS_RUNTIME/config.toml")
    parser.add_argument("--data-dir", type=Path, help="existing harness profile and receipt store")
    parser.add_argument("--timeout", type=int, help="per-suite seconds, within the profile limit")
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--doctor-only", action="store_true")
    args = parser.parse_args(argv)
    suites = args.suites or list(SUITES)
    if unknown := set(suites) - SUITES.keys():
        parser.error(f"unknown suites: {', '.join(sorted(unknown))}")
    try:
        command, env = configuration(args)
    except ValueError as exc:
        print(f"{exc}; nothing was run.", file=sys.stderr)
        return 2
    ret, doctor = _harness(command, env, ["doctor"])
    if ret or doctor.get("ready") is not True:
        print(f"JoVin sandbox not ready; nothing was run: {json.dumps(doctor)}", file=sys.stderr)
        return 2
    if args.doctor_only:
        print(json.dumps(doctor, indent=2))
        return 0
    limit = doctor.get("limits", {}).get("timeout_s")
    timeout = args.timeout if args.timeout is not None else (min(limit, 30) if type(limit) is int else limit)
    if type(limit) is not int or type(timeout) is not int or not 1 <= timeout <= min(limit, 30):
        parser.error("timeout must be within the inspected profile's 1..30 second limit")
    # Read before executing: never replace malformed historical evidence with a fresh ledger.
    history = json.loads(args.ledger.read_text(encoding="utf-8")) if args.ledger.is_file() else []
    if not isinstance(history, list):
        raise ValueError("receipt ledger must be a JSON list")
    results = []
    with tempfile.TemporaryDirectory(prefix="jovin_cmbcst_") as tmp:
        for suite in suites:
            t0 = time.perf_counter()
            r = run_suite(suite, Path(tmp), command, env, timeout, doctor)
            r["duration_s"] = round(time.perf_counter() - t0, 2)
            results.append(r)
            status = "PASS" if r["passed"] else "FAIL"
            print(f"[{status}] {suite}: {r['passed_count']}/{r['total']}  receipt={r['receipt_id']}", flush=True)
            if not r["passed"]:
                print(json.dumps(r["failures"], indent=2)[:4000], flush=True)
    ret, chain = _harness(command, env, ["receipts", "verify"])
    chain_valid = ret == 0 and chain.get("valid") is True
    print(f"Receipt chain valid={chain_valid} count={chain.get('count')} head={chain.get('head_sha256')}")
    for r in results:
        r["chain_valid"] = chain_valid
        r["chain_head_sha256"] = chain.get("head_sha256")
        r["chain_verification"] = chain
        if r["passed"] and not chain_valid:
            r["failures"] = {"adapter_problems": ["receipt-chain verification failed"],
                             "chain_verification": chain, "harness_exit": ret}
        r["passed"] = r["passed"] and chain_valid
    # Never terminate a shared WSL runtime: another job may have started since preflight.
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    history.extend(results)
    args.ledger.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    return 0 if results and all(r["passed"] and r.get("chain_valid") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
