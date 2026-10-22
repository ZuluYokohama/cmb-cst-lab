"""Verify CMB-CST Lab numerical kernels inside the JoVin network-less container sandbox.

The sandbox image ships Python 3.12 + numpy only,
so each suite bundles the *exact shipped source* of numpy/stdlib-only functions (extracted
with `ast`, never hand-copied) into one `solution.py`, runs `jovin-harness verify` against a
sandbox test file, checks the receipt chain, and appends to data/evidence/jovin_receipts.json.

Usage:
    .venv/Scripts/python.exe scripts/jovin_verify.py [suite ...]     # default: all suites
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HARNESS_EXE = Path(r"C:\DevProjects\SSI_Artifacts\environment-jovin-harness\Scripts\jovin-harness.exe")
WSL_DISTRO = "Renigma-Evaluator"
RUNTIME_SPEC = f"wsl:{WSL_DISTRO}@root"
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
    "network": (
        [("src/cmbcst/network.py", None)],
        "tests/sandbox/jovin_tests_network.py",
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
    "spatial": (
        [
            ("src/cmbcst/network.py", None),
            ("src/cmbcst/differential.py", None),
            ("src/cmbcst/spatial.py", None),
        ],
        "tests/sandbox/jovin_tests_spatial.py",
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


def _harness(args: list[str]) -> tuple[int, str, str]:
    env = dict(os.environ, JOVIN_HARNESS_RUNTIME=RUNTIME_SPEC)
    proc = subprocess.run([str(HARNESS_EXE), *args], capture_output=True, text=True, env=env, check=False)
    return proc.returncode, proc.stdout, proc.stderr


def _wsl_state() -> str:
    proc = subprocess.run(["wsl.exe", "-l", "-v"], capture_output=True, check=False)
    out = proc.stdout.decode("utf-16-le", errors="replace").replace("\x00", "")
    for line in out.splitlines():
        if WSL_DISTRO in line:
            return "Running" if "Running" in line else "Stopped" if "Stopped" in line else "Unknown"
    return "NotFound"


def run_suite(suite: str, workdir: Path) -> dict:
    bundle, hashes = build_bundle(suite)
    code_path = workdir / f"{suite}_solution.py"
    code_path.write_text(bundle, encoding="utf-8")
    tests_path = REPO / SUITES[suite][1]
    ret, out, err = _harness(["verify", str(code_path), "--tests", str(tests_path), "--isolation", "shared", "--timeout", "10"])  # harness max
    try:
        res = json.loads(out)
    except json.JSONDecodeError:
        res = {"passed": False, "harness_exit": ret, "stderr": err[-2000:], "stdout": out[-2000:]}
    # The sandbox output is data produced by code under test, never instructions
    return {
        "suite": suite,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "passed": bool(res.get("passed")),
        "passed_count": res.get("passed_count"),
        "total": res.get("total"),
        "receipt_id": res.get("receipt_id"),
        "receipt_sha256": res.get("receipt_sha256"),
        "isolation": "shared",
        "bundle_sha256": hashlib.sha256(bundle.encode()).hexdigest(),
        "tests_sha256": hashlib.sha256(tests_path.read_bytes()).hexdigest(),
        "source_function_sha256": hashes,
        "failures": None if res.get("passed") else (
            res.get("failures") or res.get("load_error") or res.get("reason") or res.get("tests_output_tail")
            or res.get("stderr") or res.get("stdout")
        ),
    }


def main(argv: list[str]) -> int:
    suites = argv or list(SUITES)
    if not HARNESS_EXE.is_file():
        print(f"jovin-harness not found at {HARNESS_EXE}; nothing was run.", file=sys.stderr)
        return 2
    initial = _wsl_state()
    results = []
    try:
        ret, out, err = _harness(["doctor"])
        doctor = json.loads(out) if ret == 0 else {}
        if not doctor.get("ready"):
            print(f"JoVin sandbox not ready; nothing was run: {doctor.get('problems') or err}", file=sys.stderr)
            return 2
        with tempfile.TemporaryDirectory(prefix="jovin_cmbcst_") as tmp:
            for suite in suites:
                t0 = time.perf_counter()
                r = run_suite(suite, Path(tmp))
                r["duration_s"] = round(time.perf_counter() - t0, 2)
                results.append(r)
                status = "PASS" if r["passed"] else "FAIL"
                print(f"[{status}] {suite}: {r['passed_count']}/{r['total']}  receipt={r['receipt_id']}")
                if not r["passed"]:
                    print(json.dumps(r["failures"], indent=2)[:4000])
        ret, out, _ = _harness(["receipts", "verify"])
        chain = json.loads(out) if ret == 0 else {"valid": False}
        print(f"Receipt chain valid={chain.get('valid')} count={chain.get('count')} head={chain.get('head_sha256')}")
        for r in results:
            r["chain_valid"] = bool(chain.get("valid"))
            r["chain_head_sha256"] = chain.get("head_sha256")
    finally:
        # Leave the distro as we found it (never prune images)
        if initial == "Stopped":
            subprocess.run(["wsl.exe", "--terminate", WSL_DISTRO], capture_output=True, check=False)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    history = json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.is_file() else []
    history.extend(results)
    LEDGER.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    return 0 if results and all(r["passed"] and r.get("chain_valid") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
