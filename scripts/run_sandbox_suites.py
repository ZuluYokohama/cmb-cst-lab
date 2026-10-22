"""Run the JoVin sandbox suites under plain pytest, without the JoVin harness (e.g. in CI).

Each suite is bundled exactly as scripts/jovin_verify.py bundles it (shipped functions extracted
with `ast` into one `solution.py`), then its tests run against that bundle. This re-runs the same
checks anywhere; it does not produce JoVin receipts, which only the sandbox issues.

Usage:
    python scripts/run_sandbox_suites.py [suite ...]     # default: all suites
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from jovin_verify import SUITES, build_bundle  # noqa: E402


def main(argv: list[str]) -> int:
    suites = argv or list(SUITES)
    failed = []
    for suite in suites:
        bundle, _ = build_bundle(suite)
        with tempfile.TemporaryDirectory(prefix=f"sbx_{suite}_") as tmp:
            (Path(tmp) / "solution.py").write_text(bundle, encoding="utf-8")
            tests = Path(tmp) / Path(SUITES[suite][1]).name
            shutil.copy(REPO / SUITES[suite][1], tests)
            ret = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tests)],
                                 cwd=tmp, check=False).returncode
        print(f"[{'PASS' if ret == 0 else 'FAIL'}] {suite}", flush=True)
        if ret:
            failed.append(suite)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
