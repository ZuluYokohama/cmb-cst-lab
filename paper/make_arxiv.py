"""Package the paper for arXiv: main.tex, numbers.tex and figures/*.pdf -> paper/arxiv.tar.gz,
then compile the unpacked tarball in a clean temporary directory to prove it is self-contained.

Run paper/make_inputs.py first so numbers.tex and the figures match results/*.json.

Usage:
    python paper/make_arxiv.py [--tectonic data/work/tools/tectonic.exe]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

PAPER = Path(__file__).resolve().parent
REPO = PAPER.parent
FILES = ["main.tex", "numbers.tex"] + sorted(f"figures/{p.name}" for p in (PAPER / "figures").glob("*.pdf"))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tectonic", default=str(REPO / "data" / "work" / "tools" / "tectonic.exe"))
    args = ap.parse_args(argv)
    out = PAPER / "arxiv.tar.gz"
    with tarfile.open(out, "w:gz") as tar:
        for name in FILES:
            # fixed metadata so the archive depends only on the file contents
            info = tar.gettarinfo(PAPER / name, arcname=name)
            info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
            with open(PAPER / name, "rb") as f:
                tar.addfile(info, f)
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(out) as tar:
            tar.extractall(tmp, filter="data")
        run = subprocess.run([args.tectonic, "-X", "compile", "main.tex"], cwd=tmp, capture_output=True, text=True)
        if run.returncode != 0 or not (Path(tmp) / "main.pdf").exists():
            print(run.stderr[-2000:], file=sys.stderr)
            raise SystemExit("arxiv.tar.gz does not compile on its own")
        pdf_size = (Path(tmp) / "main.pdf").stat().st_size
    print(f"wrote {out.relative_to(REPO)} ({out.stat().st_size / 1024:.0f} KB, {len(FILES)} files); "
          f"clean compile OK ({pdf_size / 1024:.0f} KB PDF)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
