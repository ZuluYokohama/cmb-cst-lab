"""Kernel-check the two exact basis identities with the pinned compiler."""
import hashlib
import json
from pathlib import Path
import subprocess
ROOT = Path(__file__).resolve().parents[1]


def main():
    compiler = Path('/opt/cmb-lean-4.35.0/bin/lean')
    archive = Path('/opt/cmb-lean-downloads/lean-4.35.0-linux.tar.zst')
    digest = lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    source = ROOT/'formal/PolarizationGeometry.lean'
    expected = '7289597a44543246c9227b8cae501767e095ce17f6d448b95fc4a8dc97268f6d'
    if digest(archive)!=expected:
        raise ValueError('compiler archive mismatch')
    ret = subprocess.run([str(compiler),str(source)],cwd=ROOT,text=True,capture_output=True)
    record = dict(status='passed' if ret.returncode==0 else 'failed',exit_code=ret.returncode,
        source_sha256=digest(source),compiler_sha256=digest(compiler),archive_sha256=expected,
        version=subprocess.check_output([str(compiler),'--version'],text=True).strip(),
        stdout=ret.stdout,stderr=ret.stderr,theorem_count=2,
        scope='Exact rational polarization basis identities under c*c+s*s=1; not empirical or floating-point correctness')
    with (ROOT/'data/evidence/acquisition_lean.json').open('x') as stream:
        json.dump(record,stream,indent=2)
    print(record['status'])
    if ret.returncode:
        raise SystemExit(ret.returncode)


if __name__=='__main__':
    main()
