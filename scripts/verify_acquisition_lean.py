"""Kernel-check the two exact basis identities with the pinned compiler."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
ROOT = Path(__file__).resolve().parents[1]
PINNED_COMPILER_SHA256 = 'bf8d54e4714cc4b03d3f6bb34c83b7202b87e49c8bfcbff6895c085bb90ceb38'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lean',type=Path,default=Path('/opt/cmb-lean-4.35.0/bin/lean'))
    parser.add_argument('--archive',type=Path,default=Path('/opt/cmb-lean-downloads/lean-4.35.0-linux.tar.zst'))
    parser.add_argument('--output',type=Path,default=ROOT/'data/evidence/acquisition_lean.json')
    args = parser.parse_args()
    compiler,archive = args.lean,args.archive
    digest = lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    source = ROOT/'formal/PolarizationGeometry.lean'
    expected = '7289597a44543246c9227b8cae501767e095ce17f6d448b95fc4a8dc97268f6d'
    if digest(archive)!=expected:
        raise ValueError('compiler archive mismatch')
    if digest(compiler)!=PINNED_COMPILER_SHA256:
        raise ValueError('compiler binary mismatch')
    text = source.read_text()
    if re.search(r'\b(sorry|admit)\b|^\s*(axiom|opaque)\s',text,re.MULTILINE):
        raise ValueError('proof source contains an admission or an unproved declaration')
    ret = subprocess.run([str(compiler),str(source)],cwd=ROOT,text=True,capture_output=True)
    passed = ret.returncode==0 and 'sorryAx' not in ret.stdout+ret.stderr
    record = dict(status='passed' if passed else 'failed',exit_code=ret.returncode,
        source_sha256=digest(source),compiler_sha256=digest(compiler),archive_sha256=expected,
        version=subprocess.check_output([str(compiler),'--version'],text=True).strip(),
        stdout=ret.stdout,stderr=ret.stderr,theorem_count=2,
        scope='Exact rational polarization basis identities under c*c+s*s=1; not empirical or floating-point correctness')
    with args.output.open('x') as stream:
        json.dump(record,stream,indent=2)
    print(record['status'])
    if not passed:
        raise SystemExit(ret.returncode or 1)


if __name__=='__main__':
    main()
