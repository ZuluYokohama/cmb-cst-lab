"""Compile the exact rational feature invariants with the pinned Lean kernel."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
RELEASE_SHA256 = '7289597a44543246c9227b8cae501767e095ce17f6d448b95fc4a8dc97268f6d'
RELEASE_URL = 'https://github.com/leanprover/lean4/releases/download/v4.35.0/lean-4.35.0-linux.tar.zst'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lean',required=True)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'results/multiscale_v1/lean_verification.json')
    args = parser.parse_args()
    if sha(args.archive) != RELEASE_SHA256:
        raise ValueError('compiler archive SHA256 differs from the pinned official release')
    lean = Path(shutil.which(args.lean) or args.lean).resolve()
    archived_binary = subprocess.check_output(['tar','--zstd','-xOf',str(args.archive),
        'lean-4.35.0-linux/bin/lean'])
    compiler_sha256 = sha(lean)
    if compiler_sha256 != hashlib.sha256(archived_binary).hexdigest():
        raise ValueError('Lean executable differs from the binary in the pinned archive')
    source = ROOT/'formal/ScaleInvariance.lean'
    text = source.read_text()
    if re.search(r'\b(sorry|admit)\b|^\s*(axiom|opaque)\s',text,re.MULTILINE):
        raise ValueError('proof source contains an admission or an unproved declaration')
    names = re.findall(r'^theorem\s+(\w+)',text,re.MULTILINE)
    if len(names) != 9:
        raise ValueError('expected all nine supporting invariants')
    version = subprocess.check_output([str(lean),'--version'],text=True).strip()
    if 'version 4.35.0' not in version or 'eb6cafb7290e8bd5f2937522d8d804f58f02bafc' not in version:
        raise ValueError('Lean executable version/commit differs from the pinned compiler')
    start = time.perf_counter()
    # Query every theorem without changing the preserved proof source.
    queries = '\n'.join(f'#print axioms CmbCst.{name}' for name in names)
    result = subprocess.run([str(lean),'--stdin'],input=text+'\n'+queries+'\n',
        cwd=ROOT,capture_output=True,text=True,timeout=120)
    assumptions = {}
    for name in names:
        matches = re.findall(rf"^'CmbCst\.{re.escape(name)}' "
            r'(?:depends on axioms: \[([^\]]*)\]|does not depend on any axioms)\s*$',
            result.stdout,re.MULTILINE)
        if matches:
            assumptions[name] = sorted({value.strip() for values in matches
                for value in values.split(',') if value.strip()})
    allowed = {'propext','Classical.choice','Quot.sound'}
    passed = (result.returncode == 0 and 'sorryAx' not in result.stdout+result.stderr
        and len(assumptions) == len(names)
        and all(set(values) <= allowed for values in assumptions.values()))
    evidence = {'verified_utc':datetime.now(timezone.utc).isoformat(),'status':'passed' if passed else 'failed',
        'compiler':version,'compiler_sha256':compiler_sha256,
        'compiler_archive_url':RELEASE_URL,'compiler_archive_sha256':sha(args.archive),
        'source_sha256':sha(source),'numerical_module_sha256':sha(ROOT/'src/cmbcst/multiscale.py'),
        'theorems':names,'theorem_count':len(names),'theorem_axioms':assumptions,'seconds':time.perf_counter()-start,
        'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr,
        'scope':'Exact rational algebra and finite-sample normalized moment invariance; not a physical discovery, a proof of Float64 implementation for all inputs, fractal dimension, or SHA256 collision resistance.',
        'foundation_axioms':['propext','Classical.choice','Quot.sound']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({'status':evidence['status'],'theorems':len(names),'compiler':version,'seconds':evidence['seconds']},indent=2))
    if not passed:
        print(result.stdout+result.stderr)
        raise SystemExit(1)


if __name__=='__main__':
    main()
