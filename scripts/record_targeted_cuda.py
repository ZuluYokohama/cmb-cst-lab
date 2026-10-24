"""Actual development CUDA score fixture, before scientific freezing."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cmbcst import response, targeted


def main():
    path = ROOT/'tests/sandbox/jovin_tests_targeted.py'
    source = path.read_text()
    if 'GPU_ATTESTATION = None' not in source:
        raise FileExistsError('preserve recorded fixture')
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if not isinstance(n,ast.ImportFrom)]
    ns = {**vars(response),**vars(targeted)}
    exec(compile(tree,str(path),'exec'),ns)
    b,r,labels = ns['toy_training']()
    values = targeted.score_targeted_cuda(b[:8],r[:8],targeted.fit_targeted(b,r,labels)).tolist()
    path.write_text(source.replace('GPU_ATTESTATION = None','GPU_ATTESTATION = '+repr(values)),encoding='utf-8')
    record = {'device':torch.cuda.get_device_name(0),'torch':str(torch.__version__),'numpy':np.__version__,
        'scores':values,'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
        ('src/cmbcst/targeted.py','scripts/record_targeted_cuda.py')}}
    with (ROOT/'data/evidence/targeted_cuda_development.json').open('x') as stream:
        json.dump(record,stream,indent=2)
    print(record['device'])


if __name__=='__main__':
    main()
