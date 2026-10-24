"""Record an actual CUDA development fixture before the scientific pilot freezes."""
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cmbcst.response import extract_response_cuda


def main():
    path = ROOT/'tests/sandbox/jovin_tests_response.py'
    source = path.read_text()
    if 'GPU_ATTESTATION = None' not in source:
        raise FileExistsError('preserve the actual recorded fixture')
    fields = np.random.default_rng(551940).normal(size=(2,3,4,2,32,33))
    b,r = extract_response_cuda(fields)
    fixture = {'seed':551940,'baseline':b.tolist(),'response':r.tolist()}
    path.write_text(source.replace('GPU_ATTESTATION = None','GPU_ATTESTATION = '+repr(fixture)),encoding='utf-8')
    digest = lambda file:hashlib.sha256(file.read_bytes()).hexdigest()
    record = {'created_utc':datetime.now(timezone.utc).isoformat(),'device':torch.cuda.get_device_name(0),
        'torch':str(torch.__version__),'cuda':torch.version.cuda,'numpy':np.__version__,
        'fixture':fixture,'tests_sha256':digest(path),
        'source_sha256':{name:digest(ROOT/name) for name in
            ('scripts/record_response_cuda.py','src/cmbcst/multiscale.py','src/cmbcst/response.py')}}
    target = ROOT/'data/evidence/response_cuda_development.json'
    with target.open('x',encoding='utf-8') as stream:
        json.dump(record,stream,indent=2,allow_nan=False)
    print(json.dumps({'device':record['device'],'baseline_shape':list(b.shape),'response_shape':list(r.shape)}))


if __name__ == '__main__':
    main()
