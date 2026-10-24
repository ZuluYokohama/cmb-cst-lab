"""Record independent actual GPU outputs before real-data plan freeze."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cmbcst.real_splits import cross_statistics,bootstrap_tiles


def main():
    test = ROOT/'tests/sandbox/jovin_tests_real_splits.py'
    source = test.read_text()
    if 'GPU_ATTESTATION = None' not in source:
        raise FileExistsError('preserve prior GPU fixture')
    rng = np.random.default_rng(920053)
    a,b = rng.normal(size=(2,1024))+1j*rng.normal(size=(2,1024))
    features = rng.normal(size=(4,4,5,4,8))
    indices = rng.integers(0,5,size=(2,5))
    fixture = dict(seed=920053,devices=[])
    for index in (0,1):
        device = f'cuda:{index}'
        actual = cross_statistics(a,b,device)
        boot = bootstrap_tiles(features,indices,device)
        errors = [np.max(np.abs(actual-cross_statistics(a,b))/(1+np.abs(cross_statistics(a,b)))),
                  np.max(np.abs(boot-bootstrap_tiles(features,indices))/(1+np.abs(bootstrap_tiles(features,indices))))]
        if max(errors)>1e-10:
            raise ValueError('actual GPU fixture precision failure')
        fixture['devices'].append(dict(device=device,name=torch.cuda.get_device_name(index),statistics=actual.tolist(),bootstrap=boot.tolist(),maximum_scaled_error=float(max(errors))))
    test.write_text(source.replace('GPU_ATTESTATION = None','GPU_ATTESTATION = '+repr(fixture)),encoding='utf-8')
    record = dict(fixture=fixture,torch=str(torch.__version__),cuda=torch.version.cuda,numpy=np.__version__,
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('src/cmbcst/real_splits.py','scripts/prepare_real_splits.py')})
    with (ROOT/'data/evidence/real_splits_cuda_v1.json').open('x') as stream:
        json.dump(record,stream,indent=2)
    print(json.dumps(dict(status='passed',devices=[d['name'] for d in fixture['devices']],errors=[d['maximum_scaled_error'] for d in fixture['devices']]),indent=2))


if __name__=='__main__':
    main()
