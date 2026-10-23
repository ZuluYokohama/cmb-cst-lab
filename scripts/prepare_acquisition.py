"""Validate actual GPU outputs and choose batches from measured memory headroom."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cmbcst.acquisition import AcquisitionCUDA,batch_reference


def main():
    path = ROOT/'tests/sandbox/jovin_tests_acquisition.py'
    source = path.read_text()
    if 'GPU_ATTESTATION = None' not in source:
        raise FileExistsError('preserve GPU development record')
    rng = np.random.default_rng(625983)
    sky,noise = rng.normal(size=(2,3,32,32)),rng.normal(size=(2,8,32,32))
    expected = batch_reference(sky,noise)
    fixture,profiles = dict(seed=625983,devices=[]),[]
    for index in (0,1):
        device = f'cuda:{index}'
        with torch.cuda.device(index):
            model = AcquisitionCUDA(32,device)
            actual = model.run(torch.as_tensor(sky,device=device),torch.as_tensor(noise,device=device))
            error = float(np.max(np.abs(actual['mse']-expected['mse'])/(1+np.abs(expected['mse']))))
            if error>1e-10 or actual['roll_amplitude_error']>1e-10:
                raise ValueError('GPU development precision failure')
            fixture['devices'].append(dict(device=device,name=torch.cuda.get_device_name(index),mse=actual['mse'].tolist(),
                roll_amplitude_error=actual['roll_amplitude_error'],scaled_cpu_error=error))
            del model
            torch.cuda.empty_cache()
            free,total = torch.cuda.mem_get_info(index)
            candidates = []
            model = AcquisitionCUDA(512,device)
            for batch in (8,16,32,64,128):
                # Conservative peak bound before allocation; avoid probing likely OOM sizes.
                estimate = batch*512**2*8*48+400_000_000
                if estimate>.75*free:
                    continue
                torch.cuda.reset_peak_memory_stats(index)
                generator = torch.Generator(device=device).manual_seed(626000+batch)
                white = torch.randn((batch,3,512,512),dtype=torch.float64,device=device,generator=generator)
                noise_t = torch.randn((batch,8,512,512),dtype=torch.float64,device=device,generator=generator)
                start = time.perf_counter()
                actual = model.run(white,noise_t)
                torch.cuda.synchronize(index)
                peak = torch.cuda.max_memory_allocated(index)
                candidates.append(dict(batch=batch,peak_allocated_bytes=peak,seconds=time.perf_counter()-start,
                    finite=bool(np.isfinite(actual['mse']).all()),within_headroom=peak<=.75*free))
                del white,noise_t
            legal = [row for row in candidates if row['finite'] and row['within_headroom']]
            if not legal:
                raise ValueError('no valid resident batch')
            chosen = max(legal,key=lambda row:row['batch'])
            profiles.append(dict(device=device,name=torch.cuda.get_device_name(index),free_bytes=free,total_bytes=total,
                batch=chosen['batch'],candidates=candidates))
            del model
            torch.cuda.empty_cache()
    path.write_text(source.replace('GPU_ATTESTATION = None','GPU_ATTESTATION = '+repr(fixture)),encoding='utf-8')
    record = dict(torch=str(torch.__version__),numpy=np.__version__,fixture=fixture,profiles=profiles,
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ('src/cmbcst/acquisition.py','scripts/prepare_acquisition.py')})
    with (ROOT/'data/evidence/acquisition_gpu_preflight.json').open('x') as stream:
        json.dump(record,stream,indent=2)
    print(json.dumps(profiles,indent=2))


if __name__=='__main__':
    main()
