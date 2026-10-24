"""Separate CPU regeneration of three predeclared indices in every sample role."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cmbcst.response_sky import SphericalResponseModel
from cmbcst.response import extract_response_reference
from cmbcst.copula_topology import normal_scores

OUT = ROOT/'results/response_v1'
sha = lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p:json.loads(Path(p).read_text())


def error(a,b):
    a,b = np.asarray(a),np.asarray(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('replay nonfinite/shape failure')
    return float(np.max(np.abs(a-b)/(1+np.abs(b))))


def main():
    if (OUT/'replay.json').exists():
        raise FileExistsError('preserve completed replay')
    start = time.perf_counter()
    plan,result,conditioning,ensemble = [read(OUT/name) for name in ('plan.json','result.json','conditioning.json','ensemble.json')]
    if any(sha(ROOT/name)!=digest for name,digest in plan['source_sha256'].items()):
        raise ValueError('frozen source changed')
    cache = ROOT/'data/work/response_v1/examples.npz'
    if sha(cache)!=result['example_cache_sha256']:
        raise ValueError('example cache changed')
    sky = SphericalResponseModel(cl=np.asarray(conditioning['cl_TT_uK2']))
    records = []
    offset = 0
    with np.load(cache) as examples:
        for group in plan['groups']:
            for index in (0,17,group['draws']-1):
                seed = plan['base_seed']+group['seed_offset']+index
                raw,parameters,quality = sky.draw(seed,group['kind'],group['amplitude'])
                ranks = np.stack([normal_scores(field) for field in raw.reshape(-1,64,64)]).reshape(raw.shape)
                b,r = extract_response_reference(np.stack([raw,ranks],axis=2)[None])
                eb = error(b[0],ensemble['baseline_features'][offset+index])
                er = error(r[0],ensemble['response_features'][offset+index])
                saved = ensemble['draws'][offset+index]
                ep = error([parameters[key] for key in ('foreground_amplitude_uK','noise_amplitude_uK','original_local_signal_rms_uK','observed_base_std_uK')],
                    [saved[key] for key in ('foreground_amplitude_uK','noise_amplitude_uK','original_local_signal_rms_uK','observed_base_std_uK')])
                example_error = error(raw,examples[group['name']]) if index==0 else 0.
                passed = bool(quality['passed'] and max(eb,er,ep,example_error)<=1e-10 and saved['seed']==seed)
                records.append({'group':group['name'],'index':index,'seed':seed,'passed':passed,
                    'baseline_error':eb,'response_error':er,'parameter_error':ep,'example_error':example_error})

            offset += group['draws']
    audit = {'status':'passed' if len(records)==30 and all(r['passed'] for r in records) else 'failed',
        'created_utc':datetime.now(timezone.utc).isoformat(),'seconds':time.perf_counter()-start,
        'scope':'Separate CPU regeneration using frozen forward/feature kernels and stored fiducial C_ell; not an independent physics simulator',
        'records':records,'source_sha256':sha(Path(__file__)),
        'plan_sha256':sha(OUT/'plan.json'),'ensemble_sha256':sha(OUT/'ensemble.json')}
    (OUT/'replay.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({'status':audit['status'],'replayed':len(records),'seconds':audit['seconds']}))
    if audit['status']!='passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
