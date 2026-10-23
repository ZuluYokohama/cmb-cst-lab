"""Separate CPU regeneration at the first and last index of every sampled role."""
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cmbcst.targeted_sky import TargetedSky
from cmbcst import response,targeted
from run_response_benchmark import represent,sha,save,scaled_error,now
OUT = ROOT/'results/targeted_v1'


def main():
    if (OUT/'replay.json').exists():
        raise FileExistsError('preserve completed replay')
    read = lambda name:json.loads((OUT/name).read_text())
    plan,lock,conditioning = map(read,('plan.json','selection_lock.json','conditioning.json'))
    if any(sha(ROOT/name)!=digest for name,digest in plan['source_sha256'].items()):
        raise ValueError('frozen source changed')
    start = time.perf_counter()
    sky = TargetedSky(cl=np.asarray(conditioning['cl_TT_uK2']))
    records = []
    for name,roles in (('development',plan['development_roles']),('final',lock['final_roles'])):
        data = read(name+'.json')
        offset = 0
        for role in roles:
            for index in (0,role['count']-1):
                fields,p,q = sky.draw_recipe(plan['base_seed']+role['offset']+index,role['recipe'])
                b,r = response.extract_response_reference(represent(fields[None]))
                errors = [scaled_error(b[0],data['baseline'][offset+index]),
                    scaled_error(targeted.six_responses(r)[0],data['response'][offset+index])]
                records.append(dict(phase=name,role=role['name'],index=index,seed=p['seed'],errors=errors,
                    signal_passed=q['passed'],passed=bool(q['passed'] and max(errors)<=1e-10)))
            offset += role['count']
    audit = dict(status='passed' if all(r['passed'] for r in records) else 'failed',created_utc=now(),
        records=records,seconds=time.perf_counter()-start,source_sha256=sha(Path(__file__)),
        plan_sha256=sha(OUT/'plan.json'),lock_sha256=sha(OUT/'selection_lock.json'),
        scope='Separate execution of frozen CPU kernels, not an independent physical simulator')
    save(OUT/'replay.json',audit)
    print(json.dumps(dict(status=audit['status'],samples=len(records),seconds=audit['seconds'])))
    if audit['status']!='passed':
        raise SystemExit(1)


if __name__=='__main__':
    main()
