"""Bounded exact-grid symmetry control; no new sky hypothesis or p-value scan."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cmbcst import orientation as o, multiscale as m, persistence as p
from cmbcst.copula_topology import normal_scores
from cmbcst.surrogate_topology import pd0_summary
from jovin_verify import SUITES, build_bundle
from run_spherical_topology import sha, save, now

OUT = ROOT/'results/orientation_v1'
PATCH_SOURCE = ROOT/'data/work/spherical_topology_v1/observed_patches.npz'


def scaled_error(a,b):
    a,b = np.asarray(a),np.asarray(b)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError('non-finite audit observable')
    return float(np.max(np.abs(a-b)/(1+np.abs(b))))


def main():
    import torch
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'plan.json').exists():
        raise FileExistsError('preserving exact-grid audit; no duplicate run')
    receipts = json.loads((ROOT/'data/evidence/jovin_receipts.json').read_text())
    receipt = [r for r in receipts if r['suite']=='orientation'][-1]
    if (not receipt['passed'] or not receipt['receipt_verified'] or not receipt['chain_valid']
        or receipt['bundle_sha256']!=hashlib.sha256(build_bundle('orientation')[0].encode()).hexdigest()
        or receipt['tests_sha256']!=sha(ROOT/SUITES['orientation'][1])):
        raise ValueError('orientation receipt differs from shipped code/tests')
    sources = [Path(__file__),ROOT/'docs/ORIENTATION_SPEC.md'] + [ROOT/'src/cmbcst'/name for name in
        ('orientation.py','multiscale.py','persistence.py','surrogate_topology.py','copula_topology.py')]
    plan = {'created_utc':now(),'scope':'Exact D4 representation/axis-labeling control; not an SO(3) or sky discovery experiment',
        'source_sha256':{str(path.relative_to(ROOT)).replace('\\','/'):sha(path) for path in sources},
        'patch_source_sha256':sha(PATCH_SOURCE),'gaussian_seeds':list(range(710395,710403)),
        'spike_seed':710410,'shape':[128,128],'representations':['raw','standardized','normal_scores'],
        'quarter_turns':[0,1,2,3],'reflections':[False,True],'scaled_tolerance':1e-10,
        'jovin_receipt_id':receipt['receipt_id'],'jovin_bundle_sha256':receipt['bundle_sha256'],
        'device':torch.cuda.get_device_name(0),'torch':str(torch.__version__),'numpy':np.__version__}
    with (OUT/'plan.json').open('x') as stream:
        json.dump(plan,stream,indent=2)
    start = time.perf_counter()
    with np.load(PATCH_SOURCE) as patches:
        cases = {name:patches[name].copy() for name in ('NGP','SGP','ColdSpot')}
    for i,seed in enumerate(plan['gaussian_seeds']):
        cases[f'Gaussian{i}'] = np.random.default_rng(seed).normal(size=(128,128))
    row,col = np.mgrid[:128,:128]
    cases['ramp'] = (row+2*col).astype(float)
    spike = np.random.default_rng(plan['spike_seed']).normal(size=(128,128))
    spike[64,64] += 12
    cases['spike'] = spike
    records = []
    for name,raw in cases.items():
        raw_features = m.multiscale_reference(raw)[1]
        raw_count = pd0_summary(raw)[0]
        reps = {'raw':raw,'standardized':p.standardise(raw),'normal_scores':normal_scores(raw)}
        for representation,field in reps.items():
            base_curves,base_features = m.multiscale_reference(field)
            base_topology = pd0_summary(field)
            base_ranks = normal_scores(field)
            for k in range(4):
                for reflected in (False,True):
                    view = o.d4_view(field,k,reflected)
                    curves,features = m.multiscale_reference(view)
                    gpu_curves,gpu_features = m.multiscale_cuda(view)
                    expected_c,expected_f = base_curves.copy(),base_features.copy()
                    expected_c[:,:,5] *= o.directional_sign(k)
                    expected_f[:,5] *= o.directional_sign(k)
                    topology = pd0_summary(view)
                    errors = {'cpu_gpu_curves':scaled_error(gpu_curves,curves),
                        'cpu_gpu_features':scaled_error(gpu_features,features),
                        'orientation_curves':scaled_error(curves,expected_c),
                        'orientation_features':scaled_error(features,expected_f),
                        'topology_lifetime':scaled_error(topology[1:],base_topology[1:]),
                        'rank_commutation':scaled_error(normal_scores(view),o.d4_view(base_ranks,k,reflected))}
                    if representation=='standardized':
                        errors['affine_feature_invariance'] = scaled_error(features,raw_features*np.array([1,1,1,1,1,o.directional_sign(k)]))
                    exact = {'histogram_preserved':bool(np.array_equal(np.sort(view.ravel()),np.sort(field.ravel()))),
                        'topology_count_preserved':bool(topology[0]==base_topology[0]==raw_count)}
                    passed = all(exact.values()) and all(v<=1e-10 for v in errors.values())
                    records.append({'case':name,'representation':representation,'quarter_turns':k,'reflected':reflected,
                        'passed':bool(passed),'errors':errors,'exact_checks':exact,'features':features[0].tolist(),
                        'expected_features':expected_f[0].tolist(),'finite_PD0':topology.tolist()})
        print(f'finished orientation case {name}',flush=True)
    errors = {key:max(r['errors'].get(key,0.) for r in records) for key in records[8]['errors']}
    errors['affine_feature_invariance'] = max(r['errors'].get('affine_feature_invariance',0.) for r in records)
    result = {'status':'passed' if all(r['passed'] for r in records) else 'failed','completed_utc':now(),
        'seconds':time.perf_counter()-start,'cases':len(cases),'transform_checks':len(records),
        'failures':sum(not r['passed'] for r in records),'maximum_scaled_errors':errors,
        'plan_sha256':sha(OUT/'plan.json'),'records':records,
        'accepted_cosmological_claims':[],'limits':'Exact grid relabelings only. Representation changes are different observables. This does not repair foreground/noise null assumptions or establish continuous-rotation invariance.'}
    save(OUT/'result.json',result)
    print(json.dumps({key:result[key] for key in ('status','cases','transform_checks','failures','maximum_scaled_errors')},indent=2))
    if result['status']!='passed':
        raise SystemExit(1)


if __name__=='__main__':
    main()
