"""Frozen, independently cross-checked CPU/CUDA finite-scale sky experiment."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import run_spherical_topology as sphere
from cmbcst import multiscale as m, spherical_surrogates as ss, spectra, theory, persistence as p
from jovin_verify import build_bundle, SUITES

OUT = ROOT/'results/multiscale_v1'
WORK = ROOT/'data/work/multiscale_v1'
SOURCE = ROOT/'results/spherical_topology_v1'
SOURCE_WORK = ROOT/'data/work/spherical_topology_v1'


def freeze():
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    if (OUT/'plan.json').exists():
        raise FileExistsError('preserving frozen run; no duplicate launch')
    receipt = [r for r in json.loads((ROOT/'data/evidence/jovin_receipts.json').read_text()) if r['suite']=='multiscale'][-1]
    if (not receipt['passed'] or not receipt['receipt_verified'] or not receipt['chain_valid']
        or receipt['bundle_sha256'] != hashlib.sha256(build_bundle('multiscale')[0].encode()).hexdigest()
        or receipt['tests_sha256'] != sphere.sha(ROOT/SUITES['multiscale'][1])):
        raise ValueError('current multiscale source/tests lack a matching authentic receipt')
    previous = json.loads((SOURCE/'plan.json').read_text())
    for path,digest in previous['source_sha256'].items():
        if sphere.sha(ROOT/path) != digest:
            raise ValueError(f'old frozen source changed: {path}')
    paths = [Path(__file__), ROOT/'src/cmbcst/multiscale.py', ROOT/'docs/MULTISCALE_SPEC.md']
    plan = {'created_utc':sphere.now(), 'seed':sphere.SEED, 'reference_skies':999,'control_skies':64,
        'batch_size':8,'lags':m.LAGS,'features':m.FEATURE_NAMES,'holm_family_size':18,
        'candidate_p':.05,'max_control_flags':8,'scaled_cpu_gpu_tolerance':1e-10,
        'source_sha256':{**previous['source_sha256'], **{str(path.relative_to(ROOT)).replace('\\','/'):sphere.sha(path) for path in paths}},
        'source_artifacts_sha256':{str(path.relative_to(ROOT)).replace('\\','/'):sphere.sha(path) for path in
          (SOURCE/'plan.json',SOURCE/'conditioning.json',SOURCE/'result.json',SOURCE_WORK/'conditional_cl.npy',SOURCE_WORK/'observed_patches.npz')},
        'jovin_receipt_id':receipt['receipt_id'],'jovin_bundle_sha256':receipt['bundle_sha256'],
        'numpy':np.__version__,'torch':str(torch.__version__),'cuda':torch.version.cuda,
        'device':torch.cuda.get_device_name(0),'scope':'Exploratory finite-scale diagnostics on previously examined uncleaned sky; no novelty, fractal-dimension or cosmological claim'}
    with (OUT/'plan.json').open('x') as stream:
        json.dump(plan, stream, indent=2)
    return plan


def patches_of(alm):
    import ducc0
    ell,_ = spectra.alm_ell_m(1000)
    sky = ducc0.sht.experimental.synthesis(alm=(alm*theory.gaussian_beam(10.,1000)[ell])[None],
        spin=0,lmax=1000,mmax=1000,nthreads=8,**ducc0.healpix.Healpix_Base(512,'RING').sht_info())[0]
    return np.stack([p.project_patch(sky,*center) for center in sphere.PATCHES.values()])


def measured(patches):
    start = time.perf_counter()
    curves,features = m.multiscale_reference(patches)
    cpu_s = time.perf_counter()-start
    start = time.perf_counter()
    gpu_curves,gpu_features = m.multiscale_cuda(patches)
    gpu_s = time.perf_counter()-start
    errors = {'curve_scaled_error':float(np.max(np.abs(curves-gpu_curves)/(1+np.abs(curves)))),
        'feature_scaled_error':float(np.max(np.abs(features-gpu_features)/(1+np.abs(features)))),
        'finite':bool(np.all(np.isfinite(gpu_curves)) and np.all(np.isfinite(gpu_features))),
        'cpu_seconds':cpu_s,'gpu_seconds_including_transfer':gpu_s}
    errors['passed'] = bool(errors['finite'] and errors['curve_scaled_error'] <= 1e-10 and errors['feature_scaled_error'] <= 1e-10)
    if not errors['passed']:
        np.savez(WORK/'failed_parity_batch.npz',patches=patches,cpu_curves=curves,gpu_curves=gpu_curves)
        sphere.save(OUT/'parity_failure.json',errors)
        raise ValueError('CPU/GPU parity or finite-value failure')
    return gpu_curves,gpu_features,errors


def compare(observed,references):
    raw = np.array([[p.mc_pvalue(references[:,row,col],observed[row,col]) for col in range(6)] for row in range(3)])
    return raw,p.holm(raw.ravel()).reshape(3,6)


def run():
    plan = freeze()
    start = time.perf_counter()
    stage = {'status':'running','started_utc':sphere.now(),'plan_sha256':sphere.sha(OUT/'plan.json')}
    sphere.save(OUT/'stage_audit.json',stage)
    try:
        cl = np.load(SOURCE_WORK/'conditional_cl.npy')
        nalm = 1001*1002//2
        features,curves,quality = [],[],[]
        for first in range(0,1063,8):
            ids = list(range(first,min(first+8,1063)))
            normals = np.stack([np.random.default_rng(sphere.SEED+i).normal(size=(2,nalm)) for i in ids])
            alms = ss.fixed_power_cuda(cl,normals)
            powers = [sphere.quality(alm,cl) for alm in alms]
            if not all(q['passed'] for q in powers):
                np.save(WORK/f'failed_harmonics_{first}.npy',alms)
                sphere.save(OUT/'power_failure.json',powers)
                raise ValueError('harmonic power precision failure')
            fields = np.stack([patches_of(alm) for alm in alms])
            c,f,q = measured(fields.reshape(-1,128,128))
            curves.extend(c.reshape(len(ids),3,5,6))
            features.extend(f.reshape(len(ids),3,6))
            q.update(first_index=first,count=len(ids),max_shell_error=max(x['maximum_relative_shell_error'] for x in powers))
            quality.append(q)
            sphere.save(OUT/'progress.json',{'completed':len(features),'total':1063,'elapsed_s':time.perf_counter()-start})
            print(f'sky feature batches {len(features)}/1063',flush=True)
        values = np.asarray(features)
        reference = values[:999]
        sphere.save(OUT/'ensemble.json',{'features':values.tolist(),'curves':np.asarray(curves).tolist(),'batch_quality':quality,
            'reference_skies':999,'control_skies':64})
        records = []
        for i,observed in enumerate(values[999:]):
            raw,adjusted = compare(observed,reference)
            records.append({'index':i,'features':observed.tolist(),'raw_p':raw.tolist(),'p_holm':adjusted.tolist(),'flag':bool(np.min(adjusted)<=.05)})
        flags = sum(r['flag'] for r in records)
        validation = {'status':'passed' if flags<=8 else 'failed','fields':64,'familywise_flags':flags,'cutoff':8,'records':records}
        sphere.save(OUT/'validation.json',validation)
        coherent = ss.fixed_power_cuda(cl,np.ones((1,2,nalm)))[0]
        c,f,q = measured(patches_of(coherent))
        raw,adjusted = compare(f,reference)
        sphere.save(OUT/'positive_control.json',{'curves':c.tolist(),'features':f.tolist(),'raw_p':raw.tolist(),
            'p_holm':adjusted.tolist(),'quality':q,'scope':'Phase-coherent full sphere with same C_ell; sensitivity example'})
        result = {'status':'calibration_failed','completed_utc':sphere.now(),'candidate_flags':[],
            'patches':list(sphere.PATCHES),'features':m.FEATURE_NAMES,'plan_sha256':sphere.sha(OUT/'plan.json'),
            'ensemble_sha256':sphere.sha(OUT/'ensemble.json'),'validation_sha256':sphere.sha(OUT/'validation.json'),
            'accepted_cosmological_claims':[],'fractal_dimension_established':False,'universal_novelty_established':False}
        if validation['status']=='passed':
            with np.load(SOURCE_WORK/'observed_patches.npz') as fields:
                observed_fields = np.stack([fields[name] for name in sphere.PATCHES])
            c,f,q = measured(observed_fields)
            raw,adjusted = compare(f,reference)
            result.update(status='exploratory_complete',observed_curves=c.tolist(),observed=f.tolist(),
                raw_p=raw.tolist(),p_holm=adjusted.tolist(),observed_quality=q,
                reference_mean=reference.mean(0).tolist(),reference_sd=reference.std(0,ddof=1).tolist(),
                candidate_flags=[{'patch':name,'feature':m.FEATURE_NAMES[col],'p_holm':float(adjusted[row,col])}
                    for row,name in enumerate(sphere.PATCHES) for col in range(6) if adjusted[row,col]<=.05])
        sphere.save(OUT/'result.json',result)
        stage.update(status=result['status'],completed_utc=sphere.now(),elapsed_s=time.perf_counter()-start,
            control_flags=flags,precision_failures=0,max_curve_error=max(q['curve_scaled_error'] for q in quality),
            max_feature_error=max(q['feature_scaled_error'] for q in quality),max_shell_error=max(q['max_shell_error'] for q in quality),
            cpu_feature_seconds=sum(q['cpu_seconds'] for q in quality),gpu_feature_seconds=sum(q['gpu_seconds_including_transfer'] for q in quality),
            result_sha256=sphere.sha(OUT/'result.json'))
        sphere.save(OUT/'stage_audit.json',stage)
        print(f"finished {result['status']}; controls {flags}/64; feature flags {len(result['candidate_flags'])}",flush=True)
    except Exception:
        stage.update(status='execution_failed',completed_utc=sphere.now(),error=traceback.format_exc())
        sphere.save(OUT/'stage_audit.json',stage)
        raise


if __name__=='__main__':
    run()
