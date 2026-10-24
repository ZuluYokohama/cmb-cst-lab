"""Authenticate JoVin and independently reconstruct real-coordinate moments."""
import argparse
import ast
import csv
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from jovin_verify import SUITES,build_bundle,configuration,_harness
OUT=ROOT/'results/real_splits_v1'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def close(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return bool(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all() and np.max(np.abs(a-b)/(1+np.abs(b)))<=1e-10)


def covariance_oracle(a,b):
    """Real 4x4 covariance contraction, independent of complex Wick code."""
    x=np.column_stack([a.real,a.imag,b.real,b.imag])
    x-=np.mean(x,axis=0)
    cov=x.T@x/len(x)
    va,vb=np.trace(cov[:2,:2]),np.trace(cov[2:,2:])
    re,im=cov[0,2]+cov[1,3],cov[1,2]-cov[0,3]
    fourth=np.mean(np.sum(x[:,:2]**2,axis=1)*np.sum(x[:,2:]**2,axis=1))
    k=fourth-va*vb-2*np.sum(cov[:2,2:]**2)
    return [va,vb,re,(va+vb+2*re)/4,(va+vb-2*re)/4,re/np.sqrt(va*vb),k/(va*vb),im/np.sqrt(va*vb)]


def main():
    command,env=configuration(argparse.Namespace(harness=None,harness_python=None,runtime='wsl:JovinHarness',data_dir=None))
    ret,chain=_harness(command,env,['receipts','verify'])
    history=read(ROOT/'data/evidence/jovin_receipts.json')
    suites={}
    for name,(_,tests) in SUITES.items():
        record=[r for r in history if r['suite']==name][-1]
        sr,auth=_harness(command,env,['receipts','show',record['receipt_id']])
        mr,member=_harness(command,env,['receipts','verify',record['receipt_id']])
        match=(record['passed'] and record['receipt_verified'] and record['chain_valid'] and sr==mr==0
            and auth==record['receipt'] and member.get('valid') is True and member.get('chain')=='linked'
            and hashlib.sha256(build_bundle(name)[0].encode()).hexdigest()==record['bundle_sha256'] and sha(ROOT/tests)==record['tests_sha256'])
        suites[name]=dict(matched=bool(match),receipt_id=record['receipt_id'],total=record['total'])
    frozen={str(p.relative_to(ROOT)):all(sha(ROOT/name)==value for name,value in read(p).get('source_sha256',{}).items()) for p in (ROOT/'results').glob('*/plan.json')}
    prior={str(p.relative_to(ROOT)):all(sha(p.parent/name)==value for name,value in read(p).get('artifact_sha256',{}).items()) for p in (ROOT/'results').glob('*/verification.json') if p.parent!=OUT}
    plan,result,stage,tiles,inputs,replay=[read(OUT/name) for name in ('plan.json','result.json','stage_audit.json','tiles.json','inputs.json','sampling_replay.json')]
    with np.load(OUT/'geometry.npz',allow_pickle=False) as d:
        geometry={key:d[key] for key in d.files}
    g=geometry
    with np.load(OUT/'observations.npz',allow_pickle=False) as d:
        anchor,endpoint=d['anchor'],d['endpoint']
    with np.load(OUT/'tile_statistics.npz',allow_pickle=False) as d:
        features=d['features']
    with np.load(OUT/'resampling.npz',allow_pickle=False) as d:
        indices,boot=d['indices'],d['statistics']
    checks=dict(fixed_plan=plan['maps']==['100A','100B','143A','143B','217A','217B','353A','353B']
        and plan['lmax']==512 and plan['lmin']==30 and plan['common_beam_arcmin']==60
        and plan['lags_degrees']==[.5,1.,2.,4.] and plan['bearings_degrees']==[0.,45.,90.,135.]
        and plan['min_pairs_per_tile_lag']==128 and plan['bootstrap_draws']==2000 and plan['seed']==82024102026,
        complete=stage['status']==result['status']=='exploratory_measurement_complete' and stage['numerical_failures']==0 and 0<stage['elapsed_seconds']<3600,
        plan_result_bindings=stage['plan_sha256']==result['plan_sha256']==sha(OUT/'plan.json') and stage['result_sha256']==sha(OUT/'result.json'),
        inventory_binding=plan['input_inventory_sha256']==sha(ROOT/'data/evidence/planck_dataset.json') and plan['input_audit_sha256']==sha(ROOT/'data/evidence/planck_input_audit.json'),
        fixture_binding=plan['fixture_sha256']==sha(ROOT/'data/evidence/real_splits_cuda_v1.json'),
        receipt_binding=plan['jovin_receipt_id']==suites['real_splits']['receipt_id'],
        honest_claims=result['accepted_physical_claims']==[] and result['accepted_method_superiority_claims']==[])
    # Independently verify endpoints/separations and phase factors via tangent projections.
    loc=g['loc'];end=g['end']
    def frame(v):
        t,p=v[...,0],v[...,1]
        return (np.stack([np.sin(t)*np.cos(p),np.sin(t)*np.sin(p),np.cos(t)],-1),
            np.stack([np.cos(t)*np.cos(p),np.cos(t)*np.sin(p),-np.sin(t)],-1),
            np.stack([-np.sin(p),np.cos(p),np.zeros_like(p)],-1))
    n,s,e=frame(loc);n1,s1,e1=frame(end)
    separation=np.radians([.5,1,2,4])[:,None,None]
    tangent=(np.cos(separation)[...,None]*n1-n)/np.sin(separation)[...,None]
    z=np.sum(tangent*s1,axis=-1)+1j*np.sum(tangent*e1,axis=-1)
    checks['great_circle_separation']=close(np.sum(n1*n,axis=-1),np.broadcast_to(np.cos(separation),g['valid'].shape))
    checks['endpoint_phase_projection']=close(g['phase1'],np.conj(z/np.abs(z))**2)
    checks['origin_phase']=close(g['phase0'],np.broadcast_to(np.exp(-2j*np.radians([0,45,90,135]))[None,:,None],g['phase0'].shape))
    checks['map_shapes_finite']=(anchor.shape==(8,len(loc)) and endpoint.shape==(8,4,4,len(loc)) and np.isfinite(anchor).all() and np.isfinite(endpoint).all())
    views=np.stack([endpoint*g['phase1']-anchor[:,None,None]*g['phase0'],endpoint-anchor[:,None,None],np.abs(endpoint)-np.abs(anchor[:,None,None])])
    actual_tiles=[];oracle=[]
    for tile in np.unique(g['tile_ids']):
        take=g['valid']&(g['tile_ids']==tile)[None,None,:]
        counts=take.sum(axis=(1,2))
        accepted=bool(np.all(counts>=128))
        actual_tiles.append(dict(tile=int(tile),counts=counts.tolist(),accepted=accepted))
        if not accepted:
            continue
        block=np.empty((4,4,4,8))
        for lag in range(4):
            selected=views[:,:,lag][:,:,take[lag]]
            for mode in range(4):
                for freq in range(4):
                    a,b=selected[0 if mode==3 else mode,2*freq:2*freq+2]
                    if mode==3:
                        b=np.roll(b,len(b)//3)
                    block[mode,freq,lag]=covariance_oracle(a,b)
        oracle.append(block)
    oracle=np.stack(oracle,axis=2)
    checks['geometry_only_tile_selection']=actual_tiles==tiles['records'] and tiles['accepted']==len(oracle[0,0]) and tiles['valid_pairs']==int(g['valid'].sum()) and tiles['anchors']==len(loc)
    checks['independent_real_covariance_moments']=close(oracle,features)
    checks['all_features_finite']=np.isfinite(features).all() and features.dtype==np.float64
    checks['split_identity']=close(features[...,3]-features[...,4],features[...,2])
    checks['coherence_bound']=np.max(np.abs(features[...,5]))<=1+1e-10
    expected_ix=np.random.default_rng(82024102026).integers(0,features.shape[2],size=(2000,features.shape[2]))
    checks['resample_manifest']=np.array_equal(expected_ix,indices)
    expected_boot=np.array([oracle[:,:,row].mean(axis=2) for row in indices])
    checks['all_cpu_resamples']=close(boot,expected_boot)
    checks['result_means']=close(features.mean(axis=2),result['tile_equal_weight_mean'])
    checks['result_bands']=close(np.percentile(expected_boot,[2.5,97.5],axis=0),result['tile_resampling_bands'])
    for i in (0,1):
        with np.load(OUT/f'gpu{i}_statistics.npz') as d:
            checks[f'all_gpu_{i}_statistics']=close(d['features'],oracle)
        checks[f'gpu_{i}_precision_records']=result['devices'][i]['maximum_scaled_error']<=1e-10 and result['resampling_devices'][i]['maximum_scaled_error']<=1e-10 and result['resampling_devices'][i]['common32_cpu_error']<=1e-10
    inventory={v['key']:v for v in read(ROOT/'data/evidence/planck_dataset.json')['products']}
    checks['actual_input_identities']=[v['key'] for v in inputs['maps']]==plan['maps'] and all(v['sha256']==inventory[v['key']]['sha256'] and v['bytes']==inventory[v['key']]['bytes'] and v['zero_filled_invalid_pixels']>=0 for v in inputs['maps'])
    checks['mask_binding']=inputs['mask_sha256']==plan['mask_sha256']==sha(ROOT/'data/work/rtx5070_real/mask_30.npy')
    for item in inputs['maps']:
        checks[f"retained_harmonics_{item['key']}"]=sha(OUT/f"harmonics/alm_{item['key']}.npy")==item['filtered_alm_sha256'] and np.isfinite(item['diagonal_windows']).all()
    checks['sampling_replay']=replay['status']=='passed' and len(replay['records'])==8 and all(v['scaled_error']<=1e-10 for v in replay['records']) and replay['spin2_grid_direct_error']<=1e-10 and replay['source_sha256']==sha(ROOT/'scripts/replay_real_splits.py')
    checks['raw_map_reanalysis']=replay['raw_map_replay']['map']=='143A' and replay['raw_map_replay']['sha256']==inventory['143A']['sha256'] and 0<=replay['raw_map_replay']['scaled_error']<=1e-10
    fixture=read(ROOT/'data/evidence/real_splits_cuda_v1.json')
    tree=ast.parse((ROOT/'tests/sandbox/jovin_tests_real_splits.py').read_text())
    embedded=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='GPU_ATTESTATION' for t in n.targets))
    checks['actual_gpu_fixture']=embedded==fixture['fixture'] and len(embedded['devices'])==2 and all(sha(ROOT/name)==digest for name,digest in fixture['source_sha256'].items())
    rejected=read(ROOT/'data/evidence/real_splits_fixture_admission.json')
    checks['rejected_admission_preserved']=all(hashlib.sha256(gzip.decompress((ROOT/v['archive']).read_bytes())).hexdigest()==v['sha256'] for v in rejected['files'].values()) and any(v['suite']=='real_splits' and v['receipt_id'] is None and not v['passed'] for v in history)
    with (OUT/'labeled_real_samples.csv').open(newline='') as stream:
        rows=list(csv.DictReader(stream))
    rows_ok=len(rows)==16
    for row in rows:
        m=plan['maps'].index(row['map']);direction=int(float(row['bearing_deg'])/45)
        j=int(np.flatnonzero(g['anchor_pixel']==int(row['anchor_pixel_nside64_ring']))[0])
        values=[float(row[key]) for key in ('Q_anchor_uK','U_anchor_uK','Q_end_uK','U_end_uK','transported_delta_real_uK','transported_delta_imag_uK')]
        expected=[anchor[m,j].real,anchor[m,j].imag,endpoint[m,2,direction,j].real,endpoint[m,2,direction,j].imag,views[0,m,2,direction,j].real,views[0,m,2,direction,j].imag]
        rows_ok &= close(values,expected) and abs(float(row['galactic_latitude_deg'])-(90-np.degrees(loc[j,0])))<1e-12 and abs(float(row['galactic_longitude_deg'])-np.degrees(loc[j,1]))<1e-12 and float(row['lag_deg'])==2. and (row['valid_pair']=='True')==bool(g['valid'][2,direction,j])
    checks['labeled_real_rows']=bool(rows_ok)
    association=read(OUT/'association.json')
    checks['all_foreground_associations']=len(association)==16 and all(abs(v['rank_correlation_with_353_mean_increment_power']-spearmanr(oracle[0,i,:,l,6],oracle[0,3,:,l,3]).statistic)<1e-10 for v,(i,l) in zip(association,[(i,l) for i in range(4) for l in range(4)]))
    tests=ROOT/'data/real_splits_tests.log';sandbox=ROOT/'data/real_splits_sandbox_tests.log'
    checks['local_regression']='195 passed, 3 skipped' in tests.read_text()
    checks['all_sandbox_suites']=sandbox.read_text().count('[PASS]')==24 and '[FAIL]' not in sandbox.read_text()
    checks={name:bool(value) for name,value in checks.items()}
    ok=ret==0 and chain.get('valid') is True and len(suites)==24 and all(v['matched'] for v in suites.values()) and all(frozen.values()) and all(prior.values()) and all(checks.values())
    artifacts=[p for p in OUT.rglob('*') if p.is_file() and p.name not in ('verification.json','progress.json')]
    verification=dict(status='passed' if ok else 'failed',verified_utc=datetime.now(timezone.utc).isoformat(),
        scope='Authentic source/test receipt linkage, independent real-coordinate covariance contractions and all resamples; no physical-model or anomaly proof',
        calculation_checks=checks,jovin_suites=suites,chain=chain,frozen_sources=frozen,prior_artifacts_preserved=prior,
        independent_maximum_scaled_feature_error=float(np.max(np.abs(oracle-features)/(1+np.abs(features)))),
        artifact_sha256={str(p.relative_to(OUT)).replace('\\','/'):sha(p) for p in artifacts},report_sha256=sha(ROOT/'docs/REAL_SPLITS_RUN.md'),
        supporting_source_sha256={name:sha(ROOT/name) for name in ('scripts/verify_real_splits.py','scripts/replay_real_splits.py','scripts/report_real_splits.py')},
        local_tests=dict(passed=195,skipped=3,log_sha256=sha(tests)),ci_equivalent_suites=dict(passed=24,log_sha256=sha(sandbox)))
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(dict(status=verification['status'],suites=len(suites),checks=sum(v['total'] for v in suites.values()),failed=[key for key,value in checks.items() if not value],independent_feature_error=verification['independent_maximum_scaled_feature_error']),indent=2))
    if not ok:
        raise SystemExit(1)


if __name__=='__main__':
    main()
