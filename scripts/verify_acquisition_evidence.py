"""Authenticate receipts and independently check acquisition metrics and audit inputs."""
import argparse
import ast
import csv
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cmbcst.acquisition import batch_reference,instrument_design,decoder_reference,frequency_grid,detector_table
from cmbcst.acquisition_index import analytic_phase_error
from jovin_verify import SUITES,build_bundle,configuration,_harness
OUT = ROOT/'results/acquisition_v1'
read = lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha = lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def close(a,b):
    a,b = np.asarray(a),np.asarray(b)
    return bool(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all() and np.max(np.abs(a-b)/(1+np.abs(b)))<=1e-10)


def main():
    command,env = configuration(argparse.Namespace(harness=None,harness_python=None,runtime='wsl:JovinHarness',data_dir=None))
    ret,chain = _harness(command,env,['receipts','verify'])
    history = read(ROOT/'data/evidence/jovin_receipts.json')
    suites = {}
    for name,(_,tests) in SUITES.items():
        record = [r for r in history if r['suite']==name][-1]
        sr,authentic = _harness(command,env,['receipts','show',record['receipt_id']])
        mr,membership = _harness(command,env,['receipts','verify',record['receipt_id']])
        matched = record['passed'] and record['receipt_verified'] and record['chain_valid'] and sr==mr==0 and authentic==record['receipt'] and membership.get('valid') is True and membership.get('chain')=='linked'
        matched &= hashlib.sha256(build_bundle(name)[0].encode()).hexdigest()==record['bundle_sha256'] and sha(ROOT/tests)==record['tests_sha256']
        suites[name] = dict(matched=bool(matched),receipt_id=record['receipt_id'],total=record['total'])
    frozen = {str(p.relative_to(ROOT)):all(sha(ROOT/name)==digest for name,digest in read(p).get('source_sha256',{}).items()) for p in (ROOT/'results').glob('*/plan.json')}
    prior = {str(p.relative_to(ROOT)):all(sha(p.parent/name)==digest for name,digest in read(p).get('artifact_sha256',{}).items()) for p in (ROOT/'results').glob('*/verification.json') if p.parent!=OUT}
    plan,result,stage = [read(OUT/name) for name in ('plan.json','result.json','stage_audit.json')]
    records = [read(OUT/f'device_cuda{i}.json') for i in (0,1)]
    checks = dict(fixed_design=plan['total_skies']==263168 and plan['grid_size']==512 and plan['detectors']==8
        and plan['base_seed']==986004152026 and plan['compute_cap_seconds']==3600
        and plan['precision']=='float64/complex128' and [d['production'] for d in plan['devices']]==[196608,65536],
        complete_and_in_budget=stage['status']==result['status'] and stage['numerical_failures']==0 and 0<stage['elapsed_seconds']<3600,
        honest_scope=result['accepted_cosmological_claims']==[] and result['total_skies']==263168 and result['detector_pixel_samples']==263168*8*512**2)
    a,target,mask = instrument_design(512)
    w,_ = decoder_reference(a,target,mask)
    expected_noise = .25*np.sum(np.abs(w)**2*frequency_grid(512)[5].ravel()[:,None,None],axis=(0,2))/512**2
    data_files = []
    pooled = []
    intensity_only = []
    for i,(item,record) in enumerate(zip(plan['devices'],records)):
        expected_batches = []
        total = 0
        for role,count,offset in (('noiseless',256,1000000),('intensity_only',256,2000000),('production',item['production'],0)):
            path = OUT/f'{role}_cuda{i}.npz'
            data_files.append(path.name)
            with np.load(path,allow_pickle=False) as data:
                mse = data['mse']
                checks[f'{role}_{i}_shape_finite'] = mse.shape==(count,4,4) and mse.dtype==np.float64 and bool(np.isfinite(mse).all() and np.all(mse>=0))
                if role=='noiseless':
                    maximum = float(np.sqrt(mse[:,0,:3]).max())
                    checks[f'noiseless_{i}_recovery'] = maximum<=1e-10 and maximum==record['checks']['noiseless_max_RMSE_uK']
                if role=='production':
                    ratio = mse[:,0,:3].mean(0)/expected_noise
                    checks[f'noise_model_{i}'] = close(expected_noise,record['checks']['predicted_noise_variance']) and close(ratio,record['checks']['noise_variance_ratio'])
                    checks[f'noise_gate_{i}'] = bool(max(abs(v-1) for v in ratio)<=.02)==record['checks']['noise_variance_passed']
                    pooled.append(mse.mean(0)*count)
                if role=='intensity_only':
                    intensity_only.append(mse.mean(0)*count)
            for first in range(0,count,item['batch']):
                expected_batches.append((role,first,item['batch'],986004152026+offset+i*(100000 if role=='production' else 10000)+first//item['batch']))
                total += item['batch']
        observed = [(b['role'],b['first_index'],b['count'],b['seed']) for b in record['batches']]
        checks[f'batch_domain_{i}'] = observed==expected_batches and total==item['production']+512
        checks[f'roll_finite_timing_{i}'] = all(0<=b['roll_amplitude_error']<=1e-10 and b['gpu_event_ms']>0 and b['wall_seconds']>0 for b in record['batches'])
        checks[f'roll_aggregate_{i}'] = record['checks']['max_roll_amplitude_error']==max(b['roll_amplitude_error'] for b in record['batches'])
        checks[f'device_identity_{i}'] = record['name']==item['name'] and record['batch']==item['batch']
        input_path = OUT/f'audit_input_cuda{i}.npz'
        checks[f'audit_input_{i}_sha'] = record['cpu_audit']['input_sha256']==sha(input_path)
        with np.load(input_path,allow_pickle=False) as data:
            ref = batch_reference(data['sky_white'],data['noise_white'])
        checks[f'full_resolution_cpu_audit_{i}'] = close(ref['mse'],record['cpu_audit']['reference_mse']) and close(ref['mse'],record['cpu_audit']['gpu_mse']) and record['cpu_audit']['passed'] and all(0<=v<=1e-10 for v in record['cpu_audit']['errors'].values())
        with np.load(OUT/f'production_cuda{i}.npz') as data:
            checks[f'audit_is_production_{i}'] = close(data['mse'][0],record['cpu_audit']['gpu_mse'][0])
    checks['cross_gpu_precision'] = 0<=result['cross_gpu_scaled_error']<=1e-10
    checks['model_gate_status'] = result['status']==('geometry_checks_passed' if all(r['checks']['noiseless_passed'] and r['checks']['noise_variance_passed'] for r in records) else 'model_checks_failed')
    poor_a,poor_t,poor_mask = instrument_design(32,'poor_angles')
    try:
        decoder_reference(poor_a,poor_t,poor_mask)
    except ValueError:
        checks['poor_coverage_rejected'] = True
    else:
        checks['poor_coverage_rejected'] = False
    with (OUT/'labeled_samples.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    d = detector_table()
    checks['indexed_sample_rows'] = len(rows)==32 and all(int(row['sample_index'])==j and int(row['time_index'])==j//8 and int(row['channel'])==j%8
        and abs(float(row['time_seconds'])-.001*(j//8))<1e-12
        and float(row['incidence_deg'])==d['incidence_deg'][j%8]
        and float(row['polarization_deg'])==d['polarization_deg'][j%8] for j,row in enumerate(rows))
    coordinate = read(OUT/'coordinate_export_audit.json')
    with (OUT/'labeled_samples_corrected.csv').open(newline='') as stream:
        corrected = list(csv.DictReader(stream))
    checks['coordinate_source_bound'] = coordinate['status']=='passed' and all(sha(ROOT/name)==digest for name,digest in coordinate['source_sha256'].items())
    checks['coordinate_export_hashes'] = coordinate['original_sha256']==sha(OUT/'labeled_samples.csv') and coordinate['corrected_sha256']==sha(OUT/'labeled_samples_corrected.csv')
    coordinate_rows_ok = len(corrected)==len(rows)
    for original,current in zip(rows,corrected):
        h = int(original['channel'])
        preserved = dict(original)
        preserved['registration_plus_offset_x_deg'] = preserved.pop('pointing_x_deg')
        preserved['registration_plus_offset_y_deg'] = preserved.pop('pointing_y_deg')
        coordinate_rows_ok &= all(current[key]==value for key,value in preserved.items())
        for axis in ('x','y'):
            unwrapped = (int(original[f'pixel_{axis}'])-256-d[f'offset_{axis}_pixels'][h])*20/512
            wrapped = (unwrapped+10)%20-10
            coordinate_rows_ok &= abs(float(current[f'look_{axis}_unwrapped_deg'])-unwrapped)<1e-12 and abs(float(current[f'look_{axis}_wrapped_deg'])-wrapped)<1e-12
    checks['corrected_physical_look_rows'] = bool(coordinate_rows_ok)
    negative,positive = analytic_phase_error(instrument_design(32)[0],d)
    checks['independent_phase_oracle'] = negative<1e-12 and positive>.01 and close([negative,positive],[coordinate['analytic_negative_phase_error'],coordinate['incorrect_positive_phase_error']])
    proof = read(ROOT/'data/evidence/acquisition_lean.json')
    text = (ROOT/'formal/PolarizationGeometry.lean').read_text()
    checks['lean_exact_identities'] = (proof['status']=='passed' and proof['exit_code']==0 and proof['theorem_count']==2
        and proof['source_sha256']==sha(ROOT/'formal/PolarizationGeometry.lean')
        and proof['archive_sha256']=='7289597a44543246c9227b8cae501767e095ce17f6d448b95fc4a8dc97268f6d'
        and not re.search(r'\b(sorry|admit)\b|^\s*axiom\s',text,re.MULTILINE))
    fixture = read(ROOT/'data/evidence/acquisition_gpu_preflight.json')
    tree = ast.parse((ROOT/'tests/sandbox/jovin_tests_acquisition.py').read_text())
    embedded = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='GPU_ATTESTATION' for t in node.targets))
    checks['actual_two_gpu_fixture'] = embedded==fixture['fixture'] and all(sha(ROOT/name)==digest for name,digest in fixture['source_sha256'].items())
    failure = read(ROOT/'data/evidence/acquisition_workspace_failure.json')
    archive = ROOT/'data/evidence/acquisition_workspace_failure_source.py.gz'
    checks['development_failure_preserved'] = (failure['source_sha256']==hashlib.sha256(gzip.decompress(archive.read_bytes())).hexdigest()
        and failure['source_archive_sha256']==sha(archive)
        and failure['preserved_log_sha256']==hashlib.sha256(failure['log'].encode()).hexdigest())
    telemetry = read(OUT/'telemetry.json')['samples']
    performance = read(OUT/'performance.json')['devices']
    for i in (0,1):
        rows = [d for sample in telemetry if sample.get('seconds',0)<=records[i]['elapsed_seconds'] for d in sample.get('devices',[]) if d['index']==i]
        if not rows:
            checks[f'telemetry_{i}_honest'] = performance[i]['available'] is False
        else:
            measured = [np.mean([r['utilization'] for r in rows]),np.median([r['utilization'] for r in rows]),np.mean([r['utilization']>=90 for r in rows]),np.mean([r['power_W'] for r in rows])]
            recorded = [performance[i][key] for key in ('mean_utilization','median_utilization','fraction_at_least_90_percent','mean_power_W')]
            checks[f'telemetry_{i}_recomputed'] = close(measured,recorded) and performance[i]['samples']==len(rows)
    bindings = dict(plan=stage['plan_sha256']==result['plan_sha256']==sha(OUT/'plan.json'),result=stage['result_sha256']==sha(OUT/'result.json'),
        preflight=plan['preflight_sha256']==sha(ROOT/'data/evidence/acquisition_gpu_preflight.json'),proof=plan['proof_sha256']==sha(ROOT/'data/evidence/acquisition_lean.json'),
        receipt=plan['jovin_receipt_id']==suites['acquisition']['receipt_id'])
    tests,sandbox = ROOT/'data/acquisition_tests.log',ROOT/'data/acquisition_sandbox_tests.log'
    tests_ok = '192 passed, 3 skipped' in tests.read_text()
    sandbox_ok = sandbox.read_text().count('[PASS]')==23 and '[FAIL]' not in sandbox.read_text()
    checks = {name:bool(value) for name,value in checks.items()}
    ok = ret==0 and chain.get('valid') is True and len(suites)==23 and all(v['matched'] for v in suites.values()) and all(frozen.values()) and all(prior.values()) and all(checks.values()) and all(bindings.values()) and tests_ok and sandbox_ok
    artifacts = ['plan.json','result.json','stage_audit.json','device_cuda0.json','device_cuda1.json',
        'audit_input_cuda0.npz','audit_input_cuda1.npz','labeled_samples.csv','labeled_samples_corrected.csv','coordinate_export_audit.json','telemetry.json','performance.json','hardware.png','geometry.png']+data_files
    audit = dict(status='passed' if ok else 'failed',verified_utc=datetime.now(timezone.utc).isoformat(),
        verification_environment=dict(python=sys.version,numpy=np.__version__,platform=sys.platform),
        scope='Authentic source/test receipt linkage, complete metrics and two CPU audit inputs; not proof of optics validity or novelty',
        jovin_suites=suites,chain=chain,frozen_sources=frozen,prior_artifacts_preserved=prior,calculation_checks=checks,bindings=bindings,
        local_tests=dict(passed=192,skipped=3,matched=tests_ok,log_sha256=sha(tests)),ci_equivalent_sandbox_suites=dict(passed=23,matched=sandbox_ok,log_sha256=sha(sandbox)),
        pooled_production_rmse=np.sqrt(sum(pooled)/262144).tolist(),
        pooled_intensity_only_rmse=np.sqrt(sum(intensity_only)/512).tolist(),
        artifact_sha256={name:sha(OUT/name) for name in artifacts},report_sha256=sha(ROOT/'docs/ACQUISITION_RUN.md'),
        supporting_source_sha256={name:sha(ROOT/name) for name in ('scripts/verify_acquisition_evidence.py','scripts/report_acquisition_campaign.py')})
    (OUT/'verification.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(dict(status=audit['status'],suites=len(suites),checks=sum(r['total'] for r in suites.values()),
        failed_calculations=[key for key,value in checks.items() if not value],failed_bindings=[key for key,value in bindings.items() if not value]),indent=2))
    if not ok:
        raise SystemExit(1)


if __name__=='__main__':
    main()
