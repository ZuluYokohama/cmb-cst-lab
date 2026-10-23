"""Frozen synthetic response benchmark; no observational Planck hypothesis test."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cmbcst import response, response_sky, theory, persistence
from cmbcst.copula_topology import normal_scores
from jovin_verify import SUITES, build_bundle

OUT = ROOT/'results/response_v1'
WORK = ROOT/'data/work/response_v1'
SEED = 765902132026
GROUPS = [('training',768,0,'null',0.),('calibration',999,10000,'null',0.),
    ('matched_null',512,20000,'null',0.),
    ('quadratic_0.15',128,30000,'quadratic',.15),('quadratic_0.35',128,31000,'quadratic',.35),
    ('compact_0.15',128,40000,'compact',.15),('compact_0.35',128,41000,'compact',.35),
    ('filament_0.15',128,50000,'filament',.15),('filament_0.35',128,51000,'filament',.35),
    ('stripe_stress',256,60000,'stripe_stress',0.)]
sha = lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
now = lambda:datetime.now(timezone.utc).isoformat()


def save(path, record):
    temporary = Path(path).with_name(Path(path).name+'.tmp')
    temporary.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def scaled_error(measured, expected):
    a,b = np.asarray(measured),np.asarray(expected)
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError('nonfinite or mismatched numerical output')
    return float(np.max(np.abs(a-b)/(1+np.abs(b))))


def freeze():
    import torch
    import ducc0
    import camb
    OUT.mkdir(parents=True,exist_ok=True)
    WORK.mkdir(parents=True,exist_ok=True)
    if (OUT/'plan.json').exists():
        raise FileExistsError('preserving frozen response pilot; no duplicate launch')
    record = [r for r in json.loads((ROOT/'data/evidence/jovin_receipts.json').read_text()) if r['suite']=='response'][-1]
    if (not record['passed'] or not record['receipt_verified'] or not record['chain_valid']
            or record['bundle_sha256'] != hashlib.sha256(build_bundle('response')[0].encode()).hexdigest()
            or record['tests_sha256'] != sha(ROOT/SUITES['response'][1])):
        raise ValueError('response source/tests lack matching authentic JoVin verification')
    sources = ['scripts/run_response_benchmark.py','docs/RESPONSE_SPEC.md','scripts/record_response_cuda.py',
        'src/cmbcst/response.py','src/cmbcst/response_sky.py','src/cmbcst/multiscale.py',
        'src/cmbcst/copula_topology.py','src/cmbcst/surrogate_topology.py','src/cmbcst/persistence.py',
        'src/cmbcst/spherical_surrogates.py','src/cmbcst/spectra.py','src/cmbcst/theory.py',
        'tests/sandbox/jovin_tests_response.py','tests/test_response.py']
    plan = {'created_utc':now(),'scope':'Synthetic method-development benchmark; observational Planck testing remains withheld',
        'base_seed':SEED,'groups':[{'name':g,'draws':n,'seed_offset':offset,'kind':kind,'amplitude':amp} for g,n,offset,kind,amp in GROUPS],
        'total_draws':3303,'batch_size':8,'lmax':256,'nside':128,'patch_size':64,'patch_width_degrees':20.,
        'center_radians':[float(np.pi/4),.7],'angles_degrees':[0.,22.5,45.,67.5],
        'smoothing_arcmin':[30.,60.,120.],'representations':['raw','normal_scores'],
        'baseline_dimensions':17,'response_dimensions':90,'joint_dimensions':107,
        'ridge':response.RIDGE,'shrinkage':response.SHRINKAGE,'candidate_p':.05,
        'max_null_flags':40,'power_family_size':6,'minimum_power_gain':.1,
        'paired_bootstrap_draws':2000,'paired_bootstrap_seed':833600,
        'scaled_numerical_tolerance':1e-10,'synthesis_epsilon':1e-11,'threads':8,
        'fiducial_parameters':theory.PLANCK18,'numpy':np.__version__,'torch':str(torch.__version__),
        'cuda':torch.version.cuda,'ducc0':ducc0.__version__,'camb':camb.__version__,
        'device':torch.cuda.get_device_name(0),'jovin_receipt_id':record['receipt_id'],
        'jovin_bundle_sha256':record['bundle_sha256'],'jovin_tests_sha256':record['tests_sha256'],
        'source_sha256':{name:sha(ROOT/name) for name in sources},
        'source_artifacts_sha256':{'data/evidence/response_cuda_development.json':sha(ROOT/'data/evidence/response_cuda_development.json')}}
    with (OUT/'plan.json').open('x') as stream:
        json.dump(plan,stream,indent=2,allow_nan=False)
    return plan


def represent(raw):
    ranks = np.stack([normal_scores(field) for field in raw.reshape(-1,64,64)]).reshape(raw.shape)
    return np.stack([raw,ranks],axis=3)


def measure(fields, first):
    try:
        return measured_batch(fields, first)
    except Exception:
        np.save(WORK/f'failed_feature_input_{first}.npy',fields)
        save(OUT/'feature_execution_failure.json',{'first_index':first,'count':len(fields),'error':traceback.format_exc()})
        raise


def measured_batch(fields, first):
    start = time.perf_counter()
    b,r = response.extract_response_reference(fields)
    cpu = time.perf_counter()-start
    start = time.perf_counter()
    gb,gr = response.extract_response_cuda(fields)
    gpu = time.perf_counter()-start
    q = {'first_index':first,'count':len(fields),'baseline_scaled_error':scaled_error(gb,b),
        'response_scaled_error':scaled_error(gr,r),'finite':True,
        'cpu_seconds':cpu,'gpu_seconds_including_transfer':gpu}
    q['passed'] = q['baseline_scaled_error'] <= 1e-10 and q['response_scaled_error'] <= 1e-10
    if not q['passed']:
        np.savez(WORK/f'failed_feature_batch_{first}.npz',fields=fields,cpu_baseline=b,cpu_response=r,gpu_baseline=gb,gpu_response=gr)
        save(OUT/'numerical_failure.json',q)
        raise ValueError('CPU/CUDA response precision failure')
    return b,r,q


def wilson(flags):
    n,k = len(flags),int(np.sum(flags))
    z = 1.959963984540054
    center = (k/n+z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/(1+z*z/n)
    return [float(center-half),float(center+half)]


def evaluate(baseline, features, draws):
    roles = np.asarray([d['group'] for d in draws])
    train,cal = roles=='training',roles=='calibration'
    fit_start = time.perf_counter()
    model = response.fit_response_model(baseline[train],features[train])
    fit_seconds = time.perf_counter()-fit_start
    save(OUT/'model.json',model)
    cpu = response.score_response_reference(baseline,features,model)
    gpu = response.score_response_cuda(baseline,features,model)
    score_error = scaled_error(gpu,cpu)
    if score_error > 1e-10 or max(model['condition_numbers']) > 1e8:
        save(OUT/'score_failure.json',{'scaled_error':score_error,'condition_numbers':model['condition_numbers']})
        raise ValueError('score precision or conditioning failure')
    held = ~(train|cal)
    held_p = response.upper_tail_pvalues(cpu[held],cpu[cal])
    pvalues = [None]*len(draws)
    for index,pvalue in zip(np.flatnonzero(held),held_p):
        pvalues[index] = pvalue.tolist()
    flags = np.zeros((len(draws),2),dtype=bool)
    flags[held] = held_p <= .05
    null = roles=='matched_null'
    counts = flags[null].sum(axis=0)
    null_passed = bool(np.all(counts <= 40))
    strata = {}
    for key,parameter,threshold in (('foreground','foreground_amplitude_uK',30.),('noise','noise_amplitude_uK',15.)):
        for side in ('low','high'):
            subset = null & np.array([(d[parameter] < threshold) == (side=='low') for d in draws])
            strata[key+'_'+side] = {'draws':int(subset.sum()),'flags':flags[subset].sum(axis=0).tolist()}
    validation = {'status':'passed' if null_passed else 'failed','draws':int(null.sum()),
        'methods':['baseline','baseline_plus_responses'],'flags':counts.tolist(),
        'rates':(counts/null.sum()).tolist(),'wilson_95':[wilson(flags[null,i]) for i in range(2)],
        'max_flags':40,'nuisance_subgroups':strata,
        'scope':'Matched synthetic mixture, fixed model/reference; not a Planck adequacy test'}
    save(OUT/'validation.json',validation)
    comparisons = []
    for index,(name,_,_,kind,amplitude) in enumerate(GROUPS[3:9]):
        selected = roles==name
        pairs = flags[selected]
        wins,losses,p = response.paired_win_pvalue(pairs[:,0],pairs[:,1])
        differences = pairs[:,1].astype(float)-pairs[:,0].astype(float)
        rng = np.random.default_rng(833600+index)
        bootstrap = differences[rng.integers(0,len(pairs),size=(2000,len(pairs)))].mean(axis=1)
        comparisons.append({'group':name,'kind':kind,'amplitude':amplitude,'draws':int(selected.sum()),
            'flags':pairs.sum(axis=0).tolist(),'rates':pairs.mean(axis=0).tolist(),
            'gain':float(differences.mean()),'paired_gain_bootstrap_95':np.quantile(bootstrap,[.025,.975]).tolist(),
            'combined_only_wins':wins,'baseline_only_losses':losses,'raw_p':p})
    adjusted = persistence.holm(np.array([r['raw_p'] for r in comparisons]))
    for record,pvalue in zip(comparisons,adjusted):
        record.update(p_holm=float(pvalue),provisional_gain=bool(null_passed and pvalue<=.05 and record['gain']>=.1))
    stress = roles=='stripe_stress'
    stress_record = {'draws':int(stress.sum()),'flags':flags[stress].sum(axis=0).tolist(),
        'rates':flags[stress].mean(axis=0).tolist(),
        'scope':'Unmodeled scan-stripe discrepancy; flags do not identify a cosmological signal'}
    result = {'status':'calibration_failed' if not null_passed else ('provisional_synthetic_gain' if any(r['provisional_gain'] for r in comparisons) else 'no_validated_gain'),
        'completed_utc':now(),'plan_sha256':sha(OUT/'plan.json'),'conditioning_sha256':sha(OUT/'conditioning.json'),
        'model_sha256':sha(OUT/'model.json'),'validation_sha256':sha(OUT/'validation.json'),
        'methods':['baseline','baseline_plus_responses'],'comparisons':comparisons,'stripe_stress':stress_record,
        'score_scaled_error':score_error,'fit_seconds':fit_seconds,'condition_numbers':model['condition_numbers'],
        'accepted_cosmological_claims':[],'observational_planck_test_performed':False,
        'limits':'Synthetic lensed-TT conditional Gaussian directions, simple foreground proxy and anisotropic band-limited noise. Not a calibrated Planck simulator, universal novelty test, or proof of unique information beyond all conventional statistics.'}
    return result,cpu,pvalues


def run():
    plan = freeze()
    start = time.perf_counter()
    stage = {'status':'running','started_utc':now(),'plan_sha256':sha(OUT/'plan.json')}
    save(OUT/'stage_audit.json',stage)
    try:
        t0 = time.perf_counter()
        sky = response_sky.SphericalResponseModel()
        np.savez(WORK/'conditioning.npz',cl=sky.cl,foreground_alm=sky.foreground,locations=sky.locations,noise_weight=sky.noise_weight)
        save(OUT/'conditioning.json',{'created_utc':now(),'cl_TT_uK2':sky.cl.tolist(),
            'conditioning_cache_sha256':sha(WORK/'conditioning.npz'),
            'foreground_alm_sha256':hashlib.sha256(sky.foreground.tobytes()).hexdigest(),
            'spherical_locations_sha256':hashlib.sha256(sky.locations.tobytes()).hexdigest(),
            'noise_weight_sha256':hashlib.sha256(sky.noise_weight.tobytes()).hexdigest(),
            'spectrum':'CAMB total lensed TT spectrum, Gaussian directions conditional on fixed powers',
            'nuisance_scope':'Synthetic foreground band and anisotropic band-limited noise; no physical Planck fit'})
        stage['prepare_seconds'] = time.perf_counter()-t0
        all_b,all_r,draws,quality,examples = [],[],[],[],{}
        pending_fields,pending_draws = [],[]
        forward_seconds,rank_seconds = 0.,0.

        def flush():
            nonlocal rank_seconds
            if not pending_fields:
                return
            first = len(all_b)
            raw = np.stack(pending_fields)
            rank_start = time.perf_counter()
            represented = represent(raw)
            rank_seconds += time.perf_counter()-rank_start
            b,r,q = measure(represented,first)
            all_b.extend(b)
            all_r.extend(r)
            draws.extend(pending_draws)
            quality.append(q)
            np.savez(WORK/f'feature_batch_{first:05d}.npz',baseline=b,response=r)
            pending_fields.clear()
            pending_draws.clear()
            if len(quality)%8 == 0 or len(all_b)==plan['total_draws']:
                save(OUT/'progress.json',{'completed':len(all_b),'total':plan['total_draws'],'elapsed_s':time.perf_counter()-start})
                print(f"response draws {len(all_b)}/{plan['total_draws']} ({time.perf_counter()-start:.1f}s)",flush=True)

        for group,count,offset,kind,amplitude in GROUPS:
            for index in range(count):
                tick = time.perf_counter()
                fields,parameters,q = sky.draw(SEED+offset+index,kind,amplitude)
                forward_seconds += time.perf_counter()-tick
                if not q['passed']:
                    np.save(WORK/f'failed_fields_{group}_{index}.npy',fields)
                    save(OUT/'power_failure.json',{'group':group,'index':index,'quality':q,'parameters':parameters})
                    raise ValueError('signal shell-power failure')
                parameters.update(group=group,index=index,signal_quality=q)
                if index == 0:
                    examples[group] = fields
                pending_fields.append(fields)
                pending_draws.append(parameters)
                if len(pending_fields) == 8:
                    flush()
            flush()
        if len(all_b) != 3303:
            raise ValueError('incomplete fixed sample domain')
        np.savez(WORK/'examples.npz',**examples)
        baseline,features = np.asarray(all_b),np.asarray(all_r)
        result,scores,pvalues = evaluate(baseline,features,draws)
        save(OUT/'ensemble.json',{'draws':draws,'baseline_features':baseline.tolist(),'response_features':features.tolist(),
            'scores':scores.tolist(),'held_out_pvalues':pvalues,'batch_quality':quality,
            'timings':{'forward_cpu_seconds':forward_seconds,'rank_cpu_seconds':rank_seconds,
                'feature_cpu_seconds':sum(q['cpu_seconds'] for q in quality),
                'feature_gpu_seconds_including_transfer':sum(q['gpu_seconds_including_transfer'] for q in quality)}})
        result['ensemble_sha256'] = sha(OUT/'ensemble.json')
        result['example_cache_sha256'] = sha(WORK/'examples.npz')
        save(OUT/'result.json',result)
        stage.update(status=result['status'],completed_utc=now(),elapsed_s=time.perf_counter()-start,
            draws=len(draws),numerical_failures=0,nonfinite_failures=0,
            max_baseline_error=max(q['baseline_scaled_error'] for q in quality),
            max_response_error=max(q['response_scaled_error'] for q in quality),
            max_signal_shell_error=max(d['signal_quality']['shell_relative_error'] for d in draws),
            score_scaled_error=result['score_scaled_error'],result_sha256=sha(OUT/'result.json'))
        save(OUT/'stage_audit.json',stage)
        print(json.dumps({'status':result['status'],'matched_null_flags':json.loads((OUT/'validation.json').read_text())['flags'],
            'provisional_gain_groups':[r['group'] for r in result['comparisons'] if r['provisional_gain']],
            'elapsed_s':stage['elapsed_s']},indent=2),flush=True)
    except Exception:
        stage.update(status='execution_failed',completed_utc=now(),elapsed_s=time.perf_counter()-start,error=traceback.format_exc())
        save(OUT/'stage_audit.json',stage)
        raise


if __name__ == '__main__':
    run()
