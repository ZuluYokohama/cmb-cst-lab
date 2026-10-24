"""Authenticate saved CST evidence without extending the scientific run.

Only byte identity and arithmetic on already recorded arrays are checked here.
No Torch, harmonic transform, feature extraction, search or direct replay runs.
Metadata integrity and the scientific outcome are deliberately separate fields.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cmbcst.closure_fibers import projection_indices,quantize,robust_scales
from cmbcst.closure_search import TILES,STAGES,initial_kicks,matching_bins,manifest
from jovin_verify import SUITES,build_bundle,configuration,_harness
from report_closure_fibers import collect_run,read,load_npz,pytest_summary,SUCCESS_STATES,FAILURE_STATES
OUT=ROOT/'results/closure_fibers_v1'
GATE=1e-10


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4<<20),b''):
            digest.update(block)
    return digest.hexdigest()


def close(a,b,gate=GATE):
    a,b=np.asarray(a),np.asarray(b)
    return bool(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        and (not a.size or np.max(np.abs(a-b)/(1+np.abs(b)))<=gate))


def exact(a,b):
    return bool(np.array_equal(np.asarray(a),np.asarray(b)))


class Checks:
    def __init__(self):
        self.values={};self.errors={}

    def add(self,name,value):
        self.values[name]=bool(value)

    def attempt(self,name,function):
        try:
            self.add(name,function())
        except Exception as exc:
            self.add(name,False);self.errors[name]=f'{type(exc).__name__}: {exc}'


def authenticate():
    command,env=configuration(argparse.Namespace(harness=None,harness_python=None,
        runtime='wsl:JovinHarness',data_dir=None))
    ret,chain=_harness(command,env,['receipts','verify'])
    history=read(ROOT/'data/evidence/jovin_receipts.json',[])
    suites={}
    for name,(_,tests) in SUITES.items():
        records=[r for r in history if r.get('suite')==name]
        if not records:
            suites[name]=dict(matched=False,error='no current receipt');continue
        record=records[-1]
        sr,authentic=_harness(command,env,['receipts','show',record['receipt_id']])
        mr,membership=_harness(command,env,['receipts','verify',record['receipt_id']])
        bundle,functions=build_bundle(name)
        code_hash=hashlib.sha256(bundle.encode()).hexdigest()
        text_hash=hashlib.sha256((ROOT/tests).read_text(encoding='utf-8').encode()).hexdigest()
        payload={k:v for k,v in authentic.items() if k!='sha256'}
        receipt_hash=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),
            ensure_ascii=False,allow_nan=False).encode()).hexdigest()
        inputs=authentic.get('inputs',{});result=authentic.get('result',{})
        matched=(all(record.get(k) is True for k in ('passed','receipt_verified','chain_valid'))
            and sr==mr==0 and authentic==record.get('receipt') and membership.get('valid') is True
            and membership.get('chain')=='linked' and code_hash==record.get('bundle_sha256')
            and sha(ROOT/tests)==record.get('tests_sha256')
            and functions==record.get('source_function_sha256')
            and receipt_hash==authentic.get('sha256')==record.get('receipt_sha256')
            and inputs.get('code_sha256')==code_hash and inputs.get('tests_sha256')==text_hash
            and inputs.get('params')==dict(isolation='shared',timeout_s=record.get('timeout_s'))
            and authentic.get('tool')=='verify_python' and authentic.get('integrity')=='shared_interpreter'
            and result.get('status')=='passed' and result.get('timed_out') is False
            and result.get('total')==record.get('total')==record.get('passed_count')==result.get('passed_count'))
        suites[name]=dict(matched=bool(matched),receipt_id=record['receipt_id'],
            total=record.get('total'),bundle_sha256=code_hash,tests_sha256=sha(ROOT/tests),
            fresh_membership=membership)
    return ret==0 and chain.get('valid') is True,suites,chain


def validate_witness(out,entry,reference,scales,checks):
    cert=entry['certificate'];name=entry['name'];prefix=f'witness:{name}'
    path=out/cert['arrays'];arrays=load_npz(path)
    coeff=out/cert['coefficients'];alms=load_npz(coeff)
    checks.add(prefix+':hashes',sha(path)==cert['arrays_sha256'] and sha(coeff)==cert['coefficients_sha256'])
    checks.add(prefix+':finite_shapes',arrays['reference_features'].shape==arrays['candidate_features'].shape==arrays['matrix_features'].shape==arrays['gpu0_features'].shape==arrays['gpu1_features'].shape==(4,52)
        and arrays['theta'].shape==arrays['initial_theta'].shape==(128,)
        and all(np.isfinite(value).all() for value in arrays.values())
        and alms['alms'].shape==(2,2,131841) and np.isfinite(alms['alms']).all())
    ix=projection_indices(entry['planned']['representation']);q=.001;guard=1e-7
    ref=arrays['reference_features'];candidate=arrays['candidate_features']
    checks.add(prefix+':reference',close(ref[0],reference))
    cur=matching_bins(candidate[0],ref[0],scales,ix)
    future_ref=quantize(ref[1],ref[0],scales,entry['planned']['representation'])
    future_candidate=quantize(candidate[1],ref[0],scales,entry['planned']['representation'])
    dr=(ref[1,ix]-ref[0,ix])/scales[ix];dc=(candidate[1,ix]-ref[0,ix])/scales[ix]
    ref_margin=q/2-np.abs(dr-q*future_ref);can_margin=q/2-np.abs(dc-q*future_candidate)
    safe=bool(np.any((future_ref!=future_candidate)&(ref_margin>=guard)&(can_margin>=guard)))
    checks.add(prefix+':saved_bins',exact(arrays['current_reference_bins'],np.zeros(len(ix),dtype=int))
        and exact(arrays['current_candidate_bins'],cur['bins'])
        and exact(arrays['future_reference_bins'],future_ref) and exact(arrays['future_candidate_bins'],future_candidate))
    checks.add(prefix+':declared_bins',cert['input_bins_match']==cur['matched']
        and cert['future_bins_differ']==(not exact(future_ref,future_candidate))
        and cert.get('safe_future_bin_difference')==safe
        and close(cert['current_minimum_margin'],cur['minimum_margin']))
    gap=float(abs(candidate[1,22]-ref[1,22]))
    chosen=cert['chosen_start'];case=entry['arrays'];endpoints=entry['endpoints']
    checks.add(prefix+':stored_endpoint',isinstance(chosen,int) and 0<=chosen<128 and case['started'][chosen]>0
        and close(arrays['theta'],case['best_theta'][chosen]) and close(arrays['initial_theta'],case['initial_theta'][chosen])
        and endpoints is not None and close(arrays['matrix_features'][0],endpoints['features'][chosen])
        and close(cert['direct_gap'],gap) and close(cert['perturbation_ratios'],arrays['perturbation_ratios']))
    checks.add(prefix+':direct_precision',close(candidate,arrays['matrix_features'])
        and all(np.isfinite(cert[k]) and cert[k]<=GATE for k in
            ('direct_sample_feature_error','direct_covariance_error','direct_ratio_error'))
        if cert['status']=='verified_witness' else True)
    checks.add(prefix+':direct_error_arithmetic',close(cert['direct_sample_feature_error'],
        np.max(np.abs(candidate-arrays['matrix_features'])/(1+np.abs(arrays['matrix_features']))))
        and close(cert['direct_ratio_error'],np.max(np.abs(arrays['perturbation_ratios']-endpoints['ratios'][chosen])/(1+np.abs(endpoints['ratios'][chosen])))))
    checks.add(prefix+':all_time_invariant_records',len(cert['invariant_checks'])==4
        and all(isinstance(item.get('passed'),bool) and np.isfinite(item['normalized_spectral_gram_error'])
            and np.isfinite(item['split_difference_scaled_error'])
            and close(item['spectral_minimum_margin'],.5e-6-item['normalized_spectral_gram_error'])
            for item in cert['invariant_checks']))
    invariant_pass=all(v['passed'] and v['normalized_spectral_gram_error']<=GATE
        and v['split_difference_scaled_error']<=GATE and v['spectral_bins_zero']
        and v['spectral_minimum_margin']>=guard and v['zero_reference_auto_shells_exact']
        and v['unmodified_removed_shells'] and v['mzero_real'] for v in cert['invariant_checks'])
    replay_errors=cert.get('actual_gpu_replay_errors',[])
    checks.add(prefix+':both_gpu_replay_records',len(replay_errors)==2 and {v['device'] for v in replay_errors}=={0,1}
        and all(close(v['maximum_scaled_error'],np.max(np.abs(arrays[f"gpu{v['device']}_features"]-candidate)/(1+np.abs(candidate)))) for v in replay_errors)
        and cert.get('numerical_preservation_passed')==all(v['passed'] for v in cert['invariant_checks']))
    for rep in ('baseline','capacity','repair'):
        indices=projection_indices(rep);bins=quantize(candidate[0],ref[0],scales,rep)
        delta=(candidate[0,indices]-ref[0,indices])/scales[indices]
        margins=q/2-np.abs(delta-q*bins);separated=bool(np.any((bins!=0)&(margins>=guard)))
        details=cert.get('separation_details',{}).get(rep,{})
        checks.add(prefix+':separation:'+rep,cert['distinguished_by'][rep]==separated
            and details.get('bins_differ')==bool(np.any(bins!=0))
            and details.get('robust_changed_coordinate')==separated
            and exact(details.get('bins'),bins) and close(details.get('nearest_boundary_margins'),margins))
    if cert['status']=='verified_witness':
        checks.add(prefix+':acceptance',cur['matched'] and safe and gap>=.05 and invariant_pass
            and all(v['maximum_scaled_error']<=GATE for v in replay_errors)
            and np.max(np.abs(arrays['theta']))<=np.pi/4 and arrays['perturbation_ratios'].shape==(2,)
            and np.min(arrays['perturbation_ratios'])>=0 and np.max(arrays['perturbation_ratios'])<=.25)


def validate_case(out,entry,reference,scales,checks):
    name=entry['name'];a=entry['arrays'];started=a['started']>0
    finite=np.isfinite(a['gpu_features']).all(axis=1)&np.isfinite(a['gpu_target'])&np.isfinite(a['gpu_ratios']).all(axis=1)
    checks.add(name+':checkpoint_range',entry['checkpoint_error'] is None)
    checks.add(name+':array_shapes',a['theta'].shape==a['initial_theta'].shape==a['best_theta'].shape==(128,128)
        and a['gpu_features'].shape==(128,52) and a['gpu_ratios'].shape==(128,2)
        and a['gpu_target'].shape==(128,) and a['started'].shape==a['status'].shape==(128,)
        and np.all(np.isin(a['started'],[0,1])))
    known=(*SUCCESS_STATES,*FAILURE_STATES,'not_started','incomplete','active','initializing')
    checks.add(name+':status_domain',np.isin(a['status'],known).all()
        and np.all(a['status'][~started]=='not_started'))
    checks.add(name+':accounting',all(np.asarray(a[key]).shape==(128,)
        and np.isfinite(a[key]).all() and np.all(a[key]>=0)
        for key in ('iterations','stalls','rejected','nonfinite_trials','rank'))
        and np.all(a['iterations']<=1000) and np.all(a['stalls']<=50)
        and np.all(a['rank']<=len(projection_indices(entry['planned']['representation']))))
    checks.add(name+':iteration_status',np.all(a['iterations'][a['status']=='iteration_cap']==1000)
        and np.all(a['stalls'][a['status']=='stalled']==50) and np.all(a['rejected']>=a['stalls']))
    failed=np.isin(a['status'],FAILURE_STATES)
    incomplete=np.isin(a['status'],('incomplete','active','initializing'))
    checks.add(name+':retained_endpoints_finite',not np.any(started&~finite&~(failed|incomplete)))
    checks.add(name+':unstarted_endpoints_missing',not np.any(~started&finite))
    checks.add(name+':started_angles_finite',np.isfinite(a['theta'][started]).all()
        and np.isfinite(a['best_theta'][started]).all())
    checkpoint=entry['checkpoint'];metadata=entry['checkpoint_meta']
    if checkpoint is not None and metadata is not None:
        first=metadata['first_start'];rows=len(checkpoint['theta'])
        kicks,signs=initial_kicks(entry['planned']['tile'],entry['planned']['stage'])
        checks.add(name+':checkpoint_initializations',exact(checkpoint['raw_kicks'],kicks[first:first+rows])
            and exact(checkpoint['signs'],signs[first:first+rows]))
    record=entry['record']
    if record:
        checks.add(name+':case_identity',all(record[k]==entry['planned'][k] for k in ('tile','stage','representation','device')))
        # A hard-cap checkpoint can contain starts newer than its case record.
        closed=record['status'] in ('complete','execution_failed') and not entry['checkpoint_overlay'].any()
        checks.add(name+':closed_counts',not closed or (record['starts_started']==entry['started']
            and record['starts_completed']==entry['completed'] and record['terminal_failure_starts']==entry['terminal_failures']))
        checks.add(name+':complete_accounting',record['status']!='complete' or
            (entry['started']==entry['completed']==128 and entry['terminal_failures']==0 and finite[started].all()))
    if entry['array_file_present']:
        expected_kicks,expected_signs=initial_kicks(entry['planned']['tile'],entry['planned']['stage'])
        checks.add(name+':saved_initializations',exact(a['raw_kicks'],expected_kicks) and exact(a['signs'],expected_signs))
    cert=entry['certificate'];ep=entry['endpoints']
    if cert is None:
        return
    checks.add(name+':certificate_domain',cert['status'] in ('verified_witness','no_verified_witness','precision_failure','rejected_candidate'))
    checks.add(name+':certificate_identity',all(cert['case'][key]==entry['planned'][key] for key in ('tile','stage','representation','device')))
    if 'maximum_discovered_feasible_gap' not in cert:
        checks.add(name+':failure_explicit',cert['status']=='precision_failure' and bool(cert.get('error')));return
    checks.add(name+':endpoint_audit_present',ep is not None)
    if ep is None:
        return
    checks.add(name+':endpoint_audit_shapes',ep['features'].shape==(128,52) and ep['ratios'].shape==(128,2)
        and ep['target'].shape==ep['matching'].shape==ep['gap'].shape==(128,)
        and exact(ep['started'],a['started']) and exact(ep['start_id'],np.arange(128)))
    checks.add(name+':cpu_gpu_agreement',close(ep['features'][started],a['gpu_features'][started])
        and close(ep['target'][started],a['gpu_target'][started]) and close(ep['ratios'][started],a['gpu_ratios'][started])
        and 0<=cert['independent_feature_error']<=GATE)
    matches=np.zeros(128,dtype=bool)
    for index in np.flatnonzero(started):
        matches[index]=(matching_bins(ep['features'][index],reference,scales,projection_indices(entry['planned']['representation']))['matched']
            and np.max(np.abs(a['best_theta'][index]))<=np.pi/4 and np.max(ep['ratios'][index])<=.25)
    target=entry['reference_target']
    if target is None and cert.get('arrays'):
        target=float(load_npz(out/cert['arrays'])['reference_features'][1,22])
    checks.add(name+':reference_target_recorded',target is not None)
    if target is not None:
        gaps=np.abs(ep['target']-target)
        checks.add(name+':saved_gap_arithmetic',close(ep['gap'][started],gaps[started]))
    checks.add(name+':endpoint_decisions',exact(ep['matching'],matches)
        and cert['starts_checked']==int(started.sum()) and cert['feasible_endpoints']==int(matches.sum())
        and cert['provisional_candidates']==int(np.sum(matches&(ep['gap']>=.05)))
        and close(cert['maximum_discovered_feasible_gap'],float(np.max(ep['gap'][matches])) if np.any(matches) else 0))
    if cert.get('arrays'):
        validate_witness(out,entry,reference,scales,checks)
    if cert['status']=='no_verified_witness':
        checks.add(name+':no_witness_honest',cert['provisional_candidates']==0)


def validate_timing(out,plan,stage,checks):
    """Audit recorded wall-time metadata, without extending the scientific run."""
    status=stage.get('status');elapsed=stage.get('elapsed_seconds');cap=plan['compute_cap_seconds']
    terminal=('complete','complete_with_failures','incomplete','incomplete_with_failures',
        'incomplete_hard_cap','execution_failed')
    valid_elapsed=isinstance(elapsed,(int,float)) and np.isfinite(elapsed) and elapsed>=0
    checks.add('scientific_run_terminal',status in terminal)
    checks.add('recorded_elapsed_valid',valid_elapsed)
    stages=stage.get('stages',{})
    valid_stages=all(isinstance(value,(int,float)) and np.isfinite(value) and value>=0 for value in stages.values())
    checks.add('recorded_stage_times_valid',valid_stages)
    checks.add('recorded_stage_time_accounting',valid_elapsed and valid_stages
        and sum(stages.values())<=elapsed+1e-6)
    checks.add('completion_within_declared_cap',not str(status).startswith('complete')
        or (valid_elapsed and elapsed<=cap and set(stages)=={
            'preprocessing_seconds','search_seconds','scientific_verification_seconds'}))
    checks.add('recorded_cap_excess_explicit',not valid_elapsed or elapsed<=cap
        or str(status).startswith('incomplete'))
    hard=read(out/'hard_cap.json')
    checks.add('hard_cap_marker_consistent',(hard is None and status!='incomplete_hard_cap') or
        (hard is not None and status=='incomplete_hard_cap' and hard.get('status')==status
        and hard.get('plan_sha256')==stage.get('plan_sha256')
        and hard.get('elapsed_seconds')==elapsed and valid_elapsed and elapsed>=cap
        and not (out/'result.json').exists()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=OUT)
    args=parser.parse_args();out=args.run_dir;checks=Checks()
    run=collect_run(out);plan,stage,summary=[run[k] for k in ('plan','stage','summary')]
    checks.add('frozen_protocol',plan['seed']==82024102027 and plan['selected_tiles']==list(TILES)
        and plan['cases']==[manifest(t,s,r) for t in TILES for s,r in STAGES]
        and plan['planned_starts']==6144 and plan['starts_per_case']==plan['planes']==128
        and plan['local_quantum']==.001 and plan['spectral_quantum']==1e-6
        and plan['certificate_guard']==1e-7 and plan['search_guard']==2e-7
        and plan['rms_ratio_limit']==.25 and plan['angle_limit']==np.pi/4 and plan['witness_gap']==.05
        and plan['max_iterations']==1000 and plan['max_stalls']==50 and plan['svd_relative_cutoff']==1e-10
        and plan['step_norm']==.1 and plan['newton_corrections']==5 and plan['step_halvings']==10
        and plan['compute_cap_seconds']==14400 and plan['verification_reserve_seconds']==900
        and plan['sampling_epsilon']==1e-12 and plan['numerical_gate']==1e-10 and plan['threads']==8
        and plan['shells']==np.rint(np.linspace(30,216,32)).astype(int).tolist())
    checks.add('plan_stage_binding',stage.get('plan_sha256')==sha(out/'plan.json'))
    validate_timing(out,plan,stage,checks)
    required_sources={'src/cmbcst/'+name for name in ('closure_fibers.py','closure_fibers_data.py',
        'closure_fibers_oracle.py','closure_search.py','closure_certification.py','closure_jacobian.py')}
    checks.attempt('current_scientific_source_bindings',lambda:required_sources<=set(plan['source_sha256'])
        and all(sha(ROOT/name)==digest for name,digest in plan['source_sha256'].items()))
    if plan.get('source_revision'):
        def revision_bindings():
            revision=plan['source_revision']
            if not re.fullmatch(r'[0-9a-f]{40}',revision):return False
            kind=subprocess.run(['git','cat-file','-t',revision],cwd=ROOT,capture_output=True,check=False)
            if kind.returncode or kind.stdout.strip()!=b'commit':return False
            for name,digest in plan['source_sha256'].items():
                blob=subprocess.run(['git','show',f'{revision}:{name}'],cwd=ROOT,capture_output=True,check=False)
                if blob.returncode or hashlib.sha256(blob.stdout).hexdigest()!=digest:return False
            return True
        checks.attempt('committed_source_revision_binding',revision_bindings)
    checks.add('declared_real_inputs',set(plan['input_sha256'])=={
        'results/real_splits_v1/harmonics/alm_143A.npy','results/real_splits_v1/harmonics/alm_143B.npy',
        'results/real_splits_v1/geometry.npz','results/real_splits_v1/tiles.json'})
    checks.attempt('input_hash_bindings',lambda:all(sha(ROOT/name)==digest for name,digest in plan['input_sha256'].items()))
    checks.attempt('initialization_hash',lambda:sha(out/'initializations.npz')==plan['initialization_sha256'])
    initial=load_npz(out/'initializations.npz')
    checks.add('initialization_seed_replay',initial is not None and all(
        exact(initial[f'tile_{tile}_{kind}'],initial_kicks(tile,kind)[0]) and
        exact(initial[f'signs_{tile}_{kind}'],initial_kicks(tile,kind)[1]) for tile in TILES for kind in ('source','fresh')))
    frozen={};prior={}
    for path in sorted((ROOT/'results').glob('*/plan.json')):
        key=str(path.relative_to(ROOT)).replace('\\','/')
        try:
            frozen[key]=all(sha(ROOT/name)==digest for name,digest in read(path).get('source_sha256',{}).items())
        except Exception:
            frozen[key]=False
    for path in sorted((ROOT/'results').glob('*/verification.json')):
        if path.parent.resolve()==out.resolve():continue
        key=str(path.relative_to(ROOT)).replace('\\','/')
        try:
            prior[key]=all(sha(path.parent/name)==digest for name,digest in read(path).get('artifact_sha256',{}).items())
        except Exception:
            prior[key]=False
    checks.add('all_frozen_sources',bool(frozen) and all(frozen.values()))
    checks.add('prior_artifacts_preserved',bool(prior) and all(prior.values()))
    profile_path=ROOT/plan.get('preflight_path','data/evidence/closure_gpu_preflight_v2.json')
    profile=read(profile_path,{})
    checks.attempt('preflight_source_binding',lambda:profile['status']=='passed'
        and sha(profile_path)==plan['preflight_sha256']
        and all(sha(ROOT/name)==digest for name,digest in profile['source_sha256'].items()))
    real_profile_path=ROOT/'data/evidence/closure_real_operator_preflight.json'
    real_profile=read(real_profile_path,{})
    checks.attempt('real_operator_preflight_binding',lambda:real_profile['status']=='passed'
        and real_profile['source_unchanged'] is True
        and sha(real_profile_path)==plan['real_operator_preflight_sha256']
        and real_profile['source_inputs']==plan['input_sha256']
        and all(sha(ROOT/name)==digest for name,digest in real_profile['source_sha256'].items())
        and real_profile['accepted_tiles']==79 and real_profile['invariants']['passed'] is True
        and all(np.isfinite(value) and 0<=value<=GATE for value in real_profile['direct_feature_errors'])
        and len(real_profile['gpu'])==2 and {v['index'] for v in real_profile['gpu']}=={0,1}
        and all(np.isfinite(value) and 0<=value<=GATE for device in real_profile['gpu']
            for value in device['feature_errors_u0_u1']))
    tree=ast.parse((ROOT/'tests/sandbox/jovin_tests_closure_fibers.py').read_text(encoding='utf-8'))
    assignment=next(v for v in tree.body if isinstance(v,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='GPU_ATTESTATION' for t in v.targets))
    checks.add('preflight_sandbox_actual_output_binding',ast.literal_eval(assignment.value)==profile.get('fixture'))
    fixtures=profile.get('fixture',{}).get('devices',[])
    checks.add('both_actual_devices',len(fixtures)==2 and {v['device'] for v in fixtures}=={'cuda:0','cuda:1'}
        and all(all(np.isfinite(x) and 0<=x<=(1e-7 if k=='jacobian_five_point' else GATE)
            for k,x in v['errors'].items()) for v in fixtures))
    proof=read(ROOT/'data/evidence/closure_lean.json',{})
    checks.attempt('pinned_lean_binding',lambda:proof['status']=='passed' and proof['exit_code']==0
        and proof['theorem_count']==7 and len(proof['theorem_axioms'])==7
        and proof['source_sha256']==sha(ROOT/'formal/OperationalClosure.lean')
        and proof['verifier_sha256']==sha(ROOT/'scripts/verify_closure_lean.py')
        and proof['supporting_rotation_source_sha256']==sha(ROOT/'formal/PolarizationGeometry.lean')
        and sha(ROOT/'data/evidence/closure_lean.json')==plan['proof_sha256']
        and proof['compiler_sha256']=='bf8d54e4714cc4b03d3f6bb34c83b7202b87e49c8bfcbff6895c085bb90ceb38'
        and proof['archive_sha256']=='7289597a44543246c9227b8cae501767e095ce17f6d448b95fc4a8dc97268f6d'
        and all(set(v)<={'propext','Classical.choice','Quot.sound'} for v in proof['theorem_axioms'].values())
        and not re.search(r'\b(sorry|admit)\b|^\s*(axiom|opaque)\s',(ROOT/'formal/OperationalClosure.lean').read_text(),re.MULTILINE))
    supporting_proofs=[read(ROOT/p,{}) for p in ('results/multiscale_v1/lean_verification.json','data/evidence/acquisition_lean.json')]
    checks.add('supporting_18_theorems',all(p.get('status')=='passed' and p.get('exit_code')==0 for p in supporting_proofs)
        and proof.get('theorem_count',0)+sum(p.get('theorem_count',0) for p in supporting_proofs)==18)
    audit=read(ROOT/'data/evidence/closure_search_development_audit.json',{})
    checks.attempt('historical_search_archive',lambda:sha(ROOT/audit['archive'])==audit['archive_sha256']
        and hashlib.sha256(gzip.decompress((ROOT/audit['archive']).read_bytes())).hexdigest()==audit['source_sha256'])
    archive_bytes={str(path.relative_to(ROOT)).replace('\\','/'):gzip.decompress(path.read_bytes())
        for path in sorted((ROOT/'data/evidence').glob('closure*.gz'))}
    archive_digest={hashlib.sha256(value).hexdigest():name for name,value in archive_bytes.items()}
    attested_fixtures=[('tests/sandbox/jovin_tests_closure_fibers.py',ast.literal_eval(assignment.value))]
    for name,value in archive_bytes.items():
        try:
            archived_tree=ast.parse(value.decode('utf-8'))
            node=next(v for v in archived_tree.body if isinstance(v,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='GPU_ATTESTATION' for t in v.targets))
            attested_fixtures.append((name,ast.literal_eval(node.value)))
        except (ValueError,SyntaxError,StopIteration):
            pass
    historical_profiles={}
    for path in sorted((ROOT/'data/evidence').glob('closure_gpu_preflight*.json')):
        old=read(path);bindings={}
        for name,digest in old.get('source_sha256',{}).items():
            bindings[name]=name if (ROOT/name).exists() and sha(ROOT/name)==digest else archive_digest.get(digest)
        old_fixture=next((name for name,fixture in attested_fixtures if fixture==old.get('fixture')),None)
        historical_profiles[str(path.relative_to(ROOT)).replace('\\','/')]=dict(source_bindings=bindings,
            fixture_binding=old_fixture,sha256=sha(path),recorded_status=old.get('status'))
    checks.add('historical_preflight_sources_and_outputs',all(all(v is not None for v in p['source_bindings'].values())
        and p['fixture_binding'] is not None for p in historical_profiles.values()))
    ref=run['reference'];reference_map={}
    if ref is not None:
        checks.add('fixed_reference_scales',ref['features'].shape==(79,52) and ref['tile_ids'].shape==(79,)
            and len(np.unique(ref['tile_ids']))==79 and np.isfinite(ref['features']).all()
            and close(ref['scales'],robust_scales(ref['features'])))
        normalization=read(out/'normalization_lock.json',{})
        checks.attempt('normalization_lock_binding',lambda:
            normalization['plan_sha256']==sha(out/'plan.json')
            and normalization['reference_features_sha256']==sha(out/'reference_features.npz')
            and exact(normalization['accepted_tiles'],ref['tile_ids'])
            and normalization['rule']=='max(1.4826*MAD, .001*max(1,abs(median)))'
            and datetime.fromisoformat(plan['created_utc'])<=datetime.fromisoformat(normalization['created_utc'])
            and ('preprocessing_seconds' not in stage.get('stages',{}) or
                (datetime.fromisoformat(normalization['created_utc'])-
                 datetime.fromisoformat(stage['started_utc'])).total_seconds()
                <=stage['stages']['preprocessing_seconds']+1.))
        reference_map={int(tile):v for tile,v in zip(ref['tile_ids'],ref['features'])}
        for entry in run['entries']:
            checks.attempt('case_audit:'+entry['name'],lambda e=entry:(validate_case(out,e,
                reference_map[e['planned']['tile']],ref['scales'],checks) is None))
    else:
        checks.add('reference_absence_explicit',summary['starts_started']==0 and stage['status']!='complete')
    for tile in TILES:
        operator=out/'operators'/f'family_{tile}.json';planes=out/'operators'/f'planes_{tile}.npz'
        used=any(e['planned']['tile']==tile and e['started'] for e in run['entries'])
        checks.add(f'operator:{tile}:availability',not used or (operator.exists() and planes.exists()))
        if not operator.exists():continue
        def validate_operator(tile=tile,operator=operator,planes=planes):
            meta=read(operator);config=meta['configuration'];source=config['source']
            numerical=load_npz(planes)
            cache=ROOT/'data/work/closure_fibers_v1'/f'family_{tile}.npz'
            return (sha(planes)==meta['planes_sha256'] and config['tile']==tile
                and config['seed']==plan['seed'] and config['shells']==plan['shells'] and config['planes_per_shell']==4
                and source['input_sha256']==plan['input_sha256'] and source['epsilon']==1e-12
                and source['threads']==8 and source['lmax']==512
                and source['data_source_sha256']==sha(ROOT/'src/cmbcst/closure_fibers_data.py')
                and meta['source_reference']['source']==source
                and all(np.isfinite(v) and 0<=v<=GATE for v in meta['plane_checks'].values())
                and numerical['shells'].shape==(128,) and exact(numerical['shells'],np.repeat(plan['shells'],4))
                and int(numerical['seed'])==plan['seed'] and int(numerical['tile'])==tile
                and all(np.isfinite(v).all() for v in numerical.values())
                and (not cache.exists() or sha(cache)==meta['family_sha256']))
        checks.attempt(f'operator:{tile}:bindings',validate_operator)
    result=run['result']
    if result is not None:
        checks.add('final_bindings',result['plan_sha256']==sha(out/'plan.json') and stage.get('result_sha256')==sha(out/'result.json')
            and result['status']==stage['status'])
        checks.add('final_accounting',all(result[k]==summary[k] for k in
            ('planned_starts','starts_started','starts_completed','terminal_failure_starts','verified_witnesses')))
        failures=sum(e['terminal_failures']+int(bool((e['record'] or {}).get('error')) and
            (e['record'] or {})['error'].get('type')!='TimeoutError')+int(bool(e['certificate']) and e['certificate']['status']=='precision_failure') for e in run['entries'])
        checks.add('failure_accounting',result['numerical_or_execution_failures']==failures)
        expected_status=('complete' if not failures else 'complete_with_failures') if summary['starts_completed']+summary['terminal_failure_starts']==6144 else ('incomplete_with_failures' if failures else 'incomplete')
        checks.add('final_status_accounting',result['status']==expected_status and stage['numerical_failures']==failures)
        checks.add('all_final_cases_represented',len(result['cases'])==48 and
            {(v['tile'],v['stage'],v['representation']) for v in result['cases']}==
            {(v['tile'],v['stage'],v['representation']) for v in plan['cases']})
        checks.add('all_final_started_cases_audited',len(result['certificates'])==sum(e['started']>0 for e in run['entries'])
            and all(e['certificate'] is not None for e in run['entries'] if e['started']))
        checks.add('scientific_outcome_honest',result['scientific_outcome']==
            ('verified_finite_precision_collision' if summary['verified_witnesses'] else 'no_verified_witness_found')
            and ('complete'!=result['status'] or (summary['starts_completed']==6144 and not failures)))
    else:
        checks.add('missing_final_result_explicit',stage['status'] not in ('complete','complete_with_failures')
            and summary['scientific_outcome']=='not_finalized')
    log=ROOT/'data/closure_tests.log';sandbox=ROOT/'data/closure_sandbox_tests.log'
    tests=pytest_summary(log);sandbox_text=sandbox.read_text(errors='replace') if sandbox.exists() else ''
    names=re.findall(r'^\[PASS\]\s+([^\s:]+)',sandbox_text,re.MULTILINE)
    checks.add('local_regression_log',tests['available'] and tests['passed']>0 and tests['failed']==tests['errors']==0)
    checks.add('all_current_sandbox_suites',len(names)==len(SUITES) and set(names)==set(SUITES) and '[FAIL]' not in sandbox_text)
    chain_ok,suites,chain=authenticate()
    checks.add('authentic_current_jovin',chain_ok and len(suites)==len(SUITES) and all(v['matched'] for v in suites.values()))
    checks.add('freeze_receipt_binding',suites.get('closure_fibers',{}).get('receipt_id')==plan['jovin_receipt_id']
        and suites.get('closure_fibers',{}).get('bundle_sha256')==plan['jovin_bundle_sha256'])
    report=ROOT/'docs/CLOSURE_FIBERS_RUN.md';saved_summary=read(out/'report_summary.json',{})
    checks.add('report_summary_consistent',report.exists() and all(saved_summary.get(k)==v for k,v in summary.items()))
    ok=all(checks.values.values())
    artifacts=[p for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ('verification.json','progress.json')]
    verification=dict(status='passed' if ok else 'failed',metadata_integrity='passed' if ok else 'failed',
        verified_utc=datetime.now(timezone.utc).isoformat(),scientific_status=summary['scientific_status'],
        scientific_outcome=summary['scientific_outcome'],scientific_summary=summary,
        scope='Post-run saved-evidence hash/count/bin/margin/bound audit; no new physical transforms, feature extraction, optimization or direct scientific verification',
        arithmetic_checks=checks.values,errors=checks.errors,jovin_suites=suites,chain=chain,
        jovin_suite_count=len(suites),jovin_test_count=sum(v.get('total',0) for v in suites.values()),
        frozen_sources=frozen,prior_artifacts_preserved=prior,
        historical_preflight_bindings=historical_profiles,
        artifact_sha256={str(p.relative_to(out)).replace('\\','/'):sha(p) for p in artifacts},
        report_sha256=sha(report) if report.exists() else None,
        supporting_script_sha256={name:sha(ROOT/name) for name in ('scripts/report_closure_fibers.py','scripts/verify_closure_fibers.py')},
        local_tests={**tests,'log_sha256':sha(log) if log.exists() else None},
        plain_sandbox_tests=dict(passed=len(names),suites=names,log_sha256=sha(sandbox) if sandbox.exists() else None),
        development_artifact_sha256={str(p.relative_to(ROOT)).replace('\\','/'):sha(p)
            for p in sorted((ROOT/'data/evidence').glob('closure*')) if p.is_file()})
    (out/'verification.json').write_text(json.dumps(verification,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(metadata_integrity=verification['metadata_integrity'],scientific_status=summary['scientific_status'],
        scientific_outcome=summary['scientific_outcome'],verified_witnesses=summary['verified_witnesses'],
        suites=len(suites),checks=verification['jovin_test_count'],failed=[k for k,v in checks.values.items() if not v],errors=checks.errors),indent=2))
    if not ok:raise SystemExit(1)


if __name__=='__main__':
    main()
