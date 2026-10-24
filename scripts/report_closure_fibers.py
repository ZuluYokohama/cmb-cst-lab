"""Report saved CST search evidence without running new scientific calculations.

This module deliberately does not import Torch, ducc or the scientific feature
evaluators. A post-cap report may summarize saved arrays and certificates only.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cmbcst.closure_fibers import projection_indices
from cmbcst.closure_search import matching_bins
OUT=ROOT/'results/closure_fibers_v1'
CASE_LABELS=(('source','baseline'),('fresh','baseline'),('fresh','capacity'),('fresh','repair'))
SUCCESS_STATES=('stalled','iteration_cap')
FAILURE_STATES=('nonfinite_jacobian','nonfinite_gradient')


def read(path,default=None):
    path=Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def case_name(case):
    return f"tile_{case['tile']}_{case['stage']}_{case['representation']}"


def load_npz(path):
    path=Path(path)
    if not path.exists():
        return None
    with np.load(path,allow_pickle=False) as source:
        return {key:source[key] for key in source.files}


def pytest_summary(path):
    path=Path(path)
    if not path.exists():
        return dict(available=False,passed=0,skipped=0,failed=0,errors=0)
    for line in reversed(path.read_text(encoding='utf-8',errors='replace').splitlines()):
        if re.search(r'\b\d+ passed\b',line):
            count=lambda pattern:int((re.search(pattern,line) or [None,'0'])[1])
            return dict(available=True,passed=count(r'\b(\d+) passed\b'),
                skipped=count(r'\b(\d+) skipped\b'),failed=count(r'\b(\d+) failed\b'),
                errors=count(r'\b(\d+) errors?\b'),summary=line.strip())
    return dict(available=False,passed=0,skipped=0,failed=0,errors=0)


def _empty_arrays(count=128,planes=128):
    return dict(theta=np.zeros((count,planes)),best_theta=np.zeros((count,planes)),
        initial_theta=np.zeros((count,planes)),started=np.zeros(count,dtype=int),
        initial_retraction_succeeded=np.zeros(count,dtype=bool),signs=np.zeros(count,dtype=float),
        iterations=np.zeros(count,dtype=int),stalls=np.zeros(count,dtype=int),
        rejected=np.zeros(count,dtype=int),nonfinite_trials=np.zeros(count,dtype=int),
        rank=np.zeros(count,dtype=int),status=np.full(count,'not_started',dtype='<U32'),
        gpu_features=np.full((count,52),np.nan),gpu_target=np.full(count,np.nan),
        gpu_ratios=np.full((count,2),np.nan))


def timing_evidence(plan,stage,hard_cap=None):
    """Summarize recorded times; never manufacture unfinished stage durations."""
    def finite(value):
        return float(value) if isinstance(value,(int,float)) and np.isfinite(value) else None
    elapsed=finite(stage.get('elapsed_seconds'))
    cap=float(plan['compute_cap_seconds'])
    reserve_begin=cap-float(plan['verification_reserve_seconds'])
    stages=stage.get('stages',{})
    durations=[finite(value) for value in stages.values()]
    total=sum(durations) if all(value is not None for value in durations) else None
    search_end=None
    if all(key in stages for key in ('preprocessing_seconds','search_seconds')):
        values=[finite(stages[key]) for key in ('preprocessing_seconds','search_seconds')]
        if all(value is not None for value in values):search_end=sum(values)
    return dict(declared_cap_seconds=cap,verification_reserve_seconds=float(plan['verification_reserve_seconds']),
        recorded_elapsed_seconds=elapsed,completed_stage_seconds=total,
        recorded_cap_excess_seconds=None if elapsed is None else max(0.,elapsed-cap),
        approximate_verification_start_seconds=search_end,
        verification_start_after_reserve_seconds=None if search_end is None else max(0.,search_end-reserve_begin),
        hard_cap_marker_present=hard_cap is not None,
        scope='Saved elapsed and sequential-stage metadata; unfinished stages and timer scheduling delays are not reconstructed')


def collect_run(out):
    """Reconstruct all planned cases, including interrupted in-flight batches.

    A checkpoint angle is retained as a partial search state. Its absent endpoint
    measurements remain NaN and unverified; they are never recomputed here.
    """
    out=Path(out)
    plan=read(out/'plan.json')
    if plan is None:
        raise FileNotFoundError('the frozen plan is required to report a run')
    stage=read(out/'stage_audit.json',dict(status='unknown_missing_stage'))
    result=read(out/'result.json')
    aggregate=read(out/'cases.json',dict(records=[]))
    fallback={case_name(v):v for v in aggregate.get('records',[])}
    if result:
        fallback.update({case_name(v):v for v in result.get('cases',[])})
    refs=load_npz(out/'reference_features.npz')
    reference_map={} if refs is None else {int(tile):features for tile,features in zip(refs['tile_ids'],refs['features'])}
    entries=[]
    terminal=stage.get('status')!='running'
    for expected in plan['cases']:
        name=case_name(expected)
        record=read(out/'cases'/f'{name}.json',fallback.get(name))
        arrays=load_npz(out/'cases'/f'{name}.npz')
        present=arrays is not None
        arrays=_empty_arrays(plan['starts_per_case'],plan['planes']) if arrays is None else arrays
        arrays={key:np.array(value,copy=True) for key,value in arrays.items()}
        checkpoint=load_npz(out/'checkpoints'/f'{name}.npz')
        checkpoint_meta=read(out/'checkpoints'/f'{name}.json')
        overlay=np.zeros(plan['starts_per_case'],dtype=bool)
        checkpoint_error=None
        reference_target=None
        if checkpoint is not None and checkpoint_meta is not None:
            first=checkpoint_meta.get('first_start')
            rows=len(checkpoint.get('theta',[]))
            if not isinstance(first,int) or rows<1 or first<0 or first+rows>plan['starts_per_case']:
                checkpoint_error='checkpoint row range is invalid'
            else:
                ids=np.arange(first,first+rows)
                cp_iterations=checkpoint.get('iterations',np.zeros(rows,dtype=int))
                use=(arrays['started'][ids]==0)|(cp_iterations>=arrays['iterations'][ids])
                old_theta=arrays['theta'][ids].copy()
                changed=~np.all(np.isclose(old_theta,checkpoint['theta'],rtol=0,atol=0),axis=1)
                # If a checkpoint is newer than persisted GPU endpoints, retain
                # its state and erase stale endpoints rather than pairing them.
                newer=use&((arrays['started'][ids]==0)|changed)
                for key,value in checkpoint.items():
                    if key in arrays and value.ndim>=1 and len(value)==rows and arrays[key].shape[1:]==value.shape[1:]:
                        arrays[key][ids[use]]=value[use]
                arrays['started'][ids[use]]=1
                arrays['best_theta'][ids[newer]]=checkpoint['theta'][newer]
                for key in ('gpu_features','gpu_target','gpu_ratios'):
                    arrays[key][ids[newer]]=np.nan
                overlay[ids[newer]]=True
                future=checkpoint.get('reference_target',np.empty(0))
                if future.ndim==1 and len(future) and np.isfinite(future).all():
                    reference_target=float(future[0])
        raw_status=arrays['status'].copy()
        if terminal:
            arrays['status'][np.isin(arrays['status'],('active','initializing'))]='incomplete'
        certificate=read(out/'certificates'/f'{name}.json')
        failure=read(out/'certificates'/f'{name}_failure.json')
        if certificate is None:
            certificate=failure
        endpoints=load_npz(out/'certificates'/f'{name}_endpoints.npz')
        if certificate is None and result:
            certificate=next((v for v in result.get('certificates',[]) if v.get('case') and case_name(v['case'])==name),None)
        started=np.asarray(arrays['started'])>0
        completed=np.isin(arrays['status'],SUCCESS_STATES)&started
        failed=np.isin(arrays['status'],FAILURE_STATES)&started
        finite=np.isfinite(arrays['gpu_features']).all(axis=1)&np.isfinite(arrays['gpu_target'])&np.isfinite(arrays['gpu_ratios']).all(axis=1)
        max_gap=None;gap_scope='no saved endpoint audit'
        if certificate is not None and 'maximum_discovered_feasible_gap' in certificate:
            max_gap=float(certificate['maximum_discovered_feasible_gap']);gap_scope='saved independent endpoint audit'
        elif reference_target is not None and refs is not None and int(expected['tile']) in reference_map:
            feasible=[]
            for index in np.flatnonzero(started&finite):
                values=arrays['gpu_features'][index]
                matches=matching_bins(values,reference_map[int(expected['tile'])],refs['scales'],projection_indices(expected['representation']))['matched']
                if matches and np.max(arrays['gpu_ratios'][index])<=plan['rms_ratio_limit'] and np.max(np.abs(arrays['best_theta'][index]))<=plan['angle_limit']:
                    feasible.append(abs(float(arrays['gpu_target'][index])-reference_target))
            if feasible:
                max_gap=max(feasible);gap_scope='uncertified saved GPU endpoints'
        entries.append(dict(name=name,planned=expected,record=record,arrays=arrays,array_file_present=present,
            checkpoint=checkpoint,checkpoint_meta=checkpoint_meta,checkpoint_error=checkpoint_error,
            checkpoint_overlay=overlay,raw_status=raw_status,certificate=certificate,endpoints=endpoints,
            started=int(started.sum()),completed=int(completed.sum()),terminal_failures=int(failed.sum()),
            unverified_started=int((started&~finite).sum()),nonfinite_trial_count=int(arrays['nonfinite_trials'].sum()),
            maximum_gap=max_gap,gap_scope=gap_scope,reference_target=reference_target))
    matched=[]
    for tile in plan['selected_tiles']:
        group=[e for e in entries if e['planned']['tile']==tile and e['planned']['stage']=='fresh']
        good=(len(group)==3 and all(e['record'] is not None and e['record'].get('status')=='complete'
            and e['started']==e['completed']==plan['starts_per_case'] and e['terminal_failures']==0
            and e['certificate'] is not None and e['certificate']['status'] in ('verified_witness','no_verified_witness') for e in group))
        if good:
            matched.append(int(tile))
    statuses={}
    for entry in entries:
        for status,count in zip(*np.unique(entry['arrays']['status'],return_counts=True)):
            statuses[str(status)]=statuses.get(str(status),0)+int(count)
    verified=[e for e in entries if e['certificate'] and e['certificate']['status']=='verified_witness']
    summary=dict(scientific_status=stage.get('status','unknown'),result_present=result is not None,
        scientific_outcome=(result or {}).get('scientific_outcome','not_finalized'),
        planned_starts=plan['planned_starts'],starts_started=sum(e['started'] for e in entries),
        starts_completed=sum(e['completed'] for e in entries),terminal_failure_starts=sum(e['terminal_failures'] for e in entries),
        unverified_started_rows=sum(e['unverified_started'] for e in entries),verified_witnesses=len(verified),
        nonfinite_trial_count=sum(e['nonfinite_trial_count'] for e in entries),
        matched_complete_tiles=matched,status_counts=statuses,
        independently_audited_starts=sum(int(np.sum((e['arrays']['started']>0)&np.isfinite(e['endpoints']['features']).all(axis=1)))
            for e in entries if e['endpoints'] is not None),
        elapsed_seconds=stage.get('elapsed_seconds'),timing=timing_evidence(plan,stage,read(out/'hard_cap.json')),
        certificate_statuses={name:sum(bool(e['certificate']) and e['certificate']['status']==name for e in entries)
            for name in ('verified_witness','no_verified_witness','rejected_candidate','precision_failure')})
    return dict(plan=plan,stage=stage,result=result,entries=entries,reference=refs,summary=summary)


def saved_runtime_context(run,out):
    """Read coverage, failures and timer records without new scientific work."""
    entries,summary,stage=run['entries'],run['summary'],run['stage']
    errors=[];groups=[];initial_failures=0;directions={}
    for entry in entries:
        record=entry['record'] or {};error=record.get('error')
        if error and error.get('type')!='TimeoutError':
            errors.append(dict(name=entry['name'],type=error.get('type','unknown'),
                message=error.get('error','').strip().splitlines()[-1] if error.get('error') else 'No traceback retained'))
        arrays=entry['arrays'];started=arrays['started']>0
        if 'initial_retraction_succeeded' in arrays:
            initial_failures+=int(np.sum(started&~arrays['initial_retraction_succeeded']))
        if 'signs' in arrays:
            completed=np.isin(arrays['status'],SUCCESS_STATES)&started
            for label,mask in (('maximize',arrays['signs']>0),('minimize',arrays['signs']<0)):
                totals=directions.setdefault(label,dict(started=0,completed=0))
                totals['started']+=int(np.sum(started&mask));totals['completed']+=int(np.sum(completed&mask))
    for phase,rep in CASE_LABELS:
        group=[e for e in entries if e['planned']['stage']==phase and e['planned']['representation']==rep]
        groups.append(dict(stage=phase,representation=rep,planned_cases=len(group),
            started_cases=sum(e['started']>0 for e in group),started=sum(e['started'] for e in group),
            completed=sum(e['completed'] for e in group)))
    certs=[entry['certificate'] for entry in entries if entry['certificate']]
    errors_saved=[cert['independent_feature_error'] for cert in certs if 'independent_feature_error' in cert]
    gaps=[entry['maximum_gap'] for entry in entries if entry['maximum_gap'] is not None]
    notes=read(Path(out)/'runtime_notes.json',dict(notes=[])).get('notes',[])
    log_path=Path(out)/'run.log'
    log=log_path.read_text(encoding='utf-8',errors='replace') if log_path.exists() else ''
    utc_difference=None
    if stage.get('started_utc') and stage.get('completed_utc'):
        utc_difference=(datetime.fromisoformat(stage['completed_utc'])-datetime.fromisoformat(stage['started_utc'])).total_seconds()
    elapsed=stage.get('elapsed_seconds')
    return dict(started_cases=sum(e['started']>0 for e in entries),planned_cases=len(entries),
        unstarted_cases=sum(e['started']==0 for e in entries),
        interrupted_starts=summary['starts_started']-summary['starts_completed']-summary['terminal_failure_starts'],
        case_errors=errors,groups=groups,directions=directions,initial_retraction_failures=initial_failures,
        maximum_endpoint_error=max(errors_saved) if errors_saved else None,
        maximum_saved_gap=max(gaps) if gaps else None,
        provisional_candidates=sum(cert.get('provisional_candidates',0) for cert in certs),
        direct_replay_candidates=sum(bool(cert.get('arrays')) for cert in certs),runtime_notes=notes,
        svd_fallback_warnings=log.count('UserWarning: torch.linalg.svd:'),
        utc_interval_seconds=utc_difference,
        elapsed_minus_utc_seconds=None if elapsed is None or utc_difference is None else elapsed-utc_difference)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=OUT)
    parser.add_argument('--report',type=Path,default=ROOT/'docs/CLOSURE_FIBERS_RUN.md')
    args=parser.parse_args();out=args.run_dir
    run=collect_run(out);plan,stage,entries,summary=[run[k] for k in ('plan','stage','entries','summary')]
    context=saved_runtime_context(run,out)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    tiles=plan['selected_tiles'];labels=['Source\nbaseline','Fresh\nbaseline','Fresh\npower control','Fresh\nCST repair']
    complete=np.zeros((len(tiles),4));started=np.zeros_like(complete);gaps=np.full_like(complete,np.nan)
    lookup={(e['planned']['tile'],e['planned']['stage'],e['planned']['representation']):e for e in entries}
    for row,tile in enumerate(tiles):
        for col,(phase,representation) in enumerate(CASE_LABELS):
            e=lookup[(tile,phase,representation)]
            complete[row,col]=e['completed'];started[row,col]=e['started']
            if e['maximum_gap'] is not None:
                gaps[row,col]=e['maximum_gap']
    fig,axes=plt.subplots(1,3,figsize=(16,8))
    fig.subplots_adjust(left=.055,right=.96,top=.73,bottom=.24,wspace=.38)
    fig.text(.055,.94,'CST smoothing study | recorded search evidence',fontsize=24,weight='bold',color='#25374d')
    fig.text(.055,.875,f"{summary['scientific_status']} · {summary['starts_completed']:,}/{summary['planned_starts']:,} completed starts · {summary['verified_witnesses']} verified witnesses",fontsize=12)
    fig.text(.055,.825,'Discovered gaps are lower bounds from a finite search, never worst-case upper bounds or proofs of closure.',fontsize=10)
    im=axes[0].imshow(complete/plan['starts_per_case'],vmin=0,vmax=1,cmap='Blues',aspect='auto')
    for i in range(len(tiles)):
        for j in range(4):
            axes[0].text(j,i,f'{int(started[i,j])}/{int(complete[i,j])}',ha='center',va='center',fontsize=8,color='white' if complete[i,j]>96 else '#25374d')
    axes[0].set(title='Started / completed\n(each case has 128 starts)',yticks=range(len(tiles)),yticklabels=tiles,ylabel='Fixed sky tile',xticks=range(4),xticklabels=labels)
    color=plt.get_cmap('viridis').copy();color.set_bad('#e5e7eb')
    maximum=max(.05,float(np.nanmax(gaps)) if np.isfinite(gaps).any() else .05)
    image=axes[1].imshow(gaps,vmin=0,vmax=maximum,cmap=color,aspect='auto')
    for i in range(len(tiles)):
        for j in range(4):
            axes[1].text(j,i,'—' if not np.isfinite(gaps[i,j]) else f'{gaps[i,j]:.3g}',ha='center',va='center',fontsize=8,color='white' if np.isfinite(gaps[i,j]) and gaps[i,j]<maximum*.6 else '#25374d')
    axes[1].set(title='Largest saved feasible\nendpoint gap',yticks=range(len(tiles)),yticklabels=tiles,xticks=range(4),xticklabels=labels)
    fig.colorbar(image,ax=axes[1],shrink=.7,pad=.025,label='|Future K4 − reference K4|')
    categories=(('iteration cap','#087f8c',lambda s:s=='iteration_cap'),('stalled','#64a6a6',lambda s:s=='stalled'),
        ('terminal failure','#b94e63',lambda s:s in FAILURE_STATES),('incomplete','#e5a83b',lambda s:s in ('incomplete','active','initializing')),
        ('not started','#d9dee7',lambda s:s=='not_started'))
    bottom=np.zeros(4)
    for name,color,predicate in categories:
        values=np.array([sum(sum(predicate(str(s)) for s in e['arrays']['status']) for e in entries if (e['planned']['stage'],e['planned']['representation'])==case) for case in CASE_LABELS])
        axes[2].bar(np.arange(4),values,bottom=bottom,label=name,color=color)
        bottom+=values
    axes[2].set(title='All planned starts\nincluding interruptions',ylabel='Starts',xticks=range(4),xticklabels=labels)
    axes[2].legend(frameon=False,fontsize=8)
    for ax in axes:
        ax.tick_params(axis='x',labelsize=8)
    fig.text(.055,.095,'Missing gap cells have no saved endpoint estimate. Checkpoint-only starts remain scientifically unverified.',fontsize=10)
    fig.text(.055,.052,'Counterfactual finite-precision sufficiency test; no cosmological, fractal, priority or detector-superiority claim.',fontsize=10)
    fig.savefig(out/'search_overview.png',dpi=160,facecolor='white');plt.close(fig)

    source=[e for e in entries if e['planned']['stage']=='source' and e['certificate'] and e['certificate']['status']=='verified_witness']
    fig,axes=plt.subplots(1,2,figsize=(13,5.5));fig.subplots_adjust(left=.085,right=.96,top=.8,bottom=.22,wspace=.3)
    fig.text(.085,.93,'Repair evidence and fresh matched searches',fontsize=22,weight='bold',color='#25374d')
    if source:
        separated=[sum(e['certificate'].get('distinguished_by',{}).get(name,False) for e in source) for name in ('capacity','repair')]
        axes[0].bar(['Power-response control','CST K4 response'],separated,color=['#e08b2c','#087f8c'])
        axes[0].set(title=f'Distinguishes saved source witnesses (n={len(source)})',ylabel='Witnesses distinguished',ylim=(0,len(source)+.5))
    else:
        axes[0].axis('off');axes[0].text(.5,.5,'No verified source witness\nNo repair-separation claim',ha='center',va='center',fontsize=15)
    eligible=summary['matched_complete_tiles']
    if eligible:
        for j,representation in enumerate(('baseline','capacity','repair')):
            values=[lookup[(tile,'fresh',representation)]['maximum_gap'] for tile in eligible]
            axes[1].plot(np.arange(len(eligible)),values,'o-',label=representation)
        axes[1].axhline(.05,color='#64748b',ls='--',lw=1,label='Declared witness gap')
        axes[1].set(title='Complete equal-budget fresh groups only',xticks=range(len(eligible)),xticklabels=eligible,xlabel='Sky tile',ylabel='Largest discovered feasible gap')
        axes[1].legend(frameon=False,fontsize=8)
    else:
        axes[1].axis('off');axes[1].text(.5,.5,'No complete matched fresh group\nNo between-representation comparison',ha='center',va='center',fontsize=14)
    fig.text(.085,.075,'Distinguishing an old witness is not universal repair. A fresh search finding no witness is inconclusive.',fontsize=10)
    fig.savefig(out/'repair_comparison.png',dpi=160,facecolor='white');plt.close(fig)

    telemetry=read(out/'telemetry.json',dict(samples=[]))
    fig,axes=plt.subplots(2,2,figsize=(12,8));fig.subplots_adjust(left=.085,right=.96,top=.81,bottom=.13,wspace=.3,hspace=.38)
    fig.text(.085,.94,'Recorded device counters and stage timings',fontsize=22,weight='bold',color='#25374d')
    fig.text(.085,.87,'Device-wide counters may include other applications; no saturation or CPU/GPU speedup claim.',fontsize=10)
    measured=[]
    for index,color in ((0,'#087f8c'),(1,'#e08b2c')):
        data=[(s['seconds'],d) for s in telemetry.get('samples',[]) for d in s.get('devices',[]) if d['index']==index]
        if not data:
            measured.append(dict(device=index,available=False));continue
        seconds=np.array([s[0] for s in data]);rows=[s[1] for s in data]
        label=plan['devices'][index].replace('NVIDIA GeForce ','')
        for ax,key in ((axes[0,0],'utilization'),(axes[0,1],'power_W'),(axes[1,0],'memory_MiB')):
            ax.plot(seconds,[v[key] for v in rows],color=color,label=label,lw=.8)
        measured.append(dict(device=index,name=label,available=True,samples=len(rows),
            whole_run_mean_busy=float(np.mean([v['utilization'] for v in rows])),peak_busy=max(v['utilization'] for v in rows),
            mean_power_W=float(np.mean([v['power_W'] for v in rows])),peak_memory_MiB=max(v['memory_MiB'] for v in rows),peak_temperature_C=max(v['temperature_C'] for v in rows)))
    stages=stage.get('stages',{})
    names=list(stages);values=[float(stages[name]) for name in names]
    if names:
        axes[1,1].barh(np.arange(len(names)),values,color='#64748b')
        axes[1,1].set(yticks=range(len(names)),yticklabels=[v.replace('_seconds','').replace('_',' ') for v in names],xlabel='Recorded wall seconds',title='Available stage timings')
    else:
        axes[1,1].axis('off');axes[1,1].text(.5,.5,'No completed stage timings retained',ha='center',va='center')
    boundary=0.
    for name in ('preprocessing_seconds','search_seconds','scientific_verification_seconds'):
        if name not in stages:
            break
        end=boundary+float(stages[name])
        if any(v['available'] for v in measured):
            for ax in (axes[0,0],axes[0,1],axes[1,0]):
                ax.axvline(end,color='#64748b',alpha=.45,ls='--',lw=.8)
        boundary=end
    axes[0,0].set(title='Sampled GPU busy',xlabel='Seconds from study start',ylabel='Busy (%)',ylim=(0,105))
    if any(v['available'] for v in measured):
        axes[0,0].legend(frameon=False,fontsize=8)
    axes[0,1].set(title='Sampled power',xlabel='Seconds from study start',ylabel='Watts')
    axes[1,0].set(title='Sampled device memory',xlabel='Seconds from study start',ylabel='MiB')
    if not any(v['available'] for v in measured):
        for ax in (axes[0,0],axes[0,1],axes[1,0]):
            ax.axis('off');ax.text(.5,.5,'No retained device counters',ha='center',va='center',fontsize=13)
    fig.text(.085,.045,'Dashed boundaries approximate recorded sequential stage durations; unavailable phases are not reconstructed.',fontsize=9)
    fig.savefig(out/'hardware.png',dpi=160,facecolor='white');plt.close(fig)

    tests=pytest_summary(ROOT/'data/closure_tests.log')
    sandbox=ROOT/'data/closure_sandbox_tests.log'
    sandbox_count=0 if not sandbox.exists() else len(re.findall(r'^\[PASS\]\s+(\S+)',sandbox.read_text(errors='replace'),re.MULTILINE))
    summary['hardware']=measured
    phase_summary=[];begin=0.
    for phase in ('preprocessing_seconds','search_seconds','scientific_verification_seconds'):
        if phase not in stages:break
        end=begin+float(stages[phase])
        for index in (0,1):
            counters=[d for sample in telemetry.get('samples',[]) if begin<=sample['seconds']<end
                for d in sample.get('devices',[]) if d['index']==index]
            if counters:
                phase_summary.append(dict(phase=phase.replace('_seconds',''),device=index,
                    approximate_start_seconds=begin,approximate_end_seconds=end,samples=len(counters),
                    mean_busy=float(np.mean([d['utilization'] for d in counters])),
                    mean_power_W=float(np.mean([d['power_W'] for d in counters])),
                    peak_memory_MiB=max(d['memory_MiB'] for d in counters)))
        begin=end
    summary['approximate_phase_hardware']=phase_summary
    summary['case_summary']=[dict(name=e['name'],started=e['started'],completed=e['completed'],terminal_failures=e['terminal_failures'],
        unverified_started=e['unverified_started'],largest_discovered_feasible_gap=e['maximum_gap'],gap_scope=e['gap_scope'],
        search_status=(e['record'] or {}).get('status','no_final_case_record'),certificate_status=(e['certificate'] or {}).get('status','not_verified')) for e in entries]
    (out/'report_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    rows='\n'.join(f"| {e['planned']['tile']} | {e['planned']['stage']} / {e['planned']['representation']} | {e['started']} / {e['completed']} | {e['terminal_failures']} | {e['unverified_started']} | {'—' if e['maximum_gap'] is None else format(e['maximum_gap'],'.6g')} | {(e['certificate'] or {}).get('status','not_verified')} |" for e in entries)
    source_rows='\n'.join(f"| {e['planned']['tile']} | {e['certificate']['direct_gap']:.6g} | {e['certificate']['distinguished_by']['capacity']} | {e['certificate']['distinguished_by']['repair']} |" for e in source)
    if not source_rows:
        source_rows='| — | No verified source witness | — | — |'
    witness_rows=[];trajectory_rows=[]
    for entry in entries:
        cert=entry['certificate'] or {}
        if not cert.get('arrays'):continue
        saved=load_npz(out/cert['arrays'])
        if saved is None:continue
        label=f"{entry['planned']['tile']} / {entry['planned']['stage']} / {entry['planned']['representation']}"
        invariants=cert.get('invariant_checks',[])
        spectral=max((v['normalized_spectral_gram_error'] for v in invariants),default=float('nan'))
        split=max((v['split_difference_scaled_error'] for v in invariants),default=float('nan'))
        ratios=cert.get('perturbation_ratios',[float('nan'),float('nan')])
        evidence=f"[certificate](../results/closure_fibers_v1/certificates/{entry['name']}.json), [summaries/angles/bins](../results/closure_fibers_v1/{cert['arrays']}), [harmonics](../results/closure_fibers_v1/{cert['coefficients']})"
        witness_rows.append(f"| {label} | {cert['status']} | {cert['current_minimum_margin']:.6g} | {spectral:.3g} | {split:.3g} | {cert['direct_sample_feature_error']:.3g} | {ratios[0]:.6g} / {ratios[1]:.6g} | {evidence} |")
        ref,can=saved['reference_features'],saved['candidate_features']
        pairs=[f"{ref[u,22]:.7g} / {can[u,22]:.7g}" for u in range(4)]
        trajectory_rows.append(f"| {label} | {' | '.join(pairs)} | {cert['direct_gap']:.7g} |")
    witness_rows='\n'.join(witness_rows) or '| — | No directly replayed candidate | — | — | — | — | — | — |'
    trajectory_rows='\n'.join(trajectory_rows) or '| — | No directly replayed candidate | — | — | — | — |'
    timing='\n'.join(f"| {k.replace('_',' ')} | {v:.3f} |" for k,v in stages.items()) or '| — | No completed stage timing |'
    hardware_rows='\n'.join(f"| {v['phase'].replace('_',' ')} | {v['device']} | {v['samples']} | {v['mean_busy']:.2f} | {v['mean_power_W']:.2f} | {v['peak_memory_MiB']:.0f} |" for v in phase_summary) or '| — | — | No retained phase counters | — | — | — |'
    elapsed='not finalized' if summary['elapsed_seconds'] is None else f"{summary['elapsed_seconds']:.2f} seconds"
    coverage_rows='\n'.join(f"| {v['stage']} / {v['representation']} | {v['started_cases']} / {v['planned_cases']} | {v['started']} | {v['completed']} |" for v in context['groups'])
    error_rows='\n'.join(f"| {v['name']} | {v['type']} | {v['message'].replace('|','/')} | [case record](../results/closure_fibers_v1/cases/{v['name']}.json), [saved starts](../results/closure_fibers_v1/cases/{v['name']}.npz) |" for v in context['case_errors']) or '| — | No non-timeout case error recorded | — | — |'
    runtime_rows=[]
    for note in context['runtime_notes']:
        if all(key in note for key in ('case','start_id','maximum_saved_scaled_change','inner_gate','excess_over_inner_gate','declared_bin_boundary_margin','required_certificate_margin')):
            runtime_rows.append(f"| {note['case']} / {note['start_id']} | {note['maximum_saved_scaled_change']:.17g} | {note['inner_gate']:.17g} | {note['excess_over_inner_gate']:.17g} | {note['declared_bin_boundary_margin']:.17g} | {note['required_certificate_margin']:.17g} |")
    runtime_rows='\n'.join(runtime_rows) or '| — | No saved inner-gate arithmetic note | — | — | — | — |'
    direction_text='; '.join(f"{name}: {counts['started']} started / {counts['completed']} completed" for name,counts in context['directions'].items()) or 'No direction counts saved'
    unstarted_repairs=[v['representation'] for v in context['groups'] if v['representation'] in ('capacity','repair') and v['started']==0]
    repair_scope=(f"The {', '.join(unstarted_repairs)} searches were entirely unstarted. Their resistance to fresh searches was not tested, so no comparison or repair-performance conclusion is available." if unstarted_repairs else
        'Fresh representation comparisons remain restricted to complete matched groups below.')
    audit_error='not available' if context['maximum_endpoint_error'] is None else f"{context['maximum_endpoint_error']:.17g}"
    max_gap='not available' if context['maximum_saved_gap'] is None else f"{context['maximum_saved_gap']:.17g}"
    replay_scope=('No saved certificate records a provisional candidate. No production full-harmonic direct replay or production invariant certificate is recorded. The separately recorded real-operator preflight verifies only its fixed-angle controls.'
        if context['provisional_candidates']==context['direct_replay_candidates']==0 else
        f"Saved provisional candidates: {context['provisional_candidates']}; directly replayed selected candidates: {context['direct_replay_candidates']}. Only explicit passing certificates support witness claims.")
    svd_note=(f"The [preserved run log](../results/closure_fibers_v1/run.log) contains {context['svd_fallback_warnings']} SVD convergence warning(s): the initially selected CUDA driver did not converge for a batch and PyTorch announced a more accurate fallback. The warning does not state that the fallback failed."
        if context['svd_fallback_warnings'] else 'No SVD fallback warning was found in the saved run log.')
    report=f'''# CST information-loss search under angular smoothing

**Scientific status: {summary['scientific_status']}. Outcome: {summary['scientific_outcome']}.**
Completed starts: **{summary['starts_completed']:,}/{summary['planned_starts']:,}**;
started: {summary['starts_started']:,}; terminal nonfinite failure starts: {summary['terminal_failure_starts']};
started rows without retained GPU endpoints: {summary['unverified_started_rows']}.
Separately recorded non-timeout optimizer/execution case failures: **{len(context['case_errors'])}**.
Independently audited saved endpoints: {summary['independently_audited_starts']}.
Saved verified witnesses: **{summary['verified_witnesses']}**. Elapsed study time: {elapsed}.
The result file is {'present' if summary['result_present'] else 'absent; conclusions are limited to retained partial evidence'}.

![All case outcomes](../results/closure_fibers_v1/search_overview.png)

## Actual coverage and limits

Searches began in **{context['started_cases']}/{context['planned_cases']} cases**;
{context['unstarted_cases']} cases never started. Of {summary['starts_started']}
started rows, {summary['starts_completed']} completed the solver stopping rule
and {context['interrupted_starts']} remained interrupted. Completed status counts:
{summary['status_counts'].get('stalled',0)} stalled and
{summary['status_counts'].get('iteration_cap',0)} reached the iteration cap.
Initial retraction failures among started rows: {context['initial_retraction_failures']}.
Directions: {direction_text}. The fixed order was truncated by the budget;
these unequal direction counts cannot be treated as a balanced completed search.

| Planned search | Started / planned cases | Starts begun | Starts completed |
|---|---:|---:|---:|
{coverage_rows}

{repair_scope}
The largest saved feasible endpoint gap was **{max_gap}**, against the fixed
witness threshold **{plan['witness_gap']}**. It is a finite-search discovered
lower bound, not an upper bound on undiscovered gaps or evidence of closure.
Independent CPU covariance/derivative and stored GPU endpoint audits cover
{summary['independently_audited_starts']} saved starts; their maximum recorded
scaled discrepancy was {audit_error} (acceptance gate {plan['numerical_gate']}).
{replay_scope}

## What the experiment asks

The supplied CST paper §8.1 says that a reduced deterministic operator can exist
only if equal representations have equal representations after the same transition.
Here the transition is declared angular smoothing, not cosmic or spacecraft time.
The states are bounded counterfactuals built from the existing real 143 GHz A/B
harmonics. They are not newly observed skies or a foreground/instrument model.

The [frozen protocol](CLOSURE_FIBERS_SPEC.md) fixes 12 tiles and 6,144 starts.
Each tile has a source-baseline search and fresh baseline, power-control and CST-
repair searches, with 128 starts each. The orthogonal-plane construction is
designed to preserve global E/B auto/cross spectra and the full A−B difference
at each common smoothing time; preflight controls verify those identities.
An independently replayed production invariant certificate is conditional on a
selected witness candidate, as disclosed above. Throughout the search, accepted
states must keep perturbation RMS within 25% of the reference coadd globally
and within the tile, with plane angles bounded by ±π/4.

The baseline records 32 transported increment statistics and 12 local band powers.
The power control adds four derivatives of mean increment power; the CST repair
instead adds four derivatives of connected fourth-order structure. Reference-
centered local bins have width 0.001 robust-scale units, with a 1e-7 margin;
normalized global spectral bins have width 1e-6. A witness requires matching
current bins and a future 2° K4 gap ≥ 0.05, verified on independent CPU, both
actual GPUs and direct harmonic-sampling routes inside the scientific budget.

These are **finite-precision implemented representations**. A witness does not
prove nonclosure of every exact continuous summary, establish new cosmology,
measure a fractal dimension or establish detector superiority. No witness found
means only that this finite search found none; its maximum is a discovered lower
bound, never a certified worst-case upper bound or uniform error bound under §8.2.

## All planned search cases

The status plot includes stalls, iteration caps, interruptions and unstarted
rows. Missing values are not replaced with zero discoveries. Endpoint gaps may
be uncertified when the run ended before scientific verification; their scope
is retained in [report_summary.json](../results/closure_fibers_v1/report_summary.json).
Recorded nonfinite trial events: {summary['nonfinite_trial_count']}; rejected
proposals and terminal numerical failures remain separately recorded.

| Tile | Search | Started / completed | Terminal failures | Unverified started | Largest saved feasible gap | Certificate |
|---:|---|---:|---:|---:|---:|---|
{rows}

## Preserved execution failures and numerical warnings

Case-level exceptions are separate from the terminal nonfinite-start counter.
Timeouts at the reserved verification boundary are reported as interruptions;
the following table lists other preserved case exceptions.

| Case | Exception type | Recorded final error | Evidence |
|---|---|---|---|
{error_rows}

The [runtime arithmetic notes](../results/closure_fibers_v1/runtime_notes.json)
retain the following comparisons of saved values with the stricter optimizer
inner gate. These are metadata calculations, not new feature recomputations.

| Case / start | Maximum saved scaled change | Optimizer inner gate | Excess over inner gate | Bin-boundary margin | Required certificate margin |
|---|---:|---:|---:|---:|---:|
{runtime_rows}

A tiny excess over the optimizer inner gate while the saved certificate margin
remains safe is consistent with roundoff at that stricter boundary. This is a
**provisional diagnosis** of the preserved failure. It neither overrides the
frozen solver decision nor certifies a witness; no source, seed or tolerance was
changed to turn the failure into a pass.

{svd_note}

## Does the proposed repair distinguish the examples?

![Source separation and fresh searches](../results/closure_fibers_v1/repair_comparison.png)

Only independently verified source witnesses enter the following table.
Distinguishing one known example does not establish universal repair.

| Source tile | Verified future K4 gap | Power control distinguishes | CST repair distinguishes |
|---:|---:|---|---|
{source_rows}

Fresh comparisons are restricted to complete equal-budget triples with saved
scientific endpoint audits. Eligible tiles: {summary['matched_complete_tiles']}.
All other groups remain visible but are not used to compare representations.
Failure to find a fresh collision is resistance to this particular search,
not a proof of sufficiency or a detection-power advantage.

## Directly replayed candidates and numerical margins

Every directly replayed selected candidate appears below, including rejected or
precision-failed candidates. Only rows labeled `verified_witness` support a
collision claim. The current margin is the smallest distance to a recorded local
bin boundary in robust-scale units (required ≥ 1e-7); spectral, split-difference
and direct-sampling errors use the declared scaled numerical gate 1e-10.
RMS columns are global / tile perturbation ratios (each required ≤ 0.25).
Linked arrays retain all 52 before/after summaries, response derivatives,
rotation parameters, bin IDs and both actual GPU replays. Certificates retain
the future changed-coordinate safety decision and detailed invariant checks.

| Tile / search / representation | Verification status | Current minimum margin | Maximum spectral error | Maximum split-difference error | Direct feature error | RMS ratios | Saved evidence |
|---|---|---:|---:|---:|---:|---:|---|
{witness_rows}

The following values are the primary 2° K4 summary, with each cell showing
**reference / candidate** after the indicated identical smoothing. Steps 2/3
are descriptive trajectories of the selected pair, not additional searches.

| Tile / search / representation | u=0 | u=1 | u=2 | u=3 | Absolute u=1 gap |
|---|---:|---:|---:|---:|---:|
{trajectory_rows}

## Verification, proofs and provenance

The publication audit reads saved evidence and performs hash, count, bin, margin,
bound and certificate-consistency arithmetic only. It does not run new physical
transforms, feature extraction, optimization or direct scientific checks after
the four-hour cap. Its metadata-integrity status is separate from the scientific
status above. [Verification](../results/closure_fibers_v1/verification.json) authenticates
all current JoVin receipts and current source/test bindings, plus preservation of
all earlier frozen artifacts. SHA256 identifies bytes, not physical truth.

Local regression log: {tests['passed']} passed, {tests['skipped']} skipped,
{tests['failed']} failed and {tests['errors']} errors; available: {tests['available']}.
Saved plain-CPU sandbox passes: {sandbox_count}. JoVin validates the shipped
CPU bundle and separately recorded actual CUDA fixtures; the sandbox does not
execute CUDA or the full scientific search.

The actual-device profile is `{plan.get('preflight_path','data/evidence/closure_gpu_preflight_v2.json')}`,
SHA256 `{plan.get('preflight_sha256','not recorded')}`. The frozen closure JoVin
receipt is `{plan.get('jovin_receipt_id','not recorded')}`, covering bundle SHA256
`{plan.get('jovin_bundle_sha256','not recorded')}`. Input and source SHA256 maps,
software versions, stored numerical rotation bases and initialization manifests
are bound in [plan.json](../results/closure_fibers_v1/plan.json). The supporting
proof receipt is `data/evidence/closure_lean.json`, SHA256 `{plan.get('proof_sha256','not recorded')}`.
Frozen source revision: `{plan.get('source_revision',plan.get('base_revision','not recorded'))}`.

Seven new pinned Lean kernel checks cover the conditional no-factor criterion
and an exact rational toy counterexample. Together with the earlier 11 identities,
there are 18 supporting theorems. They do not prove Float64 correctness, successful
optimization, physical sky behavior, novelty or universal closure of the repair.
Development preflight/source archives and failed checks are retained; earlier
results and their frozen numerical modules are not rewritten.

## Time and hardware evidence

![Recorded counters](../results/closure_fibers_v1/hardware.png)

| Available stage | Wall seconds |
|---|---:|
{timing}

Approximate phases below use the same cumulative completed-stage boundaries as
the figure. Counters remain actual saved device-wide samples; phase boundaries
are approximate and incomplete stages are not reconstructed.

| Approximate phase | GPU | Samples | Mean busy (%) | Mean power (W) | Peak memory (MiB) |
|---|---:|---:|---:|---:|---:|
{hardware_rows}

The scientific wall cap is 14,400 seconds, including preprocessing and scientific
verification; new batches stop at 13,500 seconds to reserve 900 seconds. Publication
formatting and metadata-only authentication are outside that cap. Device-wide
counters may include other applications. Recorded activity is not a percentage
of theoretical FLOPS and establishes neither saturation nor CPU/GPU speedup.
Saved timing audit: completed-stage sum {summary['timing']['completed_stage_seconds']}
seconds; recorded elapsed beyond the declared cap
{summary['timing']['recorded_cap_excess_seconds']} seconds; hard-cap marker
present: {summary['timing']['hard_cap_marker_present']}. The approximate recorded
verification start was {summary['timing']['approximate_verification_start_seconds']}
seconds, with {summary['timing']['verification_start_after_reserve_seconds']}
seconds beyond the planned reserve boundary. Missing timing values mean an
unfinished phase was not recorded; scheduling delays are not normalized away.
The recorded UTC interval is `{stage.get('started_utc','not recorded')}` to
`{stage.get('completed_utc','not recorded')}`, a difference of
{context['utc_interval_seconds']} seconds. The saved `perf_counter` elapsed
value is {summary['elapsed_seconds']} seconds, differing from the UTC interval
by {context['elapsed_minus_utc_seconds']} seconds (`perf_counter` minus UTC).
Both records are preserved. This report does not infer a cause for that discrepancy.

## What is known and what remains open

- **Proved:** the conditional mathematical implication and the finite rational toy.
- **Measured:** only saved, explicitly certified counterfactual witnesses and recorded search outcomes.
- **Inferred:** a verified collision rules out an autonomous factor map for this declared recorded representation on a domain containing that pair.
- **Unknown:** closure outside the tested domain, universal repair, physical origin, primordial structure and general superiority.

The supplied paper is referenced by its existing audited package hash and section
numbers. It is not redistributed or assigned new attribution by this work.
'''
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(report,encoding='utf-8')
    print(json.dumps({key:summary[key] for key in ('scientific_status','scientific_outcome','planned_starts',
        'starts_started','starts_completed','verified_witnesses','terminal_failure_starts','unverified_started_rows',
        'independently_audited_starts','matched_complete_tiles')},indent=2,allow_nan=False))


if __name__=='__main__':
    main()
