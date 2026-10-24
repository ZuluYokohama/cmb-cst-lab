"""Independent finite-precision certificates, never inferred from optimizer success."""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
from .closure_fibers import features_numpy,ratios_numpy,projection_indices,quantize
from .closure_fibers_oracle import oracle_features
from .closure_fibers_data import replay_candidate,harmonic_layout,SIGMA,invariants,sha
from .closure_search import matching_bins,GUARD,QUANTUM


def scaled_error(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return float('inf')
    return float(np.max(np.abs(a-b)/(1+np.abs(b))))


def _json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def _npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,**arrays)
    temporary.replace(path)


def bin_separation(features,reference,scales,representation):
    """Different bins and safe evidence are distinct; an unsafe zero bin is not separation."""
    ix=projection_indices(representation)
    normalized=(np.asarray(features)[ix]-np.asarray(reference)[ix])/np.asarray(scales)[ix]
    bins=quantize(features,reference,scales,representation)
    margin=QUANTUM/2-np.abs(normalized-QUANTUM*bins)
    return dict(bins_differ=bool(np.any(bins!=0)),robust_changed_coordinate=bool(np.any((bins!=0)&(margin>=GUARD))),
        bins=bins.tolist(),nearest_boundary_margins=margin.tolist())


def certify_case(root,out,work,case,family,shared,scales,archive,budget=None):
    """Audit all stored endpoints, then direct-replay the largest-gap candidate.

    A precision failure in the chosen candidate is retained; no runner-up is
    substituted to obtain a passing witness. No search maximum is an upper bound.
    """
    started=time.perf_counter();root,out,work=Path(root),Path(out),Path(work)
    check=lambda:budget() if budget is not None else None
    check()
    theta=np.asarray(archive['best_theta'])
    initial=np.asarray(archive['initial_theta'])
    finite_rows=np.isfinite(theta).all(axis=1)&(np.asarray(archive['started'])>0)
    ids=np.flatnonzero(finite_rows)
    reference=features_numpy(family,np.zeros(theta.shape[1]))
    reference_oracle=oracle_features(family,np.zeros(theta.shape[1]))
    error=scaled_error(reference,reference_oracle)
    if error>1e-10:
        raise FloatingPointError('independent reference covariance disagrees')
    reference_future=np.stack([oracle_features(family,np.zeros(theta.shape[1]),u) for u in range(4)])
    all_features=np.full((len(theta),52),np.nan)
    all_target=np.full(len(theta),np.nan)
    all_ratios=np.full((len(theta),2),np.nan)
    matches=np.zeros(len(theta),dtype=bool)
    for first in range(0,len(ids),4):
        check();take=ids[first:first+4]
        independent=oracle_features(family,theta[take])
        complex_route=features_numpy(family,theta[take])
        future=oracle_features(family,theta[take],1)
        error=max(error,scaled_error(independent,complex_route))
        if error>1e-10:
            raise FloatingPointError('independent endpoint covariance disagrees')
        ratios=ratios_numpy(family,theta[take])
        error=max(error,scaled_error(independent,np.asarray(archive['gpu_features'])[take]),
            scaled_error(future[:,22],np.asarray(archive['gpu_target'])[take]),
            scaled_error(ratios,np.asarray(archive['gpu_ratios'])[take]))
        if error>1e-10:
            raise FloatingPointError('stored actual GPU endpoint disagrees with independent CPU')
        all_features[take]=independent;all_target[take]=future[:,22];all_ratios[take]=ratios
        for index in take:
            match=matching_bins(all_features[index],reference,scales,projection_indices(case['representation']))
            matches[index]=match['matched'] and np.max(np.abs(theta[index]))<=np.pi/4 and np.max(all_ratios[index])<=.25
    gap=np.abs(all_target-reference_future[1,22])
    candidates=np.flatnonzero(matches&(gap>=.05))
    name=f"tile_{case['tile']}_{case['stage']}_{case['representation']}"
    directory=out/'certificates';directory.mkdir(exist_ok=True)
    _npz(directory/f'{name}_endpoints.npz',start_id=np.arange(len(theta)),started=archive['started'],
        features=all_features,target=all_target,ratios=all_ratios,matching=matches,gap=gap)
    max_gap=float(np.nanmax(gap[matches])) if np.any(matches) else 0.
    record=dict(status='no_verified_witness',case=case,starts_checked=len(ids),feasible_endpoints=int(matches.sum()),
        provisional_candidates=len(candidates),maximum_discovered_feasible_gap=max_gap,
        independent_feature_error=error,seconds=0.,
        note='Finite search maxima are discovered lower bounds, not certified worst-case upper bounds')
    if not len(candidates):
        record['seconds']=time.perf_counter()-started
        _json(directory/f'{name}.json',record)
        return record
    # Fixed largest-gap order, tie by start ID. Selection is not a discovery p-value.
    chosen=int(sorted(candidates,key=lambda i:(-gap[i],i))[0])
    check()
    replay=replay_candidate(root,case['tile'],work,theta[chosen],shared,budget)
    direct=replay['direct_family']
    direct_values=np.stack([oracle_features(direct,np.zeros(0),u) for u in range(4)])
    matrix_values=np.stack([oracle_features(family,theta[chosen],u) for u in range(4)])
    direct_error=scaled_error(direct_values,matrix_values)
    direct_complex=np.stack([features_numpy(direct,np.zeros(0),u) for u in range(4)])
    covariance_error=scaled_error(direct_values,direct_complex)
    direct_match=matching_bins(direct_values[0],reference,scales,projection_indices(case['representation']))
    future_ref_bins=quantize(reference_future[1],reference,scales,case['representation'])
    future_candidate_bins=quantize(direct_values[1],reference,scales,case['representation'])
    future_ix=projection_indices(case['representation'])
    normalized_ref=(reference_future[1,future_ix]-reference[future_ix])/scales[future_ix]
    normalized_candidate=(direct_values[1,future_ix]-reference[future_ix])/scales[future_ix]
    ref_margin=QUANTUM/2-np.abs(normalized_ref-QUANTUM*future_ref_bins)
    candidate_margin=QUANTUM/2-np.abs(normalized_candidate-QUANTUM*future_candidate_bins)
    safe_future_difference=bool(np.any((future_ref_bins!=future_candidate_bins)&(ref_margin>=GUARD)&(candidate_margin>=GUARD)))
    direct_gap=float(abs(direct_values[1,22]-reference_future[1,22]))
    check();ell,_=harmonic_layout();rate=.5*SIGMA**2*ell*(ell+1)
    time_checks=[invariants(shared['reference_alms']*np.exp(-u*rate),replay['candidate_alms']*np.exp(-u*rate)) for u in range(4)]
    ratio_error=scaled_error(replay['perturbation_ratios'],all_ratios[chosen])
    import torch
    from .closure_fibers import ClosureCUDA
    gpu_replays={};gpu_errors=[]
    for device in (0,1):
        check()
        with torch.cuda.device(device),torch.no_grad():
            model=ClosureCUDA(family,f'cuda:{device}')
            angle_tensor=torch.as_tensor(theta[chosen],dtype=torch.float64,device=device)
            values=np.stack([model.features(angle_tensor,u).cpu().numpy() for u in range(4)])
            gpu_replays[f'gpu{device}_features']=values
            gpu_errors.append(dict(device=device,maximum_scaled_error=scaled_error(values,direct_values)))
            del model
    preservation_ok=all(item['passed'] for item in time_checks)
    finite_precision_ok=(direct_error<=1e-10 and covariance_error<=1e-10 and ratio_error<=1e-10
        and all(item['maximum_scaled_error']<=1e-10 for item in gpu_errors) and preservation_ok)
    verified=(finite_precision_ok and direct_match['matched'] and np.max(replay['perturbation_ratios'])<=.25
        and all(item['passed'] for item in time_checks) and direct_gap>=.05
        and safe_future_difference)
    witness_directory=out/'witnesses';witness_directory.mkdir(exist_ok=True)
    arrays=witness_directory/f'{name}.npz'
    _npz(arrays,theta=theta[chosen],initial_theta=initial[chosen],
        reference_features=reference_future,candidate_features=direct_values,matrix_features=matrix_values,
        current_reference_bins=np.zeros(len(direct_match['bins']),dtype=np.int64),
        current_candidate_bins=direct_match['bins'],future_reference_bins=future_ref_bins,future_candidate_bins=future_candidate_bins,
        perturbation_ratios=replay['perturbation_ratios'],**gpu_replays)
    candidate_alms=witness_directory/f'{name}_candidate_alms.npz'
    # Store selected reconstructed coefficients, including rejected precision
    # candidates, to preserve the complete replay evidence.
    _npz(candidate_alms,alms=replay['candidate_alms'])
    separation={};separation_details={}
    for representation in ('baseline','capacity','repair'):
        details=bin_separation(direct_values[0],reference,scales,representation)
        separation[representation]=details['robust_changed_coordinate']
        separation_details[representation]=details
    record.update(status='verified_witness' if verified else ('precision_failure' if not finite_precision_ok else 'rejected_candidate'),
        chosen_start=chosen,direct_gap=direct_gap,current_minimum_margin=direct_match['minimum_margin'],
        direct_sample_feature_error=direct_error,direct_covariance_error=covariance_error,direct_ratio_error=ratio_error,
        actual_gpu_replay_errors=gpu_errors,
        numerical_preservation_passed=preservation_ok,
        invariant_checks=time_checks,distinguished_by=separation,
        separation_details=separation_details,safe_future_bin_difference=safe_future_difference,
        input_bins_match=direct_match['matched'],future_bins_differ=not np.array_equal(future_ref_bins,future_candidate_bins),
        perturbation_ratios=replay['perturbation_ratios'].tolist(),
        arrays=str(arrays.relative_to(out)),arrays_sha256=sha(arrays),
        coefficients=str(candidate_alms.relative_to(out)),coefficients_sha256=sha(candidate_alms),
        seconds=time.perf_counter()-started)
    _json(directory/f'{name}.json',record)
    return record
