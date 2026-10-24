"""Bounded CPU continuation of a specified finite-precision equality fiber.

This supplies candidates to independent observing-operator verification. A
failed finite search is not a proof that the global admissible domain is empty.
"""
from __future__ import annotations
import hashlib
import time
import numpy as np


def _continuation_diagnostic(value):
    if isinstance(value,dict):return {str(k):_continuation_diagnostic(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_continuation_diagnostic(v) for v in value]
    if isinstance(value,np.ndarray):return _continuation_diagnostic(value.tolist())
    if isinstance(value,np.generic):return _continuation_diagnostic(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def direction_manifest(nplanes,seed,global_start_id):
    if not isinstance(nplanes,(int,np.integer)) or nplanes<1:raise ValueError('positive plane count required')
    if any(not isinstance(v,(int,np.integer)) or isinstance(v,bool) or not 0<=v<2**63 for v in (seed,global_start_id)):raise ValueError('nonnegative int64 seed and global_start_id required')
    direction=np.random.default_rng(np.random.SeedSequence([int(seed),int(global_start_id)])).normal(size=nplanes)
    direction/=np.linalg.norm(direction)
    return direction,dict(seed=int(seed),global_start_id=int(global_start_id),
        derivation='SeedSequence([seed,global_start_id]); standard normal normalized to unit norm',
        direction_sha256=hashlib.sha256(np.ascontiguousarray(direction).tobytes()).hexdigest())


def continue_fiber(model,scales,family,*,direction=None,global_start_id=0,seed=82024102027,
                   max_iterations=100,max_stalls=25,step_norm=.1,rank_cutoff=1e-10,
                   corrections=5,halvings=10,restore_target=1e-12,tolerance=1e-10,
                   minimum_rms=.01,maximum_rms=.25,budget=None):
    """Start at zero, restore constraints and seek a finite nontrivial candidate.

    Raw q values and Jacobians come from ``model``; scales are caller-frozen.
    No constraint, amplitude threshold or numerical gate is relaxed on failure.
    Returned ``candidate_found`` is not independent NaMaster verification.
    """
    if not (1<=max_iterations<=100 and 1<=max_stalls<=25 and 0<=corrections<=5 and 1<=halvings<=10):raise ValueError('continuation exceeds its bounded pilot limits')
    if not (0<restore_target<=tolerance<=1e-10 and 0<minimum_rms<=maximum_rms<=.25 and 0<rank_cutoff<1 and step_norm>0):raise ValueError('invalid fixed continuation thresholds')
    p=int(family.nplanes)
    generated,manifest=direction_manifest(p,seed,global_start_id)
    if direction is None:direction=generated
    else:
        if np.iscomplexobj(direction) or np.asarray(direction).dtype.kind not in 'iuf':raise ValueError('direction must be real and non-boolean')
        direction=np.asarray(direction,dtype=np.float64)
        if direction.shape!=(p,) or not np.isfinite(direction).all() or np.linalg.norm(direction)==0:raise ValueError('finite nonzero P-vector direction required')
        direction=direction/np.linalg.norm(direction)
        manifest.update(derivation='caller-provided direction, normalized to unit norm',
            direction_sha256=hashlib.sha256(np.ascontiguousarray(direction).tobytes()).hexdigest())
    if np.iscomplexobj(scales) or np.asarray(scales).dtype.kind not in 'iuf':raise ValueError('feature scales must be real and non-boolean')
    scale=np.asarray(scales,dtype=np.float64)
    if scale.ndim!=1 or not len(scale) or not np.isfinite(scale).all() or np.any(scale<=0):raise ValueError('caller-frozen positive feature scales required')
    theta=np.zeros(p);reference=None;path=[theta.copy()];rms_path=[0.];residual_path=[0.]
    ranks=[];failures=[];stalls=0;iterations=0;accepted=0;status='no_nontrivial_found';reason='not_started'
    reference_verified=False
    started=time.perf_counter()
    def check():
        if budget is not None:budget()
    def values(angle):
        raw=model.features(angle)
        if np.iscomplexobj(raw) or np.asarray(raw).dtype.kind not in 'iuf':raise FloatingPointError('quadratic features must be real')
        actual=np.asarray(raw,dtype=np.float64)
        if actual.shape!=scale.shape or not np.isfinite(actual).all():raise FloatingPointError('nonfinite or invalid quadratic features')
        return (actual-reference)/scale
    def jacobian(angle):
        raw=model.jacobian(angle)
        if np.iscomplexobj(raw) or np.asarray(raw).dtype.kind not in 'iuf':raise FloatingPointError('quadratic Jacobian must be real')
        actual=np.asarray(raw,dtype=np.float64)
        if actual.shape!=(len(scale),p) or not np.isfinite(actual).all():raise FloatingPointError('nonfinite or invalid quadratic Jacobian')
        return actual/scale[:,None]
    def decompose(j):
        u,s,v=np.linalg.svd(j,full_matrices=False);use=s>(s[0]*rank_cutoff if len(s) else 0.)
        inverse=(v[use].T/s[use])@u[:,use].T
        return v[use],inverse,int(use.sum())
    try:
        check();raw=model.features(theta)
        if np.iscomplexobj(raw) or np.asarray(raw).dtype.kind not in 'iuf':raise FloatingPointError('reference features must be real')
        reference=np.asarray(raw,dtype=np.float64)
        if reference.shape!=scale.shape or not np.isfinite(reference).all():raise FloatingPointError('invalid reference features')
        initial_invariants=family.invariants(theta,tolerance=tolerance)
        if not initial_invariants['passed']:raise FloatingPointError('reference family invariants failed')
        reference_verified=True
        for iteration in range(max_iterations):
            check();iterations=iteration+1
            row,inverse,rank=decompose(jacobian(theta));ranks.append(rank)
            tangent=direction-row.T@(row@direction);norm=np.linalg.norm(tangent)
            found=False;reason='no_feasible_improving_step'
            if norm<=1e-14:reason='no_local_direction_in_requested_tangent'
            else:
                tangent/=norm
                for h in range(halvings):
                    check();trial=theta+step_norm*(.5**h)*tangent
                    if np.max(np.abs(trial))>np.pi/4:reason='angle_bound';continue
                    try:
                        for correction in range(corrections+1):
                            check();residual=values(trial)
                            if np.max(np.abs(residual))<=restore_target or correction==corrections:break
                            _,pinv,_=decompose(jacobian(trial));trial=trial-pinv@residual
                            if not np.isfinite(trial).all() or np.max(np.abs(trial))>np.pi/4:break
                        if not np.isfinite(trial).all():raise FloatingPointError('nonfinite Newton angles')
                        if np.max(np.abs(trial))>np.pi/4:reason='angle_restoration';continue
                        residual=values(trial);error=float(np.max(np.abs(residual)))
                        ratio=float(family.rms_ratio(trial))
                        if not np.isfinite(ratio):raise FloatingPointError('nonfinite perturbation RMS')
                        if error>tolerance:reason='equality_restoration_failed';continue
                        if not 0<=ratio<=maximum_rms:reason='rms_bound';continue
                        if ratio<=rms_path[-1]+1e-14:reason='no_amplitude_progress';continue
                        invariant=family.invariants(trial,tolerance=tolerance)
                        if not invariant['passed']:raise FloatingPointError('candidate family invariants failed')
                        theta=trial;path.append(theta.copy());rms_path.append(ratio);residual_path.append(error)
                        accepted+=1;found=True;reason='accepted';break
                    except FloatingPointError as exc:
                        failures.append(dict(iteration=iteration,halving=h,theta=trial.copy(),
                            nonfinite_angles=bool(not np.isfinite(trial).all()),error=str(exc)))
                        status='numerical_failure';reason=str(exc);break
            if status=='numerical_failure':break
            if found:
                stalls=0
                if rms_path[-1]>=minimum_rms:status='candidate_found';reason='nontrivial_threshold_reached';break
            else:
                stalls+=1
                if stalls>=max_stalls:reason='stalled:'+reason;break
        if status=='no_nontrivial_found' and iterations==max_iterations:reason='iteration_cap:'+reason
    except TimeoutError as exc:
        status='incomplete';reason=str(exc)
    except (FloatingPointError,np.linalg.LinAlgError,ValueError) as exc:
        status='numerical_failure';reason=f'{type(exc).__name__}: {exc}'
        failures.append(dict(iteration=iterations,theta=theta.copy(),
            nonfinite_angles=bool(not np.isfinite(theta).all()),error=reason))
    return _continuation_diagnostic(dict(status=status,reason=reason,theta=theta.copy(),theta_path=np.stack(path),
        rms_ratio=float(rms_path[-1]),rms_path=np.asarray(rms_path),max_scaled_residual=float(residual_path[-1]),
        residual_path=np.asarray(residual_path),rank_history=np.asarray(ranks,dtype=np.int64),
        iterations=iterations,stalls=stalls,accepted_steps=accepted,failure_artifacts=failures,
        minimum_rms=minimum_rms,maximum_rms=maximum_rms,tolerance=tolerance,restore_target=restore_target,
        seed_manifest=manifest,direction=direction.copy(),seconds=time.perf_counter()-started,
        reference_verified=reference_verified,
        feasible=reference_verified and status!='numerical_failure' and residual_path[-1]<=tolerance,
        nontrivial=bool(status=='candidate_found' and rms_path[-1]>=minimum_rms),
        independent_verification_required=True,
        interpretation='Finite continuation only; no proof of global infeasibility, cut-sky preservation or physical noise law'))
