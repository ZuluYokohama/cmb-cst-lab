"""Fixed-budget feasible projected searches, not a proof of closure on failure."""
from __future__ import annotations
import hashlib
import time
import numpy as np

SEED = 82024102027
TILES = (24,6,30,34,21,11,152,170,189,174,186,166)
STAGES = (('source','baseline'),('fresh','baseline'),('fresh','capacity'),('fresh','repair'))
QUANTUM = .001
GUARD = 1e-7


def initial_kicks(tile,stage,count=128,planes=128):
    """Fresh representations share raw starts; projection can differ by model."""
    if stage not in ('source','fresh') or count%2 or planes<1:
        raise ValueError('invalid fixed initialization domain')
    values=[]
    for index in range(count):
        generator=np.random.default_rng(np.random.SeedSequence([SEED,int(tile),0 if stage=='source' else 1,index]))
        value=generator.normal(size=planes)
        value*=.1/np.linalg.norm(value)
        if index in (0,count//2):
            value[:]=0
        values.append(value)
    return np.asarray(values),np.r_[np.ones(count//2),-np.ones(count//2)]


def tangent_reference(jacobian,direction,cutoff=1e-10):
    j,d=np.asarray(jacobian,dtype=float),np.asarray(direction,dtype=float)
    if j.ndim!=2 or d.shape!=(j.shape[1],) or not np.isfinite(j).all() or not np.isfinite(d).all():
        raise ValueError('finite Jacobian and matching direction required')
    u,s,v=np.linalg.svd(j,full_matrices=False)
    active=s>(s[0]*cutoff if len(s) else 0)
    projection=d-v[active].T@(v[active]@d)
    inverse=(v[active].T/s[active])@u[:,active].T
    return projection,inverse,int(active.sum())


def matching_bins(values,reference,scales,indices,quantum=QUANTUM,guard=GUARD):
    a,b,s=np.asarray(values),np.asarray(reference),np.asarray(scales)
    ix=np.asarray(indices,dtype=int)
    if a.shape!=b.shape or s.shape!=b.shape or np.any(s<=0) or not all(np.isfinite(v).all() for v in (a,b,s)):
        raise ValueError('finite feature/scale arrays required')
    delta=(a[ix]-b[ix])/s[ix]
    bins=np.rint(delta/quantum).astype(np.int64)
    margin=quantum/2-np.abs(delta)
    return dict(matched=bool(np.all(bins==0) and np.all(margin>=guard)),bins=bins,
        minimum_margin=float(np.min(margin)),maximum_scaled_change=float(np.max(np.abs(delta))))


def manifest(tile,stage,representation,count=128):
    if int(tile) not in TILES or (stage,representation) not in STAGES:
        raise ValueError('not a planned search case')
    position=TILES.index(int(tile))
    return dict(tile=int(tile),stage=stage,representation=representation,device=1 if position in (3,7,11) else 0,
        starts=count,seed=SEED,signs='first half maximize; second half minimize',
        seed_derivation='SeedSequence([seed,tile,source=0/fresh=1,start]); fresh representations share raw kicks')


def projected_batch(feature_function,target_function,ratio_function,kicks,signs,*,device,
                    max_iterations=1000,max_stalls=50,rank_cutoff=1e-10,step_norm=.1,
                    corrections=5,halvings=10,budget=None,checkpoint=None,
                    jacobian_function=None,target_gradient_function=None):
    """Search one small batch. Functions accept a single real angle vector.

    feature_function returns reference-centered robust-scale changes, not bins.
    target_function returns one scalar; ratio_function returns the two RMS ratios.
    Every accepted step is feasible and improving in its declared direction.
    ``rejected`` counts outer iterations without an accepted proposal. Separate
    nonfinite counters include initialization/retraction proposals; one invalid
    member never blocks finite peers. The aggregate counts failure events, not
    unique floating-point states. Snapshots never reevaluate scientific kernels.
    An optional execution hook returns the same selected/scaled Jacobian for a
    batched real angle tensor; the default computes it with unchunked jacrev.
    """
    import torch
    kicks=np.asarray(kicks,dtype=float);signs=np.asarray(signs,dtype=float)
    if kicks.ndim!=2 or signs.shape!=(len(kicks),) or not np.isfinite(kicks).all() or not np.all(np.abs(signs)==1):
        raise ValueError('finite paired starts/signs required')
    started=time.perf_counter();batch,planes=kicks.shape
    tensor=lambda x:torch.as_tensor(x,dtype=torch.float64,device=device)
    vf=torch.func.vmap(feature_function)
    vt=torch.func.vmap(target_function)
    vr=torch.func.vmap(ratio_function)
    if jacobian_function is None:
        jac=torch.func.vmap(torch.func.jacrev(feature_function))
    else:
        if not callable(jacobian_function):
            raise TypeError('jacobian_function must be callable')
        def jac(value):
            answer=jacobian_function(value)
            if (not torch.is_tensor(answer) or answer.ndim!=3
                    or answer.shape[0]!=len(value) or answer.shape[-1]!=planes
                    or answer.dtype!=torch.float64 or answer.device!=value.device):
                raise ValueError('Jacobian hook must return float64 [batch,feature,plane] on the input device')
            return answer
    if target_gradient_function is None:
        grad=torch.func.vmap(torch.func.grad(target_function))
    else:
        if not callable(target_gradient_function):
            raise TypeError('target_gradient_function must be callable')
        def grad(value):
            answer=target_gradient_function(value)
            if (not torch.is_tensor(answer) or answer.shape!=value.shape
                    or answer.dtype!=torch.float64 or answer.device!=value.device):
                raise ValueError('Target gradient hook must match float64 angle batch on input device')
            return answer
    # Leave an additional guard for independent CPU/direct-SHT replay rounding.
    half=QUANTUM/2-2*GUARD
    bound=np.pi/4
    zero=tensor(np.zeros_like(kicks));x=zero.clone()
    initialized=torch.zeros(batch,dtype=torch.bool,device=device)
    initial_values=np.empty((0,planes),dtype=np.float64)
    reference_y=None;cached_targets=None;initialization_complete=False
    iterations=np.zeros(batch,dtype=np.int64);stalls=np.zeros(batch,dtype=np.int64)
    rejected=np.zeros(batch,dtype=np.int64);nonfinite=np.zeros(batch,dtype=np.int64)
    constraint_nonfinite=np.zeros(batch,dtype=np.int64)
    ratio_nonfinite=np.zeros(batch,dtype=np.int64)
    target_nonfinite=np.zeros(batch,dtype=np.int64)
    jacobian_nonfinite=np.zeros(batch,dtype=np.int64)
    gradient_nonfinite=np.zeros(batch,dtype=np.int64)
    proposal_evaluations=np.zeros(batch,dtype=np.int64)
    retraction_corrections=np.zeros(batch,dtype=np.int64)
    ranks=np.zeros(batch,dtype=np.int64)
    status=np.full(batch,'initializing',dtype='<U32')
    history=[];interruption=None;failure_exception=None;finished=False
    all_ids=np.arange(batch,dtype=np.int64)
    def snapshot(phase):
        targets=np.empty(0,dtype=np.float64) if cached_targets is None else cached_targets.copy()
        ref=np.empty(0,dtype=np.float64) if reference_y is None else reference_y.detach().cpu().numpy().copy()
        return dict(theta=x.detach().cpu().numpy().copy(),initial_theta=initial_values.copy(),
            raw_kicks=kicks.copy(),signs=signs.copy(),reference_target=ref,target=targets,
            gap=np.empty(0,dtype=np.float64) if len(targets)==0 or len(ref)==0 else targets-ref,
            target_available=cached_targets is not None,initialization_complete=initialization_complete,
            iterations=iterations.copy(),stalls=stalls.copy(),rejected=rejected.copy(),
            nonfinite_trials=nonfinite.copy(),nonfinite_constraint_trials=constraint_nonfinite.copy(),
            nonfinite_ratio_trials=ratio_nonfinite.copy(),nonfinite_target_trials=target_nonfinite.copy(),
            nonfinite_jacobian_evaluations=jacobian_nonfinite.copy(),
            nonfinite_gradient_evaluations=gradient_nonfinite.copy(),
            proposal_evaluations=proposal_evaluations.copy(),retraction_corrections=retraction_corrections.copy(),
            rank=ranks.copy(),status=status.copy(),initial_retraction_succeeded=initialized.detach().cpu().numpy().copy(),
            history=list(history),seconds=time.perf_counter()-started,checkpoint_phase=phase,
            interruption=interruption)
    def emit(phase):
        if checkpoint is not None:
            checkpoint(snapshot(phase))
    def check_budget():
        if budget is not None:
            budget()
    def decomposed(j):
        if not bool(torch.isfinite(j).all().item()):
            raise FloatingPointError('nonfinite Jacobian at evaluated state')
        u,s,v=torch.linalg.svd(j,full_matrices=False)
        active=s>s[:,:1]*rank_cutoff
        reciprocal=torch.where(active,1/torch.clamp(s,min=torch.finfo(s.dtype).tiny),0.)
        return u,s,v,active,reciprocal
    def feasible(x):
        with torch.no_grad():
            value=vf(x);ratios=vr(x)
            finite=torch.isfinite(value).all(-1)&torch.isfinite(ratios).all(-1)
            valid=finite&(value.abs()<=half).all(-1)&(ratios<=.25).all(-1)&(x.abs()<=bound).all(-1)
        return valid,value,ratios
    def retract(proposal,original_ids):
        x=proposal.detach().clamp(-bound,bound)
        original_ids=np.asarray(original_ids,dtype=np.int64)
        proposal_evaluations[original_ids]+=1
        bad_constraint=torch.zeros(len(x),dtype=torch.bool,device=device)
        bad_ratio=bad_constraint.clone();bad_jacobian=bad_constraint.clone()
        try:
            for _ in range(corrections):
                check_budget()
                valid,values,ratios=feasible(x)
                bad_constraint|=~torch.isfinite(values).all(-1)
                bad_ratio|=~torch.isfinite(ratios).all(-1)
                blocked=bad_constraint|bad_ratio|bad_jacobian
                valid&=~blocked
                # Invalid rows remain rejected; already feasible rows are left
                # untouched. Finite peers still receive their own corrections.
                needs=(values.abs()>half).any(-1)&~blocked
                if not bool(needs.any().item()):
                    break
                indices=torch.nonzero(needs,as_tuple=False).ravel()
                j=jac(x[indices])
                finite_j=torch.isfinite(j).all(dim=(-2,-1))
                bad_jacobian[indices[~finite_j]]=True
                indices=indices[finite_j];j=j[finite_j]
                if len(indices)==0:
                    continue
                u,s,v,active,reciprocal=decomposed(j)
                residual=values[indices]-values[indices].clamp(-half,half)
                adjustment=torch.einsum('bkp,bk->bp',v,
                    torch.einsum('bdk,bd->bk',u,residual)*reciprocal)
                updated=x.clone()
                updated[indices]=x[indices]-adjustment
                x=updated.detach().clamp(-bound,bound)
                retraction_corrections[original_ids[indices.cpu().numpy()]]+=1
            valid,values,ratios=feasible(x)
            bad_constraint|=~torch.isfinite(values).all(-1)
            bad_ratio|=~torch.isfinite(ratios).all(-1)
            blocked=bad_constraint|bad_ratio|bad_jacobian
            return x,valid&~blocked,blocked
        finally:
            bc,br,bj=[value.cpu().numpy() for value in (bad_constraint,bad_ratio,bad_jacobian)]
            constraint_nonfinite[original_ids]+=bc
            ratio_nonfinite[original_ids]+=br
            jacobian_nonfinite[original_ids]+=bj
            nonfinite[original_ids]+=bc|br|bj
    try:
        check_budget()
        reference_y=vt(zero).detach()
        if not bool(torch.isfinite(reference_y).all().item()):
            failed=~torch.isfinite(reference_y).cpu().numpy()
            target_nonfinite[:]+=failed;nonfinite[:]+=failed
            raise FloatingPointError('nonfinite objective at reference state')
        j0=jac(zero)
        finite_reference_j=torch.isfinite(j0).all(dim=(-2,-1)).cpu().numpy()
        if not np.all(finite_reference_j):
            jacobian_nonfinite[:]+=~finite_reference_j;nonfinite[:]+=~finite_reference_j
            raise FloatingPointError('nonfinite Jacobian at reference state')
        _,_,v0,active0,_=decomposed(j0)
        ranks[:]=active0.sum(-1).cpu().numpy()
        raw=tensor(kicks)
        initial=raw-torch.einsum('bkp,bk->bp',v0,torch.einsum('bkp,bp->bk',v0,raw)*active0)
        for h in range(halvings):
            trial,valid,_=retract(initial*(.5**h),all_ids)
            take=valid&~initialized
            x=torch.where(take[:,None],trial,x);initialized|=valid
            if bool(initialized.all().item()):
                break
        with torch.no_grad():
            initial_y=vt(x)
            finite_target=torch.isfinite(initial_y)
            failed=~finite_target.cpu().numpy()
            target_nonfinite[:]+=failed;nonfinite[:]+=failed
            # A future-invalid kick is an explicit initialization failure. The
            # already verified zero reference is retained without a new draw.
            x=torch.where(finite_target[:,None],x,zero)
            initialized&=finite_target
            initial_y=torch.where(finite_target,initial_y,reference_y)
            cached_targets=initial_y.cpu().numpy().copy()
        initial_values=x.detach().cpu().numpy().copy()
        initialization_complete=True
        status[:]='active'
        emit('initialized')
        sign=tensor(signs)
        while np.any(status=='active'):
            check_budget()
            ids=np.flatnonzero(status=='active')
            index=torch.as_tensor(ids,device=device)
            current=x[index]
            j=jac(current)
            finite_j=torch.isfinite(j).all(dim=(-2,-1)).cpu().numpy()
            if not np.all(finite_j):
                failed=ids[~finite_j]
                jacobian_nonfinite[failed]+=1;nonfinite[failed]+=1
                status[failed]='nonfinite_jacobian'
                ids=ids[finite_j];j=j[torch.as_tensor(finite_j,device=device)]
                if len(ids)==0:
                    continue
                index=torch.as_tensor(ids,device=device);current=x[index]
            _,_,v,active,_=decomposed(j)
            ranks[ids]=active.sum(-1).cpu().numpy()
            g=grad(current)*sign[index,None]
            finite_g=torch.isfinite(g).all(-1).cpu().numpy()
            if not np.all(finite_g):
                failed=ids[~finite_g]
                gradient_nonfinite[failed]+=1;nonfinite[failed]+=1
                status[failed]='nonfinite_gradient'
                ids=ids[finite_g]
                keep=torch.as_tensor(finite_g,device=device)
                v,active,g=v[keep],active[keep],g[keep]
                if len(ids)==0:
                    continue
                index=torch.as_tensor(ids,device=device);current=x[index]
            direction=g-torch.einsum('bkp,bk->bp',v,torch.einsum('bkp,bp->bk',v,g)*active)
            norm=torch.linalg.vector_norm(direction,dim=-1,keepdim=True)
            direction=direction/torch.clamp(norm,min=1e-30)
            old_y=vt(current).detach();chosen=current.clone();chosen_y=old_y.clone()
            if not bool(torch.isfinite(old_y).all().item()):
                raise FloatingPointError('stored accepted target became nonfinite')
            accepted=torch.zeros(len(ids),dtype=torch.bool,device=device)
            for h in range(halvings):
                trial,valid,already_nonfinite=retract(current+step_norm*(.5**h)*direction,ids)
                with torch.no_grad():
                    yy=vt(trial)
                    finite=torch.isfinite(yy)
                    bad_target=~finite
                    target_nonfinite[ids]+=bad_target.cpu().numpy()
                    nonfinite[ids]+=(bad_target&~already_nonfinite).cpu().numpy()
                    improve=(yy-old_y)*sign[index]>1e-14
                    take=valid&finite&improve&~accepted
                    chosen=torch.where(take[:,None],trial,chosen)
                    chosen_y=torch.where(take,yy,chosen_y)
                    accepted|=take
                if bool(accepted.all().item()):
                    break
            x[index]=chosen.detach()
            cached_targets[ids]=chosen_y.cpu().numpy()
            accepted_np=accepted.cpu().numpy()
            iterations[ids]+=1
            stalls[ids]=np.where(accepted_np,0,stalls[ids]+1)
            rejected[ids]+=~accepted_np
            for index_np in ids:
                if iterations[index_np]>=max_iterations:
                    status[index_np]='iteration_cap'
                elif stalls[index_np]>=max_stalls:
                    status[index_np]='stalled'
            if len(history)==0 or int(iterations.max())//20>history[-1]['iteration']//20 or not np.any(status=='active'):
                valid,values,ratios=feasible(x)
                if not bool(valid.all().item()):
                    raise FloatingPointError('stored search state lost feasibility')
                record=dict(iteration=int(iterations.max()),targets=cached_targets.tolist(),
                    gaps=(cached_targets-reference_y.cpu().numpy()).tolist(),ranks=ranks.tolist(),
                    maximum_constraint=float(values.abs().max().item()),maximum_rms_ratio=float(ratios.max().item()),
                    seconds=time.perf_counter()-started)
                history.append(record)
                emit('periodic')
        valid,_,_=feasible(x)
        if not bool(valid.all().item()):
            raise FloatingPointError('final stored state lost feasibility')
        finished=True
        result=snapshot('complete')
        result['feasible']=valid.cpu().numpy()
        return result
    except BaseException as exc:
        failure_exception=exc
        interruption=f'{type(exc).__name__}: {exc}'
        status[np.isin(status,('active','initializing'))]='incomplete'
        raise
    finally:
        try:
            emit('complete' if finished else 'interrupted')
        except Exception as checkpoint_error:
            # Never conceal the original timeout or scientific failure with a
            # secondary persistence error; propagate normal callback failures.
            if interruption is None:
                raise
            failure_exception.add_note(f'Final checkpoint also failed: {checkpoint_error}')
