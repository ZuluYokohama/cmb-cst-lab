"""Native-batch, constraint-aware search; finite failure is not a closure proof.

This is a separately versioned optimizer. The v1 source, starts and evidence are
unchanged. Features are reference-centered robust-scale residuals. All numerical
callbacks consume native [batch, plane] tensors; derivative hooks are optional.
"""
from __future__ import annotations
import time
import numpy as np
from .closure_search import SEED,TILES,STAGES,initial_kicks,manifest,matching_bins

QUANTUM=.001
ANGLE_BOUND=np.pi/4
RMS_RATIO_NAMES=('tile','global')


def batched_svd(matrix,*,svd_backend='device',svd_driver=None,cpu_lock=None):
    """Exact Float64 SVD of an entire batch, optionally using the shared CPU.

    The CPU lock covers transfer, the native batched decomposition and return
    transfer. It can be shared with eight-thread direct sampling to avoid two
    independently scheduled CPU workloads oversubscribing the same machine.
    This optimizer does not differentiate through a decomposition.
    """
    import torch
    from contextlib import nullcontext
    if svd_backend not in ('device','cpu'):raise ValueError('SVD backend must be device or cpu')
    if svd_driver not in (None,'gesvd','gesvdj'):raise ValueError('only exact SVD drivers are allowed')
    if not torch.is_tensor(matrix) or matrix.ndim!=3 or matrix.dtype!=torch.float64:
        raise ValueError('SVD input must be a native Float64 matrix batch')
    if cpu_lock is not None and not (hasattr(cpu_lock,'__enter__') and hasattr(cpu_lock,'__exit__')):
        raise TypeError('cpu_lock must be a shared synchronous context manager')
    if svd_backend=='cpu':
        with cpu_lock if cpu_lock is not None else nullcontext():
            host=matrix.detach().to(device='cpu',dtype=torch.float64)
            values=torch.linalg.svd(host,full_matrices=False)
            return tuple(value.to(device=matrix.device) for value in values)
    options={} if matrix.device.type!='cuda' or svd_driver is None else dict(driver=svd_driver)
    return torch.linalg.svd(matrix,full_matrices=False,**options)


def active_tangent_numpy(jacobian,gradient,theta,ratios,rms_gradients,*,
                         cutoff=1e-10,active_rms_threshold=.24,box_tolerance=1e-10):
    """Independent small-matrix oracle for active RMS and box-face constraints."""
    j,g,x,r,q=[np.asarray(v,dtype=float) for v in (jacobian,gradient,theta,ratios,rms_gradients)]
    if j.ndim!=2 or g.shape!=x.shape or g.shape!=(j.shape[1],) or r.shape!=(2,) or q.shape!=(2,len(g)) or not all(np.isfinite(v).all() for v in (j,g,x,r,q)):
        raise ValueError('finite, compatible Jacobian/gradient/angle/RMS arrays required')
    rms=r>=active_rms_threshold;box=np.zeros(len(g),dtype=bool)
    for _ in range(len(g)+1):
        rows=np.concatenate([j,q[rms],np.eye(len(g))[box]])
        _,singular,v=np.linalg.svd(rows,full_matrices=False)
        active=singular>(singular[0]*cutoff if len(singular) else 0.)
        direction=g-v[active].T@(v[active]@g)
        outward=((x>=ANGLE_BOUND-box_tolerance)&(direction>1e-12))|((x<=-ANGLE_BOUND+box_tolerance)&(direction<-1e-12))
        added=outward&~box
        if not np.any(added):break
        box|=added
    return dict(direction=direction,rank=int(active.sum()),active_rms=rms,active_box=box,
        box_satisfied=bool(not np.any(outward)))


def active_tangent(jacobian,gradient,theta,ratios,rms_gradients,*,cutoff=1e-10,
                   active_rms_threshold=.24,box_tolerance=1e-10,svd_driver=None,
                   svd_backend='device',cpu_lock=None):
    """Batched SVD tangent with monotonically added outward box-face rows.

Adding a face can expose another outward component after reprojection. Repeat
until every active face is tangent/inward, rather than zeroing coordinates after
projection (which would generally violate the feature tangent constraints).
"""
    import torch
    if svd_driver not in (None,'gesvd','gesvdj'):raise ValueError('only exact SVD drivers are allowed')
    if svd_backend not in ('device','cpu'):raise ValueError('SVD backend must be device or cpu')
    batch,planes=theta.shape
    rms=ratios>=active_rms_threshold
    box=torch.zeros_like(theta,dtype=torch.bool)
    eye=torch.eye(planes,dtype=theta.dtype,device=theta.device)
    for _ in range(planes+1):
        # Only append faces active somewhere in this native batch. Padding all
        # P inactive identity rows would turn the common interior thin SVD into
        # a much larger tall decomposition without changing its row space.
        columns=torch.nonzero(box.any(dim=0),as_tuple=False).ravel()
        pieces=[jacobian,rms_gradients*rms[...,None]]
        if len(columns):pieces.append(eye[columns][None]*box[:,columns][...,None])
        rows=torch.cat(pieces,dim=1)
        _,singular,v=batched_svd(rows,svd_backend=svd_backend,svd_driver=svd_driver,cpu_lock=cpu_lock)
        used=singular>singular[:,:1]*cutoff
        direction=gradient-torch.einsum('bkp,bk->bp',v,torch.einsum('bkp,bp->bk',v,gradient)*used)
        outward=((theta>=ANGLE_BOUND-box_tolerance)&(direction>1e-12))|((theta<=-ANGLE_BOUND+box_tolerance)&(direction<-1e-12))
        added=outward&~box
        if not bool(added.any().item()):break
        box|=added
    return dict(direction=direction,rank=used.sum(-1),active_rms=rms,active_box=box,
        box_satisfied=~outward.any(-1))


def projected_batch(feature_function,target_function,ratio_function,kicks,signs,*,device,
                    max_iterations=1000,max_stalls=50,rank_cutoff=1e-10,step_norm=.1,
                    corrections=5,halvings=10,budget=None,checkpoint=None,
                    jacobian_function=None,target_gradient_function=None,
                    rms_gradient_function=None,center_tolerance=1e-6,
                    feasible_margin=1e-5,active_rms_threshold=.24,svd_driver=None,
                    svd_backend='device',cpu_lock=None):
    """Optimize real angles with native-batch callbacks and explicit diagnostics.

    feature: [B,P]->[B,D], target: [B,P]->[B], ratio: [B,P]->[B,2].
    RMS columns are [tile/local, global]. Jacobian and squared-RMS-gradient
    hooks return [B,D,P] and [B,2,P];
    the target-gradient hook returns [B,P]. All use float64 on ``device``.
    Newton restoration targets F=0 and stops at 1e-6. Center attainment is a
    search diagnostic; acceptance uses the declared bin interior (0.00049),
    angle/RMS bounds and signed improvement, not an extra witness condition.
    """
    import torch
    kicks=np.asarray(kicks,dtype=np.float64);signs=np.asarray(signs,dtype=np.float64)
    if kicks.ndim!=2 or not kicks.shape[0] or not kicks.shape[1] or signs.shape!=(len(kicks),) or not np.isfinite(kicks).all() or not np.all(np.abs(signs)==1):
        raise ValueError('finite nonempty paired real-angle starts/signs required')
    if not (0<center_tolerance<=QUANTUM/2-feasible_margin and 0<feasible_margin<QUANTUM/2 and 0<active_rms_threshold<.25):
        raise ValueError('invalid center, bin-interior or active RMS parameters')
    if not (1<=max_iterations<=1000 and 1<=max_stalls<=50 and 0<=corrections<=5 and 1<=halvings<=10 and 0<rank_cutoff<1 and step_norm>0):
        raise ValueError('invalid optimizer limits')
    if svd_driver not in (None,'gesvd','gesvdj'):raise ValueError('only exact SVD drivers are allowed')
    if svd_backend not in ('device','cpu'):raise ValueError('SVD backend must be device or cpu')
    started=time.perf_counter();batch,planes=kicks.shape;half=QUANTUM/2-feasible_margin
    tensor=lambda value:torch.as_tensor(value,dtype=torch.float64,device=device)
    x=tensor(np.zeros_like(kicks));sign=tensor(signs)
    def output(function,value,shape,label):
        answer=function(value)
        if not torch.is_tensor(answer) or answer.shape!=shape or answer.dtype!=torch.float64 or answer.device!=value.device:
            raise ValueError(f'{label} must return native float64 {shape} on the input device')
        return answer
    # Wrap the native API only for automatic derivative fallbacks. The production
    # execution hooks bypass these wrappers and are called once per native batch.
    fjac=jacobian_function or torch.func.vmap(torch.func.jacrev(lambda value:feature_function(value[None])[0]))
    tgrad=target_gradient_function or torch.func.vmap(torch.func.grad(lambda value:target_function(value[None])[0]))
    rgrad=rms_gradient_function or torch.func.vmap(torch.func.jacrev(lambda value:ratio_function(value[None])[0].square()))
    for callback in (fjac,tgrad,rgrad):
        if not callable(callback):raise TypeError('derivative hooks must be callable')
    dimensions=None
    def features(value):
        nonlocal dimensions
        answer=feature_function(value)
        if not torch.is_tensor(answer) or answer.ndim!=2 or answer.shape[0]!=len(value) or answer.shape[1]<1 or answer.dtype!=torch.float64 or answer.device!=value.device:
            raise ValueError('feature callback must return native float64 [batch,features]')
        if dimensions is None:dimensions=answer.shape[1]
        if answer.shape[1]!=dimensions:raise ValueError('feature dimension changed')
        return answer
    targets=lambda value:output(target_function,value,(len(value),),'target')
    ratios=lambda value:output(ratio_function,value,(len(value),2),'RMS ratios')
    jac=lambda value:output(fjac,value,(len(value),dimensions,planes),'feature Jacobian')
    grad=lambda value:output(tgrad,value,value.shape,'target gradient')
    squared_grad=lambda value:output(rgrad,value,(len(value),2,planes),'squared RMS gradient')
    count_names=('iterations','stalls','rejected','nonfinite_trials','nonfinite_constraint_trials',
        'nonfinite_ratio_trials','nonfinite_target_trials','nonfinite_jacobian_evaluations',
        'nonfinite_gradient_evaluations','nonfinite_rms_gradient_evaluations',
        'proposal_evaluations','retraction_corrections','rms_restoration_corrections','rms_restoration_failures','center_restoration_failures',
        'center_failure_rejections','rms_rejections','box_rejections','bin_rejections',
        'nonimproving_rejections','rank','active_rms_count','active_box_count',
        'maximum_active_rms_count','maximum_active_box_count')
    counts={name:np.zeros(batch,dtype=np.int64) for name in count_names}
    status=np.full(batch,'initializing',dtype='<U32')
    reason=np.full(batch,'not_evaluated',dtype='<U48')
    initial_reason=reason.copy();initial_values=np.empty((0,planes))
    initialized=np.zeros(batch,dtype=bool);initialization_complete=False
    values_cache=None;ratio_cache=None;target_cache=None;reference_y=None
    minimum_margin=np.full(batch,np.nan);max_tracking=np.zeros(batch);last_step=np.zeros(batch)
    direction_theta=np.zeros_like(kicks);projected_norm=np.full(batch,np.nan);gradient_norm=np.full(batch,np.nan)
    rms_active=np.zeros((batch,2),dtype=bool);box_active=np.zeros((batch,planes),dtype=bool)
    history=[];failures=[];interruption=None;finished=False;failure_exception=None
    all_ids=np.arange(batch)
    def check_budget():
        if budget is not None:budget()
    def failure(ids,angles,kind,raw):
        ids=np.asarray(ids,dtype=int)
        aa=angles.detach().cpu().numpy();rr=raw.detach().cpu().numpy()
        for index,angle,value in zip(ids,aa,rr):
            failures.append((int(index),int(counts['iterations'][index]),kind,angle.copy(),np.asarray(value).copy()))
    def snapshot(phase):
        ref=np.empty(0) if reference_y is None else reference_y.copy()
        yy=np.empty(0) if target_cache is None else target_cache.copy()
        residual=np.full(batch,np.nan) if values_cache is None else np.max(np.abs(values_cache),axis=1)
        raw_values=[f[4].ravel() for f in failures]
        offsets=np.r_[0,np.cumsum([len(v) for v in raw_values])].astype(np.int64)
        result=dict(theta=x.detach().cpu().numpy().copy(),initial_theta=initial_values.copy(),raw_kicks=kicks.copy(),
            signs=signs.copy(),reference_target=ref,target=yy,gap=yy-ref if len(yy) and len(ref) else np.empty(0),
            target_available=target_cache is not None,initialization_complete=initialization_complete,
            initial_retraction_succeeded=initialized.copy(),initial_failure_reason=initial_reason.copy(),
            status=status.copy(),stall_reason=reason.copy(),minimum_margin=minimum_margin.copy(),
            center_residual=residual,center_reached=residual<=center_tolerance,
            maximum_tracking_residual=max_tracking.copy(),last_accepted_step_norm=last_step.copy(),
            last_direction_theta=direction_theta.copy(),projected_gradient_norm=projected_norm.copy(),signed_gradient_norm=gradient_norm.copy(),
            active_rms=rms_active.copy(),active_box=box_active.copy(),
            active_local_rms=rms_active[:,0].copy(),active_global_rms=rms_active[:,1].copy(),
            svd_driver=svd_driver,svd_backend=svd_backend,
            svd_execution='cpu_float64' if svd_backend=='cpu' or x.device.type=='cpu' else 'cuda:'+str(svd_driver or 'default'),
            local_rms_ratio=np.full(batch,np.nan) if ratio_cache is None else ratio_cache[:,0].copy(),
            global_rms_ratio=np.full(batch,np.nan) if ratio_cache is None else ratio_cache[:,1].copy(),
            failure_start_id=np.asarray([f[0] for f in failures],dtype=np.int64),
            failure_iteration=np.asarray([f[1] for f in failures],dtype=np.int64),
            failure_kind=np.asarray([f[2] for f in failures],dtype='<U48'),
            failure_theta=np.stack([f[3] for f in failures]) if failures else np.empty((0,planes)),
            failure_value_offsets=offsets,
            failure_value_ndim=np.asarray([f[4].ndim for f in failures],dtype=np.int64),
            failure_value_shapes=np.asarray([(*f[4].shape,*([0]*(2-f[4].ndim))) for f in failures],dtype=np.int64).reshape(-1,2),
            failure_values=np.concatenate(raw_values) if raw_values else np.empty(0),
            history=list(history),seconds=time.perf_counter()-started,checkpoint_phase=phase,interruption=interruption)
        result.update({name:value.copy() for name,value in counts.items()})
        return result
    def emit(phase):
        if checkpoint is not None:checkpoint(snapshot(phase))
    def numerical(ids,angles,raw,finite,counter,kind):
        bad=~finite.detach().cpu().numpy()
        if np.any(bad):
            counts[counter][ids[bad]]+=1;counts['nonfinite_trials'][ids[bad]]+=1
            failure(ids[bad],angles[torch.as_tensor(bad,device=device)],kind,raw[torch.as_tensor(bad,device=device)])
        return ~bad
    def decompose(matrix):
        return batched_svd(matrix,svd_backend=svd_backend,svd_driver=svd_driver,cpu_lock=cpu_lock)
    def restore(proposal,ids):
        """Restore toward zero; separate center diagnostics from safe acceptance."""
        proposal=proposal.detach();ids=np.asarray(ids,dtype=int);n=len(ids)
        counts['proposal_evaluations'][ids]+=1
        vv=torch.full((n,dimensions),float('nan'),dtype=torch.float64,device=device)
        rr=torch.full((n,2),float('nan'),dtype=torch.float64,device=device)
        alive=torch.isfinite(proposal).all(-1)&(proposal.abs()<=ANGLE_BOUND).all(-1)
        why=np.full(n,'box_bound',dtype='<U48')
        for correction in range(corrections+1):
            check_budget()
            take=torch.nonzero(alive,as_tuple=False).ravel()
            if not len(take):break
            current=proposal[take]
            with torch.no_grad():
                vals=features(current);rms=ratios(current)
            vv[take]=vals;rr[take]=rms
            take_np=take.cpu().numpy()
            finite_v=torch.isfinite(vals).all(-1);finite_r=torch.isfinite(rms).all(-1)
            good_v=numerical(ids[take_np],current,vals,finite_v,'nonfinite_constraint_trials','restoration_features')
            good_r=numerical(ids[take_np],current,rms,finite_r,'nonfinite_ratio_trials','restoration_ratios')
            good=good_v&good_r
            # A single proposal with two invalid outputs is two recorded numeric
            # events; both artifacts and per-category counts remain inspectable.
            bad_take=take[torch.as_tensor(~good,device=device)]
            alive[bad_take]=False;why[take_np[~good]]='nonfinite_proposal'
            rms_excess=(rms>.25).any(-1).cpu().numpy()
            need=good&((vals.abs().amax(-1).cpu().numpy()>center_tolerance)|rms_excess)
            if correction==corrections or not np.any(need):break
            change=take[torch.as_tensor(need,device=device)]
            j=jac(proposal[change])
            finite_j=torch.isfinite(j).all(dim=(-2,-1))
            good_j=numerical(ids[change.cpu().numpy()],proposal[change],j,finite_j,
                'nonfinite_jacobian_evaluations','restoration_jacobian')
            bad=change[torch.as_tensor(~good_j,device=device)]
            alive[bad]=False;why[bad.cpu().numpy()]='nonfinite_jacobian'
            change=change[torch.as_tensor(good_j,device=device)];j=j[torch.as_tensor(good_j,device=device)]
            if not len(change):continue
            # Tangent steps along a curved RMS surface can overshoot at second
            # order. Restore only actual excess with the SAME Newton allowance;
            # inactive RMS rows contribute neither a row nor a residual.
            excess=rr[change]>.25
            channels=torch.nonzero(excess.any(dim=0),as_tuple=False).ravel()
            residual=vv[change]
            if len(channels):
                q=squared_grad(proposal[change])
                good_q=numerical(ids[change.cpu().numpy()],proposal[change],q,
                    torch.isfinite(q).all(dim=(-2,-1)),
                    'nonfinite_rms_gradient_evaluations','restoration_rms_gradient')
                bad=change[torch.as_tensor(~good_q,device=device)]
                alive[bad]=False;why[bad.cpu().numpy()]='nonfinite_rms_gradient'
                keep=torch.as_tensor(good_q,device=device)
                change,j,excess,q=change[keep],j[keep],excess[keep],q[keep]
                if not len(change):continue
                residual=vv[change]
                q=q[:,channels]*excess[:,channels][...,None]
                extra=(rr[change][:,channels].square()-.25**2).clamp(min=0.)
                j=torch.cat((j,q),dim=1);residual=torch.cat((residual,extra),dim=1)
                counts['rms_restoration_corrections'][ids[change.cpu().numpy()]]+=excess.any(-1).cpu().numpy()
            u,s,v=decompose(j)
            use=s>s[:,:1]*rank_cutoff
            inverse=torch.where(use,1/torch.clamp(s,min=torch.finfo(s.dtype).tiny),0.)
            adjustment=torch.einsum('bkp,bk->bp',v,torch.einsum('bdk,bd->bk',u,residual)*inverse)
            proposal[change]=(proposal[change]-adjustment).detach()
            counts['retraction_corrections'][ids[change.cpu().numpy()]]+=1
            inside=torch.isfinite(proposal[change]).all(-1)&(proposal[change].abs()<=ANGLE_BOUND).all(-1)
            outside=change[~inside];alive[outside]=False;why[outside.cpu().numpy()]='box_restoration'
        # Final cached values correspond to the last evaluated angles. A Newton
        # step is followed by another evaluation before this decision is made.
        center=alive&(vv.abs().amax(-1)<=center_tolerance)
        safe_bins=alive&(vv.abs().amax(-1)<=half)
        safe_rms=(rr>=0).all(-1)&(rr<=.25).all(-1)
        valid=safe_bins&safe_rms
        alive_np=alive.cpu().numpy();center_np=center.cpu().numpy();valid_np=valid.cpu().numpy()
        counts['center_restoration_failures'][ids]+=~center_np
        bins_bad=alive_np&~safe_bins.cpu().numpy();rms_bad=alive_np&~safe_rms.cpu().numpy()
        counts['bin_rejections'][ids]+=bins_bad;counts['rms_rejections'][ids]+=rms_bad
        counts['rms_restoration_failures'][ids]+=rms_bad
        counts['center_failure_rejections'][ids]+=bins_bad&~center_np
        counts['box_rejections'][ids]+=np.isin(why,('box_bound','box_restoration'))&~alive_np
        why[bins_bad]='bin_interior';why[rms_bad]='rms_bound';why[valid_np]='restored' 
        return proposal,valid,vv,rr,why
    def update_cache(ids,angles,vals,rms,yy):
        nonlocal x
        index=torch.as_tensor(ids,device=device)
        old=x[index].clone();x[index]=angles.detach()
        values_cache[ids]=vals.detach().cpu().numpy();ratio_cache[ids]=rms.detach().cpu().numpy()
        target_cache[ids]=yy.detach().cpu().numpy()
        residual=np.max(np.abs(values_cache[ids]),axis=1)
        minimum_margin[ids]=QUANTUM/2-residual
        max_tracking[ids]=np.maximum(max_tracking[ids],residual)
        last_step[ids]=torch.linalg.vector_norm(angles-old,dim=-1).cpu().numpy()
    try:
        emit('before_initialization');check_budget()
        with torch.no_grad():
            base_f=features(x);base_r=ratios(x);base_y=targets(x)
        values_cache=base_f.cpu().numpy().copy();ratio_cache=base_r.cpu().numpy().copy()
        target_cache=base_y.cpu().numpy().copy();reference_y=target_cache.copy()
        fv=numerical(all_ids,x,base_f,torch.isfinite(base_f).all(-1),'nonfinite_constraint_trials','reference_features')
        fr=numerical(all_ids,x,base_r,torch.isfinite(base_r).all(-1),'nonfinite_ratio_trials','reference_ratios')
        fy=numerical(all_ids,x,base_y,torch.isfinite(base_y),'nonfinite_target_trials','reference_target')
        healthy=fv&fr&fy
        status[~healthy]='nonfinite_reference';reason[~healthy]='nonfinite_reference'
        residual=np.max(np.abs(values_cache),axis=1);minimum_margin[:]=QUANTUM/2-residual
        max_tracking[:]=np.where(np.isfinite(residual),residual,0.)
        reference_valid=healthy&(residual<=center_tolerance)&(ratio_cache>=0).all(-1)&(ratio_cache<=.25).all(-1)
        status[healthy&~reference_valid]='invalid_reference';reason[healthy&~reference_valid]='reference_outside_center'
        initial_reason[~reference_valid]=reason[~reference_valid]
        ids=np.flatnonzero(reference_valid)
        if len(ids):
            angles=x[torch.as_tensor(ids,device=device)];j=jac(angles)
            good=numerical(ids,angles,j,torch.isfinite(j).all(dim=(-2,-1)),
                'nonfinite_jacobian_evaluations','initial_jacobian')
            status[ids[~good]]='nonfinite_jacobian';reason[ids[~good]]='nonfinite_jacobian'
            ids=ids[good];j=j[torch.as_tensor(good,device=device)]
            if len(ids):
                _,s,v=decompose(j);used=s>s[:,:1]*rank_cutoff
                raw=tensor(kicks[ids]);initial=raw-torch.einsum('bkp,bk->bp',v,torch.einsum('bkp,bp->bk',v,raw)*used)
                counts['rank'][ids]=used.sum(-1).cpu().numpy()
                pending=np.ones(len(ids),dtype=bool)
                for h in range(halvings):
                    take=np.flatnonzero(pending)
                    if not len(take):break
                    trial,valid,vals,rms,why=restore(initial[torch.as_tensor(take,device=device)]*(.5**h),ids[take])
                    valid_np=valid.cpu().numpy();initial_reason[ids[take]]=why
                    positions=np.flatnonzero(valid_np)
                    if not len(positions):continue
                    ti=torch.as_tensor(positions,device=device);yy=targets(trial[ti])
                    finite=numerical(ids[take[positions]],trial[ti],yy,torch.isfinite(yy),'nonfinite_target_trials','initial_target')
                    accepted=positions[finite]
                    if len(accepted):
                        ai=torch.as_tensor(accepted,device=device)
                        update_cache(ids[take[accepted]],trial[ai],vals[ai],rms[ai],yy[torch.as_tensor(finite,device=device)])
                        initialized[ids[take[accepted]]]=True;pending[take[accepted]]=False
                initial_reason[ids[~initialized[ids]]]='fallback_verified_zero'
                status[ids]='active';reason[ids]='initialized'
        initial_values=x.cpu().numpy().copy();initialization_complete=True
        emit('initialized')
        while np.any(status=='active'):
            check_budget();ids=np.flatnonzero(status=='active');index=torch.as_tensor(ids,device=device)
            current=x[index];j=jac(current)
            good=numerical(ids,current,j,torch.isfinite(j).all(dim=(-2,-1)),
                'nonfinite_jacobian_evaluations','direction_jacobian')
            status[ids[~good]]='nonfinite_jacobian';reason[ids[~good]]='nonfinite_jacobian'
            ids=ids[good];j=j[torch.as_tensor(good,device=device)]
            if not len(ids):continue
            index=torch.as_tensor(ids,device=device);current=x[index];g=grad(current)*sign[index,None]
            good=numerical(ids,current,g,torch.isfinite(g).all(-1),'nonfinite_gradient_evaluations','direction_gradient')
            status[ids[~good]]='nonfinite_gradient';reason[ids[~good]]='nonfinite_gradient'
            ids=ids[good];j=j[torch.as_tensor(good,device=device)];g=g[torch.as_tensor(good,device=device)]
            if not len(ids):continue
            index=torch.as_tensor(ids,device=device);current=x[index];rms=tensor(ratio_cache[ids])
            q=torch.zeros((len(ids),2,planes),dtype=torch.float64,device=device)
            needed=torch.nonzero((rms>=active_rms_threshold).any(-1),as_tuple=False).ravel()
            if len(needed):
                q[needed]=squared_grad(current[needed])
            good=numerical(ids,current,q,torch.isfinite(q).all(dim=(-2,-1)),
                'nonfinite_rms_gradient_evaluations','direction_rms_gradient')
            status[ids[~good]]='nonfinite_rms_gradient';reason[ids[~good]]='nonfinite_rms_gradient'
            ids=ids[good];keep=torch.as_tensor(good,device=device);j,g,q,rms=j[keep],g[keep],q[keep],rms[keep]
            if not len(ids):continue
            index=torch.as_tensor(ids,device=device);current=x[index]
            tangent=active_tangent(j,g,current,rms,q,cutoff=rank_cutoff,active_rms_threshold=active_rms_threshold,
                svd_driver=svd_driver,svd_backend=svd_backend,cpu_lock=cpu_lock)
            direction=tangent['direction'];norm=torch.linalg.vector_norm(direction,dim=-1)
            direction_theta[ids]=current.detach().cpu().numpy();projected_norm[ids]=norm.cpu().numpy()
            gradient_norm[ids]=torch.linalg.vector_norm(g,dim=-1).cpu().numpy()
            counts['rank'][ids]=tangent['rank'].cpu().numpy()
            rms_active[ids]=tangent['active_rms'].cpu().numpy();box_active[ids]=tangent['active_box'].cpu().numpy()
            counts['active_rms_count'][ids]=rms_active[ids].sum(-1);counts['active_box_count'][ids]=box_active[ids].sum(-1)
            for key in ('rms','box'):
                counts[f'maximum_active_{key}_count'][ids]=np.maximum(counts[f'maximum_active_{key}_count'][ids],counts[f'active_{key}_count'][ids])
            direction=direction/torch.clamp(norm[:,None],min=1e-30)
            eligible=(norm>1e-14)&tangent['box_satisfied'];eligible_np=eligible.cpu().numpy()
            reason[ids]='projected_gradient_zero';reason[ids[~tangent['box_satisfied'].cpu().numpy()]]='box_tangent_not_resolved'
            reason[ids[eligible_np]]='no_feasible_improving_step'
            accepted=np.zeros(len(ids),dtype=bool)
            for h in range(halvings):
                pending=np.flatnonzero(eligible_np&~accepted)
                if not len(pending):break
                pi=torch.as_tensor(pending,device=device)
                trial,valid,vals,trial_rms,why=restore(current[pi]+step_norm*(.5**h)*direction[pi],ids[pending])
                reason[ids[pending]]=why
                feasible=np.flatnonzero(valid.cpu().numpy())
                if not len(feasible):continue
                fi=torch.as_tensor(feasible,device=device);yy=targets(trial[fi])
                finite=numerical(ids[pending[feasible]],trial[fi],yy,torch.isfinite(yy),'nonfinite_target_trials','proposal_target')
                old=tensor(target_cache[ids[pending[feasible]]])
                improve=((yy-old)*sign[torch.as_tensor(ids[pending[feasible]],device=device)]>1e-14).cpu().numpy()&finite
                nonimprove=finite&~improve
                counts['nonimproving_rejections'][ids[pending[feasible[nonimprove]]]]+=1
                reason[ids[pending[feasible[nonimprove]]]]='no_signed_improvement'
                reason[ids[pending[feasible[~finite]]]]='nonfinite_target'
                chosen=feasible[improve]
                if len(chosen):
                    ci=torch.as_tensor(chosen,device=device)
                    update_cache(ids[pending[chosen]],trial[ci],vals[ci],trial_rms[ci],yy[torch.as_tensor(improve,device=device)])
                    accepted[pending[chosen]]=True;reason[ids[pending[chosen]]]='accepted'
            counts['iterations'][ids]+=1;counts['rejected'][ids]+=~accepted
            counts['stalls'][ids]=np.where(accepted,0,counts['stalls'][ids]+1)
            status[ids[counts['iterations'][ids]>=max_iterations]]='iteration_cap'
            stall=ids[(counts['stalls'][ids]>=max_stalls)&(status[ids]=='active')];status[stall]='stalled'
            if not history or int(counts['iterations'].max())//20>history[-1]['iteration']//20 or not np.any(status=='active'):
                plain=lambda value:[float(v) if np.isfinite(v) else None for v in value]
                finite_extreme=lambda value,fn:float(fn(value[np.isfinite(value)])) if np.isfinite(value).any() else None
                history.append(dict(iteration=int(counts['iterations'].max()),targets=plain(target_cache),
                    gaps=plain(target_cache-reference_y),ranks=counts['rank'].tolist(),
                    maximum_constraint=finite_extreme(np.abs(values_cache),np.max),maximum_rms_ratio=finite_extreme(ratio_cache,np.max),
                    minimum_margin=finite_extreme(minimum_margin,np.min),active_rms_counts=counts['active_rms_count'].tolist(),
                    active_box_counts=counts['active_box_count'].tolist(),stall_reasons=reason.tolist(),
                    center_reached=(np.max(np.abs(values_cache),axis=1)<=center_tolerance).tolist(),seconds=time.perf_counter()-started))
                emit('periodic')
        feasible=np.isfinite(values_cache).all(-1)&np.isfinite(ratio_cache).all(-1)&np.isfinite(target_cache)
        feasible&=(np.max(np.abs(values_cache),axis=1)<=half)&(ratio_cache>=0).all(-1)&(ratio_cache<=.25).all(-1)
        feasible&=(np.abs(x.cpu().numpy())<=ANGLE_BOUND).all(-1)
        terminal_good=np.isin(status,('stalled','iteration_cap'))
        if np.any(terminal_good&~feasible):raise FloatingPointError('a retained search endpoint lost feasibility')
        finished=True;result=snapshot('complete');result['feasible']=feasible
        return result
    except BaseException as exc:
        failure_exception=exc;interruption=f'{type(exc).__name__}: {exc}'
        status[np.isin(status,('active','initializing'))]='incomplete'
        raise
    finally:
        try:emit('complete' if finished else 'interrupted')
        except Exception as checkpoint_error:
            if interruption is None:raise
            failure_exception.add_note(f'Final checkpoint also failed: {checkpoint_error}')


projected_batch_v2=projected_batch
