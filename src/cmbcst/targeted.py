"""Six fixed physical response contrasts and three supervised linear heads."""
from __future__ import annotations
import numpy as np
from .response import fit_response_model

RESPONSE_INDICES = (7,16,73,79,82,88)
FAMILIES = ('quadratic','compact','filament')


def six_responses(values):
    a = np.asarray(values,dtype=np.float64)
    if a.ndim!=2 or a.shape[1]!=90 or not np.isfinite(a).all():
        raise ValueError('need finite 90-column responses')
    return a[:,RESPONSE_INDICES]


def ensure_budget(elapsed, limit=1800.):
    if not np.isfinite(elapsed) or elapsed>=limit:
        raise TimeoutError('fixed accumulated benchmark budget exhausted')


def coordinates(b,r,model):
    b,r = np.asarray(b,dtype=float),np.asarray(r,dtype=float)
    if b.ndim!=2 or r.shape!=(len(b),6) or b.shape[1]!=17 or not np.isfinite(b).all() or not np.isfinite(r).all():
        raise ValueError('need finite 17 baseline and six response values')
    bz = (b-model['baseline_mean'])/model['baseline_scale']
    rz = (r-model['response_mean'])/model['response_scale']
    return bz,np.concatenate([bz,rz-bz@model['coefficients']],axis=1)


def fit_targeted(b,r,labels):
    labels = np.asarray(labels)
    if len(labels)!=len(b) or set(labels)!=set(('null',)+FAMILIES):
        raise ValueError('require null and every physical family in training')
    null = labels=='null'
    model = fit_response_model(np.asarray(b)[null],np.asarray(r)[null])
    bz,joint = coordinates(b,r,model)
    detectors = []
    for x in (bz,joint):
        center = x[null].mean(0)
        cov = np.cov(x[null],rowvar=False)
        regularized = .9*cov+.1*np.eye(x.shape[1])
        weights = np.stack([np.linalg.solve(regularized,x[labels==family].mean(0)-center) for family in FAMILIES],axis=1)
        null_scores = (x[null]-center)@weights
        scale = null_scores.std(0,ddof=1)
        if np.any(scale<=1e-12):
            raise ValueError('degenerate family head')
        detectors.append({'center':center.tolist(),'weights':weights.tolist(),'scale':scale.tolist(),
            'condition_number':float(np.linalg.cond(regularized))})
    omni = np.einsum('bi,ij,bj->b',bz[null],model['baseline_precision'],bz[null])
    model.update(detectors=detectors,omnibus_mean=float(omni.mean()),omnibus_scale=float(omni.std(ddof=1)),
        labels=FAMILIES,training_counts={label:int(np.sum(labels==label)) for label in ('null',)+FAMILIES})
    return model


def score_targeted(b,r,model):
    bz,joint = coordinates(b,r,model)
    scores = [(np.einsum('bi,ij,bj->b',bz,model['baseline_precision'],bz)-model['omnibus_mean'])/model['omnibus_scale']]
    for x,head in zip((bz,joint),model['detectors']):
        scores.append(np.max(((x-head['center'])@head['weights'])/head['scale'],axis=1))
    return np.column_stack(scores)


def score_targeted_cuda(b,r,model,device='cuda:0'):
    import torch
    tensor = lambda value:torch.as_tensor(value,dtype=torch.float64,device=device)
    # Validate through the same public input contract, then compute separately.
    coordinates(b,r,model)
    bz = (tensor(b)-tensor(model['baseline_mean']))/tensor(model['baseline_scale'])
    rz = (tensor(r)-tensor(model['response_mean']))/tensor(model['response_scale'])
    joint = torch.cat([bz,rz-bz@tensor(model['coefficients'])],dim=1)
    scores = [(torch.einsum('bi,ij,bj->b',bz,tensor(model['baseline_precision']),bz)-model['omnibus_mean'])/model['omnibus_scale']]
    for x,head in zip((bz,joint),model['detectors']):
        scores.append(torch.max(((x-tensor(head['center']))@tensor(head['weights']))/tensor(head['scale']),dim=1).values)
    return torch.stack(scores,dim=1).cpu().numpy()


def choose_blind_spots(rows):
    selected = {}
    for family in FAMILIES:
        eligible = [row for row in rows if row['family']==family and row['baseline_rate']<=.15]
        selected[family] = min(eligible,key=lambda row:(-row['gain'],row['amplitude'],row['id']))['id'] if eligible else None
    return selected


def wilson_interval(flags):
    a = np.asarray(flags,dtype=bool)
    if a.ndim!=1 or not len(a):
        raise ValueError('need nonempty flags')
    n,p,z = len(a),float(a.mean()),1.959963984540054
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [float(center-half),float(center+half)]
