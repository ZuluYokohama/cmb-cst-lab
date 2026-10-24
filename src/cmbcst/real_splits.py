"""Exploratory split-polarization geometry and connected fourth moments.

Known spherical geometry and Wick subtractions, not a novelty or anomaly test.
"""
from __future__ import annotations
import numpy as np

MODES = ('transported', 'naive_QU', 'amplitude_only', 'misaligned_B')
STATISTICS = ('auto_A', 'auto_B', 'cross_S2', 'mean_S2', 'difference_S2',
              'coherence', 'connected4_normalized', 'imaginary_cross_normalized')


def local_frame(loc):
    loc = np.asarray(loc, dtype=float)
    if loc.shape[-1] != 2 or not np.isfinite(loc).all() or np.any((loc[...,0]<0)|(loc[...,0]>np.pi)):
        raise ValueError('finite spherical colatitude/longitude required')
    t,p = loc[...,0],loc[...,1]
    normal = np.stack([np.sin(t)*np.cos(p),np.sin(t)*np.sin(p),np.cos(t)],-1)
    south = np.stack([np.cos(t)*np.cos(p),np.cos(t)*np.sin(p),-np.sin(t)],-1)
    east = np.stack([-np.sin(p),np.cos(p),np.zeros_like(p)],-1)
    return normal,south,east


def spherical_pairs(loc,lags_degrees=(.5,1.,2.,4.),bearings_degrees=(0.,45.,90.,135.)):
    """Geodesic endpoint and exp(-2i chi) pair-frame factors at both ends."""
    loc = np.asarray(loc,dtype=float)
    lag,bearing = np.radians(lags_degrees),np.radians(bearings_degrees)
    if loc.ndim!=2 or not len(loc) or lag.ndim!=1 or bearing.ndim!=1 or not len(lag) or not len(bearing) or not np.isfinite(lag).all() or not np.isfinite(bearing).all() or np.any((lag<=0)|(lag>=np.pi)):
        raise ValueError('nondegenerate spherical pairs required')
    n,s,e = local_frame(loc)
    direction = np.cos(bearing)[:,None,None]*s+np.sin(bearing)[:,None,None]*e
    endpoint = np.cos(lag)[:,None,None,None]*n+np.sin(lag)[:,None,None,None]*direction
    tangent = -np.sin(lag)[:,None,None,None]*n+np.cos(lag)[:,None,None,None]*direction
    endloc = np.stack([np.arccos(np.clip(endpoint[...,2],-1,1)),np.mod(np.arctan2(endpoint[...,1],endpoint[...,0]),2*np.pi)],-1)
    _,s1,e1 = local_frame(endloc)
    chi1 = np.arctan2(np.sum(tangent*e1,-1),np.sum(tangent*s1,-1))
    phase0 = np.broadcast_to(np.exp(-2j*bearing)[None,:,None],chi1.shape).copy()
    return endloc,phase0,np.exp(-2j*chi1)


def increment_views(anchor,endpoint,phase0,phase1):
    anchor,endpoint = np.asarray(anchor),np.asarray(endpoint)
    if anchor.ndim!=2 or endpoint.ndim!=4 or endpoint.shape[0]!=len(anchor) or endpoint.shape[-1]!=anchor.shape[-1] or endpoint.shape[1:]!=np.shape(phase0) or np.shape(phase0)!=np.shape(phase1) or not all(np.isfinite(x).all() for x in (anchor,endpoint,phase0,phase1)):
        raise ValueError('matching finite map/pair arrays required')
    return np.stack([endpoint*phase1-anchor[:,None,None,:]*phase0,
        endpoint-anchor[:,None,None,:],np.abs(endpoint)-np.abs(anchor[:,None,None,:])],axis=0)


def cross_statistics(a,b,device=None):
    """Centered two-point and connected four-point plug-in estimates.

    Wick identity: E|a|²|b|² - E|a|²E|b|² - |Eab*|² - |Eab|².
    No finite-sample unbiasedness or spatial Gaussian null is asserted.
    """
    a,b = np.asarray(a,dtype=np.complex128),np.asarray(b,dtype=np.complex128)
    if a.shape!=b.shape or a.ndim!=1 or len(a)<4 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('matching finite complex samples required')
    if device is None:
        a,b = a-a.mean(),b-b.mean()
        aa,bb = np.abs(a)**2,np.abs(b)**2
        va,vb = aa.mean(),bb.mean()
        z,w = np.mean(a*b.conj()),np.mean(a*b)
        if min(va,vb)<=0:
            raise ValueError('zero split variance')
        norm = np.sqrt(va*vb)
        return np.array([va,vb,z.real,np.mean(np.abs((a+b)/2)**2),np.mean(np.abs((a-b)/2)**2),
            z.real/norm,(np.mean(aa*bb)-va*vb-abs(z)**2-abs(w)**2)/(va*vb),z.imag/norm])
    import torch
    x,y = torch.as_tensor(a,device=device),torch.as_tensor(b,device=device)
    x,y = x-x.mean(),y-y.mean()
    aa,bb = torch.abs(x)**2,torch.abs(y)**2
    va,vb = aa.mean(),bb.mean()
    z,w = (x*y.conj()).mean(),(x*y).mean()
    if min(va.item(),vb.item())<=0:
        raise ValueError('zero CUDA split variance')
    norm = torch.sqrt(va*vb)
    return torch.stack([va,vb,z.real,(torch.abs((x+y)/2)**2).mean(),(torch.abs((x-y)/2)**2).mean(),
        z.real/norm,((aa*bb).mean()-va*vb-torch.abs(z)**2-torch.abs(w)**2)/(va*vb),z.imag/norm]).cpu().numpy()


def tile_statistics(views,valid,tile_ids,min_pairs=128,device=None):
    """Same geometry-only tile selection for every frequency and representation."""
    x,valid,tile_ids = np.asarray(views),np.asarray(valid),np.asarray(tile_ids)
    if x.ndim!=5 or x.shape[:2]!=(3,8) or valid.dtype!=bool or valid.shape!=x.shape[2:] or tile_ids.shape!=(x.shape[-1],) or not np.isfinite(x).all():
        raise ValueError('need [3,8,lag,direction,anchor] views and shared mask')
    records,output = [],[]
    for tile in np.unique(tile_ids):
        take = valid & (tile_ids==tile)[None,None,:]
        counts = take.sum(axis=(1,2))
        accepted = bool(np.all(counts>=min_pairs))
        records.append(dict(tile=int(tile),counts=counts.tolist(),accepted=accepted))
        if not accepted:
            continue
        block = np.empty((4,4,x.shape[2],8))
        for lag in range(x.shape[2]):
            v = x[:,:,lag][:,:,take[lag]]
            for mode in range(4):
                for frequency in range(4):
                    aa,bb = v[0 if mode==3 else mode,2*frequency:2*frequency+2]
                    if mode==3:
                        bb = np.roll(bb,len(bb)//3)
                    block[mode,frequency,lag] = cross_statistics(aa,bb,device=device)
        output.append(block)
    if len(output)<4:
        raise ValueError('insufficient geometry-selected tiles')
    return np.stack(output,axis=2),records


def bootstrap_tiles(features,indices,device=None):
    f,ix = np.asarray(features),np.asarray(indices)
    if f.ndim!=5 or f.shape[:2]!=(4,4) or f.shape[-1]!=8 or ix.ndim!=2 or ix.shape[1]!=f.shape[2] or ix.dtype.kind not in 'iu' or np.any((ix<0)|(ix>=f.shape[2])) or not np.isfinite(f).all():
        raise ValueError('invalid tile-resampling input')
    if device is None:
        return np.stack([f[:,:,row].mean(axis=2) for row in ix])
    import torch
    tensor = torch.as_tensor(f,device=device)
    return torch.cat([tensor[:,:,torch.as_tensor(ix[first:first+128],device=device)].mean(dim=3).permute(2,0,1,3,4)
        for first in range(0,len(ix),128)]).cpu().numpy()
