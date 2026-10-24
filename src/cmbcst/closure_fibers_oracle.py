"""Independent real-coordinate moment contractions for closure certificates.

This module does not call the complex feature or moment kernels. It shares the
declared sampled linear operator, then contracts a real four-coordinate
covariance matrix and its derivative. Physical sampling is replayed separately.
"""
from __future__ import annotations
import numpy as np


def covariance_moments(values, derivatives):
    """Eight statistics and derivatives from X=(Re A,Im A,Re B,Im B)."""
    if np.iscomplexobj(values) or np.iscomplexobj(derivatives):
        raise ValueError('real covariance coordinates required')
    x, dx = np.asarray(values, dtype=np.float64), np.asarray(derivatives, dtype=np.float64)
    if (x.ndim < 2 or x.shape[-1] != 4 or x.shape[-2] < 4 or dx.shape != x.shape
            or not np.isfinite(x).all() or not np.isfinite(dx).all()):
        raise ValueError('matching finite [...,samples,4] covariance coordinates required')
    x, dx = x-x.mean(axis=-2, keepdims=True), dx-dx.mean(axis=-2, keepdims=True)
    count = x.shape[-2]
    cov = np.einsum('...ni,...nj->...ij', x, x)/count
    dcov = (np.einsum('...ni,...nj->...ij', dx, x)
            + np.einsum('...ni,...nj->...ij', x, dx))/count
    va, vb = cov[...,0,0]+cov[...,1,1], cov[...,2,2]+cov[...,3,3]
    dva, dvb = dcov[...,0,0]+dcov[...,1,1], dcov[...,2,2]+dcov[...,3,3]
    if np.any(va <= 0) or np.any(vb <= 0):
        raise ValueError('zero covariance-block variance')
    re, im = cov[...,0,2]+cov[...,1,3], cov[...,1,2]-cov[...,0,3]
    dre, dim = dcov[...,0,2]+dcov[...,1,3], dcov[...,1,2]-dcov[...,0,3]
    ra, rb = np.sum(x[...,:2]**2,axis=-1), np.sum(x[...,2:]**2,axis=-1)
    dra = 2*np.sum(x[...,:2]*dx[...,:2],axis=-1)
    drb = 2*np.sum(x[...,2:]*dx[...,2:],axis=-1)
    den, dden = va*vb, dva*vb+va*dvb
    cross, dcross = cov[...,:2,2:], dcov[...,:2,2:]
    fourth = (ra*rb).mean(axis=-1)
    dfourth = (dra*rb+ra*drb).mean(axis=-1)
    numerator = fourth-den-2*np.sum(cross*cross,axis=(-2,-1))
    dnumerator = dfourth-dden-4*np.sum(cross*dcross,axis=(-2,-1))
    root = np.sqrt(den)
    root_rate = .5*(dva/va+dvb/vb)
    stat = np.stack([va,vb,re,.25*(va+vb+2*re),.25*(va+vb-2*re),
                     re/root,numerator/den,im/root],axis=-1)
    derivative = np.stack([dva,dvb,dre,.25*(dva+dvb+2*dre),.25*(dva+dvb-2*dre),
                          (dre-re*root_rate)/root,
                          dnumerator/den-numerator*dden/(den*den),
                          (dim-im*root_rate)/root],axis=-1)
    if not np.isfinite(stat).all() or not np.isfinite(derivative).all():
        raise ValueError('nonfinite covariance contraction')
    return stat, derivative


def oracle_features(family, theta, u=0):
    """Independent [...,52] features at stored integer smoothing time 0..3."""
    if (isinstance(u,(bool,np.bool_)) or not np.isscalar(u) or not np.isfinite(u)
            or int(u) != u or not 0 <= int(u) <= 3):
        raise ValueError('integer smoothing time 0 through 3 required')
    u = int(u)
    if np.iscomplexobj(theta):
        raise ValueError('real rotation angles required')
    theta = np.asarray(theta,dtype=np.float64)
    rates = np.asarray(family['rates'],dtype=np.float64)
    valid = np.asarray(family['valid'])
    if (theta.ndim < 1 or rates.ndim != 1 or theta.shape[-1] != len(rates)
            or valid.dtype != bool or valid.ndim != 3 or valid.shape[:2] != (4,4)
            or not np.isfinite(theta).all() or not np.isfinite(rates).all()
            or np.any(valid.sum(axis=(1,2)) < 4)):
        raise ValueError('invalid real angles, rates or shared mask')
    n = valid.shape[-1]
    base = np.asarray(family['base'],dtype=np.complex128)
    dbase = np.asarray(family['base_derivative'],dtype=np.complex128)
    band_base = np.asarray(family['base_band'],dtype=np.complex128)
    cosine = np.asarray(family['cos_basis'],dtype=np.complex128)
    sine = np.asarray(family['sin_basis'],dtype=np.complex128)
    p0,p1 = np.asarray(family['phase0']),np.asarray(family['phase1'])
    band = np.asarray(family['band_id'])
    if (n < 1 or base.shape != (4,2,17*n) or dbase.shape != base.shape
            or band_base.shape != (4,2,4,n) or cosine.shape != (len(rates),17*n)
            or sine.shape != cosine.shape or p0.shape != valid.shape or p1.shape != valid.shape
            or band.shape != rates.shape or band.dtype.kind not in 'iu'
            or np.any((band<0)|(band>3))
            or not all(np.isfinite(x).all() for x in (base,dbase,band_base,cosine,sine,p0,p1))):
        raise ValueError('inconsistent finite sampled-operator arrays')
    decay = np.exp(-u*rates)
    weights_c,weights_s = (np.cos(theta)-1)*decay,np.sin(theta)*decay
    # Einsum contraction instead of invoking the production matrix evaluator.
    delta = (np.einsum('...p,pm->...m',weights_c,cosine)
             + np.einsum('...p,pm->...m',weights_s,sine))
    ddelta = (np.einsum('...p,pm->...m',-weights_c*rates,cosine)
              + np.einsum('...p,pm->...m',-weights_s*rates,sine))
    y,dy = base[u]+delta[...,None,:],dbase[u]+ddelta[...,None,:]
    # Geodesic-frame subtraction is applied to shared samples, not recoded moments.
    endpoint = y[...,n:].reshape(y.shape[:-1]+(4,4,n))
    dendpoint = dy[...,n:].reshape(dy.shape[:-1]+(4,4,n))
    increment = endpoint*p1-y[...,:n][...,None,None,:]*p0
    dincrement = dendpoint*p1-dy[...,:n][...,None,None,:]*p0
    statistics,responses = [],[]
    for lag in range(4):
        selected = increment[...,lag,:,:][...,valid[lag]]
        dselected = dincrement[...,lag,:,:][...,valid[lag]]
        a,b = selected[...,0,:],selected[...,1,:]
        da,db = dselected[...,0,:],dselected[...,1,:]
        x = np.stack([a.real,a.imag,b.real,b.imag],axis=-1)
        dx = np.stack([da.real,da.imag,db.real,db.imag],axis=-1)
        stat,response = covariance_moments(x,dx)
        statistics.append(stat);responses.append(response)
    statistics,responses = np.stack(statistics,axis=-2),np.stack(responses,axis=-2)
    band_power = []
    for index in range(4):
        mask = (band==index).astype(np.float64)
        band_delta = (np.einsum('...p,pm,p->...m',weights_c,cosine[:,:n],mask)
                      + np.einsum('...p,pm,p->...m',weights_s,sine[:,:n],mask))
        ab = band_base[u,:,index,:]+band_delta[...,None,:]
        ar,ai,br,bi = ab[...,0,:].real,ab[...,0,:].imag,ab[...,1,:].real,ab[...,1,:].imag
        band_power.append(np.stack([(ar*ar+ai*ai).mean(axis=-1),
                                    (br*br+bi*bi).mean(axis=-1),
                                    (ar*br+ai*bi).mean(axis=-1)],axis=-1))
    band_power = np.stack(band_power,axis=-2)
    result = np.concatenate([statistics.reshape(statistics.shape[:-2]+(32,)),
                             band_power.reshape(band_power.shape[:-2]+(12,)),
                             responses[..., :,3],responses[..., :,6]],axis=-1)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite independent closure features')
    return result
