"""Spin-2 HEALPix synthesis and area-weighted adjoints, never inverse claims."""
from __future__ import annotations

import hashlib
import importlib
import os
from pathlib import Path
import numpy as np

from .models import MissingOptionalDependency,ORDERINGS,canonical_json,validate_geometry


def _dependency(name):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise MissingOptionalDependency(f'{name} is required for this requested operator; no substitute calculation ran') from exc


def alm_layout(lmax):
    """Complete healpy/ducc m-major storage: real m=0, both E/B rows."""
    if isinstance(lmax,(bool,np.bool_)) or not isinstance(lmax,(int,np.integer)) or lmax<2:
        raise ValueError('spin-2 lmax must be an integer at least two')
    ell=np.concatenate([np.arange(m,lmax+1) for m in range(lmax+1)])
    m=np.concatenate([np.full(lmax+1-order,order,dtype=int) for order in range(lmax+1)])
    return ell,m


def _alms(values,lmax):
    values=np.asarray(values)
    count=(lmax+1)*(lmax+2)//2
    if (values.dtype.kind!='c' or values.dtype.itemsize>16 or values.shape!=(2,count)
            or not np.isfinite(values).all()):
        raise ValueError('finite complex E/B alms with shape [2,(lmax+1)*(lmax+2)//2] required')
    ell,_=alm_layout(lmax)
    if np.any(values[:,:lmax+1].imag!=0.) or np.any(values[:,ell<2]!=0.):
        raise ValueError('real m=0 and zero ell<spin coefficients required')
    return np.ascontiguousarray(values,dtype=np.complex128)


def _maps(values,nside):
    values=np.asarray(values)
    if (values.dtype.kind!='f' or values.dtype.itemsize>8 or values.shape!=(2,12*nside**2)
            or not np.isfinite(values).all()):
        raise ValueError('finite floating RING COSMO Q/U maps with shape [2,12*nside**2] required')
    return np.ascontiguousarray(values,dtype=np.float64)


def harmonic_inner_product(a,b,lmax):
    """Real-field coefficient metric, counting positive m twice."""
    a,b=_alms(a,lmax),_alms(b,lmax)
    _,m=alm_layout(lmax)
    return float(np.sum((a.conj()*b).real*np.where(m==0,1.,2.)))


def pixel_inner_product(a,b,nside):
    """Equal-area HEALPix pixel metric over both real Q/U components."""
    if isinstance(nside,(bool,np.bool_)) or not isinstance(nside,(int,np.integer)) or nside<1 or nside&(nside-1):
        raise ValueError('positive power-of-two nside required')
    a,b=_maps(a,nside),_maps(b,nside)
    return float(np.sum(a*b)*(4*np.pi/(12*nside**2)))


def reorder_healpix(values,nside,source,target):
    """An exact pixel permutation; no interpolation or coordinate rotation."""
    if source not in ORDERINGS or target not in ORDERINGS:
        raise ValueError('explicit RING or NESTED source/target required')
    if isinstance(nside,(bool,np.bool_)) or not isinstance(nside,(int,np.integer)) or nside<1 or nside&(nside-1):
        raise ValueError('positive power-of-two nside required')
    values=np.asarray(values)
    if values.ndim<1 or values.shape[-1]!=12*nside**2:
        raise ValueError('last axis must contain the full HEALPix pixel count')
    if source==target:
        return values.copy()
    ducc=_dependency('ducc0')
    nested_to_ring=ducc.healpix.Healpix_Base(int(nside),'NESTED').nest2ring(np.arange(values.shape[-1],dtype=np.int64))
    if source=='RING':
        return values[...,nested_to_ring].copy()
    result=np.empty_like(values);result[...,nested_to_ring]=values
    return result


class DuccHealpixOperator:
    """Spin-2 E/B ↔ Q/U operators in declared RING/COSMO conventions.

    ``adjoint`` is the adjoint of ``synthesize`` under the real-field harmonic
    metric and equal-area pixel metric. It is not a sky reconstruction inverse.
    Analysis masks/weights must be applied explicitly by the caller; these raw
    array operators never infer masks or instrument responses from pixel values.
    """
    backend_name='ducc0'

    def __init__(self,nside,lmax,nthreads=8):
        self.nside,self.lmax=validate_geometry(nside,lmax)
        if isinstance(nthreads,(bool,np.bool_)) or not isinstance(nthreads,(int,np.integer)) or not 1<=nthreads<=8:
            raise ValueError('nthreads must be an integer from one through eight')
        self.nthreads=int(nthreads)
        self.npix=12*self.nside**2
        self.nalm=(self.lmax+1)*(self.lmax+2)//2
        self.pixel_area=4*np.pi/self.npix

    def synthesize(self,alms):
        alms=_alms(alms,self.lmax)
        ducc=_dependency('ducc0')
        geometry=ducc.healpix.Healpix_Base(self.nside,'RING').sht_info()
        result=ducc.sht.synthesis(alm=alms,spin=2,lmax=self.lmax,mmax=self.lmax,
            nthreads=self.nthreads,**geometry)
        return _maps(result,self.nside)

    def adjoint(self,maps):
        maps=_maps(maps,self.nside)
        ducc=_dependency('ducc0')
        geometry=ducc.healpix.Healpix_Base(self.nside,'RING').sht_info()
        result=ducc.sht.adjoint_synthesis(map=maps,spin=2,lmax=self.lmax,mmax=self.lmax,
            nthreads=self.nthreads,**geometry)*self.pixel_area
        return _alms(result,self.lmax)

    def provenance(self):
        try:
            backend=_dependency(self.backend_name)
            availability=dict(available=True,version=str(getattr(backend,'__version__','unknown')))
        except MissingOptionalDependency as exc:
            availability=dict(available=False,version=None,unavailable_reason=str(exc))
        direct_threads=self.backend_name=='ducc0'
        record=dict(backend=self.backend_name,**availability,nside=self.nside,lmax=self.lmax,mmax=self.lmax,
            nthreads=self.nthreads if direct_threads else None,requested_nthreads=self.nthreads,
            thread_control='explicit nthreads argument' if direct_threads else 'caller-controlled OpenMP; healpy spin API has no nthreads argument',
            openmp_environment=None if direct_threads else os.environ.get('OMP_NUM_THREADS'),
            spin=2,ordering='RING',polarization_convention='COSMO',
            harmonic_layout='complete healpy m-major E/B; real m=0; ell<2 zero',
            harmonic_metric='sum_EB,m>=0 (1 if m=0 else 2)*Re(conj(a)*b)',
            pixel_metric='4*pi/npix * sum_QU,pixel(a*b)',
            adjoint_kind='area-weighted synthesis adjoint; not an inverse',
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        record['operator_sha256']=hashlib.sha256(canonical_json(record).encode()).hexdigest()
        return record


class HealpyHealpixOperator(DuccHealpixOperator):
    """Optional independent HEALPix spin-2 reference; no ducc fallback.

    Healpy's spin functions expose no per-call thread count. The caller must set
    the process OpenMP limit before backend initialization; provenance reports
    that distinction rather than claiming the requested count was enforced.
    """
    backend_name='healpy'

    def synthesize(self,alms):
        alms=_alms(alms,self.lmax)
        healpy=_dependency('healpy')
        result=healpy.alm2map_spin(alms,self.nside,spin=2,lmax=self.lmax,mmax=self.lmax)
        return _maps(np.asarray(result),self.nside)

    def adjoint(self,maps):
        maps=_maps(maps,self.nside)
        healpy=_dependency('healpy')
        result=healpy.map2alm_spin(maps,spin=2,lmax=self.lmax,mmax=self.lmax)
        return _alms(np.asarray(result),self.lmax)
