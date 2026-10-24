"""Stored full-sky spectral rotations with an empirical split-difference lock.

These are counterfactual harmonic states, not draws from a physical noise law.
No cut-sky spectrum is automatically preserved by this parameterization.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np


def harmonic_layout(lmax):
    """Healpy/ducc m-major, m>=0 coefficient layout."""
    if not isinstance(lmax,(int,np.integer)) or lmax<2:raise ValueError('lmax must be at least two')
    m=np.concatenate([np.full(lmax+1-k,k,dtype=np.int64) for k in range(lmax+1)])
    ell=np.concatenate([np.arange(k,lmax+1,dtype=np.int64) for k in range(lmax+1)])
    return ell,m


def _family_hash(value):
    value=np.ascontiguousarray(value)
    digest=hashlib.sha256(value.dtype.str.encode()+str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _family_diagnostic(value):
    """Plain JSON diagnostics; nonfinite errors remain explicit null values."""
    if isinstance(value,dict):return {str(k):_family_diagnostic(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_family_diagnostic(v) for v in value]
    if isinstance(value,np.ndarray):return _family_diagnostic(value.tolist())
    if isinstance(value,np.generic):return _family_diagnostic(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def _real_shell(alms,shell,ell):
    value=alms[...,ell==shell]
    return np.concatenate((value[...,:1].real,np.sqrt(2)*value[...,1:].real,
        np.sqrt(2)*value[...,1:].imag),axis=-1)


def _complex_shell(value,shell):
    result=np.empty(value.shape[:-1]+(shell+1,),dtype=np.complex128)
    result[...,0]=value[...,0]
    result[...,1:]=(value[...,1:shell+1]+1j*value[...,shell+1:])/np.sqrt(2)
    return result


def _spectral_gram(alms,ell,m,lmax):
    rows=np.asarray(alms).reshape(np.shape(alms)[:-3]+(4,len(ell)))
    result=np.empty(rows.shape[:-2]+(lmax+1,4,4))
    for shell in range(lmax+1):
        take=ell==shell;values=rows[...,take];weights=np.where(m[take]==0,1.,2.)
        result[...,shell,:,:]=np.einsum('...in,...jn,n->...ij',values,values.conj(),weights).real/(2*shell+1)
    return result


class RotationFamily:
    """Disjoint real-shell rotations fixing both E/B difference vectors.

    ``cos_basis``/``sin_basis`` have shape [P,2 E/B,nalm] and are common
    increments to both splits. Numerical planes are serialized explicitly;
    ``from_arrays`` never reruns QR or an RNG. Angles are restricted to +/-pi/4.
    """
    FORMAT='cmbcst.integration.rotation_family.v1'
    ANGLE_LIMIT=np.pi/4

    def __init__(self,reference_alms,shells,planes_per_shell=4,seed=82024102027):
        self._reference(reference_alms,shells,planes_per_shell,seed)
        width=2*int(self.shells.max())+1
        self.plane_shells=np.repeat(self.shells,self.planes_per_shell)
        self.plane_u=np.zeros((self.nplanes,width));self.plane_v=np.zeros_like(self.plane_u)
        self.noise_rank=np.zeros(len(self.shells),dtype=np.int64)
        difference=self.difference_alms
        for shell_index,shell in enumerate(self.shells):
            d=_real_shell(difference,int(shell),self.ell)
            _,singular,v=np.linalg.svd(d,full_matrices=False)
            rank=int(np.sum(singular>(singular[0]*1e-12 if len(singular) else 0.)))
            if 2*int(shell)+1-rank<2*self.planes_per_shell:
                raise ValueError(f'insufficient difference-complement dimension at ell={shell}: rank={rank}')
            self.noise_rank[shell_index]=rank
            generator=np.random.default_rng(np.random.SeedSequence([self.seed,int(shell)]))
            normal=generator.normal(size=(2*int(shell)+1,2*self.planes_per_shell))
            for _ in range(2):normal-=v[:rank].T@(v[:rank]@normal)
            q,upper=np.linalg.qr(normal,mode='reduced')
            if np.min(np.abs(np.diag(upper)))<=1e-12:raise ValueError('degenerate numerical rotation basis')
            q*=np.where(np.diag(upper)<0,-1.,1.)
            for j in range(self.planes_per_shell):
                row=shell_index*self.planes_per_shell+j
                self.plane_u[row,:q.shape[0]]=q[:,2*j]
                self.plane_v[row,:q.shape[0]]=q[:,2*j+1]
        self.cos_basis,self.sin_basis=self._basis_from_planes()
        self._finish()

    def _reference(self,reference_alms,shells,planes_per_shell,seed):
        value=np.asarray(reference_alms,dtype=np.complex128)
        if value.ndim!=3 or value.shape[:2]!=(2,2) or not np.isfinite(value).all():
            raise ValueError('finite [2 splits,2 E/B,nalm] reference required')
        self.nalm=value.shape[-1];self.lmax=int(round((np.sqrt(8*self.nalm+1)-3)/2))
        if (self.lmax+1)*(self.lmax+2)//2!=self.nalm or self.lmax<2:raise ValueError('incomplete m-major harmonic storage')
        self.ell,self.m=harmonic_layout(self.lmax)
        if np.any(value[...,self.m==0].imag!=0):raise ValueError('real-field m=0 coefficients must be real')
        if np.any(value[...,self.ell<2]!=0):raise ValueError('spin-2 ell<2 coefficients must be zero')
        raw_shells=np.asarray(shells)
        if raw_shells.ndim!=1 or not len(raw_shells) or not np.issubdtype(raw_shells.dtype,np.integer):raise ValueError('nonempty integer shells required')
        if np.any(raw_shells<2) or np.any(raw_shells>self.lmax) or len(np.unique(raw_shells))!=len(raw_shells):raise ValueError('invalid distinct ell shells')
        if not isinstance(planes_per_shell,(int,np.integer)) or isinstance(planes_per_shell,bool) or planes_per_shell<1:raise ValueError('positive integer planes_per_shell required')
        if not isinstance(seed,(int,np.integer)) or isinstance(seed,bool) or not 0<=seed<2**63:raise ValueError('nonnegative int64 seed required')
        self.reference_alms=value.copy();self.shells=raw_shells.astype(np.int64,copy=True)
        self.planes_per_shell=int(planes_per_shell);self.seed=int(seed)
        self.nplanes=len(self.shells)*self.planes_per_shell;self.planes=self.nplanes
        self.coadd_alms=(value[0]+value[1])/2;self.difference_alms=value[0]-value[1]
        self._weights=np.where(self.m==0,1.,2.)
        self.reference_energy=float(np.sum(np.abs(self.coadd_alms)**2*self._weights))
        if not np.isfinite(self.reference_energy) or self.reference_energy<=0:raise ValueError('nondegenerate finite coadd energy required')

    def _basis_from_planes(self):
        cosine=np.zeros((self.nplanes,2,self.nalm),dtype=np.complex128);sine=np.zeros_like(cosine)
        for index,shell in enumerate(self.plane_shells):
            width=2*int(shell)+1;u=self.plane_u[index,:width];v=self.plane_v[index,:width]
            mean=_real_shell(self.coadd_alms,int(shell),self.ell)
            mu,mv=mean@u,mean@v
            cosine[index,:,self.ell==shell]=_complex_shell(mu[:,None]*u+mv[:,None]*v,int(shell)).T
            sine[index,:,self.ell==shell]=_complex_shell(mv[:,None]*u-mu[:,None]*v,int(shell)).T
        return cosine,sine

    def _finish(self):
        orthogonal=0.;annihilation=0.
        for shell in self.shells:
            ids=np.flatnonzero(self.plane_shells==shell);width=2*int(shell)+1
            q=np.stack((self.plane_u[ids,:width],self.plane_v[ids,:width]),axis=1).reshape(-1,width).T
            d=_real_shell(self.difference_alms,int(shell),self.ell)
            singular=np.linalg.svd(d,compute_uv=False)
            rank=int(np.sum(singular>(singular[0]*1e-12 if len(singular) else 0.)))
            if width-rank<2*self.planes_per_shell:raise ValueError('insufficient stored difference-complement dimension')
            if self.noise_rank[np.flatnonzero(self.shells==shell)[0]]!=rank:raise ValueError('stored difference rank disagrees')
            if np.any(self.plane_u[ids,width:]!=0) or np.any(self.plane_v[ids,width:]!=0):raise ValueError('nonzero padding in stored planes')
            orthogonal=max(orthogonal,float(np.max(np.abs(q.T@q-np.eye(q.shape[1])))))
            annihilation=max(annihilation,float(np.max(np.abs(d@q))/(1+np.max(np.abs(d)))))
        if max(orthogonal,annihilation)>1e-10:raise ValueError('stored planes violate orthogonality or empirical difference lock')
        self.plane_checks=dict(orthogonality_error=orthogonal,difference_annihilation_error=annihilation)
        self.plane_energy=np.sum(np.abs(self.cos_basis)**2*self._weights,axis=(1,2))
        self._reference_gram=_spectral_gram(self.reference_alms,self.ell,self.m,self.lmax)
        for name in ('reference_alms','shells','plane_shells','plane_u','plane_v','noise_rank','cos_basis',
                     'sin_basis','coadd_alms','difference_alms','plane_energy','ell','m','_weights','_reference_gram'):
            getattr(self,name).setflags(write=False)
        self._array_hashes={name:_family_hash(value) for name,value in self.to_arrays().items()}

    def _angles(self,theta):
        if np.iscomplexobj(theta) or np.asarray(theta).dtype.kind not in 'iuf':raise ValueError('angles must be real and non-boolean')
        value=np.asarray(theta,dtype=np.float64)
        if value.ndim<1 or value.shape[-1]!=self.nplanes or not np.isfinite(value).all():raise ValueError('finite real angles ending in P required')
        if np.any(np.abs(value)>self.ANGLE_LIMIT):raise ValueError('angle outside +/-pi/4')
        return value

    def coordinates(self,theta):
        value=self._angles(theta)
        return np.concatenate((np.ones(value.shape[:-1]+(1,)),-2*np.sin(value/2)**2,np.sin(value)),axis=-1)

    @property
    def basis_alms(self):
        value=np.concatenate((self.coadd_alms[None],self.cos_basis,self.sin_basis))
        value.setflags(write=False);return value

    def candidate_alms(self,theta):
        z=self.coordinates(theta)
        delta=np.einsum('...p,pen->...en',z[...,1:1+self.nplanes],self.cos_basis)
        delta+=np.einsum('...p,pen->...en',z[...,1+self.nplanes:],self.sin_basis)
        return self.reference_alms+delta[...,None,:,:]

    def rms_ratio(self,theta):
        value=self._angles(theta)
        return np.sqrt(np.sum(4*np.sin(value/2)**2*self.plane_energy,axis=-1)/self.reference_energy)

    def invariants(self,theta,tolerance=1e-10):
        if not np.isfinite(tolerance) or tolerance<=0:raise ValueError('positive finite invariant tolerance required')
        candidate=self.candidate_alms(theta)
        gram=_spectral_gram(candidate,self.ell,self.m,self.lmax)
        diagonal=np.diagonal(self._reference_gram,axis1=-2,axis2=-1)
        scale=np.sqrt(np.maximum(diagonal,0)[...,None]*np.maximum(diagonal,0)[...,None,:])
        delta=np.abs(gram-self._reference_gram)
        normalized=np.divide(delta,scale,out=np.zeros_like(delta),where=scale>0)
        normalized=np.where((scale==0)&(delta!=0),np.inf,normalized)
        difference=(candidate[...,0,:,:]-candidate[...,1,:,:])-self.difference_alms
        derror=np.max(np.abs(difference)/(1+np.abs(self.difference_alms)),axis=(-2,-1))
        gerror=np.max(normalized,axis=(-3,-2,-1))
        mzero=np.all(candidate[...,self.m==0].imag==0,axis=(-3,-2,-1))
        passed=np.isfinite(gerror)&np.isfinite(derror)&(gerror<=tolerance)&(derror<=tolerance)&mzero
        return _family_diagnostic(dict(passed=bool(np.all(passed)),per_state_passed=passed,
            finite=np.isfinite(gerror)&np.isfinite(derror),
            normalized_spectral_gram_error=gerror,split_difference_scaled_error=derror,
            mzero_real=mzero,maximum_spectral_error=float(np.max(gerror)),
            maximum_split_difference_error=float(np.max(derror)),tolerance=float(tolerance),
            scope='Full-sky spectra and empirical A-B only; no cut-sky or physical-noise guarantee'))

    def to_arrays(self):
        names=('reference_alms','shells','plane_shells','plane_u','plane_v','noise_rank','cos_basis','sin_basis')
        result={name:getattr(self,name).copy() for name in names}
        result.update(seed=np.array(self.seed,dtype=np.int64),planes_per_shell=np.array(self.planes_per_shell,dtype=np.int64))
        return result

    def provenance(self):
        current={name:_family_hash(value) for name,value in self.to_arrays().items()}
        if current!=self._array_hashes:raise ValueError('stored family bytes changed after construction')
        source=Path(__file__)
        return _family_diagnostic(dict(format=self.FORMAT,array_sha256=current,reference_sha256=current['reference_alms'],
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None,
            seed=self.seed,seed_derivation='SeedSequence([seed,ell]); stored planes authoritative',
            lmax=self.lmax,nplanes=self.nplanes,planes_per_shell=self.planes_per_shell,
            shells=self.shells.tolist(),difference_rank_by_shell=self.noise_rank.tolist(),
            angle_limit=self.ANGLE_LIMIT,plane_checks=dict(self.plane_checks),
            hash_rule='SHA256(dtype.str ASCII || shape tuple ASCII || C-contiguous raw bytes)',
            scope='Empirical difference-locked full-sky counterfactual parameterization'))

    @classmethod
    def from_arrays(cls,arrays,expected_sha256=None):
        obj=cls.__new__(cls)
        for name in ('seed','planes_per_shell','plane_shells','noise_rank'):
            if not np.issubdtype(np.asarray(arrays[name]).dtype,np.integer):raise ValueError('stored integer metadata required')
        for name in ('seed','planes_per_shell'):
            if np.asarray(arrays[name]).ndim!=0:raise ValueError('stored seed and planes_per_shell must be scalar integers')
        if any(np.iscomplexobj(arrays[name]) or np.asarray(arrays[name]).dtype.kind not in 'iuf' for name in ('plane_u','plane_v')):raise ValueError('stored real planes required')
        obj._reference(arrays['reference_alms'],arrays['shells'],int(np.asarray(arrays['planes_per_shell'])),int(np.asarray(arrays['seed'])))
        for name in ('plane_shells','noise_rank'):
            setattr(obj,name,np.asarray(arrays[name],dtype=np.int64).copy())
        for name in ('plane_u','plane_v'):
            setattr(obj,name,np.asarray(arrays[name],dtype=np.float64).copy())
        width=2*int(obj.shells.max())+1
        if (obj.plane_u.shape!=(obj.nplanes,width) or obj.plane_v.shape!=obj.plane_u.shape
            or not np.array_equal(obj.plane_shells,np.repeat(obj.shells,obj.planes_per_shell))
            or obj.noise_rank.shape!=(len(obj.shells),)
            or not np.isfinite(obj.plane_u).all() or not np.isfinite(obj.plane_v).all()):raise ValueError('invalid stored numerical planes')
        obj.cos_basis,obj.sin_basis=obj._basis_from_planes()
        for name in ('cos_basis','sin_basis'):
            stored=np.asarray(arrays[name],dtype=np.complex128)
            expected=getattr(obj,name);scale=float(np.max(np.abs(expected)))
            if stored.shape!=expected.shape or not np.isfinite(stored).all():raise ValueError('invalid stored harmonic basis')
            if np.max(np.abs(stored-expected))>(1e-12*scale if scale else 0.):raise ValueError('stored basis disagrees with stored planes/reference')
            setattr(obj,name,stored.copy())
        obj._finish()
        if expected_sha256 is not None and obj.provenance()['array_sha256']!=expected_sha256:raise ValueError('family SHA256 binding mismatch')
        return obj
