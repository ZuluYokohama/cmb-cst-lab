"""Symmetric quadratic response tensors for stored trigonometric coordinates.

The constant coordinate carries linear and constant terms as well as quadratic
ones. No assertion is made that a supplied tensor represents a physical sky.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np


def _quadratic_hash(value):
    value=np.ascontiguousarray(value)
    digest=hashlib.sha256(value.dtype.str.encode()+str(value.shape).encode())
    digest.update(value.tobytes());return digest.hexdigest()


class SymmetricQuadraticModel:
    def __init__(self,Q):
        if np.iscomplexobj(Q) or np.asarray(Q).dtype.kind not in 'iuf':raise ValueError('Q must be real and non-boolean')
        value=np.asarray(Q,dtype=np.float64)
        if value.ndim!=3 or not len(value) or value.shape[1]!=value.shape[2] or value.shape[1]%2!=1 or not np.isfinite(value).all():raise ValueError('finite Q[K,1+2P,1+2P] required')
        scale=float(np.max(np.abs(value)))
        if np.max(np.abs(value-value.transpose(0,2,1)))>1e-13*scale:raise ValueError('Q must be symmetric')
        self.Q=((value+value.transpose(0,2,1))/2).copy();self.Q.setflags(write=False)
        self.nplanes=(value.shape[1]-1)//2;self.planes=self.nplanes;self.nfeatures=len(value)
        self.sha256=_quadratic_hash(self.Q)

    def _angles(self,theta):
        if np.iscomplexobj(theta) or np.asarray(theta).dtype.kind not in 'iuf':raise ValueError('angles must be real and non-boolean')
        value=np.asarray(theta,dtype=np.float64)
        if value.ndim<1 or value.shape[-1]!=self.nplanes or not np.isfinite(value).all():raise ValueError('finite real angles ending in P required')
        return value

    def coordinates(self,theta):
        value=self._angles(theta)
        return np.concatenate((np.ones(value.shape[:-1]+(1,)),-2*np.sin(value/2)**2,np.sin(value)),axis=-1)

    def features(self,theta):
        z=self.coordinates(theta)
        return np.einsum('...i,kij,...j->...k',z,self.Q,z)

    def jacobian(self,theta):
        value=self._angles(theta);z=self.coordinates(value)
        product=2*np.einsum('kij,...j->...ki',self.Q,z)
        return (-product[...,1:1+self.nplanes]*np.sin(value)[...,None,:]
                +product[...,1+self.nplanes:]*np.cos(value)[...,None,:])

    jac=jacobian

    def to_arrays(self):return dict(Q=self.Q.copy())

    @classmethod
    def from_arrays(cls,arrays,expected_sha256=None):
        result=cls(arrays['Q'])
        if expected_sha256 is not None and result.sha256!=expected_sha256:raise ValueError('quadratic SHA256 binding mismatch')
        return result

    def provenance(self):
        if _quadratic_hash(self.Q)!=self.sha256:raise ValueError('quadratic response bytes changed')
        source=Path(__file__)
        return dict(format='cmbcst.integration.quadratic.v1',Q_sha256=self.sha256,nplanes=self.nplanes,
            nfeatures=self.nfeatures,coordinate_order='[1,cos(theta)-1,sin(theta)]',
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None)

    def torch(self,device='cpu'):return TorchQuadraticModel(self,device)


class TorchQuadraticModel:
    """Optional actual-device float64 evaluator; no per-start Python loop."""
    def __init__(self,model,device='cpu'):
        import torch
        if not isinstance(model,SymmetricQuadraticModel):model=SymmetricQuadraticModel(model)
        self.torch=torch;self.Q=torch.tensor(model.Q.copy(),dtype=torch.float64,device=device)
        self.nplanes=model.nplanes;self.planes=model.nplanes;self.nfeatures=model.nfeatures
        self.device=self.Q.device;self.Q_sha256=model.sha256

    def coordinates(self,theta):
        torch=self.torch
        if not torch.is_tensor(theta) or theta.ndim<1 or theta.shape[-1]!=self.nplanes or theta.dtype!=torch.float64 or theta.device!=self.device:raise ValueError('native float64 angles on the evaluator device required')
        return torch.cat((torch.ones_like(theta[...,:1]) if self.nplanes else torch.ones(theta.shape[:-1]+(1,),dtype=theta.dtype,device=theta.device),
            -2*torch.sin(theta/2).square(),torch.sin(theta)),dim=-1)

    def features(self,theta):
        z=self.coordinates(theta)
        return self.torch.einsum('...i,kij,...j->...k',z,self.Q,z)

    def jacobian(self,theta):
        torch=self.torch;z=self.coordinates(theta)
        product=2*torch.einsum('kij,...j->...ki',self.Q,z)
        return (-product[...,1:1+self.nplanes]*torch.sin(theta)[...,None,:]
                +product[...,1+self.nplanes:]*torch.cos(theta)[...,None,:])

    jac=jacobian


QuadraticModel=SymmetricQuadraticModel
