"""Native batched GPU evaluation of the unchanged closure-fiber features.

This is a separate engineering implementation. It caches transported, centered
sample operators on the selected device and contracts their moment weights in
batch. In particular it never constructs a [batch, planes, samples] tangent or
uses vmap/jacrev in the optimization-facing analytic paths. No spherical
transform, data loading, or scientific search happens in these methods.
"""
from __future__ import annotations

import numpy as np

from .closure_fibers import ClosureCUDA, _time_index, projection_indices

RMS_RATIO_NAMES = ('tile','global')


class ClosureBatchCUDA:
    """Resident complex128 operators with explicit [B,P] angle batches.

    ``features`` returns [B,52]; ``feature_jacobian`` returns [B,K,P]
    (K=52 or a selected representation). ``target``/``target_gradient`` use
    the future, two-degree connected fourth moment by default. ``ratio_squared``
    and ``rms_gradient`` return [B,2] and [B,2,P], including at zero angles.
    Construction also supports device='cpu' for independent small fixtures.
    """

    def __init__(self, family, device='cuda:0'):
        model = ClosureCUDA(family, device)
        self.torch, self.device = model.torch, model.device
        self.planes, self.n = model.planes, model.n
        self.rates = model.rates
        self.decays = self.torch.exp(-self.torch.arange(4, device=device,
            dtype=self.torch.float64).unsqueeze(-1)*self.rates)
        self.base, self.derivative, self.cosine, self.sine = [], [], [], []
        for lag in range(4):
            def increments(value):
                end = value[..., self.n:].reshape(value.shape[:-1]+(4,4,self.n))[...,lag,:,:]
                result = (end*model.phase1[lag]
                    - value[...,:self.n].unsqueeze(-2)*model.phase0[lag])
                result = result.flatten(-2).index_select(-1, model.indices[lag])
                return (result-result.mean(dim=-1, keepdim=True)).contiguous()
            self.base.append(increments(model.base))
            self.derivative.append(increments(model.base_derivative))
            self.cosine.append(increments(model.cos_basis))
            self.sine.append(increments(model.sin_basis))
        self.anchor_cosine = model.cos_basis[:,:self.n].contiguous()
        self.anchor_sine = model.sin_basis[:,:self.n].contiguous()
        self.band_cosine = [(self.anchor_cosine*mask.unsqueeze(-1)).contiguous()
                            for mask in model.band_masks]
        self.band_sine = [(self.anchor_sine*mask.unsqueeze(-1)).contiguous()
                          for mask in model.band_masks]
        self.base_band = model.base_band
        self.global_weights = model.sqrt_weights.square()
        self.local_denominator = self.n*model.reference_local_rms**2
        self.global_denominator = model.reference_global_rms**2
        # Deliberately retain only resident operators used by the batched path.

    def _angles(self, theta):
        t = self.torch
        if (theta.ndim != 2 or theta.shape[1] != self.planes
                or theta.dtype != t.float64 or str(theta.device) != str(t.device(self.device))):
            raise ValueError('device-local float64 [batch,planes] angles required')
        return t.cos(theta), t.sin(theta)

    def _coefficients(self, theta, u):
        cosine, sine = self._angles(theta)
        decay = self.decays[_time_index(u)]
        return [(v*decay).to(self.torch.complex128)
                for v in (cosine-1, sine, -sine, cosine)]

    def _dots(self, matrix_c, matrix_s, weights, dc, ds):
        """Contract fixed matrices without allocating per-start sample tangents."""
        t = self.torch
        wc = t.einsum('pn,bnw->bpw', matrix_c, weights)
        ws = t.einsum('pn,bnw->bpw', matrix_s, weights)
        return (dc.unsqueeze(-1)*wc+ds.unsqueeze(-1)*ws)/weights.shape[-2]

    def _lag(self, theta_coefficients, u, lag, jacobian):
        t = self.torch
        c,s,dc,ds = theta_coefficients
        cosine,sine = self.cosine[lag],self.sine[lag]
        value = self.base[lag][u]+(c@cosine+s@sine).unsqueeze(-2)
        derivative = self.derivative[lag][u]+(
            (-c*self.rates)@cosine+(-s*self.rates)@sine).unsqueeze(-2)
        a,b,da,db = [v-v.mean(dim=-1,keepdim=True)
                    for v in (value[:,0],value[:,1],derivative[:,0],derivative[:,1])]
        aa,bb = a.real.square()+a.imag.square(),b.real.square()+b.imag.square()
        daa,dbb = 2*(da*a.conj()).real,2*(db*b.conj()).real
        va,vb,dva,dvb = [v.mean(dim=-1) for v in (aa,bb,daa,dbb)]
        z,w = (a*b.conj()).mean(dim=-1),(a*b).mean(dim=-1)
        dz,dw = (da*b.conj()+a*db.conj()).mean(dim=-1),(da*b+a*db).mean(dim=-1)
        f,df = (aa*bb).mean(dim=-1),(daa*bb+aa*dbb).mean(dim=-1)
        denominator = va*vb
        ddenominator = dva*vb+va*dvb
        numerator = f-denominator-t.abs(z).square()-t.abs(w).square()
        dnumerator = df-ddenominator-2*(dz*z.conj()+dw*w.conj()).real
        k = numerator/denominator
        k_rate = (dnumerator-k*ddenominator)/denominator
        average,difference = (a+b)/2,(a-b)/2
        curve = t.stack([va,vb,z.real,
            (average.real.square()+average.imag.square()).mean(dim=-1),
            (difference.real.square()+difference.imag.square()).mean(dim=-1),
            z.real/t.sqrt(denominator),k,z.imag/t.sqrt(denominator),
            .25*(dva+dvb+2*dz.real),k_rate],dim=-1)
        if not jacobian:
            return curve,None
        weights = t.stack([a.conj(),b.conj(),da.conj(),db.conj(),a+b,da+db,
            a.conj()*bb+b.conj()*aa,
            da.conj()*bb+a.conj()*dbb+db.conj()*aa+b.conj()*daa],dim=-1)
        weights = weights-weights.mean(dim=-2,keepdim=True)
        ja,jb,jda,jdb,jw,jdw,jf,jdf = self._dots(cosine,sine,weights,dc,ds).unbind(dim=-1)
        jva,jvb = 2*ja.real,2*jb.real
        jz = jb+ja.conj()
        jdva,jdvb = -self.rates*jva+2*jda.real,-self.rates*jvb+2*jdb.real
        jdz,jdw = -self.rates*jz+jdb+jda.conj(),-self.rates*jw+jdw
        jf = 2*jf.real
        jdf = -self.rates*jf+2*jdf.real
        va,vb,dva,dvb,z,w,dz,dw,denominator,ddenominator,k,k_rate = [v.unsqueeze(-1)
            for v in (va,vb,dva,dvb,z,w,dz,dw,denominator,ddenominator,k,k_rate)]
        jdenominator = jva*vb+va*jvb
        jddenominator = jdva*vb+dva*jvb+jva*dvb+va*jdvb
        jnumerator = jf-jdenominator-2*(jz*z.conj()+jw*w.conj()).real
        jdnumerator = jdf-jddenominator-2*(jdz*z.conj()+dz*jz.conj()+jdw*w.conj()+dw*jw.conj()).real
        jk = (jnumerator-k*jdenominator)/denominator
        jk_rate = ((jdnumerator-jk*ddenominator-k*jddenominator)/denominator
                   -k_rate*jdenominator/denominator)
        root = t.sqrt(denominator)
        relative = .5*jdenominator/denominator
        jac = t.stack([jva,jvb,jz.real,.25*(jva+jvb+2*jz.real),
            .25*(jva+jvb-2*jz.real),(jz.real-z.real*relative)/root,jk,
            (jz.imag-z.imag*relative)/root,.25*(jdva+jdvb+2*jdz.real),jk_rate],dim=-2)
        return curve,jac

    def _evaluate(self, theta, u, jacobian):
        t = self.torch
        u = _time_index(u)
        coeff = self._coefficients(theta,u)
        curves,jacobians = [],[]
        for lag in range(4):
            curve,jac = self._lag(coeff,u,lag,jacobian)
            curves.append(curve)
            if jacobian:
                jacobians.append(jac)
        curves = t.stack(curves,dim=-2)
        powers,power_jacobians = [],[]
        c,s,dc,ds = coeff
        for band,(cosine,sine) in enumerate(zip(self.band_cosine,self.band_sine)):
            ab = self.base_band[u,:,band]+(c@cosine+s@sine).unsqueeze(-2)
            a,b = ab[:,0],ab[:,1]
            powers.append(t.stack([(a.real.square()+a.imag.square()).mean(dim=-1),
                (b.real.square()+b.imag.square()).mean(dim=-1),
                (a*b.conj()).real.mean(dim=-1)],dim=-1))
            if jacobian:
                weights = t.stack([2*a.conj(),2*b.conj(),(a+b).conj()],dim=-1)
                power_jacobians.append(self._dots(cosine,sine,weights,dc,ds).real.transpose(-2,-1))
        features = t.cat([curves[...,:8].flatten(-2),t.stack(powers,dim=-2).flatten(-2),
                          curves[...,8],curves[...,9]],dim=-1)
        if not jacobian:
            return features,None
        jacobians = t.stack(jacobians,dim=-3)
        jac = t.cat([jacobians[...,:8,:].flatten(-3,-2),
            t.stack(power_jacobians,dim=-3).flatten(-3,-2),
            jacobians[...,8,:],jacobians[...,9,:]],dim=-2)
        return features,jac

    def features(self, theta, u=0):
        return self._evaluate(theta,u,False)[0]

    def features_and_jacobian(self, theta, u=0, indices=None):
        """One batched evaluation; selection applies only to returned Jacobian."""
        feature,jacobian = self._evaluate(theta,u,True)
        if isinstance(indices,str):
            indices = projection_indices(indices)
        if indices is not None:
            indices = self.torch.as_tensor(indices,dtype=self.torch.int64,device=self.device)
            jacobian = jacobian.index_select(-2,indices)
        return feature,jacobian

    def feature_jacobian(self, theta, u=0, indices=None):
        return self.features_and_jacobian(theta,u,indices)[1]

    def target_and_gradient(self, theta, u=1, lag=2):
        """Future K4 and its analytic real-angle gradient for every batch row."""
        if type(lag) is not int or not 0 <= lag < 4:
            raise ValueError('lag index zero through three required')
        return self._target(theta,u,lag,True)

    def _target(self, theta, u, lag, gradient):
        t = self.torch
        c,s,dc,ds = self._coefficients(theta,u)
        cosine,sine = self.cosine[lag],self.sine[lag]
        value = self.base[lag][_time_index(u)]+(c@cosine+s@sine).unsqueeze(-2)
        value = value-value.mean(dim=-1,keepdim=True)
        a,b = value[:,0],value[:,1]
        aa,bb = a.real.square()+a.imag.square(),b.real.square()+b.imag.square()
        va,vb = aa.mean(dim=-1),bb.mean(dim=-1)
        z,w = (a*b.conj()).mean(dim=-1),(a*b).mean(dim=-1)
        denominator = va*vb
        k = ((aa*bb).mean(dim=-1)-denominator-t.abs(z).square()-t.abs(w).square())/denominator
        if not gradient:
            return k
        weights = t.stack([a.conj(),b.conj(),a+b,a.conj()*bb+b.conj()*aa],dim=-1)
        weights = weights-weights.mean(dim=-2,keepdim=True)
        ja,jb,jw,jf = self._dots(cosine,sine,weights,dc,ds).unbind(dim=-1)
        jva,jvb = 2*ja.real,2*jb.real
        jz = jb+ja.conj()
        jdenominator = jva*vb.unsqueeze(-1)+va.unsqueeze(-1)*jvb
        jnumerator = 2*jf.real-jdenominator-2*(jz*z.conj().unsqueeze(-1)+jw*w.conj().unsqueeze(-1)).real
        return k,(jnumerator-k.unsqueeze(-1)*jdenominator)/denominator.unsqueeze(-1)

    def target(self, theta, u=1, lag=2):
        if type(lag) is not int or not 0 <= lag < 4:
            raise ValueError('lag index zero through three required')
        return self._target(theta,u,lag,False)

    def target_gradient(self, theta, u=1, lag=2):
        return self.target_and_gradient(theta,u,lag)[1]

    def ratio_squared_and_gradient(self, theta):
        """Smooth RMS-squared constraints, with a finite zero-angle gradient."""
        t = self.torch
        cosine,sine = self._angles(theta)
        c,s = (cosine-1).to(t.complex128),sine.to(t.complex128)
        dc,ds = -sine.to(t.complex128),cosine.to(t.complex128)
        delta = c@self.anchor_cosine+s@self.anchor_sine
        local = (delta.real.square()+delta.imag.square()).sum(dim=-1)/self.local_denominator
        local_gradient = (dc*t.einsum('pn,bn->bp',self.anchor_cosine,delta.conj())
            +ds*t.einsum('pn,bn->bp',self.anchor_sine,delta.conj())).real*2/self.local_denominator
        # Squared sine/cosine coefficients avoid cancellation near theta=0.
        weights = self.global_weights/self.global_denominator
        global_ratio = ((cosine-1).square()+sine.square())@weights
        global_gradient = 2*sine*weights
        return t.stack([local,global_ratio],dim=-1),t.stack([local_gradient,global_gradient],dim=-2)

    def ratio_squared(self, theta):
        t = self.torch
        cosine,sine = self._angles(theta)
        c,s = cosine-1,sine
        delta = c.to(t.complex128)@self.anchor_cosine+s.to(t.complex128)@self.anchor_sine
        local = (delta.real.square()+delta.imag.square()).sum(dim=-1)/self.local_denominator
        global_ratio = (c.square()+s.square())@(self.global_weights/self.global_denominator)
        return t.stack([local,global_ratio],dim=-1)

    def ratios(self, theta):
        return self.torch.sqrt(self.ratio_squared(theta))

    def rms_gradient(self, theta):
        return self.ratio_squared_and_gradient(theta)[1]


# Explicit version alias for callers; v1 ClosureCUDA remains untouched.
ClosureGPUV2 = ClosureBatchCUDA
