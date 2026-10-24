"""Sample-tangent Jacobians for unchanged closure feature equations.

The angle-to-sample derivative is analytic. Eight complex sample contractions
give moment primitive derivatives, followed by explicit scalar chain rules.
No scientific representation is changed.
"""
from __future__ import annotations


def _moment_jacobian(t, a,b,da,db,tangent,rates):
    """Real-angle derivative from common complex sample tangent [...,P,N]."""
    a,b,da,db = [v-v.mean(dim=-1,keepdim=True) for v in (a,b,da,db)]
    aa,bb = a.real*a.real+a.imag*a.imag,b.real*b.real+b.imag*b.imag
    daa,dbb = 2*(da*a.conj()).real,2*(db*b.conj()).real
    va,vb,dva,dvb = [v.mean(dim=-1) for v in (aa,bb,daa,dbb)]
    z,w = (a*b.conj()).mean(dim=-1),(a*b).mean(dim=-1)
    dz,dw = (da*b.conj()+a*db.conj()).mean(dim=-1),(da*b+a*db).mean(dim=-1)
    f,df = (aa*bb).mean(dim=-1),(daa*bb+aa*dbb).mean(dim=-1)

    # Center weights: E[(J-mean J)W] = E[J(W-mean W)]. This accounts for
    # moving split means without centered [...,P,N] tangent allocations.
    weights = t.stack([a.conj(),b.conj(),da.conj(),db.conj(),a+b,da+db,
        a.conj()*bb+b.conj()*aa,
        da.conj()*bb+a.conj()*dbb+db.conj()*aa+b.conj()*daa],dim=-1)
    weights = weights-weights.mean(dim=-2,keepdim=True)
    dots = tangent@weights/a.shape[-1]
    ja,jb,jda,jdb,jw,jdw,jf,jdf = dots.unbind(dim=-1)
    jva,jvb = 2*ja.real,2*jb.real
    jz = jb+ja.conj()
    jdv_a,jdv_b = -rates*jva+2*jda.real,-rates*jvb+2*jdb.real
    jdz = -rates*jz+jdb+jda.conj()
    jdw = -rates*jw+jdw
    jf = 2*jf.real
    jdf = -rates*jf+2*jdf.real
    # Scalar chain rules. A final singleton axis broadcasts primitive values
    # against their plane derivatives, without a reverse pass per output.
    va,vb,dva,dvb,z,w,dz,dw,f,df = [v.unsqueeze(-1) for v in (va,vb,dva,dvb,z,w,dz,dw,f,df)]
    denominator = va*vb
    ddenominator = dva*vb+va*dvb
    jdenominator = jva*vb+va*jvb
    jddenominator = jdv_a*vb+dva*jvb+jva*dvb+va*jdv_b
    numerator = f-denominator-z.real*z.real-z.imag*z.imag-w.real*w.real-w.imag*w.imag
    dnumerator = df-ddenominator-2*(dz*z.conj()+dw*w.conj()).real
    jnumerator = jf-jdenominator-2*(jz*z.conj()+jw*w.conj()).real
    jdnumerator = jdf-jddenominator-2*(jdz*z.conj()+dz*jz.conj()+jdw*w.conj()+dw*jw.conj()).real
    k = numerator/denominator
    k_rate = (dnumerator-k*ddenominator)/denominator
    jk = (jnumerator-k*jdenominator)/denominator
    jk_rate = ((jdnumerator-jk*ddenominator-k*jddenominator)/denominator
               - k_rate*jdenominator/denominator)
    root = t.sqrt(denominator)
    root_relative = .5*jdenominator/denominator
    return t.stack([jva,jvb,jz.real,.25*(jva+jvb+2*jz.real),.25*(jva+jvb-2*jz.real),
                    (jz.real-z.real*root_relative)/root,jk,
                    (jz.imag-z.imag*root_relative)/root,
                    .25*(jdv_a+jdv_b+2*jdz.real),jk_rate],dim=-2)


def feature_jacobian(model, theta, u=0):
    """Return [...,52,P] d(features)/d(theta), supporting outer vmap."""
    t = model.torch
    if type(u) is not int or not 0 <= u <= 3:
        raise ValueError('stored integer smoothing time 0 through 3 required')
    if theta.ndim < 1 or theta.shape[-1] != model.planes or theta.dtype != t.float64:
        raise ValueError('float64 angles with one coordinate per plane required')
    if model.planes == 0:
        return theta.new_zeros(theta.shape[:-1]+(52,0))
    decay = t.exp(-u*model.rates)
    cosine,sine = t.cos(theta),t.sin(theta)
    c,s = ((cosine-1)*decay).to(t.complex128),(sine*decay).to(t.complex128)
    delta = c@model.cos_basis+s@model.sin_basis
    ddelta = (-c*model.rates)@model.cos_basis+(-s*model.rates)@model.sin_basis
    value,derivative = model.base[u]+delta.unsqueeze(-2),model.base_derivative[u]+ddelta.unsqueeze(-2)
    shape = value.shape[:-1]+(4,4,model.n)
    inc = (value[...,model.n:].reshape(shape)*model.phase1
           - value[...,:model.n].unsqueeze(-2).unsqueeze(-2)*model.phase0)
    dinc = (derivative[...,model.n:].reshape(shape)*model.phase1
            - derivative[...,:model.n].unsqueeze(-2).unsqueeze(-2)*model.phase0)
    sample_jac = ((-sine*decay).unsqueeze(-1)*model.cos_basis
                  + (cosine*decay).unsqueeze(-1)*model.sin_basis)
    jshape = sample_jac.shape[:-1]+(4,4,model.n)
    jinc = (sample_jac[...,model.n:].reshape(jshape)*model.phase1
            - sample_jac[...,:model.n].unsqueeze(-2).unsqueeze(-2)*model.phase0)

    moments = []
    for lag in range(4):
        ix = model.indices[lag]
        pair = inc[...,lag,:,:].flatten(-2).index_select(-1,ix)
        dpair = dinc[...,lag,:,:].flatten(-2).index_select(-1,ix)
        tangent = jinc[...,lag,:,:].flatten(-2).index_select(-1,ix)
        moments.append(_moment_jacobian(t,pair[...,0,:],pair[...,1,:],
            dpair[...,0,:],dpair[...,1,:],tangent,model.rates))
    moments = t.stack(moments,dim=-3)
    curves = moments[...,:8,:].flatten(-3,-2)
    mean_rates,k_rates = moments[...,8,:],moments[...,9,:]

    band_jacobians = []
    anchor_jac = sample_jac[...,:model.n]
    for band,mask in enumerate(model.band_masks):
        band_delta = ((c*mask)@model.cos_basis[:,:model.n]
                      + (s*mask)@model.sin_basis[:,:model.n])
        ab = model.base_band[u,:,band,:]+band_delta.unsqueeze(-2)
        a,b = ab[...,0,:],ab[...,1,:]
        aa = 2*(anchor_jac*a.conj().unsqueeze(-2)).real.mean(dim=-1)*mask
        bb = 2*(anchor_jac*b.conj().unsqueeze(-2)).real.mean(dim=-1)*mask
        cross = (anchor_jac*(a+b).conj().unsqueeze(-2)).real.mean(dim=-1)*mask
        band_jacobians.append(t.stack([aa,bb,cross],dim=-2))
    powers = t.stack(band_jacobians,dim=-3).flatten(-3,-2)
    return t.cat([curves,powers,mean_rates,k_rates],dim=-2)


class ClosureTargetCUDA:
    """One transported-lag K4 target and gradient, with no unused features."""
    def __init__(self, model, u=1, lag=2):
        if type(u) is not int or not 0 <= u <= 3 or type(lag) is not int or not 0 <= lag <= 3:
            raise ValueError('stored smoothing time and lag index required')
        self.torch,self.planes = model.torch,model.planes
        self.u,self.lag = u,lag
        t = self.torch
        n = model.n
        ix = model.indices[lag]
        def increments(value):
            end = value[...,n:].reshape(value.shape[:-1]+(4,4,n))[...,lag,:,:]
            result = end*model.phase1[lag]-value[...,:n].unsqueeze(-2)*model.phase0[lag]
            result = result.flatten(-2).index_select(-1,ix)
            return result-result.mean(dim=-1,keepdim=True)
        self.base = increments(model.base[u])
        decay = t.exp(-u*model.rates).unsqueeze(-1)
        self.cosine = increments(model.cos_basis)*decay
        self.sine = increments(model.sin_basis)*decay

    def _statistics(self, theta):
        t = self.torch
        if theta.ndim < 1 or theta.shape[-1] != self.planes or theta.dtype != t.float64:
            raise ValueError('float64 angles with one coordinate per plane required')
        c,s = (t.cos(theta)-1).to(t.complex128),t.sin(theta).to(t.complex128)
        values = self.base+(c@self.cosine+s@self.sine).unsqueeze(-2)
        # Recenter residual rounding as in the production moment definition.
        values = values-values.mean(dim=-1,keepdim=True)
        a,b = values[...,0,:],values[...,1,:]
        aa,bb = a.real*a.real+a.imag*a.imag,b.real*b.real+b.imag*b.imag
        va,vb = aa.mean(dim=-1),bb.mean(dim=-1)
        z,w = (a*b.conj()).mean(dim=-1),(a*b).mean(dim=-1)
        denominator = va*vb
        numerator = (aa*bb).mean(dim=-1)-denominator-t.abs(z)**2-t.abs(w)**2
        return a,b,aa,bb,va,vb,z,w,denominator,numerator/denominator

    def value(self, theta):
        """Return the scalar K4 target with any leading angle batches."""
        return self._statistics(theta)[-1]

    def gradient(self, theta):
        """Return [...,P] exact analytic derivatives of the K4 target."""
        t = self.torch
        a,b,aa,bb,va,vb,z,w,denominator,k = self._statistics(theta)
        tangent = (-t.sin(theta).unsqueeze(-1)*self.cosine
                   + t.cos(theta).unsqueeze(-1)*self.sine)
        weights = t.stack([a.conj(),b.conj(),a+b,a.conj()*bb+b.conj()*aa],dim=-1)
        weights = weights-weights.mean(dim=-2,keepdim=True)
        ja,jb,jw,jf = (tangent@weights/a.shape[-1]).unbind(dim=-1)
        jva,jvb = 2*ja.real,2*jb.real
        jz = jb+ja.conj()
        jdenominator = jva*vb.unsqueeze(-1)+va.unsqueeze(-1)*jvb
        jnumerator = 2*jf.real-jdenominator-2*(jz*z.conj().unsqueeze(-1)+jw*w.conj().unsqueeze(-1)).real
        return (jnumerator-k.unsqueeze(-1)*jdenominator)/denominator.unsqueeze(-1)
