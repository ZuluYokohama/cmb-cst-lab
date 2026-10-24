"""Finite-precision moment projections under a declared angular smoothing map.

The functions evaluate observed/counterfactual samples supplied by a separately
audited spherical operator. They do not establish physical validity or closure.
All derivatives are analytic derivatives with respect to the smoothing step u.
"""
from __future__ import annotations
import numpy as np


STATISTICS = ('auto_A', 'auto_B', 'cross_S2', 'mean_S2', 'difference_S2',
              'coherence', 'connected4_normalized', 'imaginary_cross_normalized')
QUANTUM = .001


def projection_indices(name):
    """Indices into the common 52-coordinate feature vector."""
    if name == 'baseline':
        return np.arange(44, dtype=np.int64)
    if name == 'capacity':
        return np.arange(48, dtype=np.int64)
    if name == 'repair':
        return np.r_[np.arange(44), np.arange(48, 52)].astype(np.int64)
    raise ValueError('unknown closure representation')


def robust_scales(reference_features):
    x = np.asarray(reference_features, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or x.shape[1] != 52 or not np.isfinite(x).all():
        raise ValueError('need finite reference feature rows with 52 coordinates')
    median = np.median(x, axis=0)
    mad = np.median(np.abs(x - median), axis=0)
    return np.maximum(1.4826 * mad, .001 * np.maximum(1., np.abs(median)))


def quantize(features, reference, scales, name='baseline', quantum=QUANTUM):
    """Reference-centered nearest-even integer bins; not exact real equality."""
    x, reference, scales = [np.asarray(v, dtype=np.float64) for v in (features, reference, scales)]
    if (x.ndim < 1 or x.shape[-1] != 52 or reference.shape != (52,) or scales.shape != (52,)
            or not all(np.isfinite(v).all() for v in (x, reference, scales))
            or np.any(scales <= 0) or not np.isfinite(quantum) or quantum <= 0):
        raise ValueError('invalid reference-centered quantization input')
    ix = projection_indices(name)
    normalized = (x[..., ix] - reference[ix]) / scales[ix] / quantum
    if np.any(np.abs(normalized) >= 2.**63):
        raise ValueError('quantized feature exceeds int64 range')
    return np.rint(normalized).astype(np.int64)


def _checked_family(family):
    """Validate the shared operator samples without modifying the input arrays."""
    complex_names = ('base', 'base_derivative', 'base_band', 'cos_basis', 'sin_basis', 'phase0', 'phase1')
    f = {key: np.asarray(family[key], dtype=np.complex128) for key in complex_names}
    for key in ('rates', 'global_weights'):
        f[key] = np.asarray(family[key], dtype=np.float64)
    raw_band = np.asarray(family['band_id'])
    if raw_band.dtype.kind not in 'iu':
        raise ValueError('integer band IDs required')
    f['band_id'] = raw_band.astype(np.int64, copy=False)
    f['valid'] = np.asarray(family['valid'])
    if f['valid'].dtype != bool or f['valid'].ndim != 3 or f['valid'].shape[:2] != (4, 4):
        raise ValueError('need a boolean [4 lags,4 directions,N] mask')
    n = f['valid'].shape[-1]
    p = len(f['rates']) if f['rates'].ndim == 1 else -1
    if (n < 1 or p < 0 or f['base'].shape != (4, 2, 17*n)
            or f['base_derivative'].shape != f['base'].shape
            or f['base_band'].shape != (4, 2, 4, n)
            or f['cos_basis'].shape != (p, 17*n) or f['sin_basis'].shape != (p, 17*n)
            or f['global_weights'].shape != (p,) or f['band_id'].shape != (p,)
            or f['phase0'].shape != f['valid'].shape or f['phase1'].shape != f['valid'].shape):
        raise ValueError('inconsistent closure family shapes')
    if not all(np.isfinite(v).all() for v in f.values()):
        raise ValueError('nonfinite closure family')
    if np.any(f['rates'] < 0) or np.any(f['global_weights'] < 0) or np.any((f['band_id'] < 0) | (f['band_id'] > 3)):
        raise ValueError('invalid rates, weights or spectral bands')
    if np.any(f['valid'].sum(axis=(1, 2)) < 4):
        raise ValueError('at least four valid pairs per lag required')
    if any(np.max(np.abs(np.abs(f[key]) - 1.)) > 1e-10 for key in ('phase0', 'phase1')):
        raise ValueError('unit spin-2 phase factors required')
    for key in ('reference_local_rms', 'reference_global_rms'):
        value = np.asarray(family[key])
        if value.ndim or not np.isfinite(value) or float(value) <= 0:
            raise ValueError('positive scalar reference RMS required')
        f[key] = float(value)
    return f


def _checked_angles(theta, planes):
    raw = np.asarray(theta)
    if np.iscomplexobj(raw):
        raise ValueError('real rotation angles required')
    value = np.asarray(raw, dtype=np.float64)
    if value.ndim < 1 or value.shape[-1] != planes or not np.isfinite(value).all():
        raise ValueError('finite angles with one coordinate per plane required')
    return value


def _time_index(u):
    if isinstance(u, (bool, np.bool_)) or not np.isscalar(u):
        raise ValueError('u must select a precomputed integer time 0 through 3')
    if not np.isfinite(u) or int(u) != u or not 0 <= int(u) <= 3:
        raise ValueError('u must select a precomputed integer time 0 through 3')
    return int(u)


def moments_with_derivatives(a, b, da, db):
    """Eight centered complex moments and their analytic smoothing derivatives.

    The last axis contains samples; arbitrary matching leading dimensions are
    supported. Centering the derivative is necessary because the means move.
    """
    a, b, da, db = [np.asarray(v, dtype=np.complex128) for v in (a, b, da, db)]
    if (a.ndim < 1 or a.shape[-1] < 4 or any(v.shape != a.shape for v in (b, da, db))
            or not all(np.isfinite(v).all() for v in (a, b, da, db))):
        raise ValueError('matching finite complex moment samples required')
    a, b, da, db = [v - v.mean(axis=-1, keepdims=True) for v in (a, b, da, db)]
    aa, bb = a.real*a.real + a.imag*a.imag, b.real*b.real + b.imag*b.imag
    daa, dbb = 2*(da*a.conj()).real, 2*(db*b.conj()).real
    va, vb = aa.mean(axis=-1), bb.mean(axis=-1)
    if np.any(va <= 0) or np.any(vb <= 0):
        raise ValueError('zero split increment variance')
    dva, dvb = daa.mean(axis=-1), dbb.mean(axis=-1)
    z, w = (a*b.conj()).mean(axis=-1), (a*b).mean(axis=-1)
    dz = (da*b.conj() + a*db.conj()).mean(axis=-1)
    dw = (da*b + a*db).mean(axis=-1)
    product = va*vb
    dproduct = dva*vb + va*dvb
    root = np.sqrt(product)
    numerator = (aa*bb).mean(axis=-1) - product - np.abs(z)**2 - np.abs(w)**2
    dnumerator = ((daa*bb + aa*dbb).mean(axis=-1) - dproduct
                  - 2*(dz*z.conj()).real - 2*(dw*w.conj()).real)
    average, difference = (a+b)/2, (a-b)/2
    stats = np.stack([va, vb, z.real,
        (average.real**2 + average.imag**2).mean(axis=-1),
        (difference.real**2 + difference.imag**2).mean(axis=-1),
        z.real/root, numerator/product, z.imag/root], axis=-1)
    logarithmic_root_rate = .5*(dva/va + dvb/vb)
    derivative = np.stack([dva, dvb, dz.real,
        .25*(dva+dvb+2*dz.real), .25*(dva+dvb-2*dz.real),
        (dz.real-z.real*logarithmic_root_rate)/root,
        (dnumerator - numerator*dproduct/product)/product,
        (dz.imag-z.imag*logarithmic_root_rate)/root], axis=-1)
    if not np.isfinite(stats).all() or not np.isfinite(derivative).all():
        raise ValueError('nonfinite closure moments or derivatives')
    return stats, derivative


def _numpy_delta(f, theta, u):
    decay = np.exp(-u*f['rates'])
    c, s = (np.cos(theta)-1)*decay, np.sin(theta)*decay
    delta = c@f['cos_basis'] + s@f['sin_basis']
    derivative = (-c*f['rates'])@f['cos_basis'] + (-s*f['rates'])@f['sin_basis']
    return c, s, delta, derivative


def features_numpy(family, theta, u=0):
    """Return [...,52] features at a precomputed integer smoothing time."""
    f = _checked_family(family)
    theta = _checked_angles(theta, len(f['rates']))
    u = _time_index(u)
    n = f['valid'].shape[-1]
    c, s, delta, ddelta = _numpy_delta(f, theta, u)
    values = f['base'][u] + delta[..., None, :]
    derivatives = f['base_derivative'][u] + ddelta[..., None, :]
    shape = values.shape[:-1] + (4, 4, n)
    inc = values[..., n:].reshape(shape)*f['phase1'] - values[..., :n][..., None, None, :]*f['phase0']
    dinc = derivatives[..., n:].reshape(shape)*f['phase1'] - derivatives[..., :n][..., None, None, :]*f['phase0']
    curves, responses = [], []
    for lag in range(4):
        take = f['valid'][lag]
        x, dx = inc[..., lag, :, :][..., take], dinc[..., lag, :, :][..., take]
        curve, response = moments_with_derivatives(x[..., 0, :], x[..., 1, :], dx[..., 0, :], dx[..., 1, :])
        curves.append(curve); responses.append(response)
    curves, responses = np.stack(curves, axis=-2), np.stack(responses, axis=-2)
    band_delta = np.stack([(c*(f['band_id']==band))@f['cos_basis'][:, :n]
                         + (s*(f['band_id']==band))@f['sin_basis'][:, :n] for band in range(4)], axis=-2)
    bands = f['base_band'][u] + band_delta[..., None, :, :]
    a, b = bands[..., 0, :, :], bands[..., 1, :, :]
    powers = np.stack([(a.real*a.real+a.imag*a.imag).mean(axis=-1),
                      (b.real*b.real+b.imag*b.imag).mean(axis=-1), (a*b.conj()).real.mean(axis=-1)], axis=-1)
    result = np.concatenate([curves.reshape(curves.shape[:-2]+(32,)),
        powers.reshape(powers.shape[:-2]+(12,)), responses[..., :, 3], responses[..., :, 6]], axis=-1)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite closure features')
    return result


def ratios_numpy(family, theta):
    f = _checked_family(family)
    theta = _checked_angles(theta, len(f['rates']))
    n = f['valid'].shape[-1]
    c, s = np.cos(theta)-1, np.sin(theta)
    delta = c@f['cos_basis'][:, :n] + s@f['sin_basis'][:, :n]
    local = np.linalg.norm(delta, axis=-1)/np.sqrt(n)/f['reference_local_rms']
    weighted = np.concatenate([c*np.sqrt(f['global_weights']), s*np.sqrt(f['global_weights'])], axis=-1)
    global_ratio = np.linalg.norm(weighted, axis=-1)/f['reference_global_rms']
    return np.stack([local, global_ratio], axis=-1)


class ClosureCUDA:
    """Differentiable complex128 evaluation; pure tensor methods support jacrev/vmap.

    Validation happens on construction or outside differentiation. A degenerate
    candidate produces nonfinite features rather than a clamped scientific value.
    """
    def __init__(self, family, device='cuda:0'):
        import torch
        f = _checked_family(family)
        # Check every fixed reference trajectory, including empty-plane families.
        for u in range(4):
            features_numpy(f, np.zeros(len(f['rates'])), u)
        self.torch, self.device = torch, device
        self.n = f['valid'].shape[-1]
        self.planes = len(f['rates'])
        for key in ('base', 'base_derivative', 'base_band', 'cos_basis', 'sin_basis', 'phase0', 'phase1', 'rates'):
            setattr(self, key, torch.as_tensor(f[key], device=device))
        self.sqrt_weights = torch.as_tensor(np.sqrt(f['global_weights']), device=device)
        self.reference_local_rms = f['reference_local_rms']
        self.reference_global_rms = f['reference_global_rms']
        self.band_masks = [torch.as_tensor((f['band_id']==band).astype(float), device=device) for band in range(4)]
        self.indices = [torch.as_tensor(np.flatnonzero(f['valid'][lag].ravel()), device=device) for lag in range(4)]

    def _moments(self, a, b, da, db):
        t = self.torch
        a, b, da, db = [v-v.mean(dim=-1, keepdim=True) for v in (a,b,da,db)]
        aa, bb = a.real*a.real+a.imag*a.imag, b.real*b.real+b.imag*b.imag
        daa, dbb = 2*(da*a.conj()).real, 2*(db*b.conj()).real
        va, vb = aa.mean(dim=-1), bb.mean(dim=-1)
        dva, dvb = daa.mean(dim=-1), dbb.mean(dim=-1)
        z, w = (a*b.conj()).mean(dim=-1), (a*b).mean(dim=-1)
        dz = (da*b.conj()+a*db.conj()).mean(dim=-1)
        dw = (da*b+a*db).mean(dim=-1)
        product = va*vb
        dproduct = dva*vb+va*dvb
        root = t.sqrt(product)
        numerator = (aa*bb).mean(dim=-1)-product-t.abs(z)**2-t.abs(w)**2
        dnumerator = (daa*bb+aa*dbb).mean(dim=-1)-dproduct-2*(dz*z.conj()).real-2*(dw*w.conj()).real
        average, difference = (a+b)/2, (a-b)/2
        curve = t.stack([va,vb,z.real,
            (average.real**2+average.imag**2).mean(dim=-1),
            (difference.real**2+difference.imag**2).mean(dim=-1),
            z.real/root,numerator/product,z.imag/root], dim=-1)
        return curve, .25*(dva+dvb+2*dz.real), (dnumerator-numerator*dproduct/product)/product

    def features(self, theta, u=0):
        t = self.torch
        if theta.dtype != t.float64 or theta.ndim < 1 or theta.shape[-1] != self.planes:
            raise ValueError('float64 angle tensors with one coordinate per plane required')
        u = _time_index(u)
        decay = t.exp(-u*self.rates)
        c, s = (t.cos(theta)-1)*decay, t.sin(theta)*decay
        # Real-to-complex conversion is explicit: matmul requires equal dtypes.
        c, s = c.to(t.complex128), s.to(t.complex128)
        delta = c@self.cos_basis+s@self.sin_basis
        ddelta = (-c*self.rates)@self.cos_basis+(-s*self.rates)@self.sin_basis
        value = self.base[u]+delta.unsqueeze(-2)
        derivative = self.base_derivative[u]+ddelta.unsqueeze(-2)
        shape = value.shape[:-1]+(4,4,self.n)
        inc = value[...,self.n:].reshape(shape)*self.phase1-value[...,:self.n].unsqueeze(-2).unsqueeze(-2)*self.phase0
        dinc = derivative[...,self.n:].reshape(shape)*self.phase1-derivative[...,:self.n].unsqueeze(-2).unsqueeze(-2)*self.phase0
        curves, mean_derivatives, k_derivatives = [], [], []
        for lag in range(4):
            x = inc[...,lag,:,:].flatten(-2).index_select(-1,self.indices[lag])
            dx = dinc[...,lag,:,:].flatten(-2).index_select(-1,self.indices[lag])
            curve, dmean, dk = self._moments(x[...,0,:],x[...,1,:],dx[...,0,:],dx[...,1,:])
            curves.append(curve);mean_derivatives.append(dmean);k_derivatives.append(dk)
        curves = t.stack(curves,dim=-2).flatten(-2)
        band_delta = t.stack([(c*mask)@self.cos_basis[:,:self.n]+(s*mask)@self.sin_basis[:,:self.n]
                              for mask in self.band_masks],dim=-2)
        bands = self.base_band[u]+band_delta.unsqueeze(-3)
        a,b = bands[...,0,:,:],bands[...,1,:,:]
        powers = t.stack([(a.real*a.real+a.imag*a.imag).mean(dim=-1),
                          (b.real*b.real+b.imag*b.imag).mean(dim=-1),(a*b.conj()).real.mean(dim=-1)],dim=-1).flatten(-2)
        return t.cat([curves,powers,t.stack(mean_derivatives,dim=-1),t.stack(k_derivatives,dim=-1)],dim=-1)

    def ratios(self, theta):
        t = self.torch
        if theta.dtype != t.float64 or theta.ndim < 1 or theta.shape[-1] != self.planes:
            raise ValueError('float64 angle tensors with one coordinate per plane required')
        c, s = t.cos(theta)-1,t.sin(theta)
        delta = c.to(t.complex128)@self.cos_basis[:,:self.n]+s.to(t.complex128)@self.sin_basis[:,:self.n]
        local = t.linalg.vector_norm(delta,dim=-1)/np.sqrt(self.n)/self.reference_local_rms
        weighted = t.cat([c*self.sqrt_weights,s*self.sqrt_weights],dim=-1)
        global_ratio = t.linalg.vector_norm(weighted,dim=-1)/self.reference_global_rms
        return t.stack([local,global_ratio],dim=-1)
