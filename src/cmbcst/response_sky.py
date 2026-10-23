"""Explicit synthetic spherical forward model; no Planck foreground/noise claim."""
from __future__ import annotations
import numpy as np
from . import spectra, theory, spherical_surrogates
from .response import spherical_patch_locations


def foreground_proxy(theta, phi):
    latitude = np.pi/2-np.asarray(theta)
    return np.exp(-.5*(latitude/.45)**2)*(1+.35*np.cos(5*np.asarray(phi))+.15*np.sin(11*np.asarray(phi))*np.cos(latitude))


def localized_template(theta, phi, center, kind, rng):
    theta, phi = np.asarray(theta), np.asarray(phi)
    st, ct, cp, sp = np.sin(center[0]), np.cos(center[0]), np.cos(center[1]), np.sin(center[1])
    vectors = np.stack([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)], axis=-1)
    normal = np.array([st*cp, st*sp, ct])
    east, north = np.array([-sp, cp, 0.]), np.array([-ct*cp, -ct*sp, st])
    dot = vectors@normal
    good = dot > np.cos(np.radians(12.))
    x, y = vectors@east/np.maximum(dot, .1), vectors@north/np.maximum(dot, .1)
    if kind == 'compact':
        value = np.zeros_like(theta)
        for _ in range(3):
            px, py = rng.uniform(-.08, .08, 2)
            value += np.exp(-.5*((x-px)**2+(y-py)**2)/.009**2)
    elif kind == 'filament':
        angle = rng.uniform(0, 2*np.pi)
        value = np.tanh((x*np.cos(angle)+y*np.sin(angle))/.009)*np.exp(-.5*(x*x+y*y)/.08**2)
    else:
        raise ValueError('unknown localized alternative')
    return np.where(good, value, 0.)


class SphericalResponseModel:
    """CPU spherical forward operator with explicit, varied synthetic nuisances."""
    def __init__(self, lmax=256, nside=128, size=64, cl=None, threads=8):
        import ducc0
        self.ducc, self.lmax, self.threads = ducc0, lmax, threads
        self.center = (np.pi/4, .7)
        self.size = size
        self.base = ducc0.healpix.Healpix_Base(nside, 'RING')
        self.info = self.base.sht_info()
        self.theta, self.phi = self.base.pix2ang(np.arange(self.base.npix())).T
        self.ell, self.m = spectra.alm_ell_m(lmax)
        self.locations = spherical_patch_locations(*self.center, np.radians([0,22.5,45,67.5]), size=size)
        self.beams = [theory.gaussian_beam(width, lmax)[self.ell] for width in (30.,60.,120.)]
        self.camb_version = None
        if cl is None:
            import camb
            self.camb_version = camb.__version__
            parameters = camb.set_params(lmax=lmax, **theory.PLANCK18)
            cl = camb.get_results(parameters).get_cmb_power_spectra(lmax=lmax, raw_cl=True, CMB_unit='muK')['total'][:, 0]
        self.cl = np.asarray(cl, dtype=np.float64).copy()
        if self.cl.shape != (lmax+1,) or not np.all(np.isfinite(self.cl)) or np.any(self.cl[2:] <= 0):
            raise ValueError('invalid benchmark TT spectrum')
        self.cl[:2] = 0
        self.foreground = self.unit_rms(self.adjoint(foreground_proxy(self.theta, self.phi)))
        self.noise_weight = 1+.6*np.sin(self.theta)*np.cos(2*self.phi)
        center_vector = np.array([np.sin(self.center[0])*np.cos(self.center[1]),
            np.sin(self.center[0])*np.sin(self.center[1]), np.cos(self.center[0])])
        self.local = self.base.pix2vec(np.arange(self.base.npix()))@center_vector > np.cos(np.radians(10.))

    def adjoint(self, values):
        alm = self.ducc.sht.adjoint_synthesis(map=np.asarray(values, dtype=np.float64)[None], spin=0,
            lmax=self.lmax, mmax=self.lmax, nthreads=self.threads, **self.info)[0]*(4*np.pi/self.base.npix())
        alm[self.ell < 2] = 0
        return alm

    def full_map(self, alm):
        return self.ducc.sht.synthesis(alm=alm[None], spin=0, lmax=self.lmax,
            mmax=self.lmax, nthreads=self.threads, **self.info)[0]

    def unit_rms(self, alm):
        power = spectra.cross_cl(alm, alm, self.lmax)
        variance = np.sum((2*np.arange(self.lmax+1)+1)*power)/(4*np.pi)
        if not np.isfinite(variance) or variance <= 0:
            raise ValueError('invalid component variance')
        return alm/np.sqrt(variance)

    def signal_quality(self, alm):
        power = spectra.cross_cl(alm, alm, self.lmax)
        error = float(np.max(np.abs(power[2:]/self.cl[2:]-1)))
        q = {'shell_relative_error':error, 'finite':bool(np.all(np.isfinite(alm))),
            'zero_shells_exact':bool(np.all(power[:2] == 0)), 'mzero_real':bool(np.all(alm[self.m == 0].imag == 0))}
        q['passed'] = q['finite'] and q['zero_shells_exact'] and q['mzero_real'] and error <= 1e-10
        return q

    def views(self, alm):
        return np.stack([self.ducc.sht.synthesis_general(alm=(alm*beam)[None], spin=0,
            lmax=self.lmax, mmax=self.lmax, loc=self.locations.reshape(-1, 2),
            epsilon=1e-11, nthreads=self.threads)[0].reshape(4, self.size, self.size) for beam in self.beams])

    def draw(self, seed, kind='null', amplitude=0.):
        rng = np.random.default_rng(seed)
        normals = rng.normal(size=(1, 2, len(self.ell)))
        signal = spherical_surrogates.fixed_power_reference(self.cl, normals)[0]
        foreground_amplitude, noise_amplitude = float(rng.uniform(0, 60)), float(rng.uniform(5, 25))
        raw_noise = (rng.normal(size=len(self.ell))+1j*rng.normal(size=len(self.ell))*(self.m > 0))/np.sqrt(np.where(self.m == 0, 1., 2.))
        raw_noise[self.ell < 2] = 0
        noise = self.unit_rms(self.adjoint(self.full_map(raw_noise)*self.noise_weight))
        clean_map = self.full_map(signal)
        local_rms = float(clean_map[self.local].std())
        if kind in ('quadratic', 'compact', 'filament'):
            if kind == 'quadratic':
                z = (clean_map-clean_map.mean())/clean_map.std()
                template = z*z-1
            else:
                template = localized_template(self.theta, self.phi, self.center, kind, rng)
            template = template/template[self.local].std()
            modified = signal+float(amplitude)*local_rms*self.adjoint(template)
            power = spectra.cross_cl(modified, modified, self.lmax)
            if np.any(power[2:] <= 0):
                raise ValueError('zero alternative shell direction')
            factor = np.zeros_like(self.cl)
            factor[2:] = np.sqrt(self.cl[2:]/power[2:])
            signal = modified*factor[self.ell]
        elif kind not in ('null', 'stripe_stress'):
            raise ValueError('unknown sky role')
        quality = self.signal_quality(signal)
        observed = signal+foreground_amplitude*self.foreground+noise_amplitude*noise
        if kind == 'stripe_stress':
            stripe = np.sin(18*self.phi+rng.uniform(0, 2*np.pi))*np.sin(self.theta)**2
            observed += 40*self.adjoint(stripe/stripe[self.local].std())
        fields = self.views(observed)
        parameters = {'seed':int(seed), 'kind':kind, 'amplitude':float(amplitude),
            'foreground_amplitude_uK':foreground_amplitude, 'noise_amplitude_uK':noise_amplitude,
            'original_local_signal_rms_uK':local_rms, 'observed_base_std_uK':float(fields[0, 0].std())}
        return fields, parameters, quality
