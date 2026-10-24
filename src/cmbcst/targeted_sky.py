"""Physical-pattern variants for the targeted synthetic pilot; not Planck fits."""
import numpy as np
from .response_sky import SphericalResponseModel
from . import spectra, spherical_surrogates


def physical_template(theta,phi,center,family,width_degrees,rng):
    vectors = np.stack([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)],axis=-1)
    t,p = center
    dot = vectors@np.array([np.sin(t)*np.cos(p),np.sin(t)*np.sin(p),np.cos(t)])
    x = vectors@np.array([-np.sin(p),np.cos(p),0.])/np.maximum(dot,.1)
    y = vectors@np.array([-np.cos(t)*np.cos(p),-np.cos(t)*np.sin(p),np.sin(t)])/np.maximum(dot,.1)
    width = np.radians(width_degrees)
    if family=='compact':
        field = np.zeros_like(theta)
        for _ in range(3):
            px,py = rng.uniform(-.08,.08,2)
            field += np.exp(-.5*((x-px)**2+(y-py)**2)/width**2)
    elif family=='filament':
        angle = rng.uniform(0,2*np.pi)
        field = np.tanh((x*np.cos(angle)+y*np.sin(angle))/width)*np.exp(-.5*(x*x+y*y)/.08**2)
    else:
        raise ValueError('unknown localized family')
    return np.where(dot>np.cos(np.radians(12.)),field,0.)


class TargetedSky(SphericalResponseModel):
    def draw_recipe(self,seed,recipe=None):
        rng = np.random.default_rng(seed)
        signal = spherical_surrogates.fixed_power_reference(self.cl,rng.normal(size=(1,2,len(self.ell))))[0]
        fa,na = float(rng.uniform(0,60)),float(rng.uniform(5,25))
        raw_noise = (rng.normal(size=len(self.ell))+1j*rng.normal(size=len(self.ell))*(self.m>0))/np.sqrt(np.where(self.m==0,1.,2.))
        raw_noise[self.ell<2] = 0
        noise = self.unit_rms(self.adjoint(self.full_map(raw_noise)*self.noise_weight))
        clean = self.full_map(signal)
        local_rms = float(clean[self.local].std())
        if recipe is not None:
            if recipe['family']=='quadratic':
                z = (clean-clean.mean())/clean.std()
                template = recipe['variant']*(z*z-1)
            else:
                template = physical_template(self.theta,self.phi,self.center,recipe['family'],recipe['variant'],rng)
            template /= template[self.local].std()
            modified = signal+recipe['amplitude']*local_rms*self.adjoint(template)
            power = spectra.cross_cl(modified,modified,self.lmax)
            if np.any(power[2:]<=0):
                raise ValueError('zero alternative shell')
            factor = np.zeros_like(self.cl)
            factor[2:] = np.sqrt(self.cl[2:]/power[2:])
            signal = modified*factor[self.ell]
        quality = self.signal_quality(signal)
        fields = self.views(signal+fa*self.foreground+na*noise)
        return fields,{'seed':int(seed),'recipe':None if recipe is None else recipe['id'],
            'foreground_amplitude_uK':fa,'noise_amplitude_uK':na,'local_signal_rms_uK':local_rms},quality
