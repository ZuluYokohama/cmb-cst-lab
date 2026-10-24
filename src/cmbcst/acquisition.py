"""Controlled tangent-plane polarized acquisition, not a Planck TOD simulator.

Indices describe angular position, detector look/roll and beam footprint. No
radial distance is inferred. Periodic boundaries and a fixed local Stokes basis
are explicit approximations of this synthetic model.
"""
from __future__ import annotations
import numpy as np

MODES = ('correct','roll_error','pointing_error','circular_beam')


def detector_table():
    h = np.arange(8,dtype=float)
    return dict(channel=np.arange(8),incidence_deg=np.array([0.,10.,20.,30.,40.,50.,55.,60.]),
        beam_angle_deg=22.5*h,polarization_deg=20+22.5*h,
        offset_x_pixels=.75*np.cos(2*np.pi*h/8),offset_y_pixels=.75*np.sin(2*np.pi*h/8),
        gain=1+.02*np.sin(2*np.pi*h/8),efficiency=np.full(8,.95))


def frequency_grid(n):
    if type(n) is not int or n<16 or n%2:
        raise ValueError('need an even grid of at least16')
    fy,fx = np.meshgrid(np.fft.fftfreq(n),np.fft.rfftfreq(n),indexing='ij')
    radius = np.hypot(fx,fy)
    mask = radius<=.2
    target = np.exp(-2*np.pi**2*9*radius**2)*mask
    spectral_filter = (1+(radius/.04)**4)**(-.5)*mask
    spectral_filter[0,0] = 0
    weight = np.full_like(radius,2.)
    weight[:,0] = weight[:,-1] = 1.
    return fx,fy,mask,target,spectral_filter,weight


def instrument_design(n,mode='correct'):
    if mode not in MODES+('poor_angles',):
        raise ValueError('unknown instrument assumption')
    fx,fy,mask,target,_,_ = frequency_grid(n)
    d = detector_table()
    psi = np.radians(d['polarization_deg']+(10. if mode=='roll_error' else 0.))
    if mode=='poor_angles':
        psi = np.zeros(8)
    rows = []
    for h in range(8):
        angle = np.radians(d['beam_angle_deg'][h])
        parallel = fx*np.cos(angle)+fy*np.sin(angle)
        perpendicular = -fx*np.sin(angle)+fy*np.cos(angle)
        major = 1.2 if mode=='circular_beam' else 1.2/np.cos(np.radians(d['incidence_deg'][h]))
        beam = np.exp(-2*np.pi**2*(major**2*parallel**2+1.2**2*perpendicular**2))
        phase = np.ones_like(fx,dtype=complex) if mode=='pointing_error' else np.exp(-2j*np.pi*(fx*d['offset_x_pixels'][h]+fy*d['offset_y_pixels'][h]))
        p = np.array([1.,d['efficiency'][h]*np.cos(2*psi[h]),d['efficiency'][h]*np.sin(2*psi[h])])
        rows.append((d['gain'][h]*beam*phase*mask)[...,None]*p)
    return np.stack(rows,axis=-2).reshape(-1,8,3),target.ravel(),mask.ravel()


def decoder_reference(design,target,mask):
    a = np.asarray(design,dtype=np.complex128)[mask]
    gram = np.einsum('fhi,fhj->fij',a.conj(),a)
    eigen = np.linalg.eigvalsh(gram)
    if not np.isfinite(eigen).all() or np.any(eigen[:,0]<=0):
        raise ValueError('unidentifiable polarization geometry')
    condition = float(np.max(eigen[:,-1]/eigen[:,0]))
    if condition>1e8:
        raise ValueError('ill-conditioned polarization geometry')
    w = np.zeros((len(mask),3,8),dtype=np.complex128)
    w[mask] = np.linalg.solve(gram,a.conj().transpose(0,2,1))*target[mask,None,None]
    return w,condition


def reference_models(n):
    models,conditions = {},{}
    for mode in MODES:
        a,target,mask = instrument_design(n,mode)
        models[mode],conditions[mode] = decoder_reference(a,target,mask)
    return models,conditions


def checked_white(sky,noise):
    x,z = np.asarray(sky,dtype=np.float64),np.asarray(noise,dtype=np.float64)
    if x.ndim!=4 or x.shape[1]!=3 or x.shape[-1]!=x.shape[-2] or z.shape!=(len(x),8,*x.shape[-2:]) or not len(x) or not np.isfinite(x).all() or not np.isfinite(z).all():
        raise ValueError('need finite [batch,3,n,n] sky and [batch,8,n,n] noise')
    frequency_grid(x.shape[-1])
    return x,z


def source_reference(white,i_only=False):
    n = white.shape[-1]
    filt = frequency_grid(n)[4]
    spectral = np.fft.rfft2(white,norm='ortho')*filt
    image = np.fft.irfft2(spectral,s=(n,n),norm='ortho')
    std = image.std(axis=(-2,-1),keepdims=True)
    if np.any(std<=0):
        raise ValueError('degenerate source')
    spectral *= np.array([100.,5.,5.])[None,:,None,None]/std
    if i_only:
        spectral[:,1:] = 0
    return spectral.reshape(len(white),3,-1)


def batch_reference(sky_white,noise_white,sigma=.5,i_only=False,return_maps=False):
    sky_white,noise_white = checked_white(sky_white,noise_white)
    n = sky_white.shape[-1]
    a,target,_ = instrument_design(n)
    models,conditions = reference_models(n)
    source = source_reference(sky_white,i_only)
    observed = np.einsum('fhc,bcf->bhf',a,source)+sigma*np.fft.rfft2(noise_white,norm='ortho').reshape(len(source),8,-1)
    truth = np.fft.irfft2((source*target).reshape(-1,3,n,n//2+1),s=(n,n),norm='ortho')
    metrics,maps = [],[]
    for mode in MODES:
        recovered = np.fft.irfft2(np.einsum('fch,bhf->bcf',models[mode],observed).reshape(-1,3,n,n//2+1),s=(n,n),norm='ortho')
        errors = np.mean((recovered-truth)**2,axis=(-2,-1))
        p = np.hypot(recovered[:,1],recovered[:,2])
        truth_p = np.hypot(truth[:,1],truth[:,2])
        metrics.append(np.c_[errors,np.mean((p-truth_p)**2,axis=(-2,-1))])
        maps.append(recovered)
    p0,p1 = [np.hypot(m[:,1],m[:,2]) for m in maps[:2]]
    invariant = float(np.max(np.abs(p1-p0)/(1+np.abs(p0))))
    result = dict(mse=np.stack(metrics,axis=1),roll_amplitude_error=invariant,conditions=conditions)
    if return_maps:
        result.update(truth=truth,recovered=np.stack(maps,axis=1),
            observed=np.fft.irfft2(observed.reshape(-1,8,n,n//2+1),s=(n,n),norm='ortho'))
    return result


class AcquisitionCUDA:
    """Resident CUDA forward and inverse operators, complex128/float64 throughout."""
    def __init__(self,n=512,device='cuda:0'):
        import torch
        self.torch,self.n,self.device = torch,n,device
        self.tensor = lambda value:torch.as_tensor(value,device=device)
        a,target,mask = instrument_design(n)
        self.a,self.target = self.tensor(a),self.tensor(target)
        self.filter = self.tensor(frequency_grid(n)[4])
        self.models,self.conditions = {},{}
        for mode in MODES:
            design,_,_ = instrument_design(n,mode)
            reference_gram = np.einsum('fhi,fhj->fij',design[mask].conj(),design[mask])
            reference_eigen = np.linalg.eigvalsh(reference_gram)
            condition = float(np.max(reference_eigen[:,-1]/reference_eigen[:,0]))
            if np.any(reference_eigen[:,0]<=0) or not np.isfinite(condition) or condition>1e8:
                raise ValueError('unidentifiable CUDA geometry')
            active = self.tensor(design[mask])
            gram = active.conj().transpose(-1,-2)@active
            weights = torch.zeros((len(mask),3,8),dtype=torch.complex128,device=device)
            # Bound cuSOLVER workspace for many tiny3x3 systems. Conditioning is
            # static CPU geometry work; device inverse weights are computed here.
            solved = torch.cat([torch.linalg.solve(gram[first:first+256],active[first:first+256].conj().transpose(-1,-2))
                for first in range(0,len(active),256)])
            weights[self.tensor(mask)] = solved*self.target[self.tensor(mask),None,None]
            self.models[mode],self.conditions[mode] = weights,condition
        weight = self.tensor(frequency_grid(n)[5].ravel())
        self.predicted_noise_variance = (.25*torch.sum(torch.abs(self.models['correct'])**2*weight[:,None,None],dim=(0,2))/(n*n)).cpu().numpy()

    def run(self,white,noise,sigma=.5,i_only=False,return_maps=False):
        torch,n = self.torch,self.n
        spectral = torch.fft.rfft2(white,norm='ortho')*self.filter
        image = torch.fft.irfft2(spectral,s=(n,n),norm='ortho')
        std = image.std(dim=(-2,-1),correction=0,keepdim=True)
        if bool(torch.any(std<=0).item()):
            raise ValueError('degenerate CUDA source')
        spectral = spectral*(self.tensor(np.array([100.,5.,5.]))[None,:,None,None]/std)
        del image
        if i_only:
            spectral[:,1:] = 0
        source = spectral.reshape(len(white),3,-1)
        observed = torch.einsum('fhc,bcf->bhf',self.a,source)+sigma*torch.fft.rfft2(noise,norm='ortho').reshape(len(white),8,-1)
        truth = torch.fft.irfft2((source*self.target).reshape(-1,3,n,n//2+1),s=(n,n),norm='ortho')
        truth_p = torch.hypot(truth[:,1],truth[:,2])
        metrics,maps,p0 = [],[],None
        invariant = 0.
        for mode in MODES:
            recovered = torch.fft.irfft2(torch.einsum('fch,bhf->bcf',self.models[mode],observed).reshape(-1,3,n,n//2+1),s=(n,n),norm='ortho')
            p = torch.hypot(recovered[:,1],recovered[:,2])
            errors = torch.mean((recovered-truth)**2,dim=(-2,-1))
            metrics.append(torch.cat([errors,torch.mean((p-truth_p)**2,dim=(-2,-1))[:,None]],dim=1))
            if mode=='correct':
                p0 = p
            elif mode=='roll_error':
                invariant = float(torch.max(torch.abs(p-p0)/(1+torch.abs(p0))).item())
            if return_maps:
                maps.append(recovered.cpu().numpy())
        mse = torch.stack(metrics,dim=1)
        if not bool(torch.isfinite(mse).all().item()):
            raise ValueError('nonfinite CUDA metrics')
        result = dict(mse=mse.cpu().numpy(),roll_amplitude_error=invariant)
        if return_maps:
            result.update(truth=truth.cpu().numpy(),recovered=np.stack(maps,axis=1),
                observed=torch.fft.irfft2(observed.reshape(-1,8,n,n//2+1),s=(n,n),norm='ortho').cpu().numpy())
        return result
