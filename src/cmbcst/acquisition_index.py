"""Derived physical look coordinates for the frozen negative Fourier phase."""
import numpy as np


def look_coordinates(x,y,offset_x,offset_y,n=512,width=20.):
    """exp(-2pi*i*f*offset) samples f(x-offset), with periodic wrapping."""
    values = np.asarray([x,y,offset_x,offset_y],dtype=float)
    if not np.isfinite(values).all() or type(n) is not int or n<2 or width<=0:
        raise ValueError('invalid geometry index')
    unwrapped = (values[:2]-n/2-values[2:])*width/n
    wrapped = (unwrapped+width/2)%width-width/2
    return unwrapped,wrapped


def analytic_phase_error(design,table,n=32):
    """Independent cosine oracle for the sign of a single Fourier mode."""
    source = np.zeros((n,n//2+1),dtype=complex)
    source[0,1] = n/2
    expected,opposite = [],[]
    actual = []
    grid = np.arange(n)
    for h in range(8):
        spectrum = design[:,h,0].reshape(n,n//2+1)*source
        actual.append(np.fft.irfft2(spectrum,s=(n,n),norm='ortho')[0])
        eta = np.radians(table['beam_angle_deg'][h])
        major = 1.2/np.cos(np.radians(table['incidence_deg'][h]))
        transfer = np.exp(-2*np.pi**2*(major**2*np.cos(eta)**2+1.2**2*np.sin(eta)**2)/n**2)
        for values,sign in ((expected,-1),(opposite,1)):
            values.append(table['gain'][h]*transfer*np.cos(2*np.pi*(grid+sign*table['offset_x_pixels'][h])/n))
    return float(np.max(np.abs(np.asarray(actual)-expected))),float(np.max(np.abs(np.asarray(actual)-opposite)))
