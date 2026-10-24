import numpy as np
from cmbcst import closure_fibers_data as d


def fixture(lmax=48):
    ell,m=d.harmonic_layout(lmax)
    rng=np.random.default_rng(918331)
    value=rng.normal(size=(2,2,len(ell)))+1j*rng.normal(size=(2,2,len(ell)))*(m>0)
    value[...,ell<30]=0
    return value


def test_real_harmonic_coordinates_preserve_shell_inner_products():
    values=fixture()
    x=d.real_shell(values,35)
    back=d.complex_shell(x,35)
    ell,m=d.harmonic_layout(48)
    assert np.max(np.abs(back-values[...,ell==35]))<1e-14
    flat=x.reshape(4,-1)
    assert np.allclose(flat@flat.T/(2*35+1),d.spectral_gram(values)[35],atol=1e-14)


def test_rotation_family_preserves_all_spectra_and_fixed_difference():
    values=fixture()
    planes=d.build_planes(values,24,shells=[30,35,40],planes_per_shell=2)
    angles=np.linspace(-.7,.7,6)
    changed=d.apply_angles(values,planes,angles)
    audit=d.invariants(values,changed)
    assert audit['passed']
    assert audit['normalized_spectral_gram_error']<1e-12
    expected=np.sum(2*(1-np.cos(angles))*planes['global_weights'])
    delta=(changed[0]+changed[1]-values[0]-values[1])/2
    assert abs(d.global_rms(delta)**2-expected)<1e-12


def test_nonorthogonal_or_noise_unconstrained_rotation_is_rejected():
    values=fixture()
    planes=d.build_planes(values,24,shells=[30],planes_per_shell=1)
    # A nonorthogonal replacement must not receive a preservation certificate.
    planes['u'][0]*=1.01
    planes['coadd_u'][0]*=1.01
    changed=d.apply_angles(values,planes,np.array([.5]))
    assert not d.invariants(values,changed)['passed']
    # Rotating a direction along the split difference without fixing it changes
    # the difference when all four rows are transformed, and fails the gate.
    full=values.copy()
    shell=30
    rows=d.real_shell(values,shell).reshape(4,-1)
    difference=rows[0]-rows[2]
    u=difference/np.linalg.norm(difference)
    rng=np.random.default_rng(444)
    v=rng.normal(size=len(u));v-=u*(v@u);v/=np.linalg.norm(v)
    rotation=np.eye(len(u))+(np.cos(.3)-1)*(np.outer(u,u)+np.outer(v,v))+np.sin(.3)*(np.outer(v,u)-np.outer(u,v))
    ell,_=d.harmonic_layout(48)
    full[...,ell==shell]=d.complex_shell((rows@rotation).reshape(2,2,-1),shell)
    check=d.invariants(values,full)
    assert not check['passed'] and check['split_difference_scaled_error']>.001


def test_smoothing_spectral_representation_is_closed_and_composes():
    values=fixture()
    ell,_=d.harmonic_layout(48)
    rate=.5*d.SIGMA**2*ell*(ell+1)
    first,second=.7,1.2
    combined=values*np.exp(-(first+second)*rate)
    sequential=values*np.exp(-first*rate)*np.exp(-second*rate)
    assert np.max(np.abs(combined-sequential))<1e-14
    expected=d.spectral_gram(values)*np.exp(-(first+second)*d.SIGMA**2*np.arange(49)*(np.arange(49)+1))[:,None,None]
    assert np.max(np.abs(d.spectral_gram(combined)-expected))<1e-14
