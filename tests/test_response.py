import ast
from pathlib import Path
import numpy as np
from cmbcst import multiscale, response, response_sky


def test_response_shared_jovin_checks():
    path = Path(__file__).parent/'sandbox/jovin_tests_response.py'
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if not isinstance(node,ast.ImportFrom)]
    namespace = {**vars(multiscale),**vars(response),**vars(response_sky)}
    exec(compile(tree,str(path),'exec'),namespace)
    for name,fn in list(namespace.items()):
        if name.startswith('test_'):
            fn()


def test_direct_spherical_synthesis_matches_analytic_dipole():
    import ducc0
    loc = response.spherical_patch_locations(.8,.7,np.radians([0,22.5,45,67.5]),size=21)
    alm = np.zeros(6,dtype=complex)
    alm[1] = 1.
    values = ducc0.sht.synthesis_general(alm=alm[None],spin=0,lmax=2,
        loc=loc.reshape(-1,2),epsilon=1e-11,nthreads=2)[0].reshape(loc.shape[:-1])
    assert np.allclose(values,np.sqrt(3/(4*np.pi))*np.cos(loc[...,0]),rtol=1e-10,atol=1e-10)


def test_synthetic_forward_model_all_roles_keep_signal_shell_power():
    cl = np.zeros(25)
    cl[2:] = 1/(np.arange(2,25)*(np.arange(2,25)+1))
    model = response_sky.SphericalResponseModel(lmax=24,nside=16,size=32,cl=cl,threads=2)
    for i,kind in enumerate(('null','quadratic','compact','filament','stripe_stress')):
        fields,parameters,quality = model.draw(936560+i,kind,.15)
        assert fields.shape == (3,4,32,32) and np.all(np.isfinite(fields))
        assert quality['passed'] and 0 <= parameters['foreground_amplitude_uK'] <= 60
        assert 5 <= parameters['noise_amplitude_uK'] <= 25
