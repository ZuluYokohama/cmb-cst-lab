"""Independent fixed-fixture checks for native batch closure operators."""
import numpy as np
import pytest

from cmbcst.closure_fibers import features_numpy,ratios_numpy,projection_indices


def fixture_family(n=8,planes=4):
    rng = np.random.default_rng(929173)
    m = 17*n
    rates = np.array([.04,.11,.23,.41,.67])
    components = rng.normal(size=(5,2,m))+1j*rng.normal(size=(5,2,m))
    times = np.arange(4)
    base = np.einsum('tk,ksm->tsm',np.exp(-times[:,None]*rates),components)
    derivative = np.einsum('tk,ksm->tsm',-rates*np.exp(-times[:,None]*rates),components)
    bands = np.stack([np.einsum('tk,ksn->tsn',np.exp(-times[:,None]*rates[np.arange(5)%4==band]),
        components[np.arange(5)%4==band,:,:n]) for band in range(4)],axis=2)
    return dict(base=base,base_derivative=derivative,base_band=bands,
        cos_basis=.1*(rng.normal(size=(planes,m))+1j*rng.normal(size=(planes,m))),
        sin_basis=.1*(rng.normal(size=(planes,m))+1j*rng.normal(size=(planes,m))),
        rates=np.linspace(.05,.5,planes),band_id=np.arange(planes)%4,
        phase0=np.exp(-2j*rng.uniform(-np.pi,np.pi,size=(4,4,n))),
        phase1=np.exp(-2j*rng.uniform(-np.pi,np.pi,size=(4,4,n))),
        valid=rng.uniform(size=(4,4,n))>.2,
        reference_local_rms=float(np.sqrt(np.mean(np.abs(base[0].mean(axis=0)[:n])**2))),
        reference_global_rms=2.,global_weights=np.linspace(.01,.04,planes))


def test_batch_features_and_analytic_derivatives_match_independent_paths():
    torch = pytest.importorskip('torch')
    from cmbcst.closure_gpu_v2 import ClosureBatchCUDA
    from cmbcst.closure_fibers import ClosureCUDA
    family = fixture_family()
    theta = np.array([[.02,-.04,.01,.07],[-.03,.05,.06,-.02],[0.,0.,0.,0.]])
    x = torch.as_tensor(theta,dtype=torch.float64)
    model = ClosureBatchCUDA(family,'cpu')
    legacy = ClosureCUDA(family,'cpu')
    for u in range(4):
        feature,jac = model.features_and_jacobian(x,u)
        assert feature.shape==(3,52) and jac.shape==(3,52,4)
        assert np.allclose(feature.numpy(),features_numpy(family,theta,u),rtol=1e-12,atol=1e-12)
        automatic = torch.stack([torch.func.jacrev(lambda row:legacy.features(row,u))(row) for row in x])
        assert torch.allclose(jac,automatic,atol=1e-12,rtol=1e-12)
        assert torch.allclose(model.feature_jacobian(x,u,'repair'),jac[:,projection_indices('repair')])
    target,gradient = model.target_and_gradient(x)
    assert torch.allclose(target,legacy.features(x,1)[:,22],atol=1e-12,rtol=1e-12)
    auto_gradient = torch.stack([torch.func.grad(lambda row:legacy.features(row,1)[22])(row) for row in x])
    assert torch.allclose(gradient,auto_gradient,atol=1e-12,rtol=1e-12)


def test_ratio_squared_gradient_and_zero_angles():
    torch = pytest.importorskip('torch')
    from cmbcst.closure_gpu_v2 import ClosureBatchCUDA
    from cmbcst.closure_fibers import ClosureCUDA
    family = fixture_family()
    theta = np.array([[.02,-.04,.01,.07],[-.03,.05,.06,-.02],[0.,0.,0.,0.]])
    x = torch.as_tensor(theta,dtype=torch.float64)
    model = ClosureBatchCUDA(family,'cpu')
    legacy = ClosureCUDA(family,'cpu')
    squared,gradient = model.ratio_squared_and_gradient(x)
    assert np.allclose(squared.numpy(),ratios_numpy(family,theta)**2,rtol=1e-12,atol=1e-14)
    automatic = torch.stack([torch.func.jacrev(lambda row:legacy.ratios(row).square())(row) for row in x[:2]])
    assert torch.allclose(gradient[:2],automatic,atol=1e-12,rtol=1e-12)
    assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient[-1])==0


def test_zero_plane_and_explicit_batch_contract():
    torch = pytest.importorskip('torch')
    from cmbcst.closure_gpu_v2 import ClosureBatchCUDA
    family = fixture_family(planes=0)
    model = ClosureBatchCUDA(family,'cpu')
    x = torch.zeros((2,0),dtype=torch.float64)
    feature,jac = model.features_and_jacobian(x)
    assert feature.shape==(2,52) and jac.shape==(2,52,0)
    assert torch.isfinite(feature).all()
    assert model.rms_gradient(x).shape==(2,2,0)
    assert torch.count_nonzero(model.ratios(x))==0
    with pytest.raises(ValueError,match='batch'):
        model.features(x[0])
    with pytest.raises(ValueError,match='float64'):
        model.features(x.float())
