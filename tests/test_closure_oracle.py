import numpy as np
import pytest
from cmbcst.closure_fibers_oracle import covariance_moments,oracle_features
from cmbcst.closure_fibers import features_numpy


def fixture(time_shift=0.):
    rng=np.random.default_rng(601240)
    n,p=9,6
    rates=np.linspace(.06,.35,p)
    component_rates=np.array([.04,.12,.23,.43])
    components=rng.normal(size=(4,2,17*n))+1j*rng.normal(size=(4,2,17*n))
    decay=np.exp(-(np.arange(4)[:,None]+time_shift)*component_rates)
    base=np.einsum('tb,bsm->tsm',decay,components)
    derivative=np.einsum('tb,bsm->tsm',-decay*component_rates,components)
    band=np.einsum('tb,bsn->tsbn',decay,components[:,:,:n])
    cosine=.05*(rng.normal(size=(p,17*n))+1j*rng.normal(size=(p,17*n)))
    sine=.05*(rng.normal(size=(p,17*n))+1j*rng.normal(size=(p,17*n)))
    phase0=np.exp(1j*rng.normal(size=(4,4,n)))
    phase1=np.exp(1j*rng.normal(size=(4,4,n)))
    valid=rng.random((4,4,n))>.15
    return dict(base=base,base_derivative=derivative,base_band=band,
                cos_basis=cosine*np.exp(-time_shift*rates)[:,None],
                sin_basis=sine*np.exp(-time_shift*rates)[:,None],rates=rates,
                band_id=np.arange(p)%4,phase0=phase0,phase1=phase1,valid=valid,
                reference_local_rms=np.array(1.),reference_global_rms=np.array(2.),
                global_weights=np.full(p,.01))


def test_exact_real_covariance_and_scaling_derivative():
    z=np.array([-1.,1.,-1.,1.])
    x=np.column_stack([z,z*0,z,z*0])
    stat,derivative=covariance_moments(x,.2*x)
    assert np.array_equal(stat,[1,1,1,1,0,1,-2,0])
    assert np.allclose(derivative,[.4,.4,.4,.4,0,0,0,0],atol=2e-15)


def test_independent_covariance_features_match_complex_route():
    family=fixture()
    angles=np.array([.1,-.2,.3,-.12,.04,.2])
    for u in range(4):
        expected=features_numpy(family,angles,u)
        actual=oracle_features(family,angles,u)
        assert np.max(np.abs(actual-expected)/(1+np.abs(expected)))<1e-12
    batch=np.stack([angles,angles*.3])
    assert np.allclose(oracle_features(family,batch),features_numpy(family,batch),rtol=1e-12,atol=1e-12)


def test_independent_smoothing_derivatives_have_finite_difference_limit():
    angles=np.array([.1,-.2,.3,-.12,.04,.2])
    current=oracle_features(fixture(),angles)
    h=1e-5
    difference=(oracle_features(fixture(h),angles)-oracle_features(fixture(-h),angles))/(2*h)
    assert np.allclose(current[44:48],difference[[3,11,19,27]],rtol=1e-7,atol=1e-8)
    assert np.allclose(current[48:52],difference[[6,14,22,30]],rtol=1e-7,atol=1e-8)


def test_covariance_oracle_rejects_invalid_or_degenerate_samples():
    with pytest.raises(ValueError):
        covariance_moments(np.zeros((8,4)),np.zeros((8,4)))
    family=fixture()
    with pytest.raises(ValueError):
        oracle_features(family,np.zeros(6),u=.5)
    with pytest.raises(ValueError):
        oracle_features(family,np.zeros(6,dtype=complex))
