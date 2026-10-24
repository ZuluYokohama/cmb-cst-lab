import json
import numpy as np
import pytest
from cmbcst.integration.quadratic import QuadraticModel


def model():
    rng=np.random.default_rng(741)
    raw=rng.normal(size=(5,7,7))
    return QuadraticModel((raw+raw.transpose(0,2,1))/2)


def test_quadratic_values_and_analytic_jacobian_against_independent_differences():
    q=model();theta=np.array([.1,-.2,.25]);z=np.r_[1,np.cos(theta)-1,np.sin(theta)]
    expected=np.array([z@matrix@z for matrix in q.Q])
    assert np.allclose(q.features(theta),expected,rtol=1e-14,atol=1e-14)
    h=1e-3;fd=[]
    for direction in np.eye(3):
        fd.append((-q.features(theta+2*h*direction)+8*q.features(theta+h*direction)
            -8*q.features(theta-h*direction)+q.features(theta-2*h*direction))/(12*h))
    assert np.allclose(q.jacobian(theta),np.array(fd).T,rtol=1e-9,atol=1e-10)
    batch=np.stack([theta,-theta]);assert q.features(batch).shape==(2,5)
    assert q.jacobian(batch).shape==(2,5,3)
    assert np.allclose(q.jac(batch)[0],q.jacobian(theta))


def test_constant_coordinate_preserves_linear_and_constant_terms_and_replay():
    raw=np.zeros((1,3,3));raw[0,0,0]=2;raw[0,0,2]=raw[0,2,0]=1.5
    q=QuadraticModel(raw)
    assert q.features([.2])[0]==pytest.approx(2+3*np.sin(.2))
    assert q.jacobian([.2])[0,0]==pytest.approx(3*np.cos(.2))
    restored=QuadraticModel.from_arrays(q.to_arrays(),expected_sha256=q.sha256)
    assert np.array_equal(restored.Q,q.Q) and not restored.Q.flags.writeable
    assert json.loads(json.dumps(q.provenance(),allow_nan=False))['Q_sha256']==q.sha256
    with pytest.raises(ValueError,match='SHA256'):QuadraticModel.from_arrays(dict(Q=q.Q+np.eye(3)),expected_sha256=q.sha256)
    constant=QuadraticModel(np.array([[[3.]]]))
    assert constant.features(np.empty((2,0))).shape==(2,1)
    assert constant.jacobian(np.empty((2,0))).shape==(2,1,0)


def test_invalid_quadratic_inputs_rejected():
    with pytest.raises(ValueError):QuadraticModel(np.zeros((1,4,4)))
    with pytest.raises(ValueError):QuadraticModel(np.full((1,3,3),np.nan))
    raw=np.zeros((1,3,3));raw[0,0,1]=1
    with pytest.raises(ValueError,match='symmetric'):QuadraticModel(raw)
    with pytest.raises(ValueError):model().features([np.nan,0,0])
    with pytest.raises(ValueError):QuadraticModel(np.zeros((1,3,3),dtype=complex))
    with pytest.raises(ValueError):model().features([.1j,0,0])
    with pytest.raises(ValueError):model().jacobian(np.array([.1+.2j,0,0]))
    with pytest.raises(ValueError):model().features(np.array([True,False,False]))
    with pytest.raises(ValueError):QuadraticModel(np.ones((1,3,3),dtype=bool))
    with pytest.raises(ValueError):model().features(np.array([np.complex128(.1+.2j),0,0],dtype=object))


def test_optional_native_torch_cpu_matches_numpy_and_autograd():
    torch=pytest.importorskip('torch');q=model();actual=q.torch('cpu')
    angles=torch.tensor([[.1,-.2,.25],[-.1,.2,-.25]],dtype=torch.float64)
    assert np.allclose(actual.features(angles).numpy(),q.features(angles.numpy()),rtol=1e-13,atol=1e-13)
    assert np.allclose(actual.jacobian(angles).numpy(),q.jacobian(angles.numpy()),rtol=1e-13,atol=1e-13)
    jac=torch.func.vmap(torch.func.jacrev(lambda value:actual.features(value[None])[0]))(angles)
    assert torch.allclose(actual.jacobian(angles),jac,rtol=1e-13,atol=1e-13)
