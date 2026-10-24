import json
import numpy as np
import pytest
from cmbcst.integration.family import RotationFamily,harmonic_layout


def reference(lmax=8,seed=9123):
    ell,m=harmonic_layout(lmax);rng=np.random.default_rng(seed)
    value=rng.normal(size=(2,2,len(ell)))+1j*rng.normal(size=(2,2,len(ell)))
    value[...,m==0]=value[...,m==0].real;value[...,ell<2]=0
    return value


def independent_gram(value,ell,m):
    rows=value.reshape(4,-1)
    return np.stack([(np.einsum('in,jn,n->ij',rows[:,ell==l],rows[:,ell==l].conj(),
        np.where(m[ell==l]==0,1.,2.)).real/(2*l+1)) for l in range(ell.max()+1)])


def test_stored_rotations_preserve_full_spectra_difference_and_rms():
    value=reference();family=RotationFamily(value,[6,8],planes_per_shell=4,seed=19)
    theta=np.linspace(-.3,.3,family.nplanes)
    candidate=family.candidate_alms(theta)
    assert family.cos_basis.shape==(8,2,value.shape[-1])
    assert np.array_equal(family.candidate_alms(np.zeros(8)),value)
    assert np.allclose(candidate[0]-candidate[1],value[0]-value[1],rtol=0,atol=1e-15)
    assert np.allclose(independent_gram(candidate,family.ell,family.m),independent_gram(value,family.ell,family.m),rtol=1e-13,atol=1e-13)
    delta=(candidate[0]+candidate[1]-value[0]-value[1])/2
    measured=np.sqrt(np.sum(np.abs(delta)**2*np.where(family.m==0,1.,2.))/family.reference_energy)
    assert family.rms_ratio(theta)==pytest.approx(measured,rel=1e-13,abs=1e-15)
    assert family.invariants(theta)['passed']
    batch=family.candidate_alms(np.stack([theta,-theta]))
    assert batch.shape==(2,2,2,value.shape[-1])
    assert all(family.invariants(np.stack([theta,-theta]))['per_state_passed'])
    assert np.array_equal(candidate[...,~np.isin(family.ell,[6,8])],value[...,~np.isin(family.ell,[6,8])])


def test_serialized_planes_replay_without_qr_or_rng(monkeypatch):
    family=RotationFamily(reference(),[6,8],4,17)
    arrays=family.to_arrays();binding=family.provenance()['array_sha256'];theta=np.full(8,.15)
    expected=family.candidate_alms(theta)
    monkeypatch.setattr(np.linalg,'qr',lambda *a,**k:pytest.fail('QR must not be repeated'))
    monkeypatch.setattr(np.random,'default_rng',lambda *a,**k:pytest.fail('RNG must not be repeated'))
    restored=RotationFamily.from_arrays(arrays,expected_sha256=binding)
    assert np.array_equal(restored.candidate_alms(theta),expected)
    assert restored.provenance()['array_sha256']==binding
    assert not restored.cos_basis.flags.writeable and not restored.reference_alms.flags.writeable
    corrupted={k:v.copy() for k,v in arrays.items()};corrupted['plane_u'][0,0]+=.01
    with pytest.raises(ValueError):RotationFamily.from_arrays(corrupted)
    changed={k:v.copy() for k,v in arrays.items()};changed['seed']=np.array(18,dtype=np.int64)
    with pytest.raises(ValueError,match='SHA256'):RotationFamily.from_arrays(changed,expected_sha256=binding)


def test_spin_two_low_shell_rank_and_input_rejections():
    low=RotationFamily(reference(3),[2],planes_per_shell=1,seed=1)
    assert low.invariants([.2])['passed']
    with pytest.raises(ValueError,match='insufficient'):RotationFamily(reference(3),[2],planes_per_shell=2)
    with pytest.raises(ValueError):RotationFamily(reference(),[1])
    with pytest.raises(ValueError):RotationFamily(reference(),[6,6])
    with pytest.raises(ValueError):RotationFamily(reference(),[6.])
    value=reference();value[...,0]=1
    with pytest.raises(ValueError,match='ell<2'):RotationFamily(value,[6])
    value=reference();value[0,0,6]+=.1j
    with pytest.raises(ValueError,match='m=0'):RotationFamily(value,[6])
    with pytest.raises(ValueError):low.candidate_alms([np.pi/4+1e-8])
    with pytest.raises(ValueError):low.rms_ratio([np.nan])
    with pytest.raises(ValueError):low.candidate_alms([.1j])
    with pytest.raises(ValueError):low.coordinates(np.array([.1+.2j]))
    with pytest.raises(ValueError):low.candidate_alms(np.array([True]))
    with pytest.raises(ValueError):low.candidate_alms(np.array([np.complex128(.1+.2j)],dtype=object))


def test_coordinates_include_constant_and_stable_small_angle_terms():
    family=RotationFamily(reference(3),[2],1,1)
    theta=np.array([1e-12]);z=family.coordinates(theta)
    assert z.shape==(3,) and z[0]==1 and z[1]<0
    assert z[1]==pytest.approx(-.5e-24,rel=1e-12,abs=0)
    assert np.allclose(np.einsum('p,pen->en',z,family.basis_alms),(family.candidate_alms(theta)[0]+family.candidate_alms(theta)[1])/2,rtol=1e-14,atol=1e-14)


def test_public_family_diagnostics_are_strict_json_and_stored_seed_is_scalar():
    family=RotationFamily(reference(),[6,8],planes_per_shell=np.int64(4),seed=np.int64(19))
    payload=dict(provenance=family.provenance(),scalar=family.invariants(np.zeros(8)),
        batch=family.invariants(np.array([[.1]*8,[-.1]*8])))
    restored=json.loads(json.dumps(payload,allow_nan=False))
    assert restored['scalar']['passed'] is True and restored['scalar']['per_state_passed'] is True
    assert restored['batch']['per_state_passed']==[True,True]
    assert restored['batch']['finite']==[True,True]
    assert type(restored['provenance']['seed']) is int
    arrays=family.to_arrays()
    replay=RotationFamily.from_arrays(arrays,expected_sha256=restored['provenance']['array_sha256'])
    assert json.loads(json.dumps(replay.provenance(),allow_nan=False))==restored['provenance']
    arrays['seed']=np.array([19],dtype=np.int64)
    with pytest.raises(ValueError,match='scalar'):RotationFamily.from_arrays(arrays)


def test_failed_invariant_diagnostics_serialize_without_hiding_nonfinite_errors(monkeypatch):
    family=RotationFamily(reference(3),[2],planes_per_shell=1,seed=19)
    broken=family.reference_alms.copy();broken[0,0,0]=.1
    monkeypatch.setattr(family,'candidate_alms',lambda theta:broken.copy())
    record=json.loads(json.dumps(family.invariants([0.]),allow_nan=False))
    assert record['passed'] is False and record['finite'] is False
    assert record['maximum_spectral_error'] is None
    assert record['split_difference_scaled_error']>0
