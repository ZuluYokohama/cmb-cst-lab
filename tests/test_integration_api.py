"""Map-facing contract checks and actual-backend residual-preserving controls."""
import json
import numpy as np
import pytest

from cmbcst.integration import api
from cmbcst.integration.models import MissingOptionalDependency,ObservingResponse,SplitPair,array_sha256
from cmbcst.integration.operators import alm_layout


def map_pair(*,responses=None,different_masks=False):
    rng=np.random.default_rng(443921)
    a,b=rng.normal(size=(2,2,48))
    mask=np.ones(48,dtype=bool);mask[::7]=False
    # Retain finite nonzero invalid samples so a zero-filled reconstruction
    # cannot accidentally satisfy the original-observation residual checks.
    a[:,~mask]+=10.;b[:,~mask]-=7.
    ma,mb=mask.copy(),mask.copy()
    if different_masks:
        mb[1]=False
    if responses is None:
        response=ObservingResponse('identity',{})
        responses=(response,response)
    return SplitPair.create(a,b,nside=2,lmax=3,ordering='RING',coordinate_frame='G',
        units='uK_CMB',polarization_convention='COSMO',mask_a=ma,mask_b=mb,
        response_a=responses[0],response_b=responses[1])


def prepare(pair,**kwargs):
    options=dict(n_planes=2,shells=[2,3],planes_per_shell=1,seed=9217,backend='ducc0',nthreads=2)
    options.update(kwargs)
    return api.generate_counterfactual_family(pair,**options)


def _forbid_backend(monkeypatch):
    def forbidden(*args,**kwargs):
        pytest.fail('invalid preparation must not instantiate a numerical backend')
    monkeypatch.setattr(api,'DuccHealpixOperator',forbidden)
    monkeypatch.setattr(api,'HealpyHealpixOperator',forbidden)


def test_reject_unvalidated_pair_and_unknown_or_unequal_observing_responses(monkeypatch):
    _forbid_backend(monkeypatch)
    with pytest.raises(TypeError,match='validated SplitPair'):
        prepare(None)
    unknown=ObservingResponse('unknown',{'reason':'response not provided'})
    with pytest.raises(ValueError,match='common known observing response'):
        prepare(map_pair(responses=(unknown,unknown)))
    identity=ObservingResponse('identity',{})
    different=ObservingResponse('diagonal',{'asset_sha256':'e'*64,'description':'different split beam'})
    with pytest.raises(ValueError,match='common known observing response'):
        prepare(map_pair(responses=(identity,different)))
    with pytest.raises(ValueError,match='common split validity'):
        prepare(map_pair(different_masks=True))


@pytest.mark.parametrize('options',[
    {'n_planes':0},{'n_planes':-1},{'n_planes':True},{'n_planes':2.5},
    {'planes_per_shell':0},{'planes_per_shell':True},
    {'n_planes':3,'planes_per_shell':2},
    {'n_planes':3,'shells':[2,3],'planes_per_shell':1},
    {'n_planes':128,'shells':None,'planes_per_shell':4},
])
def test_reject_invalid_plane_budgets_before_backend_execution(monkeypatch,options):
    _forbid_backend(monkeypatch)
    with pytest.raises(ValueError,match='plane count|shell manifest'):
        prepare(map_pair(),**options)


@pytest.mark.parametrize('backend',['auto','shtns','pixell','DUCC0',None])
def test_backend_selection_must_be_explicit_and_supported(monkeypatch,backend):
    _forbid_backend(monkeypatch)
    with pytest.raises(ValueError,match='supported first-profile backends'):
        prepare(map_pair(),backend=backend)


@pytest.mark.parametrize('backend',['ducc0','healpy'])
def test_explicit_backend_dispatch_and_missing_dependency_have_no_fallback(monkeypatch,backend):
    calls=[]
    def unavailable(nside,lmax,nthreads):
        calls.append((nside,lmax,nthreads))
        raise MissingOptionalDependency(f'requested {backend} absent')
    def other(*args,**kwargs):
        pytest.fail('a missing requested backend must not fall back to another')
    monkeypatch.setattr(api,'DuccHealpixOperator',unavailable if backend=='ducc0' else other)
    monkeypatch.setattr(api,'HealpyHealpixOperator',unavailable if backend=='healpy' else other)
    with pytest.raises(MissingOptionalDependency,match=f'requested {backend} absent'):
        prepare(map_pair(),backend=backend)
    assert calls==[(2,3,2)]


def _gram(alms,lmax):
    ell,m=alm_layout(lmax)
    rows=alms.reshape(4,-1)
    return np.stack([np.einsum('in,jn,n->ij',rows[:,ell==degree],rows[:,ell==degree].conj(),
        np.where(m[ell==degree]==0,1.,2.)).real/(2*degree+1) for degree in range(lmax+1)])


def test_actual_ducc_preserves_original_residual_full_pixel_difference_and_retained_gram():
    pytest.importorskip('ducc0',reason='actual map-facing spin-2 integration requires ducc0')
    pair=map_pair();source_hashes=[array_sha256(pair.a),array_sha256(pair.b)]
    prepared=prepare(pair)
    theta=np.array([.22,-.17])
    candidate_alms=prepared.family.candidate_alms(theta)
    common_delta=prepared.operator.synthesize(candidate_alms.mean(axis=0)-prepared.family.coadd_alms)
    candidate_maps=prepared.candidate_maps(theta)
    assert candidate_maps.shape==(2,2,48)
    assert np.allclose(candidate_maps,prepared.reference_maps+common_delta[None],rtol=0,atol=1e-13)
    assert np.allclose(candidate_maps[0]-candidate_maps[1],pair.a-pair.b,rtol=0,atol=1e-13)
    modeled=np.stack([prepared.operator.synthesize(row) for row in candidate_alms])
    assert np.allclose(candidate_maps-modeled,prepared.residual_maps,rtol=0,atol=1e-12)
    assert np.linalg.norm(prepared.residual_maps)>1.
    assert np.allclose(_gram(candidate_alms,3),_gram(prepared.family.reference_alms,3),rtol=1e-12,atol=1e-13)
    assert prepared.family.invariants(theta)['passed']
    candidate_pair=prepared.candidate_pair(theta)
    assert candidate_pair.metadata.fingerprint==pair.metadata.fingerprint
    assert np.array_equal(candidate_pair.mask_a,pair.mask_a)
    assert np.array_equal(candidate_pair.mask_b,pair.mask_b)
    assert np.allclose(candidate_pair.a,candidate_maps[0],rtol=0,atol=0)
    # Reanalysis of masked observed pixels is not an inverse. The canonical
    # retained Gram claim is deliberately not extended to this new analysis.
    reanalyzed=np.stack([prepared.operator.adjoint(candidate_pair.operator_maps(split)) for split in ('A','B')])
    assert np.linalg.norm(reanalyzed-candidate_alms)>1e-4
    assert np.linalg.norm(_gram(reanalyzed,3)-_gram(candidate_alms,3))>1e-5
    assert np.array_equal(prepared.candidate_maps(np.zeros(2)),prepared.reference_maps)
    assert source_hashes==[array_sha256(pair.a),array_sha256(pair.b)]
    provenance=prepared.provenance()
    assert provenance['spectral_invariant_scope']=='canonical retained alms only'
    assert provenance['physical_noise_model_validated'] is False
    assert 'not an inverse' in provenance['operator']['adjoint_kind']
    assert provenance['residual_maps_sha256']==array_sha256(prepared.residual_maps)
    json.dumps(provenance,allow_nan=False)


def test_actual_canonicalization_requires_explicit_supported_units_sign_and_order():
    pytest.importorskip('ducc0',reason='actual canonicalization/permutation requires ducc0')
    canonical=map_pair()
    source=canonical.converted(units='K_CMB',polarization_convention='IAU',ordering='NESTED')
    before=source.provenance()
    prepared=prepare(source)
    assert prepared.pair.metadata.units=='uK_CMB'
    assert prepared.pair.metadata.polarization_convention=='COSMO'
    assert prepared.pair.metadata.ordering=='RING'
    assert np.array_equal(prepared.pair.mask_a,canonical.mask_a)
    assert np.allclose(prepared.pair.a,canonical.a,rtol=1e-15,atol=1e-15)
    assert np.allclose(prepared.pair.b,canonical.b,rtol=1e-15,atol=1e-15)
    expected=np.stack([prepared.operator.adjoint(prepared.pair.operator_maps(split)) for split in ('A','B')])
    assert np.array_equal(prepared.family.reference_alms,expected)
    assert before==source.provenance()


def test_actual_dimension_bounds_and_candidate_angle_rejections():
    pytest.importorskip('ducc0',reason='actual rank/dimension control requires ducc0')
    pair=map_pair()
    with pytest.raises(ValueError,match='insufficient.*dimension'):
        prepare(pair,n_planes=4,shells=[2],planes_per_shell=4)
    for shells in ([2,2],[1],[4],[2.5]):
        with pytest.raises(ValueError):
            prepare(pair,n_planes=len(shells),shells=shells)
    prepared=prepare(pair)
    for theta in ([.1],[[.1,.2]],[np.nan,0.],[np.pi/4+1e-8,0.],[.1j,0.],np.array([.1j,0.])):
        with pytest.raises((ValueError,TypeError)):
            prepared.candidate_maps(theta)


def test_optional_healpy_map_adapter_agrees_without_changing_scope():
    pytest.importorskip('ducc0',reason='ducc reference needed for map-adapter comparison')
    pytest.importorskip('healpy',reason='actual optional healpy adapter unavailable')
    pair=map_pair()
    direct=prepare(pair,backend='ducc0')
    reference=prepare(pair,backend='healpy')
    assert reference.operator.provenance()['backend']=='healpy'
    assert np.allclose(direct.family.reference_alms,reference.family.reference_alms,rtol=1e-12,atol=1e-12)
    theta=np.array([.22,-.17])
    assert np.allclose(direct.candidate_maps(theta),reference.candidate_maps(theta),rtol=1e-12,atol=1e-12)
    assert reference.provenance()['physical_noise_model_validated'] is False
