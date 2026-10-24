"""Spin-2 basis and weighted-adjoint controls, with honest optional skips."""
import importlib
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np
import pytest

from cmbcst.integration.models import MissingOptionalDependency
from cmbcst.integration import operators as op


def coefficient_fixture(lmax=3):
    rng=np.random.default_rng(81025)
    ell,m=op.alm_layout(lmax)
    value=rng.normal(size=(2,len(ell)))+1j*rng.normal(size=(2,len(ell)))
    value[:,m==0]=value[:,m==0].real
    value[:,ell<2]=0
    return value


@pytest.mark.parametrize('nside,lmax,threads',[(0,2,8),(3,2,8),(2,6,8),(2,1,8),(2,3,0),(2,3,9)])
def test_rejected_geometry_and_thread_contract(nside,lmax,threads):
    with pytest.raises(ValueError):
        op.DuccHealpixOperator(nside,lmax,threads)


def test_raw_array_contract_rejects_spin_storage_and_invalid_maps():
    operator=op.DuccHealpixOperator(2,3)
    value=coefficient_fixture();value[0,0]=1.
    with pytest.raises(ValueError,match='ell<spin'):
        operator.synthesize(value)
    value=coefficient_fixture();value[0,2]+=1j
    with pytest.raises(ValueError,match='real m=0'):
        operator.synthesize(value)
    with pytest.raises(ValueError,match='finite'):
        operator.adjoint(np.full((2,48),np.nan))
    with pytest.raises(ValueError,match='shape'):
        operator.adjoint(np.zeros((3,48)))


def test_absent_backends_are_explicit_and_never_replaced(monkeypatch):
    original=importlib.import_module
    def unavailable(name,*args,**kwargs):
        if name in ('ducc0','healpy'):
            raise ImportError('deliberately unavailable')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(op.importlib,'import_module',unavailable)
    for cls in (op.DuccHealpixOperator,op.HealpyHealpixOperator):
        operator=cls(2,3)
        assert operator.provenance()['available'] is False
        with pytest.raises(MissingOptionalDependency,match='no substitute'):
            operator.synthesize(coefficient_fixture())


def test_coefficient_metric_counts_positive_m_twice():
    value=np.zeros((2,10),dtype=np.complex128)
    ell,m=op.alm_layout(3)
    value[0,(ell==2)&(m==0)]=2
    value[1,(ell==3)&(m==2)]=1+2j
    assert op.harmonic_inner_product(value,value,3)==14.


def test_ducc_spin2_area_adjoint_dot_product_and_not_inverse():
    pytest.importorskip('ducc0',reason='actual spin-2 backend requires ducc0')
    operator=op.DuccHealpixOperator(4,8,nthreads=2)
    alms=coefficient_fixture(8)
    maps=np.random.default_rng(191).normal(size=(2,operator.npix))
    left=op.pixel_inner_product(operator.synthesize(alms),maps,operator.nside)
    right=op.harmonic_inner_product(alms,operator.adjoint(maps),operator.lmax)
    assert abs(left-right)/(1+abs(left)+abs(right))<1e-12
    # HEALPix equal-area quadrature is not an exact inverse even for this small
    # band-limited input. This guards against naming the adjoint an inverse.
    assert np.linalg.norm(operator.adjoint(operator.synthesize(alms))-alms)>1e-6
    assert 'not an inverse' in operator.provenance()['adjoint_kind']


def _wigner_d(ell,row,column,theta):
    prefactor=math.sqrt(math.factorial(ell+column)*math.factorial(ell-column)
        *math.factorial(ell+row)*math.factorial(ell-row))
    result=np.zeros_like(theta)
    for k in range(2*ell+1):
        terms=(ell+column-k,k,row-column+k,ell-row-k)
        if min(terms)<0:
            continue
        denominator=math.prod(math.factorial(n) for n in terms)
        result+=((-1)**(row-column+k)*prefactor/denominator
            *np.cos(theta/2)**(2*ell+column-row-2*k)
            *np.sin(theta/2)**(row-column+2*k))
    return result


def test_pure_e_b_and_mixed_modes_match_independent_spin2_formula():
    ducc=pytest.importorskip('ducc0',reason='actual pure-mode control requires ducc0')
    operator=op.DuccHealpixOperator(4,3,nthreads=2)
    locations=ducc.healpix.Healpix_Base(4,'RING').pix2ang(np.arange(operator.npix,dtype=np.int64))
    theta,phi=locations.T
    alms=np.zeros((2,operator.nalm),dtype=np.complex128)
    alms[0,2]=1
    expected=-np.sqrt(15/(32*np.pi))*np.sin(theta)**2
    actual=operator.synthesize(alms)
    assert np.max(np.abs(actual[0]-expected))<1e-12 and np.max(np.abs(actual[1]))<1e-12
    alms[:]=0;alms[1,2]=1
    actual=operator.synthesize(alms)
    assert np.max(np.abs(actual[1]-expected))<1e-12 and np.max(np.abs(actual[0]))<1e-12
    alms=coefficient_fixture(3);ell,orders=op.alm_layout(3)
    polarization=np.zeros(operator.npix,dtype=np.complex128)
    for degree in (2,3):
        for order in range(-degree,degree+1):
            index=np.flatnonzero((ell==degree)&(orders==abs(order)))[0]
            e,b=alms[:,index]
            if order<0:
                e,b=(-1)**order*e.conjugate(),(-1)**order*b.conjugate()
            harmonic=np.sqrt((2*degree+1)/(4*np.pi))*_wigner_d(degree,order,-2,theta)*np.exp(1j*order*phi)
            polarization-=(e+1j*b)*harmonic
    actual=operator.synthesize(alms)
    assert np.max(np.abs(actual[0]+1j*actual[1]-polarization))<1e-11


def test_healpy_is_an_independent_reference_when_installed():
    pytest.importorskip('ducc0',reason='ducc0 needed for cross-backend comparison')
    pytest.importorskip('healpy',reason='independent healpy reference is unavailable')
    direct=op.DuccHealpixOperator(4,3,nthreads=2)
    reference=op.HealpyHealpixOperator(4,3,nthreads=2)
    alms=coefficient_fixture()
    assert np.max(np.abs(direct.synthesize(alms)-reference.synthesize(alms)))<1e-12
    maps=np.random.default_rng(8801).normal(size=(2,direct.npix))
    assert np.max(np.abs(direct.adjoint(maps)-reference.adjoint(maps)))<1e-12


def test_actual_backend_provenance_binds_execution_and_thread_limit_semantics():
    ducc=pytest.importorskip('ducc0',reason='actual ducc provenance requires ducc0')
    healpy=pytest.importorskip('healpy',reason='actual reference provenance requires healpy')
    direct=op.DuccHealpixOperator(4,3,nthreads=2)
    reference=op.HealpyHealpixOperator(4,3,nthreads=2)
    declared=[direct.provenance(),reference.provenance()]
    alms=coefficient_fixture()
    assert np.max(np.abs(direct.synthesize(alms)-reference.synthesize(alms)))<1e-12
    assert declared==[direct.provenance(),reference.provenance()]
    expected_source=hashlib.sha256(Path(op.__file__).read_bytes()).hexdigest()
    for record,backend in zip(declared,(ducc,healpy)):
        assert record['available'] is True and record['version']==backend.__version__
        assert record['source_sha256']==expected_source
        assert json.loads(json.dumps(record,allow_nan=False))==record
        assert record['spin']==2 and record['ordering']=='RING'
        assert record['polarization_convention']=='COSMO'
        assert 'not an inverse' in record['adjoint_kind']
        payload={key:value for key,value in record.items() if key!='operator_sha256'}
        assert record['operator_sha256']==hashlib.sha256(op.canonical_json(payload).encode()).hexdigest()
    assert declared[0]['nthreads']==2 and declared[0]['thread_control']=='explicit nthreads argument'
    assert declared[1]['nthreads'] is None and declared[1]['requested_nthreads']==2
    assert 'caller-controlled OpenMP' in declared[1]['thread_control']
    assert declared[1]['openmp_environment']==os.environ.get('OMP_NUM_THREADS')
    assert declared[0]['operator_sha256']!=op.DuccHealpixOperator(4,2,2).provenance()['operator_sha256']
