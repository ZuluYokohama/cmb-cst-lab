import json
import numpy as np
import pytest
from cmbcst.integration.family import RotationFamily,harmonic_layout
from cmbcst.integration.quadratic import QuadraticModel
from cmbcst.integration.continuation import continue_fiber,direction_manifest


def family():
    ell,m=harmonic_layout(6);rng=np.random.default_rng(4924)
    alms=rng.normal(size=(2,2,len(ell)))+1j*rng.normal(size=(2,2,len(ell)))
    alms[...,m==0]=alms[...,m==0].real;alms[...,ell<2]=0
    return RotationFamily(alms,[6],planes_per_shell=2,seed=17)


def test_finite_nontrivial_continuation_keeps_strict_equalities_and_invariants():
    f=family();Q=np.zeros((1,5,5));Q[0,0,3]=Q[0,3,0]=.5
    # sin(theta0)+sin(theta1)^2=0 is a curved equality fiber.
    Q[0,4,4]=1;q=QuadraticModel(Q)
    result=continue_fiber(q,[1.],f,direction=[0,1],global_start_id=9)
    assert result['status']=='candidate_found' and result['nontrivial'] and result['feasible']
    assert .01<=result['rms_ratio']<=.25
    assert np.max(np.abs(q.features(result['theta'])-q.features(np.zeros(2))))<=1e-10
    assert f.invariants(result['theta'])['passed']
    assert np.max(np.abs(result['theta_path']))<=np.pi/4
    assert np.all(np.diff(result['rms_path'])>0)
    assert result['independent_verification_required']
    assert result['minimum_rms']==.01 and result['tolerance']==1e-10 and result['restore_target']==1e-12
    encoded=json.loads(json.dumps(result,allow_nan=False))
    assert encoded==result and type(encoded['feasible']) is bool
    assert isinstance(encoded['theta'],list) and isinstance(encoded['theta_path'],list)


def test_full_rank_and_singular_isolated_controls_do_not_claim_infeasibility():
    f=family();full=np.zeros((2,5,5));full[0,0,3]=full[0,3,0]=.5;full[1,0,4]=full[1,4,0]=.5
    closed=continue_fiber(QuadraticModel(full),np.ones(2),f,max_stalls=2,max_iterations=3)
    assert closed['status']=='no_nontrivial_found' and not closed['nontrivial']
    assert np.array_equal(closed['theta'],np.zeros(2))
    isolated=np.zeros((1,5,5));isolated[0,3,3]=isolated[0,4,4]=1
    result=continue_fiber(QuadraticModel(isolated),[1.],f,max_stalls=2,max_iterations=6)
    assert result['status']=='no_nontrivial_found' and result['rms_ratio']<.01
    assert result['minimum_rms']==.01 and not result['nontrivial']


def test_direction_identity_and_bounded_budget_and_failure_outcomes():
    a,manifest=direction_manifest(3,18,7);b,same=direction_manifest(3,18,7)
    assert np.array_equal(a,b) and manifest==same
    assert not np.array_equal(a,direction_manifest(3,18,8)[0])
    f=family();q=QuadraticModel(np.zeros((1,5,5)))
    def timeout():raise TimeoutError('synthetic budget')
    incomplete=continue_fiber(q,[1.],f,budget=timeout)
    assert incomplete['status']=='incomplete' and not incomplete['feasible'] and not incomplete['nontrivial']
    assert json.loads(json.dumps(incomplete,allow_nan=False))==incomplete
    class Broken:
        def features(self,theta):return np.array([np.nan])
        def jacobian(self,theta):return np.zeros((1,2))
    failed=continue_fiber(Broken(),[1.],f)
    assert failed['status']=='numerical_failure' and failed['failure_artifacts'] and not failed['feasible']
    assert json.loads(json.dumps(failed,allow_nan=False))==failed
    assert json.loads(json.dumps(manifest,allow_nan=False))==manifest
    with pytest.raises(ValueError):continue_fiber(q,[1.],f,max_iterations=101)
    with pytest.raises(ValueError):continue_fiber(q,[1.],f,tolerance=1e-9)
    with pytest.raises(ValueError):continue_fiber(q,[1.],f,minimum_rms=0)


def test_complex_and_boolean_parameters_fail_before_model_evaluation():
    class MustNotEvaluate:
        def features(self,theta):pytest.fail('invalid parameters must fail before evaluating any candidate')
        def jacobian(self,theta):pytest.fail('invalid parameters must fail before Jacobian evaluation')
    f=family();model=MustNotEvaluate()
    for direction in (np.array([.1+.2j,1.]),np.array([True,False]),np.array([np.complex128(.1+.2j),1.],dtype=object)):
        with pytest.raises(ValueError,match='real and non-boolean'):
            continue_fiber(model,[1.],f,direction=direction)
    for scales in (np.array([1.+.2j]),np.array([True]),np.array([np.complex128(1.+.2j)],dtype=object)):
        with pytest.raises(ValueError,match='real and non-boolean'):
            continue_fiber(model,scales,f,direction=[0.,1.])
