"""Pure compact H1 controls plus explicitly recorded actual GPU outputs.

No sky transform or external component-separation dependency runs in JoVin.
The actual NaMaster checks and their provenance belong in the outer preflight.
"""
import hashlib
import numpy as np
import pytest

from solution import RotationFamily, harmonic_layout, SymmetricQuadraticModel
from solution import continue_fiber, direction_manifest
from solution import certify_candidate, evidence_status, harmonic_gram, provenance_reference


# Must be replaced with source-bound actual outputs from BOTH supported GPUs
# before the suite can pass or an authentic receipt can be issued.
GPU_ATTESTATION = {'fixture_id': 'integration_cutsky_v1_quadratic_741', 'source_sha256': {'src/cmbcst/integration/__init__.py': '29ab1978dd10f681796f7cc9959e41db3dd977e76e8ddc17af98e546ec564ed1', 'src/cmbcst/integration/api.py': 'f243f4f39929fa68566d1ac0aa5f75d21facd4e23f6e608e3aa6630e4bdef200', 'src/cmbcst/integration/certificates.py': 'ee858ba8924aae23d82380c826c6ba8b3aa64260d2f611366a136b2751cc7953', 'src/cmbcst/integration/continuation.py': 'accf1582a2053447068f5e66125992ceb2782cbc1b97afc90ae7d73fb164b322', 'src/cmbcst/integration/family.py': '2dae884c35846eadbff6df3069d1271267ad48ed88c6864fe6f0891ddb279957', 'src/cmbcst/integration/fixtures.py': 'bb650ac8ec85c5c27e1ba04509d29a2aaa0fdff4e351cfad5733491d5b5d5c7b', 'src/cmbcst/integration/io.py': '14f0cc52f8a4751449b4fc7f3532f2ba7d690ae2e6acc0bb25db149fa94d1961', 'src/cmbcst/integration/models.py': '80adae38dd873c13ed04971a61ae105eb6470317598f6061073253e7f1d7b598', 'src/cmbcst/integration/namaster.py': 'b028fb32709ea0ae7e67fa3d2294e03a903804ec40e37385eadeb4f5cc8f4cc7', 'src/cmbcst/integration/operators.py': '8779448c1b975c9b99d8a0721f2a930c214a59eea873d860a8b6bb0b23d4233a', 'src/cmbcst/integration/quadratic.py': '2ec4efe119ad80809ddf4508ff55d0e8440958f0ba03d067a39ad1243becf02b', 'scripts/preflight_integration_v1.py': '82f6e387bd51f9ee2f1ce996d4f04acf0561f98b1349e724cadde9e2a565379b'}, 'devices': [{'device': 'cuda:0', 'name': 'NVIDIA GeForce RTX 5070', 'angles': [[0.1, -0.2, 0.25], [-0.12, 0.07, 0.18]], 'features': [[1.0105166354227988, -0.3944496780896968, -1.6111553329865715, 0.27574066040088285, -0.2936589521617599], [0.002778498762900164, -0.4818497284128162, -1.2957617498697376, -0.3146583685558577, -0.47213955291345266]], 'jacobian': [[[0.7254565422139854, -3.5619861558426202, -0.5760403474337611], [2.784552203909329, 2.3584768111156746, 1.176515905761689], [0.15515872425140814, 0.9957436310283582, -0.21907421813463612], [0.5916575921344296, -1.7454646405412158, 0.22825860759852645], [-0.7844521834329293, -1.7263210255680577, -0.31407005969934804]], [[0.9031492406150763, -2.7433562345969467, -0.3290164844630629], [2.157844572054353, 1.9404397368572526, 2.1293917612711146], [-0.04803515260398551, 1.3217087906839102, 0.03347700298121992], [0.9499085264941516, -1.412238954615049, -0.2709876395528843], [-1.0756283613204562, -1.2661357348998528, -0.5456636679821127]]], 'Q_sha256': '2b5f157dc2c6befd22ec0c852e40dce2ff814b075dd860641592824907278ea9', 'errors': {'features': 4.440892098500626e-16, 'jacobian': 4.440892098500626e-16, 'autograd': 1.1102230246251565e-16}}, {'device': 'cuda:1', 'name': 'NVIDIA GeForce GTX 1660 Ti', 'angles': [[0.1, -0.2, 0.25], [-0.12, 0.07, 0.18]], 'features': [[1.010516635422799, -0.39444967808969666, -1.6111553329865722, 0.2757406604008828, -0.2936589521617599], [0.0027784987629001154, -0.48184972841281615, -1.2957617498697376, -0.31465836855585777, -0.4721395529134526]], 'jacobian': [[[0.7254565422139854, -3.5619861558426202, -0.5760403474337612], [2.7845522039093287, 2.3584768111156746, 1.1765159057616887], [0.15515872425140814, 0.9957436310283582, -0.21907421813463612], [0.5916575921344296, -1.7454646405412155, 0.22825860759852634], [-0.7844521834329293, -1.7263210255680577, -0.31407005969934804]], [[0.9031492406150763, -2.7433562345969467, -0.3290164844630629], [2.157844572054353, 1.9404397368572526, 2.1293917612711146], [-0.04803515260398554, 1.32170879068391, 0.03347700298121997], [0.9499085264941516, -1.412238954615049, -0.2709876395528844], [-1.0756283613204562, -1.2661357348998528, -0.545663667982113]]], 'Q_sha256': '2b5f157dc2c6befd22ec0c852e40dce2ff814b075dd860641592824907278ea9', 'errors': {'features': 8.881784197001252e-16, 'jacobian': 4.440892098500626e-16, 'autograd': 4.440892098500626e-16}}]}


def compact_family():
    ell, m = harmonic_layout(6)
    rng = np.random.default_rng(4924)
    value = rng.normal(size=(2, 2, len(ell)))+1j*rng.normal(size=(2, 2, len(ell)))
    value[..., m == 0] = value[..., m == 0].real
    value[..., ell < 2] = 0
    return RotationFamily(value, [6], planes_per_shell=2, seed=17)


def quadratic_fixture():
    rng = np.random.default_rng(741)
    raw = rng.normal(size=(5, 7, 7))
    return dict(Q=(raw+raw.transpose(0, 2, 1))/2,
                angles=np.array([[.1, -.2, .25], [-.12, .07, .18]]))


def five_point(function, theta):
    step = 1e-3
    return np.stack([(-function(theta+2*step*d)+8*function(theta+step*d)
                      -8*function(theta-step*d)+function(theta-2*step*d))/(12*step)
                     for d in np.eye(len(theta))], axis=-1)


def error(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    return float(np.max(np.abs(a-b)/(1+np.abs(b)), initial=0.))


def test_recorded_actual_gpu_native_batch_values_and_jacobians():
    fixture = quadratic_fixture()
    model = SymmetricQuadraticModel(fixture['Q'])
    assert GPU_ATTESTATION['fixture_id'] == 'integration_cutsky_v1_quadratic_741'
    assert GPU_ATTESTATION['source_sha256']
    assert all(isinstance(value, str) and len(value) == 64 and set(value) <= set('0123456789abcdef')
               for value in GPU_ATTESTATION['source_sha256'].values())
    assert len(GPU_ATTESTATION['devices']) == 2
    assert {d['device'] for d in GPU_ATTESTATION['devices']} == {'cuda:0', 'cuda:1'}
    for device in GPU_ATTESTATION['devices']:
        theta = np.asarray(device['angles'])
        assert np.array_equal(theta, fixture['angles'])
        assert device['Q_sha256'] == model.sha256
        assert error(device['features'], model.features(theta)) <= 1e-10
        assert error(device['jacobian'], model.jacobian(theta)) <= 1e-10
        independent = np.stack([five_point(model.features, x) for x in theta])
        assert error(device['jacobian'], independent) <= 1e-10


def test_stored_family_preserves_full_gram_and_empirical_difference():
    family = compact_family()
    theta = np.array([.12, -.18])
    candidate = family.candidate_alms(theta)
    before, after = harmonic_gram(family.reference_alms), harmonic_gram(candidate)
    assert error(after, before) <= 1e-12
    assert error(candidate[0]-candidate[1], family.reference_alms[0]-family.reference_alms[1]) <= 1e-12
    assert family.invariants(theta)['passed']
    arrays = family.to_arrays()
    restored = RotationFamily.from_arrays(arrays, expected_sha256=family.provenance()['array_sha256'])
    assert np.array_equal(restored.candidate_alms(theta), candidate)
    arrays['plane_u'][0, 0] += .01
    with pytest.raises(ValueError):
        RotationFamily.from_arrays(arrays)


def test_curved_fiber_yields_a_finite_candidate_without_self_certification():
    family = compact_family()
    Q = np.zeros((1, 5, 5))
    Q[0, 0, 3] = Q[0, 3, 0] = .5
    Q[0, 4, 4] = 1.
    model = SymmetricQuadraticModel(Q)
    result = continue_fiber(model, [1.], family, direction=[0, 1], global_start_id=9)
    assert result['status'] == 'candidate_found' and result['independent_verification_required']
    assert .01 <= result['rms_ratio'] <= .25
    theta = result['theta']
    record = certify_candidate(family.reference_alms, family.candidate_alms(theta),
        model.features(np.zeros(2)), model.features(theta), [1.], angle_values=theta,
        direct_feature_error=0.)
    assert record['status'] == 'finite_feasible_candidate_verified'
    assert not record['counterexample_verified'] and not record['universal_closure_verified']


def test_gram_preservation_without_the_full_difference_lock_is_rejected():
    family = compact_family()
    reference = family.reference_alms
    # A common longitude phase preserves every full-sky spectral Gram entry,
    # while changing the measured split-difference field.
    candidate = reference*np.exp(.02j*family.m)
    record = certify_candidate(reference, candidate, [1.], [1.], [1.],
                               angle_values=[.1, .1], direct_feature_error=0.)
    assert record['checks']['full_harmonic_gram']
    assert not record['checks']['full_split_difference'] and not record['accepted']


def test_zero_amplitude_missing_independent_evidence_and_drift_cannot_pass():
    family = compact_family()
    reference = family.reference_alms
    zero = certify_candidate(reference, reference, [1.], [1.], [1.],
                              angle_values=[0., 0.], direct_feature_error=0.)
    assert not zero['checks']['nontrivial_harmonic_rms'] and not zero['accepted']
    theta = np.array([.15, -.1])
    candidate = family.candidate_alms(theta)
    missing = certify_candidate(reference, candidate, [1.], [1.], [1.], angle_values=theta)
    assert not missing['checks']['independent_replay_evidence']
    mismatch = certify_candidate(reference, candidate, [1.], [1.+1e-8], [1.],
                                  angle_values=theta, direct_feature_error=0.)
    assert mismatch['checks']['current_recorded_bins_match']
    assert not mismatch['checks']['realized_feature_locks'] and not mismatch['accepted']


def test_full_rank_control_and_budget_failure_remain_inconclusive_or_incomplete():
    family = compact_family()
    Q = np.zeros((2, 5, 5))
    Q[0, 0, 3] = Q[0, 3, 0] = .5
    Q[1, 0, 4] = Q[1, 4, 0] = .5
    model = SymmetricQuadraticModel(Q)
    result = continue_fiber(model, np.ones(2), family, max_iterations=3, max_stalls=2)
    assert result['status'] == 'no_nontrivial_found' and not result['nontrivial']
    state = evidence_status('none_found_in_budget', numerical_status='passed')
    assert state['closure_status'] == 'inconclusive' and not state['universal_closure_verified']
    def stop():
        raise TimeoutError('compact synthetic cap')
    unfinished = continue_fiber(model, np.ones(2), family, budget=stop)
    assert unfinished['status'] == 'incomplete' and not unfinished['feasible']
    with pytest.raises(ValueError):
        evidence_status('none_found_in_budget', execution_status='incomplete')


def test_independent_hash_references_do_not_claim_authentication_or_physics():
    payload = b'actual input bytes'
    digest = hashlib.sha256(payload).hexdigest()
    record = provenance_reference('operator', payload, expected_sha256=digest,
        trust_anchor={'kind': 'receipt', 'identifier': 'externally-verified-anchor'})
    assert record['sha256'] == digest and record['expected_hash_matches']
    assert record['authentication_status'] == 'not_established_by_this_helper'
    assert record['anchor_status'] == 'caller_declared'
    with pytest.raises(ValueError):
        evidence_status('counterexample_verified', numerical_status='failed')


def test_global_start_identity_is_deterministic_and_strict_gate_is_not_relaxed():
    one, meta = direction_manifest(3, 18, 7)
    two, same = direction_manifest(3, 18, 7)
    assert np.array_equal(one, two) and meta == same
    assert not np.array_equal(one, direction_manifest(3, 18, 8)[0])
    family = compact_family()
    with pytest.raises(ValueError):
        continue_fiber(SymmetricQuadraticModel(np.zeros((1, 5, 5))), [1.], family, tolerance=1e-9)
