"""Optional actual NaMaster/ducc checks on small deterministic spin-2 skies."""
import numpy as np
import pytest

from cmbcst.integration.family import RotationFamily, harmonic_layout
from cmbcst.integration.namaster import CutSkyProfile, NaMasterCutSky, cap_mask


def reference(lmax=12):
    ell, m = harmonic_layout(lmax)
    generator = np.random.default_rng(191178)
    value = (generator.normal(size=(2, 2, len(ell)))
             + 1j*generator.normal(size=(2, 2, len(ell))))/(ell+1)
    value[..., m == 0] = value[..., m == 0].real
    value[..., ell < 2] = 0
    return value


@pytest.fixture(scope='module')
def actual():
    pytest.importorskip('pymaster', reason='optional actual NaMaster adapter is unavailable')
    pytest.importorskip('ducc0', reason='optional actual ducc spin-2 backend is unavailable')
    family = RotationFamily(reference(), [6, 9], planes_per_shell=2, seed=917)
    profile = CutSkyProfile(8, 12, (2, 5, 9, 13), purify_b=True, nthreads=1)
    mask = cap_mask(8, .6, 20., holes=False)
    ell = np.arange(13)
    response = np.exp(-.5*ell*(ell+1)*.03**2)
    adapter = NaMasterCutSky(profile, mask, response_window=response)
    model, scales, audit = adapter.build_model(family)
    return family, adapter, model, scales, audit


def test_actual_coupled_quadratic_values_match_reconstructed_namaster_fields(actual):
    family, adapter, model, scales, audit = actual
    assert audit['zero_model_scaled_error'] <= 1e-10
    assert adapter.profile.manifest()['representation'] == 'coupled_binned_pseudo_cl'
    angles = np.array([[.12, -.17, .07, .2], [-.1, .16, .21, -.13]])
    expected = np.stack([adapter.direct_features(family.candidate_alms(theta)) for theta in angles])
    assert np.max(np.abs(model.features(angles)-expected)/scales) <= 1e-10
    candidate = family.candidate_alms(angles[0])
    maps = np.stack([adapter.observe(split) for split in candidate])
    assert np.max(np.abs(adapter.direct_features_from_maps(maps)-expected[0])/scales) <= 1e-10
    assert len(adapter.labels) == 3*4*3 == model.nfeatures


def test_actual_namaster_finite_differences_check_cached_analytic_jacobian(actual):
    family, adapter, model, scales, _ = actual
    theta = np.array([.12, -.17, .07, .2])
    step = 1e-3
    columns = []
    for direction in np.eye(family.nplanes):
        direct = lambda x: adapter.direct_features(family.candidate_alms(x))
        columns.append((-direct(theta+2*step*direction)+8*direct(theta+step*direction)
                        -8*direct(theta-step*direction)+direct(theta-2*step*direction))/(12*step))
    # Finite differences have their own truncation/cancellation error budget;
    # this does not relax the 1e-10 candidate feature-lock or replay gate.
    assert np.max(np.abs(model.jacobian(theta)-np.stack(columns, axis=-1))/scales[:, None]) <= 1e-7


def test_full_sky_rotation_preservation_does_not_imply_realized_cut_sky_locks(actual):
    family, adapter, _, scales, _ = actual
    theta = np.array([.25, -.2, .15, -.18])
    assert family.invariants(theta)['passed']
    before = adapter.direct_features(family.reference_alms)
    after = adapter.direct_features(family.candidate_alms(theta))
    assert np.max(np.abs(after-before)/scales) > 1e-6


def test_actual_zero_field_and_invalid_scalar_mask_or_map_inputs(actual):
    family, adapter, _, _, _ = actual
    assert np.array_equal(adapter.direct_features(np.zeros_like(family.reference_alms)), np.zeros(len(adapter.labels)))
    for mask in (np.zeros_like(adapter.mask), np.full_like(adapter.mask, np.nan),
                 np.full_like(adapter.mask, -1.), np.ones((2, len(adapter.mask)))):
        with pytest.raises(ValueError):
            NaMasterCutSky(adapter.profile, mask)
    with pytest.raises(ValueError):
        adapter.direct_features_from_maps(np.zeros((2, len(adapter.mask))))
    with pytest.raises(ValueError):
        NaMasterCutSky(adapter.profile, adapter.mask, response_window=np.ones(12))
    with pytest.raises(ValueError):
        NaMasterCutSky(adapter.profile, adapter.mask, response_window=-np.ones(13))


@pytest.mark.parametrize('kwargs', [dict(nside=3), dict(lmax=24), dict(band_edges=(2, 5, 5, 13)),
                                    dict(band_edges=(1, 5, 13)), dict(purify_b=1), dict(nthreads=9),
                                    dict(mask_lmax=23, purify_b=True)])
def test_invalid_profiles_rejected_without_optional_runtime(kwargs):
    options = dict(nside=8, lmax=12, band_edges=(2, 5, 9, 13))
    options.update(kwargs)
    with pytest.raises(ValueError):
        CutSkyProfile(**options)
