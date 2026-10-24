"""Actual masked spectra of map interventions retain the original residual."""
import numpy as np
import pytest

from cmbcst.integration import SplitPair, ObservingResponse, generate_counterfactual_family
from cmbcst.integration.fixtures import simulated_split_alms
from cmbcst.integration.operators import DuccHealpixOperator
from cmbcst.integration.namaster import CutSkyProfile, NaMasterCutSky, cap_mask


def test_actual_observed_reference_residual_is_in_the_cached_and_direct_models():
    pytest.importorskip('pymaster')
    pytest.importorskip('ducc0')
    op = DuccHealpixOperator(8, 12, nthreads=1)
    maps = np.stack([op.synthesize(a) for a in simulated_split_alms(12, 321)])
    maps += .02*np.sin(np.arange(maps.shape[-1]))[None, None]
    response = ObservingResponse('identity', {})
    pair = SplitPair.create(*maps, nside=8, lmax=12, units='uK_CMB',
        ordering='RING', polarization_convention='COSMO', coordinate_frame='G',
        response_a=response, response_b=response)
    prepared = generate_counterfactual_family(pair, n_planes=4, shells=[6, 9],
                                              planes_per_shell=2, nthreads=1)
    adapter = NaMasterCutSky(CutSkyProfile(8, 12, (2, 6, 10, 13), nthreads=1),
                            cap_mask(8, .6, 20., holes=False))
    model, scales, audit = adapter.build_prepared_model(prepared)
    assert audit['reference_mode'] == 'original_observed_maps_with_unchanged_residual'
    assert audit['zero_model_scaled_error'] <= 1e-10
    theta = np.array([.11, -.14, .18, -.2])
    candidate_maps = prepared.candidate_maps(theta)
    assert np.max(np.abs(candidate_maps-adapter.candidate_maps(prepared.family, theta, prepared.reference_maps))) <= 1e-12
    direct = adapter.direct_features_from_maps(candidate_maps)
    assert np.max(np.abs(direct-model.features(theta))/scales) <= 1e-10
    assert np.max(np.abs((candidate_maps[0]-candidate_maps[1])-(maps[0]-maps[1]))) <= 1e-12
    nonunit = NaMasterCutSky(adapter.profile, adapter.mask, response_window=np.full(13, .9))
    with pytest.raises(ValueError, match='already carried'):
        nonunit.build_model(prepared.family, reference_maps=maps)
    invalid = np.ones(maps.shape[-1], dtype=bool)
    invalid[0] = False
    bad_pair = SplitPair.create(*maps, nside=8, lmax=12, units='uK_CMB',
        ordering='RING', polarization_convention='COSMO', coordinate_frame='G',
        response_a=response, response_b=response, mask_a=invalid, mask_b=invalid)
    bad_prepared = generate_counterfactual_family(bad_pair, n_planes=4, shells=[6, 9],
                                                  planes_per_shell=2, nthreads=1)
    with pytest.raises(ValueError, match='invalid split pixel'):
        adapter.build_prepared_model(bad_prepared)


@pytest.mark.parametrize('bad', [np.array([1+0j], dtype=object), np.array([1+1j]), np.array(['1.0'])])
def test_real_map_boundary_rejects_coercion_before_evaluation(bad):
    from cmbcst.integration.namaster import _real_values
    with pytest.raises(ValueError):
        _real_values(bad, 'maps')
