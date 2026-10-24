import numpy as np
from solution import d4_view, directional_sign, multiscale_reference, pd0_summary, normal_scores


def test_permutation_identity_and_group_relations():
    x = np.arange(32*33, dtype=float).reshape(32,33)
    assert np.array_equal(d4_view(x),x)
    assert np.array_equal(d4_view(d4_view(x,1),3),x)
    assert np.array_equal(d4_view(d4_view(x,0,True),0,True),x)
    assert np.array_equal(d4_view(d4_view(x,1),0,True),d4_view(d4_view(x,0,True),3))
    for k in range(4):
        for reflect in (False,True):
            y = d4_view(x,k,reflect)
            assert y.flags.c_contiguous
            assert np.array_equal(np.sort(y.ravel()),np.sort(x.ravel()))


def test_all_eight_symmetries_have_expected_feature_covariance():
    row,col = np.mgrid[:32,:33]
    x = row+2*col+np.random.default_rng(716289).normal(size=row.shape)
    curves,features = multiscale_reference(x)
    for k in range(4):
        for reflect in (False,True):
            c,f = multiscale_reference(d4_view(x,k,reflect))
            expected_c,expected_f = curves.copy(),features.copy()
            expected_c[:,:,5] *= directional_sign(k)
            expected_f[:,5] *= directional_sign(k)
            assert np.allclose(c,expected_c,rtol=1e-12,atol=1e-12)
            assert np.allclose(f,expected_f,rtol=1e-12,atol=1e-12)


def test_rank_and_topology_on_plateaus_commute_with_grid_symmetries():
    x = np.random.default_rng(613245).integers(-3,4,size=(10,11)).astype(float)
    expected = pd0_summary(x)
    ranks = normal_scores(x)
    for k in range(4):
        for reflect in (False,True):
            y = d4_view(x,k,reflect)
            assert np.allclose(normal_scores(y),d4_view(ranks,k,reflect),atol=1e-14)
            assert np.allclose(pd0_summary(y),expected,rtol=1e-13,atol=1e-13)


def test_invalid_transform_parameters_rejected():
    for k,reflection in ((-1,False),(4,False),(True,False),(1,'yes')):
        try:
            d4_view(np.arange(9.).reshape(3,3),k,reflection)
        except ValueError:
            continue
        raise AssertionError('invalid transform accepted')
