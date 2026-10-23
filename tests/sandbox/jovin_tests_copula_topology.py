import numpy as np
from solution import normal_scores, pd0_summary


def test_strict_monotonic_brightness_transform_cannot_change_result():
    rng = np.random.default_rng(841230)
    field = rng.normal(size=(31, 32))
    assert np.array_equal(normal_scores(field), normal_scores(np.exp(.5*field)))
    assert pd0_summary(field)[0] == pd0_summary(normal_scores(field))[0]


def test_ties_are_preserved_and_output_standardized():
    field = np.array([[0., 1., 2.], [0., 4., 2.], [7., 3., 3.]])
    out = normal_scores(field)
    assert out[0, 0] == out[1, 0]
    assert out[0, 2] == out[1, 2]
    assert abs(out.mean()) < 1e-14
    assert abs(out.std()-1) < 1e-14
    assert np.array_equal(np.argsort(field.ravel(), kind="stable"), np.argsort(out.ravel(), kind="stable"))


def test_constant_or_nonfinite_rejected():
    for field in (np.ones((5, 5)), np.array([[1., 2.], [3., np.inf]])):
        try:
            normal_scores(field)
        except ValueError:
            continue
        raise AssertionError("invalid field accepted")
