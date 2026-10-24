import numpy as np
from solution import look_coordinates,analytic_phase_error,instrument_design,detector_table


def test_actual_fourier_response_uses_negative_sample_offset():
    error,opposite = analytic_phase_error(instrument_design(32)[0],detector_table())
    assert error<1e-12 and opposite>.01


def test_look_index_wrap_and_zero_offset():
    unwrapped,wrapped = look_coordinates(0,0,.75,0.)
    assert np.allclose(unwrapped,[-10.029296875,-10.])
    assert np.allclose(wrapped,[9.970703125,-10.])
    assert np.array_equal(look_coordinates(256,256,0.,0.)[0],[0.,0.])


def test_invalid_coordinate_rejected():
    try:
        look_coordinates(np.nan,0,0,0)
    except ValueError:
        pass
    else:
        raise AssertionError('nonfinite geometry accepted')
