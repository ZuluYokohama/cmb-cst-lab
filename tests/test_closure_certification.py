import numpy as np
from cmbcst.closure_certification import bin_separation,scaled_error


def test_unsafe_zero_bin_does_not_receive_repair_credit():
    reference=np.zeros(52);scales=np.ones(52);value=reference.copy()
    value[48]=.00049995
    assert not bin_separation(value,reference,scales,'repair')['bins_differ']
    assert not bin_separation(value,reference,scales,'repair')['robust_changed_coordinate']
    value[48]=.00050005
    assert bin_separation(value,reference,scales,'repair')['bins_differ']
    assert not bin_separation(value,reference,scales,'repair')['robust_changed_coordinate']
    value[48]=.001
    assert bin_separation(value,reference,scales,'repair')['robust_changed_coordinate']
    assert not bin_separation(value,reference,scales,'baseline')['bins_differ']


def test_numerical_gate_rejects_shape_and_nonfinite_values():
    assert np.isinf(scaled_error([np.nan],[0]))
    assert np.isinf(scaled_error([1,2],[1]))
    assert scaled_error([1.],[1.])==0
