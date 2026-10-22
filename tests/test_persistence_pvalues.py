"""Single-statistic Monte Carlo p-value corrections (not multiplicity correction)."""
import numpy as np
import pytest
from cmbcst.persistence import empirical_two_sided_mc_pvalue


def test_never_zero_with_finite_null():
    vals=np.arange(200,dtype=float)
    assert empirical_two_sided_mc_pvalue(-1.0,vals)==2/201
    assert empirical_two_sided_mc_pvalue(201.0,vals)==2/201


def test_midrange_is_bounded():
    vals=np.arange(200,dtype=float)
    p=empirical_two_sided_mc_pvalue(99.5,vals)
    assert 0.0<p<=1.0
    assert p==1.0


def test_input_validation():
    with pytest.raises(ValueError): empirical_two_sided_mc_pvalue(1.0,np.array([]))
    with pytest.raises(ValueError): empirical_two_sided_mc_pvalue(float("nan"),np.arange(10))
    with pytest.raises(ValueError): empirical_two_sided_mc_pvalue(1,np.array([1.0,float("nan")]))
