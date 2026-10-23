"""AST-bundled rank calibration checks."""
import numpy as np
from solution import upper_tail_rank_pvalues


def test_rank_reference_ties():
    x=upper_tail_rank_pvalues(np.array([0.,2.,4.]),np.array([1.,2.,2.,3.]))
    assert np.max(np.abs(x-np.array([1.,.8,.2])))<1e-15


def test_rank_minimum_conservative_fraction():
    x=upper_tail_rank_pvalues(np.array([999.]),np.arange(19.))
    assert x[0]==1/20


def test_reference_bank_required():
    try:
        upper_tail_rank_pvalues(np.array([1.]),np.zeros(0))
    except ValueError:
        pass
    else:
        raise AssertionError("empty reference must be rejected")
