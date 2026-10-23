"""AST-extracted numerical tests for independent finite covariance calibration."""
import numpy as np
from solution import graph_design, calibrated_cycle_statistic


def test_cycle_f_scaling():
    rng=np.random.default_rng(152)
    edges=[(0,1),(1,2),(2,3),(3,4),(0,4),(1,4),(0,2)]
    a,z=graph_design(5,edges)
    assert z.shape==(3,7)
    assert np.max(np.abs(z@a))==0
    cal=rng.standard_normal((12,7))
    observed=rng.standard_normal(7)
    out=calibrated_cycle_statistic(observed,cal,5,edges)
    cov=np.cov(cal@z.T,rowvar=False,ddof=1)
    score=(z@observed)@np.linalg.solve(cov,z@observed)
    assert abs(out["chi2_statistic"]-score)<1e-9
    assert abs(out["f_statistic"]-score*(12-3)/(3*(12-1)))<1e-9
    assert (out["f_numerator_dof"],out["f_denominator_dof"])==(3,9)


def test_insufficient_calibration_rejected():
    edges=[(0,1),(1,2),(0,2)]
    try:
        calibrated_cycle_statistic(np.zeros(3),np.zeros((3,3)),3,edges)
    except ValueError:
        pass
    else:
        raise AssertionError("insufficient calibration must be rejected")
