"""Empirical rank calibration tests for independent whole sessions."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from cmbcst.empirical_calibration import upper_tail_rank_pvalues

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("audit_session_rank",ROOT/"scripts"/"audit_session_rank.py")
MOD=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_conservative_tie_handling_and_resolution():
    p=upper_tail_rank_pvalues(np.array([0.,2.,4.]),np.array([1.,2.,2.,3.]))
    np.testing.assert_allclose(p,[1.,.8,.2])
    assert upper_tail_rank_pvalues(np.array([999.]),np.array([1.,2.]))[0] == pytest.approx(1/3)


def test_matched_sessions_rank_calibration_and_wrong_F():
    out=MOD.audit(reference_count=1200,test_count=1000,seed=2026)
    assert .025 < out["matched_session_rank_rejection_fraction"] < .08
    assert out["matched_session_wrong_F_rejection_fraction"] > .17
    assert out["matched_session_wrong_F_rejection_fraction"] > 3*out["matched_session_rank_rejection_fraction"]


def test_shifted_session_distribution_loses_rank_guarantee():
    out=MOD.audit(reference_count=1200,test_count=1000,seed=2026)
    assert out["shifted_session_rank_rejection_fraction"] > .14
    assert out["shifted_session_rank_rejection_fraction"] > 2*out["matched_session_rank_rejection_fraction"]


def test_exact_seed_replay():
    a=MOD.audit(reference_count=100,test_count=100,seed=47)
    b=MOD.audit(reference_count=100,test_count=100,seed=47)
    assert a==b


@pytest.mark.parametrize("test,ref",[
    ([1.],[]),([], [1.]),([[1.]], [1.]),([1.],[[2.]]),
    ([float("nan")],[1.]),([2.],[float("inf")])
])
def test_invalid_rank_inputs_rejected(test,ref):
    with pytest.raises(ValueError):
        upper_tail_rank_pvalues(test,ref)


@pytest.mark.parametrize("kwargs",[
    {"reference_count": 99}, {"test_count": 99}, {"calibration_count": 5},
    {"rho":1.0}, {"shifted_rho":float("nan")}, {"reference_count":True}
])
def test_audit_invalid_inputs_rejected(kwargs):
    with pytest.raises(ValueError):
        MOD.audit(**kwargs)
