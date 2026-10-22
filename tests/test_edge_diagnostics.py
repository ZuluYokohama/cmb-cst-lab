"""Tests for the exploratory covariance-weighted single-edge GLR diagnostic."""
import numpy as np
import pytest
from cmbcst.edge_diagnostics import edge_glrt, null_max_scores, monte_carlo_pvalue


def graph(n=8):
    from itertools import combinations
    edges=list(combinations(range(n),2))
    a=np.zeros((len(edges),n))
    for k,(i,j) in enumerate(edges):
        a[k,i],a[k,j]=1,-1
    return a


def covariance(m,seed=4):
    rng=np.random.default_rng(seed)
    z=rng.normal(size=(m,m))
    return z@z.T/m+np.eye(m)*0.4


def test_node_offsets_invisible():
    a=graph();c=covariance(len(a));rng=np.random.default_rng(5)
    r=edge_glrt(a@rng.normal(size=a.shape[1]),c,a)
    assert np.max(r["delta_chi2"])<1e-20


def test_single_edge_known_bias_exact():
    a=graph();c=covariance(len(a))
    y=np.zeros(len(a));k=11;y[k]=4
    r=edge_glrt(y,c,a)
    assert r["index"]==k
    assert abs(r["amplitude"][k]-4)<1e-12


def test_fixed_edge_null_variance():
    a=graph();c=covariance(len(a));rng=np.random.default_rng(11)
    p=edge_glrt(np.zeros(len(a)),c,a)["precision_residual"]
    noise=rng.normal(size=(5000,len(a)))@np.linalg.cholesky(c).T
    z=(noise@p.T)/np.sqrt(np.diag(p))
    assert np.max(np.abs(np.mean(z*z,axis=0)-1))<0.08


def test_independent_null_fwer():
    a=graph();c=covariance(len(a))
    calibration=null_max_scores(c,a,1200,seed=9)
    validation=null_max_scores(c,a,1200,seed=10)
    threshold=np.quantile(calibration,0.95,method="higher")
    assert 0.025<float(np.mean(validation>threshold))<0.075
    assert monte_carlo_pvalue(float(calibration.max()+1),calibration)==1/(len(calibration)+1)


def test_incompatible_shape_fails():
    a=graph();c=covariance(len(a))
    with pytest.raises(ValueError):
        edge_glrt(np.zeros(27),c,a)
