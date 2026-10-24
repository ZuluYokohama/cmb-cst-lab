import numpy as np
import pytest
from cmbcst.closure_search import initial_kicks,tangent_reference,matching_bins,manifest


def test_fixed_paired_start_manifest():
    a,s=initial_kicks(24,'fresh')
    b,t=initial_kicks(24,'fresh')
    assert np.array_equal(a,b) and np.array_equal(s,t)
    assert np.array_equal(a[[0,64]],np.zeros((2,128)))
    assert np.allclose(np.linalg.norm(a[[1,65]],axis=1),.1)
    assert not np.array_equal(a,initial_kicks(24,'source')[0])
    assert manifest(34,'source','baseline')['device']==1
    assert manifest(24,'fresh','repair')['device']==0


def test_redundant_tangent_constraints_and_retraction():
    j=np.array([[1.,0.,0.],[2.,0.,0.]])
    projected,inverse,rank=tangent_reference(j,np.array([1.,2.,3.]))
    assert rank==1 and np.allclose(projected,[0,2,3])
    assert np.allclose(j@inverse@j,j)


def test_reference_centered_quantization_and_margin():
    ref=np.array([2.,-1.]);scale=np.array([2.,.1])
    good=matching_bins(ref+np.array([.0002,.00002]),ref,scale,[0,1])
    assert good['matched']
    assert not matching_bins(ref+np.array([.002,0]),ref,scale,[0,1])['matched']
    assert not matching_bins(ref+np.array([.001-1e-9,0]),ref,scale,[0,1])['matched']
    with pytest.raises(ValueError):
        matching_bins([np.nan,1],ref,scale,[0,1])
