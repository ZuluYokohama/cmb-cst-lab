import ast
from pathlib import Path
import numpy as np
import pytest
from cmbcst import closure_fibers as c
from cmbcst import closure_fibers_data as d,closure_search as s
from cmbcst import closure_certification as cert


def shared_namespace():
    path=Path(__file__).parent/'sandbox/jovin_tests_closure_fibers.py'
    tree=ast.parse(path.read_text())
    tree.body=[n for n in tree.body if not isinstance(n,ast.ImportFrom)]
    namespace={**vars(c),**vars(d),**vars(s),**vars(cert)}
    exec(compile(tree,str(path),'exec'),namespace)
    return namespace


def test_shared_closure_checks():
    for name,fn in shared_namespace().items():
        if name.startswith('test_'):
            fn()


def test_feature_time_derivative_from_independent_exponential_sampling():
    family=shared_namespace()['fixture_family']()
    theta=np.array([.02,-.04,.01,.07])
    expected=c.features_numpy(family,theta)
    # Differentiate by changing sampled values using their independently
    # supplied first derivative. Symmetric differences have O(h²) truncation.
    h=1e-5
    values=[]
    for sign in (-1,1):
        f={key:np.array(value,copy=True) if isinstance(value,np.ndarray) else value for key,value in family.items()}
        f['base'][0]+=sign*h*f['base_derivative'][0]
        decay=np.exp(-sign*h*family['rates'])
        f['cos_basis']=family['cos_basis']*decay[:,None]
        f['sin_basis']=family['sin_basis']*decay[:,None]
        values.append(c.features_numpy(f,theta))
    numerical=(values[1]-values[0])/(2*h)
    assert np.max(np.abs(numerical[np.arange(4)*8+3]-expected[44:48]))<1e-8
    assert np.max(np.abs(numerical[np.arange(4)*8+6]-expected[48:52]))<1e-8


def test_fractional_precomputed_time_rejected():
    with pytest.raises(ValueError):
        c.features_numpy(shared_namespace()['fixture_family'](),np.zeros(4),.5)
