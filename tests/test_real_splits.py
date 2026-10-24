import ast
from pathlib import Path
import numpy as np
import pytest
from cmbcst import real_splits as r


def test_shared_real_split_checks():
    path = Path(__file__).parent/'sandbox/jovin_tests_real_splits.py'
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if not isinstance(n,ast.ImportFrom)]
    namespace = dict(vars(r))
    exec(compile(tree,str(path),'exec'),namespace)
    for name,fn in list(namespace.items()):
        if name.startswith('test_'):
            fn()


def test_geodesic_pair_separations():
    loc = np.array([[.2,.7],[2.9,5.3]])
    end,_,_ = r.spherical_pairs(loc)
    n = r.local_frame(loc)[0]
    v = r.local_frame(end)[0]
    assert np.allclose(np.sum(v*n,-1),np.cos(np.radians([.5,1,2,4]))[:,None,None],atol=1e-14)


def test_malformed_pairs_rejected():
    with pytest.raises(ValueError):
        r.spherical_pairs([[.2,.4]],[0.])
