import ast
from pathlib import Path
import numpy as np
import pytest
from cmbcst import multiscale as m


def test_multiscale_shared_jovin_checks():
    path = Path(__file__).parent/'sandbox/jovin_tests_multiscale.py'
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if not isinstance(node, ast.ImportFrom)]
    namespace = dict(vars(m))
    exec(compile(tree, str(path), 'exec'), namespace)
    for name, fn in list(namespace.items()):
        if name.startswith('test_'):
            fn()


def test_actual_cpu_gpu_multiscale_parity_at_sky_patch_size():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('no CUDA device')
    x = np.random.default_rng(714829).normal(size=(6,128,128))
    c,f = m.multiscale_reference(x)
    d,g = m.multiscale_cuda(x)
    assert np.max(np.abs(c-d)/(1+np.abs(c))) < 1e-10
    assert np.max(np.abs(f-g)/(1+np.abs(f))) < 1e-10
