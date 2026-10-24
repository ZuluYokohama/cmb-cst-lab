import ast
from pathlib import Path
import numpy as np
import pytest

from cmbcst import persistence, surrogate_topology, copula_topology, spherical_surrogates, spectra


def test_spherical_reference_suite_shared_with_jovin():
    path = Path(__file__).parent / "sandbox/jovin_tests_spherical_surrogates.py"
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if not isinstance(n, ast.ImportFrom)]
    ns = {**vars(spectra), **vars(persistence), **vars(surrogate_topology),
          **vars(copula_topology), **vars(spherical_surrogates)}
    exec(compile(tree, str(path), "exec"), ns)
    for name, fn in list(ns.items()):
        if name.startswith("test_"):
            fn()


def test_cuda_spherical_projection_matches_cpu_at_actual_bandlimit():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    lmax = 1000
    cl = 1/(np.arange(lmax+1)+1.)**2
    cl[:2] = 0
    normals = np.random.default_rng(620134).normal(size=(2, 2, (lmax+1)*(lmax+2)//2))
    cpu = spherical_surrogates.fixed_power_reference(cl, normals)
    gpu = spherical_surrogates.fixed_power_cuda(cl, normals)
    assert np.allclose(gpu, cpu, rtol=1e-11, atol=1e-12)
    for a in gpu:
        assert np.allclose(spectra.cross_cl(a, a, lmax), cl, rtol=1e-10, atol=1e-16)
