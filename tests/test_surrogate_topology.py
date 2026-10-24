import ast
from pathlib import Path

import numpy as np
import pytest

from cmbcst import persistence, surrogate_topology as s


def test_same_reference_checks_as_jovin():
    path = Path(__file__).parent / "sandbox/jovin_tests_surrogate_topology.py"
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if not isinstance(n, ast.ImportFrom)]
    ns = {**vars(persistence), **vars(s)}
    exec(compile(tree, str(path), "exec"), ns)
    for name, fn in list(ns.items()):
        if name.startswith("test_"):
            fn()


def test_cuda_projection_rank_matches_cpu_reference():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    rng = np.random.default_rng(9231)
    target = rng.normal(size=(16, 16))
    initial = rng.permutation(target.ravel()).reshape(target.shape)
    cpu_s, cpu_h = s.iaaft_reference(target, initial, 1)
    gpu_s, gpu_h = s.iaaft_gpu(target, initial[None], 1)
    assert np.allclose(cpu_s, gpu_s[0], rtol=1e-11, atol=1e-11)
    assert np.array_equal(cpu_h, gpu_h[0])
    assert s.surrogate_quality(gpu_s[0], gpu_h[0], target)["spectrum_stage_relative_error"] < 1e-12
