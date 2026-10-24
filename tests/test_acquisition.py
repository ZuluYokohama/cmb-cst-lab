import ast
from pathlib import Path
import numpy as np
from cmbcst import acquisition


def test_shared_acquisition_sandbox_checks():
    path = Path(__file__).parent/'sandbox/jovin_tests_acquisition.py'
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if not isinstance(node,ast.ImportFrom)]
    ns = dict(vars(acquisition))
    exec(compile(tree,str(path),'exec'),ns)
    for name,fn in list(ns.items()):
        if name.startswith('test_'):
            fn()


def test_invalid_and_nonfinite_inputs_fail():
    for sky,noise in ((np.zeros((1,3,16,16)),np.zeros((1,8,16,16))),
                      (np.full((1,3,16,16),np.nan),np.zeros((1,8,16,16)))):
        try:
            acquisition.batch_reference(sky,noise)
        except ValueError:
            continue
        raise AssertionError('invalid input accepted')
