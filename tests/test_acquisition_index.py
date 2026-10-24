import ast
from pathlib import Path
from cmbcst import acquisition,acquisition_index


def test_shared_coordinate_checks():
    path = Path(__file__).parent/'sandbox/jovin_tests_acquisition_index.py'
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if not isinstance(n,ast.ImportFrom)]
    ns = {**vars(acquisition),**vars(acquisition_index)}
    exec(compile(tree,str(path),'exec'),ns)
    for name,fn in list(ns.items()):
        if name.startswith('test_'):
            fn()
