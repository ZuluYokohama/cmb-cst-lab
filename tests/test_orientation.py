import ast
from pathlib import Path
from cmbcst import orientation, persistence, surrogate_topology, copula_topology, multiscale


def test_orientation_shared_jovin_checks():
    path = Path(__file__).parent/'sandbox/jovin_tests_orientation.py'
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if not isinstance(node,ast.ImportFrom)]
    namespace = {**vars(persistence),**vars(surrogate_topology),**vars(copula_topology),
                 **vars(multiscale),**vars(orientation)}
    exec(compile(tree,str(path),'exec'),namespace)
    for name,fn in list(namespace.items()):
        if name.startswith('test_'):
            fn()
