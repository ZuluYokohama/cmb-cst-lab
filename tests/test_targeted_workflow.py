"""Targeted orchestration checks without CUDA or receipt-harness dependencies."""
import ast
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('preparation,initialization', [(1800.,0.),(5.,1795.),(0.,1800.)])
def test_preparation_counts_toward_budget_and_elapsed_time(tmp_path,monkeypatch,preparation,initialization):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    import run_targeted_benchmark as runner

    clock = [100.]
    saved = {}

    def freeze():
        clock[0] += preparation
        return dict(development_roles=[dict(name='train_null',count=1,offset=0,recipe=None)])

    def sky():
        clock[0] += initialization
        return SimpleNamespace(cl=np.zeros(2),draw_recipe=lambda *args:pytest.fail('draw after budget exhaustion'))

    monkeypatch.setattr(runner,'freeze',freeze)
    monkeypatch.setattr(runner,'time',SimpleNamespace(perf_counter=lambda:clock[0]))
    monkeypatch.setattr(runner,'TargetedSky',sky)
    monkeypatch.setattr(runner,'OUT',tmp_path)
    monkeypatch.setattr(runner,'WORK',tmp_path)
    monkeypatch.setattr(runner,'sha',lambda path:'test digest')
    monkeypatch.setattr(runner,'save',lambda path,record:saved.update({path.name:dict(record)}))

    with pytest.raises(TimeoutError,match='budget exhausted'):
        runner.run()

    audit = saved['stage_audit.json']
    assert audit['status']=='incomplete'
    assert audit['elapsed_seconds']==1800.
    assert audit['draws']==0
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('summary,expected', [
    ('189 passed, 3 skipped in 1.0s',True),
    ('189 passed, 3 skipped, 2 warnings in 1.0s',True),
    ('1 failed, 189 passed, 3 skipped in 1.0s',False),
    ('189 passed, 3 skipped, 1 error in 1.0s',False),
    ('189 passed, 3 skipped, 2 errors in 1.0s',False),
    ('189 passed, 3 skipped\nERROR during collection',False),
    ('FAILED test_example\n189 passed, 3 skipped',False),
    ('188 passed, 3 skipped in 1.0s',False),
    ('189 passed, 2 skipped in 1.0s',False),
    ('1189 passed, 3 skipped in 1.0s',False),
    ('189 passed, 3 skipped0 in 1.0s',False),
    ('',False),
])
@pytest.mark.parametrize('sandbox,expected_sandbox', [
    ('[PASS]\n'*21,True),
    ('[PASS]\n'*20,False),
    ('[PASS]\n'*21+'[FAIL]\n',False),
],ids=['all_pass','missing_pass','explicit_failure'])
def test_local_summary_and_independent_sandbox_gate(tmp_path,summary,expected,sandbox,expected_sandbox):
    data = tmp_path/'data'
    data.mkdir()
    (data/'targeted_tests.log').write_text(summary)
    (data/'targeted_sandbox_tests.log').write_text(sandbox)
    source = ROOT/'scripts/verify_targeted_evidence.py'
    tree = ast.parse(source.read_text())
    main = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='main')
    start = next(i for i,node in enumerate(main.body) if isinstance(node,ast.Assign)
        and isinstance(node.targets[0],ast.Tuple) and [target.id for target in node.targets[0].elts]==['tests','sandbox'])
    end = next(i for i,node in enumerate(main.body) if isinstance(node,ast.Assign)
        and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='ok')
    # Execute the shipped log gates, isolated from the full scientific reconstruction.
    gates = ast.Module(body=main.body[start:end],type_ignores=[])
    namespace = {'REPO':tmp_path,'re':re}
    exec(compile(gates,str(source),'exec'),namespace)
    assert namespace['tests_ok'] is expected
    assert namespace['sandbox_ok'] is expected_sandbox
