import ast
from pathlib import Path
import numpy as np
from cmbcst import response, targeted
from cmbcst.targeted_sky import TargetedSky


def test_targeted_shared_sandbox():
    path = Path(__file__).parent/'sandbox/jovin_tests_targeted.py'
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if not isinstance(n,ast.ImportFrom)]
    ns = {**vars(response),**vars(targeted)}
    exec(compile(tree,str(path),'exec'),ns)
    for name,fn in list(ns.items()):
        if name.startswith('test_'):
            fn()


def test_variant_shell_power_and_noise_are_preserved():
    cl = np.zeros(25)
    cl[2:] = 1/np.arange(2,25)**2
    sky = TargetedSky(lmax=24,nside=16,size=32,cl=cl,threads=2)
    for i,(family,variant) in enumerate((('quadratic',-1),('quadratic',1),('compact',.5),('compact',1.),('filament',.5),('filament',1.))):
        fields,p,q = sky.draw_recipe(939100+i,dict(id=str(i),family=family,variant=variant,amplitude=.2))
        assert q['passed'] and fields.shape==(3,4,32,32) and np.isfinite(fields).all()


def test_six_feature_indices_match_view_definitions():
    from cmbcst.multiscale import multiscale_reference
    x = np.random.default_rng(688311).normal(size=(1,3,4,2,32,32))
    _,r = response.extract_response_reference(x)
    geometry = np.empty((3,4,2,9))
    for s in range(3):
        for a in range(4):
            for p in range(2):
                field = x[0,s,a,p]
                geometry[s,a,p] = np.r_[multiscale_reference(field)[1][0],response.excursion_euler_reference(field)[0]]
    expected = [geometry[0,1:,p,7].mean()-geometry[0,0,p,7] for p in range(2)]
    expected += [geometry[2,0,p,j]-geometry[0,0,p,j] for p,j in ((0,1),(0,7),(1,1),(1,7))]
    assert np.allclose(targeted.six_responses(r)[0],expected,atol=1e-13)


def test_seed_roles_are_disjoint_and_lock_detects_changes(tmp_path,monkeypatch):
    import sys
    import json
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    import run_targeted_benchmark as runner
    selected = dict(zip(targeted.FAMILIES,('q00','c00','f00')))
    roles = runner.development_roles()+runner.final_roles(selected)
    seeds = [role['offset']+i for role in roles for i in range(role['count'])]
    assert len(seeds)==6682 and len(set(seeds))==6682
    monkeypatch.setattr(runner,'OUT',tmp_path)
    bindings = {}
    for key,name in (('plan_sha256','plan.json'),('model_sha256','model.json'),('development_sha256','development.json')):
        (tmp_path/name).write_text('{}')
        bindings[key] = runner.sha(tmp_path/name)
    (tmp_path/'selection_lock.json').write_text(json.dumps(dict(**bindings,selected=selected,final_roles=runner.final_roles(selected))))
    runner.check_lock()
    (tmp_path/'model.json').write_text('{"changed":true}')
    try:
        runner.check_lock()
    except ValueError:
        pass
    else:
        raise AssertionError('changed selection binding accepted')
