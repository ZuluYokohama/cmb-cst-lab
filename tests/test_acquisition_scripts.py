"""Regression checks for acquisition exports and verification failure paths."""
import ast
from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def load_script(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))

    def load(name):
        path = ROOT/f'scripts/{name}.py'
        spec = importlib.util.spec_from_file_location(name,path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return load


def execute(nodes,namespace):
    """Exercise selected production statements without GPU/science dependencies."""
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<script statements>','exec'),namespace)


def main_body(name):
    tree = ast.parse((ROOT/f'scripts/{name}.py').read_text())
    return next(n.body for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')


@pytest.mark.parametrize('source',['','sample_index,channel\n'])
def test_empty_export_does_not_create_output(load_script,monkeypatch,tmp_path,source):
    module = load_script('export_acquisition_coordinates')
    monkeypatch.setattr(module,'OUT',tmp_path)
    (tmp_path/'labeled_samples.csv').write_text(source)
    with pytest.raises(ValueError,match='no sample rows'):
        module.main()
    assert not (tmp_path/'labeled_samples_corrected.csv').exists()
    assert not (tmp_path/'coordinate_export_audit.json').exists()


def test_export_preserves_existing_output(load_script,monkeypatch,tmp_path):
    module = load_script('export_acquisition_coordinates')
    monkeypatch.setattr(module,'OUT',tmp_path)
    source = ROOT/'results/acquisition_v1/labeled_samples.csv'
    (tmp_path/source.name).write_bytes(source.read_bytes())
    target = tmp_path/'labeled_samples_corrected.csv'
    target.write_text('existing data')
    with pytest.raises(FileExistsError):
        module.main()
    assert target.read_text()=='existing data'


@pytest.fixture
def lean_verifier(load_script,monkeypatch,tmp_path):
    module = load_script('verify_acquisition_lean')
    (tmp_path/'formal').mkdir()
    source = tmp_path/'formal/PolarizationGeometry.lean'
    source.write_bytes((ROOT/'formal/PolarizationGeometry.lean').read_bytes())
    compiler,archive,output = (tmp_path/name for name in ('lean','lean.tar.zst','receipt.json'))
    compiler.write_bytes(b'compiler')
    archive.write_bytes(b'archive')
    pins = {
        b'compiler': module.PINNED_COMPILER_SHA256,
        b'archive': '7289597a44543246c9227b8cae501767e095ce17f6d448b95fc4a8dc97268f6d',
    }
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(module,'hashlib',SimpleNamespace(
        sha256=lambda data:SimpleNamespace(hexdigest=lambda:pins.get(data,hashlib.sha256(data).hexdigest()))))
    monkeypatch.setattr(sys,'argv',['verify_acquisition_lean.py','--lean',str(compiler),
        '--archive',str(archive),'--output',str(output)])
    calls = []

    def run(*args,**kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args[0],0,stdout='standard foundations',stderr='')

    monkeypatch.setattr(module,'subprocess',SimpleNamespace(run=run,check_output=lambda *a,**kw:'Lean 4.35.0'))
    return module,source,compiler,output,calls


def test_lean_rejects_changed_compiler_before_execution(lean_verifier):
    module,source,compiler,output,calls = lean_verifier
    compiler.write_bytes(b'tampered compiler')
    with pytest.raises(ValueError,match='compiler binary mismatch'):
        module.main()
    assert calls==[] and not output.exists()


@pytest.mark.parametrize('declaration',['sorry','admit','axiom extra : True','opaque extra : True'])
def test_lean_rejects_unproved_source(lean_verifier,declaration):
    module,source,compiler,output,calls = lean_verifier
    source.write_text(declaration)
    with pytest.raises(ValueError,match='admission|unproved'):
        module.main()
    assert calls==[] and not output.exists()


@pytest.mark.parametrize('stream',['stdout','stderr'])
def test_lean_rejects_reported_sorry_axiom(lean_verifier,monkeypatch,stream):
    module,source,compiler,output,calls = lean_verifier
    result = subprocess.CompletedProcess([],0,stdout='',stderr='')
    setattr(result,stream,"theorem depends on axioms: [sorryAx]")
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**kw:result)
    with pytest.raises(SystemExit) as failure:
        module.main()
    assert failure.value.code==1
    assert json.loads(output.read_text())['status']=='failed'


def test_lean_valid_proof_and_existing_evidence(lean_verifier):
    module,source,compiler,output,calls = lean_verifier
    module.main()
    proof = json.loads(output.read_text())
    assert proof['status']=='passed' and proof['compiler_sha256']==module.PINNED_COMPILER_SHA256
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        module.main()
    assert output.read_bytes()==original


@pytest.mark.parametrize('summary,expected',[
    ('192 passed, 3 skipped',True),
    ('191 passed, 3 skipped',False),
    ('192 passed, 3 skipped\n1 FAILED',False),
    ('192 passed, 3 skipped, 1 error',False),
    ('ERRORS\n192 passed, 3 skipped',False),
])
def test_evidence_rejects_failed_test_logs(load_script,tmp_path,summary,expected):
    module = load_script('verify_acquisition_evidence')
    tests = tmp_path/'tests.log'
    tests.write_text(summary)
    namespace = dict(vars(module),tests=tests)
    nodes = [n for n in main_body('verify_acquisition_evidence') if isinstance(n,ast.Assign)
             and any(isinstance(t,ast.Name) and t.id in ('tests_text','tests_ok') for t in n.targets)]
    execute(nodes,namespace)
    assert bool(namespace['tests_ok']) is expected


def test_receipts_missing_suite_and_latest_match(load_script,tmp_path):
    module = load_script('verify_acquisition_evidence')
    receipt = dict(suite='present',passed=True,receipt_verified=True,chain_valid=True,
        receipt_id='latest',receipt={'authentic':True},bundle_sha256=hashlib.sha256(b'bundle').hexdigest(),
        tests_sha256=hashlib.sha256(b'tests').hexdigest(),total=4)
    (tmp_path/'tests.py').write_bytes(b'tests')
    calls = []

    def harness(command,env,args):
        calls.append(args)
        assert args[-1]=='latest'
        return 0,receipt['receipt'] if args[1]=='show' else dict(valid=True,chain='linked')

    suites = {}
    namespace = dict(vars(module),ROOT=tmp_path,command=[],env={},suites=suites,
        SUITES={'missing':('', 'tests.py'),'present':('', 'tests.py')},
        history=[dict(receipt,receipt_id='older'),receipt],_harness=harness,build_bundle=lambda name:('bundle',{}))
    loop = next(n for n in main_body('verify_acquisition_evidence') if isinstance(n,ast.For))
    execute([loop],namespace)
    assert suites['missing']==dict(matched=False,receipt_id=None,total=0)
    assert suites['present']==dict(matched=True,receipt_id='latest',total=4)
    assert len(calls)==2
    # Run the actual writer and failure exit with the unmatched suite present.
    body = main_body('verify_acquisition_evidence')
    start = next(i for i,n in enumerate(body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
                 and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='write_text')
    namespace.update(OUT=tmp_path,audit={'status':'failed','jovin_suites':suites},ok=False,checks={},bindings={})
    with pytest.raises(SystemExit) as failure:
        execute(body[start:],namespace)
    assert failure.value.code==1
    assert json.loads((tmp_path/'verification.json').read_text())['jovin_suites']['missing']['matched'] is False


@pytest.mark.parametrize('change',['none','compiler','stdout','source'])
def test_evidence_binds_compiler_and_admissions(load_script,change):
    module = load_script('verify_acquisition_evidence')
    proof = json.loads((ROOT/'data/evidence/acquisition_lean.json').read_text())
    text = (ROOT/'formal/PolarizationGeometry.lean').read_text()
    if change=='compiler':
        proof['compiler_sha256']='0'*64
    elif change=='stdout':
        proof['stdout']+='\n[sorryAx]'
    elif change=='source':
        text+='\naxiom extra : True\n'
    namespace = dict(vars(module),proof=proof,text=text,checks={})
    node = next(n for n in main_body('verify_acquisition_evidence') if isinstance(n,ast.Assign)
                and isinstance(n.targets[0],ast.Subscript) and isinstance(n.targets[0].slice,ast.Constant)
                and n.targets[0].slice.value=='lean_exact_identities')
    execute([node],namespace)
    assert bool(namespace['checks']['lean_exact_identities']) is (change=='none')


def test_telemetry_uses_names_and_elapsed_window(load_script,tmp_path):
    module = load_script('report_acquisition_campaign')
    devices = [dict(name='GPU A',elapsed_seconds=2,batches=[dict(role='production',count=10,gpu_event_ms=1000)]),
               dict(name='GPU B',elapsed_seconds=1,batches=[dict(role='production',count=20,gpu_event_ms=1000)])]
    samples = [dict(seconds=second,devices=[dict(index=index,name=name,utilization=util,power_W=50,
        power_limit_W=100,memory_used_MiB=1024,temperature_C=40)
        for index,name,util in [(1,'GPU A',90),(0,'GPU B',20)]]) for second in (1,2,3)]
    inputs = dict(zip(('plan.json','result.json','stage_audit.json','telemetry.json'),
        ({},{'total_skies':30,'detector_pixel_samples':100},{'elapsed_seconds':2},{'samples':samples})))
    inputs.update({f'device_cuda{i}.json':d for i,d in enumerate(devices)})
    namespace = dict(vars(module),OUT=tmp_path,read=inputs.__getitem__)
    body = main_body('report_acquisition_campaign')
    end = next(i for i,n in enumerate(body) if isinstance(n,ast.Assign)
               and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='all_metrics')
    execute(body[:end],namespace)
    performance = json.loads((tmp_path/'performance.json').read_text())['devices']
    assert [(p['samples'],p['mean_utilization']) for p in performance]==[(2,90),(1,20)]
    verifier = load_script('verify_acquisition_evidence')
    namespace = dict(vars(verifier),records=devices,telemetry=samples,performance=performance,checks={})
    loop = next(n for n in main_body('verify_acquisition_evidence') if isinstance(n,ast.For)
                and isinstance(n.target,ast.Name) and n.target.id=='i')
    execute([loop],namespace)
    assert namespace['checks']=={'telemetry_0_recomputed':True,'telemetry_1_recomputed':True}


@pytest.mark.parametrize('before_range',[False,True])
def test_role_failure_does_not_reuse_previous_seed(load_script,tmp_path,before_range):
    module = load_script('run_acquisition_campaign')
    tree = ast.parse((ROOT/'scripts/run_acquisition_campaign.py').read_text())
    run = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    worker = next(n for n in run.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
    event = SimpleNamespace(record=lambda:None,synchronize=lambda:None,elapsed_time=lambda other:1.)
    torch = SimpleNamespace(cuda=SimpleNamespace(device=lambda i:nullcontext(),reset_peak_memory_stats=lambda i:None,
        Event=lambda **kw:event),Generator=lambda **kw:SimpleNamespace(manual_seed=lambda s:None),
        randn=lambda *a,**kw:None,float64=np.float64)
    ticks = []

    def budget():
        ticks.append(None)
        if len(ticks)==3:
            raise TimeoutError('budget expired at next role')

    ranges = []

    def role_range(*args):
        ranges.append(None)
        if before_range and len(ranges)==2:
            raise TimeoutError('failure before first index')
        return range(*args)

    saved = {}
    namespace = dict(vars(module),torch=torch,OUT=tmp_path,WORK=tmp_path,range=role_range,budget=budget,
        AcquisitionCUDA=lambda *a:SimpleNamespace(run=lambda *a,**kw:dict(mse=np.zeros((256,4,4)),roll_amplitude_error=0)),
        start=module.time.perf_counter(),progress={},progress_lock=nullcontext(),
        save=lambda path,value:saved.update({path.name:value}))
    execute([worker],namespace)
    with pytest.raises(TimeoutError):
        namespace['worker'](dict(index=0,batch=256,production=256))
    failure = saved['failure_cuda0.json']
    assert failure['role']=='intensity_only' and failure['seed'] is None
    assert failure['first_index']==(None if before_range else 0)
