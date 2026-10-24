"""Compact actual-device fixtures and hostile solver controls before freeze."""
import ast
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cmbcst.closure_fibers import ClosureCUDA,features_numpy,ratios_numpy,moments_with_derivatives
from cmbcst.closure_search import projected_batch
from cmbcst.closure_jacobian import feature_jacobian,ClosureTargetCUDA


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'data/evidence/closure_gpu_preflight.json')
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve earlier preflight output; select a new version')
    test=ROOT/'tests/sandbox/jovin_tests_closure_fibers.py'
    source=test.read_text()
    tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fixture_family')
    namespace={'np':np}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(test),'exec'),namespace)
    family=namespace['fixture_family']()
    theta=np.array([.02,-.04,.01,.07])
    expected=features_numpy(family,theta)
    h=1e-3
    finite=np.column_stack([(-features_numpy(family,theta+2*np.eye(4)[j]*h)+8*features_numpy(family,theta+np.eye(4)[j]*h)
        -8*features_numpy(family,theta-np.eye(4)[j]*h)+features_numpy(family,theta-2*np.eye(4)[j]*h))/(12*h) for j in range(4)])
    rng=np.random.default_rng(371063)
    a,b,da,db=rng.normal(size=(4,64))+1j*rng.normal(size=(4,64))
    _,rates=moments_with_derivatives(a,b,da,db)
    fixture=dict(seed=929173,devices=[])
    for i in (0,1):
        device=f'cuda:{i}'
        with torch.cuda.device(i):
            model=ClosureCUDA(family,device)
            x=torch.as_tensor(theta,device=device)
            actual=model.features(x).detach().cpu().numpy()
            ratios=model.ratios(x).detach().cpu().numpy()
            automatic_jac=torch.func.jacrev(model.features)(x).detach().cpu().numpy()
            jac=feature_jacobian(model,x).detach().cpu().numpy()
            optimized_target=ClosureTargetCUDA(model,1,2)
            target_value=float(optimized_target.value(x).item())
            target_gradient=optimized_target.gradient(x).detach().cpu().numpy()
            reference_target=float(features_numpy(family,theta,1)[22])
            automatic_target=torch.func.grad(lambda angle:model.features(angle,1)[22])(x).detach().cpu().numpy()
            tx=[torch.as_tensor(v,device=device) for v in (a,b,da,db)]
            def moment(u,index):
                curve,mean_rate,k_rate=model._moments(tx[0]+u*tx[2],tx[1]+u*tx[3],tx[2],tx[3])
                return curve[index]
            automatic=np.array([torch.func.grad(lambda u:moment(u,j))(torch.tensor(0.,dtype=torch.float64,device=device)).item() for j in (3,6)])
            errors=dict(features=float(np.max(np.abs(actual-expected)/(1+np.abs(expected)))),
                ratios=float(np.max(np.abs(ratios-ratios_numpy(family,theta)))) ,
                jacobian_five_point=float(np.max(np.abs(jac-finite)/(1+np.abs(finite)))),
                optimized_jacobian_autograd=float(np.max(np.abs(jac-automatic_jac)/(1+np.abs(automatic_jac)))),
                optimized_target_value=abs(target_value-reference_target)/(1+abs(reference_target)),
                optimized_target_gradient=float(np.max(np.abs(target_gradient-automatic_target)/(1+np.abs(automatic_target)))),
                analytic_response=float(np.max(np.abs(automatic-rates[[3,6]])/(1+np.abs(rates[[3,6]])))))
            if max(errors.values())>1e-10:
                raise ValueError(f'preflight precision failure: {errors}')
            # R(x,y)=x, observed next coordinate y: exact paper counterexample.
            feature=lambda point:point[:1]
            target=lambda point:point[1]
            ratio=lambda point:torch.stack([point[0]*0,point[0]*0])
            kicks=np.array([[0.,0.],[.04,-.02]])
            signs=np.array([1.,-1.])
            nonclosed=projected_batch(feature,target,ratio,kicks,signs,device=device,max_iterations=12,max_stalls=5)
            identity=projected_batch(feature,lambda p:p[0],ratio,kicks,signs,device=device,max_iterations=6,max_stalls=3)
            if not (np.max(np.abs(nonclosed['theta'][:,0]))<.0005 and np.min(np.abs(nonclosed['theta'][:,1]))>=.05 and np.max(np.abs(identity['gap']))<.001):
                raise ValueError('projected optimizer missed mandatory closure controls')
            fixture['devices'].append(dict(device=device,name=torch.cuda.get_device_name(i),
                features=actual.tolist(),ratios=ratios.tolist(),jacobian=jac.tolist(),errors=errors,
                target_value=target_value,target_gradient=target_gradient.tolist(),
                nonclosed_control_theta=nonclosed['theta'][0].tolist(),identity_control_gap=identity['gap'].tolist(),
                solver_control_seconds=nonclosed['seconds']+identity['seconds']))
    assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='GPU_ATTESTATION' for t in n.targets))
    lines=source.splitlines(keepends=True)
    replacement=''.join(lines[:assignment.lineno-1])+'GPU_ATTESTATION = '+repr(fixture)+'\n'+''.join(lines[assignment.end_lineno:])
    if len(replacement.encode())>200*1024:
        raise ValueError('keep actual-device fixture within unchanged JoVin input limit')
    if ast.literal_eval(assignment.value) is not None:
        import gzip
        archive=args.output.with_suffix('.previous_test.py.gz')
        archive.write_bytes(gzip.compress(source.encode(),mtime=0))
    test.write_text(replacement,encoding='utf-8')
    paths=('src/cmbcst/closure_fibers.py','src/cmbcst/closure_fibers_data.py','src/cmbcst/closure_search.py',
           'src/cmbcst/closure_jacobian.py','src/cmbcst/closure_fibers_oracle.py','scripts/prepare_closure_fibers.py')
    record=dict(status='passed',fixture=fixture,python=sys.version,numpy=np.__version__,torch=str(torch.__version__),cuda=torch.version.cuda,
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in paths})
    with args.output.open('x') as stream:
        json.dump(record,stream,indent=2)
    print(json.dumps(dict(status='passed',devices=[r['name'] for r in fixture['devices']],errors=[r['errors'] for r in fixture['devices']]),indent=2))


if __name__=='__main__':
    main()
