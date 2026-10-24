"""Fixed-fixture engineering preflight; never runs scientific optimization.

Outputs a compact actual-device numerical attestation and bounded native-batch
time/memory measurements on both GPUs. Existing evidence is never overwritten.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cmbcst.closure_fibers import ClosureCUDA,features_numpy,ratios_numpy,projection_indices
from cmbcst.closure_gpu_v2 import ClosureBatchCUDA,RMS_RATIO_NAMES
from cmbcst.closure_jacobian import feature_jacobian
from cmbcst.closure_search_v2 import active_tangent,projected_batch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture_builder():
    test = ROOT/'tests/test_closure_gpu_v2.py'
    node = next(n for n in ast.parse(test.read_text()).body
                if isinstance(n,ast.FunctionDef) and n.name=='fixture_family')
    namespace = {'np':np}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(test),'exec'),namespace)
    return namespace['fixture_family']


def scaled_error(actual,expected):
    return float(np.max(np.abs(np.asarray(actual)-expected)/(1+np.abs(expected)),initial=0.))


def benchmark(function,device,repeats):
    """Synchronized end-to-end wall timing, including Python/kernel dispatch."""
    with torch.no_grad():
        function()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        samples = []
        for _ in range(repeats):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            event_start,event_end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            event_start.record()
            output = function()
            event_end.record()
            torch.cuda.synchronize(device)
            samples.append(time.perf_counter()-started)
            del output
        return dict(wall_seconds=samples,median_seconds=float(np.median(samples)),
            final_cuda_event_milliseconds=event_start.elapsed_time(event_end),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device))


def compact_checks(family,device):
    model = ClosureBatchCUDA(family,device)
    legacy = ClosureCUDA(family,device)
    angles = np.array([[.02,-.04,.01,.07],[-.03,.05,.06,-.02],[0.,0.,0.,0.]])
    x = torch.as_tensor(angles,device=device,dtype=torch.float64)
    features,jac = model.features_and_jacobian(x)
    actual = features.detach().cpu().numpy()
    analytic = jac.detach().cpu().numpy()
    automatic = torch.func.jacrev(legacy.features)(x[0]).detach().cpu().numpy()
    h = 1e-3
    finite = np.column_stack([(-features_numpy(family,angles[0]+2*np.eye(4)[p]*h)
        +8*features_numpy(family,angles[0]+np.eye(4)[p]*h)
        -8*features_numpy(family,angles[0]-np.eye(4)[p]*h)
        +features_numpy(family,angles[0]-2*np.eye(4)[p]*h))/(12*h) for p in range(4)])
    target,target_gradient = model.target_and_gradient(x)
    expected_target_gradient = torch.func.grad(lambda row:legacy.features(row,1)[22])(x[0])
    squared,rms_gradient = model.ratio_squared_and_gradient(x)
    expected_rms_gradient = torch.func.jacrev(lambda row:legacy.ratios(row).square())(x[0])
    errors = dict(features=scaled_error(actual,features_numpy(family,angles)),
        jacobian_autograd=scaled_error(analytic[0],automatic),
        jacobian_five_point=scaled_error(analytic[0],finite),
        target=scaled_error(target.detach().cpu().numpy(),features_numpy(family,angles,1)[:,22]),
        target_gradient=scaled_error(target_gradient[0].detach().cpu().numpy(),expected_target_gradient.detach().cpu().numpy()),
        ratio_squared=scaled_error(squared.detach().cpu().numpy(),ratios_numpy(family,angles)**2),
        rms_gradient=scaled_error(rms_gradient[0].detach().cpu().numpy(),expected_rms_gradient.detach().cpu().numpy()))
    for u in (1,2,3):
        errors[f'features_u{u}'] = scaled_error(model.features(x,u).detach().cpu().numpy(),features_numpy(family,angles,u))
    if not bool(torch.isfinite(rms_gradient).all()) or bool(torch.count_nonzero(rms_gradient[-1])):
        raise ValueError('zero-angle squared-RMS derivative must be finite and exactly zero')
    if max(errors.values())>1e-10:
        raise ValueError(f'numerical preflight failed: {errors}')
    return dict(angles=angles[0].tolist(),features=actual[0].tolist(),jacobian=analytic[0].tolist(),
        target=float(target[0]),target_gradient=target_gradient[0].detach().cpu().tolist(),
        ratio_squared=squared[0].detach().cpu().tolist(),
        rms_gradient=rms_gradient[0].detach().cpu().tolist(),errors=errors)


def svd_checks(j,gradient):
    """Check exact CUDA backends against an independently factored CPU batch."""
    cpu = j.cpu()
    cpu_u,cpu_s,cpu_v = torch.linalg.svd(cpu,full_matrices=False)
    cpu_used = cpu_s>cpu_s[:,:1]*1e-10
    cpu_g = gradient.cpu()
    cpu_projection = cpu_g-torch.einsum('bkp,bk->bp',cpu_v,
        torch.einsum('bkp,bp->bk',cpu_v,cpu_g)*cpu_used)
    checks = {}
    for driver in ('gesvd','gesvdj'):
        u,s,v = torch.linalg.svd(j,full_matrices=False,driver=driver)
        used = s>s[:,:1]*1e-10
        projected = gradient-torch.einsum('bkp,bk->bp',v,
            torch.einsum('bkp,bp->bk',v,gradient)*used)
        reconstruction = (u*s.unsqueeze(-2))@v
        eye = torch.eye(v.shape[-2],dtype=v.dtype,device=v.device)
        errors = dict(reconstruction=float(torch.max(torch.linalg.vector_norm(reconstruction-j,dim=(-2,-1))
            /(1+torch.linalg.vector_norm(j,dim=(-2,-1))))),
            right_orthogonality=float(torch.max(torch.abs(v@v.transpose(-2,-1)-eye))),
            projection_cpu=scaled_error(projected.cpu().numpy(),cpu_projection.numpy()),
            projected_constraint=float(torch.max(torch.linalg.vector_norm(j@projected.unsqueeze(-1),dim=(-2,-1))
                /(1+torch.linalg.vector_norm(j,dim=(-2,-1))*torch.linalg.vector_norm(gradient,dim=-1)))))
        rank_equal = bool(torch.equal(used.sum(-1).cpu(),cpu_used.sum(-1)))
        if max(errors.values())>1e-10 or not rank_equal:
            raise ValueError(f'exact SVD backend failed cutoff/projection check: {driver}: {errors}')
        checks[driver] = dict(errors=errors,rank_matches_cpu=rank_equal,
            rank_min=int(used.sum(-1).min()),rank_max=int(used.sum(-1).max()),
            minimum_relative_cutoff_distance=float(torch.min(torch.abs(s/s[:,:1]-1e-10))))
    return checks


def synthetic_solver_smoke(fixture,device,batch,cpu_lock):
    """Two iterations on artificial samples and a duplicated RMS-sphere control.

    This exercises the actual GPU feature pipeline and native CPU-SVD solver.
    Neither fixture contains a measured sky or yields a scientific candidate.
    """
    family = fixture(n=66,planes=128)
    model = ClosureBatchCUDA(family,device)
    indices = torch.as_tensor(projection_indices('repair'),device=device)
    zero = torch.zeros((1,128),dtype=torch.float64,device=device)
    reference = model.features(zero)[0]
    scales = torch.maximum(torch.abs(reference),torch.full_like(reference,.001))
    feature = lambda angle:((model.features(angle)-reference)/scales).index_select(-1,indices)
    jacobian = lambda angle:model.feature_jacobian(angle,indices=indices)/scales[indices][None,:,None]
    kicks = .02*np.sin((np.arange(batch)[:,None]+1)*(np.arange(128)[None,:]+.5))
    signs = np.r_[np.ones(batch//2),-np.ones(batch-batch//2)]
    torch.cuda.reset_peak_memory_stats(device)
    full = projected_batch(feature,model.target,model.ratios,kicks,signs,device=device,
        max_iterations=2,jacobian_function=jacobian,target_gradient_function=model.target_gradient,
        rms_gradient_function=model.rms_gradient,svd_backend='cpu',cpu_lock=cpu_lock)
    def sphere_feature(angle):
        return angle[:,:1]
    def sphere_jacobian(angle):
        value = angle.new_zeros((len(angle),1,128));value[:,0,0]=1
        return value
    def sphere_ratio(angle):
        value = torch.linalg.vector_norm(angle[:,1:3],dim=-1)
        return torch.stack([value,value],dim=-1)
    def sphere_rms_gradient(angle):
        value = torch.zeros_like(angle);value[:,1:3]=2*angle[:,1:3]
        return torch.stack([value,value],dim=-2)
    def sphere_target_gradient(angle):
        value = torch.zeros_like(angle);value[:,1]=1
        return value
    sphere_kicks = np.zeros((batch,128));sphere_kicks[:,2]=.24
    sphere = projected_batch(sphere_feature,lambda angle:angle[:,1],sphere_ratio,sphere_kicks,signs,
        device=device,max_iterations=2,jacobian_function=sphere_jacobian,
        target_gradient_function=sphere_target_gradient,rms_gradient_function=sphere_rms_gradient,
        svd_backend='cpu',cpu_lock=cpu_lock)
    summary = {}
    for label,result in (('sample_pipeline',full),('duplicated_rms_sphere_control',sphere)):
        counters = {key:int(np.sum(result[key])) for key in (
            'iterations','accepted','rejected','retraction_corrections','center_restoration_failures',
            'center_failure_rejections','rms_restoration_corrections','rms_restoration_failures',
            'nonfinite_trials') if key in result}
        summary[label] = dict(seconds=result['seconds'],batch=batch,planes=128,
            status_counts={str(key):int(np.sum(result['status']==key)) for key in np.unique(result['status'])},
            center_reached=int(np.sum(result['center_reached'])),
            active_tile=int(np.sum(result['active_local_rms'])),
            active_global=int(np.sum(result['active_global_rms'])),
            maximum_center_residual=float(np.max(result['center_residual'])),
            maximum_tile_ratio=float(np.max(result['local_rms_ratio'])),
            maximum_global_ratio=float(np.max(result['global_rms_ratio'])),counters=counters,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
        if (not np.isfinite(result['theta']).all() or counters.get('nonfinite_trials',0)
                or not result['initialization_complete'] or np.max(result['center_residual'])>.00049
                or np.max(result['local_rms_ratio'])>.25 or np.max(result['global_rms_ratio'])>.25):
            raise ValueError(f'bounded synthetic solver smoke failed: {label}')
    if not np.all(sphere['active_local_rms']) or not np.all(sphere['active_global_rms']):
        raise ValueError('synthetic RMS-sphere control did not exercise both active RMS rows')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'data/evidence/closure_v2_preflight_final_hybrid.json')
    parser.add_argument('--repeats',type=int,default=3)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve previous evidence; choose a new output path')
    fixture_path = args.output.with_name(args.output.stem+'_fixture.json')
    if fixture_path.exists():
        raise FileExistsError('preserve previous compact fixture; choose a new output path')
    if not 1 <= args.repeats <= 5:
        raise ValueError('bounded preflight requires one through five timing repetitions')
    if torch.cuda.device_count()<2:
        raise RuntimeError('both approved actual GPUs are required')
    sources = ('src/cmbcst/closure_gpu_v2.py','scripts/prepare_closure_fibers_v2.py',
        'tests/test_closure_gpu_v2.py','src/cmbcst/closure_fibers.py','src/cmbcst/closure_jacobian.py',
        'src/cmbcst/closure_search_v2.py')
    before_hashes = {name:sha(ROOT/name) for name in sources}
    fixture = fixture_builder()
    compact = fixture()
    stress = fixture(n=264,planes=128)
    real_path = ROOT/'data/work/closure_integration_preflight_v1/family_24.npz'
    manifest_path = real_path.with_suffix('.json')
    manifest = json.loads(manifest_path.read_text())
    if sha(real_path)!=manifest['family_sha256']:
        raise ValueError('actual fixed-angle operator cache failed manifest hash')
    with np.load(real_path,allow_pickle=False) as archive:
        real = {name:archive[name] for name in archive.files}
    record = dict(status='running',purpose='fixed-angle engineering accuracy, performance and synthetic solver smoke; no real-data optimization or scientific search',
        ratio_names=RMS_RATIO_NAMES,seed=929173,source_sha256=before_hashes,
        python=sys.version,numpy=np.__version__,torch=str(torch.__version__),cuda=torch.version.cuda,
        cpu_threads=8,fixtures=dict(compact=dict(n=8,planes=4),performance=dict(n=264,planes=128),
            actual_tile_24=dict(n=66,planes=128,family_sha256=sha(real_path),
                manifest_sha256=sha(manifest_path),input_sha256=manifest['configuration']['source']['input_sha256'])),devices=[])
    historical = ROOT/'data/evidence/closure_v2_preflight_final.json'
    record['historical_exact_gpu_driver_profile'] = dict(path=str(historical.relative_to(ROOT)),sha256=sha(historical))
    record['svd_backend']='cpu'
    record['cpu_lock']='one shared threading.Lock; benchmark devices run sequentially'
    torch.set_num_threads(8)
    cpu_lock = threading.Lock()
    for index in (0,1):
        device = f'cuda:{index}'
        with torch.cuda.device(index):
            free,total = torch.cuda.mem_get_info(index)
            if free<512*1024**2:
                raise RuntimeError('insufficient free GPU memory for bounded preflight')
            evidence = dict(device=device,name=torch.cuda.get_device_name(index),
                free_bytes_before=free,total_bytes=total,recommended_batch=128 if index==0 else 64,
                compact=compact_checks(compact,device),benchmarks=[])
            for fixture_name,family in (('actual_tile_24',real),('synthetic_n264',stress)):
                model = ClosureBatchCUDA(family,device)
                old_model = ClosureCUDA(family,device)
                for batch in (4,64,128):
                    angles = .015*np.sin((np.arange(batch)[:,None]+1)*(np.arange(128)[None,:]+.5))
                    x = torch.as_tensor(angles,device=device,dtype=torch.float64)
                    def bundle():
                        return (model.features_and_jacobian(x),model.target_and_gradient(x),
                                model.ratio_squared_and_gradient(x))
                    measured = benchmark(bundle,device,args.repeats)
                    measured.update(fixture=fixture_name,batch=batch,starts_per_second=batch/measured['median_seconds'])
                    actual = model.features(x[[0,-1]]).detach().cpu().numpy()
                    measured['feature_cpu_error'] = scaled_error(actual,features_numpy(family,angles[[0,-1]]))
                    if measured['feature_cpu_error']>1e-10:
                        raise ValueError('native batch feature precision failure')
                    if measured['peak_allocated_bytes']>.75*free:
                        raise MemoryError('preflight exceeded declared 75% free-memory envelope')
                    # A finite representative [B,48,128] Jacobian, normalized by
                    # fixed positive feature scales, for the solver's exact SVD.
                    scale = torch.as_tensor(np.maximum(.001,np.abs(features_numpy(family,np.zeros(128))))[projection_indices('repair')],device=device)
                    j = model.feature_jacobian(x,indices='repair')/scale[None,:,None]
                    if not bool(torch.isfinite(j).all()):
                        raise ValueError('nonfinite benchmark Jacobian')
                    def hybrid_svd():
                        with cpu_lock:
                            u,s,v = torch.linalg.svd(j.cpu(),full_matrices=False)
                            return u.to(device),s.to(device),v.to(device)
                    measured['thin_svd'] = benchmark(hybrid_svd,device,args.repeats)
                    def projection_bundle():
                        feature,jac = model.features_and_jacobian(x,indices='repair')
                        value,gradient = model.target_and_gradient(x)
                        squared,rms_gradient = model.ratio_squared_and_gradient(x)
                        projected = active_tangent(jac/scale[None,:,None],gradient,x,
                            torch.sqrt(squared),rms_gradient,svd_backend='cpu',cpu_lock=cpu_lock)
                        return feature,value,squared,projected
                    measured['bundle_and_active_tangent'] = benchmark(projection_bundle,device,args.repeats)
                    measured['projection_scope'] = 'one full feature/gradient evaluation and actual active-set SVD projection; excludes restoration and backtracking'
                    if batch==(128 if index==0 else 64):
                        # The historical profile retains full-batch GPU driver
                        # timings. Recheck current exact projections on four
                        # rows without spending minutes repeating discovery.
                        measured['exact_svd_checks_first_four'] = svd_checks(j[:4],model.target_gradient(x[:4]))
                        gradient = model.target_gradient(x)
                        rms_squared,rms_gradient = model.ratio_squared_and_gradient(x)
                        hybrid = active_tangent(j,gradient,x,torch.sqrt(rms_squared),rms_gradient,
                            svd_backend='cpu',cpu_lock=cpu_lock)
                        native_cpu = active_tangent(j.cpu(),gradient.cpu(),x.cpu(),
                            torch.sqrt(rms_squared).cpu(),rms_gradient.cpu(),svd_backend='cpu',cpu_lock=cpu_lock)
                        measured['hybrid_projection_cpu_error'] = scaled_error(
                            hybrid['direction'].cpu().numpy(),native_cpu['direction'].numpy())
                        measured['hybrid_rank_matches_cpu'] = bool(torch.equal(hybrid['rank'].cpu(),native_cpu['rank']))
                        if measured['hybrid_projection_cpu_error']>1e-10 or not measured['hybrid_rank_matches_cpu']:
                            raise ValueError('hybrid batched active tangent differs from native CPU')
                    if batch==4:
                        measured['v1_native_batch_analytic_jacobian'] = benchmark(
                            lambda:feature_jacobian(old_model,x),device,args.repeats)
                    evidence['benchmarks'].append(measured)
                    print(json.dumps(dict(device=device,fixture=fixture_name,batch=batch,seconds=measured['median_seconds'],
                        svd_seconds=measured['thin_svd']['median_seconds'],
                        projection_seconds=measured['bundle_and_active_tangent']['median_seconds'],
                        starts_per_second=measured['starts_per_second'],peak_bytes=measured['peak_allocated_bytes'])),flush=True)
                del model,old_model,x,j
            evidence['synthetic_solver_two_iterations'] = synthetic_solver_smoke(fixture,device,
                evidence['recommended_batch'],cpu_lock)
            record['devices'].append(evidence)
    after_hashes = {name:sha(ROOT/name) for name in sources}
    record['source_unchanged']=before_hashes==after_hashes
    record['status']='passed' if record['source_unchanged'] else 'invalidated_by_concurrent_source_change'
    if not record['source_unchanged']:
        record['source_sha256_after']=after_hashes
    # Only a few explicit Jacobian rows are embedded in JoVin, while the full
    # actual-device numerical outputs remain in this evidence record.
    selected = [0,6,22,32,44,48,51]
    compact_fixture = dict(seed=929173,family=dict(n=8,planes=4),ratio_names=RMS_RATIO_NAMES,
        source_sha256=before_hashes,devices=[])
    for device in record['devices']:
        values = device['compact']
        compact_fixture['devices'].append(dict(device=device['device'],name=device['name'],
            **{key:values[key] for key in ('angles','features','target','target_gradient','ratio_squared','rms_gradient','errors')},
            jacobian_indices=selected,jacobian=np.asarray(values['jacobian'])[selected].tolist()))
    compact_payload = json.dumps(compact_fixture,separators=(',',':'),allow_nan=False)
    if len(compact_payload.encode())>10*1024:
        raise ValueError('compact JoVin fixture exceeds 10 KiB')
    record['compact_fixture_bytes']=len(compact_payload.encode())
    record['compact_fixture_sha256']=hashlib.sha256((compact_payload+'\n').encode()).hexdigest()
    payload = json.dumps(record,indent=2,allow_nan=False)
    if len(payload.encode())>180*1024:
        raise ValueError('compact actual-device attestation exceeds sandbox input envelope')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:
        stream.write(payload+'\n')
    with fixture_path.open('x',encoding='utf-8') as stream:
        stream.write(compact_payload+'\n')
    if not record['source_unchanged']:
        raise RuntimeError('source changed during preflight; invalidated measurements preserved')
    print(json.dumps(dict(status='passed',output=str(args.output),sha256=sha(args.output))))


if __name__=='__main__':
    main()
