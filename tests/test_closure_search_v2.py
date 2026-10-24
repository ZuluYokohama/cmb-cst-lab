"""Synthetic positive/negative controls; no sky transforms or GPU jobs."""
import json
import numpy as np
import pytest
from cmbcst.closure_search_v2 import active_tangent_numpy,active_tangent,batched_svd,projected_batch,initial_kicks


def test_active_rms_tangent_and_iterative_outward_box_faces():
    result=active_tangent_numpy(np.zeros((1,3)),[1,2,0],[0,0,0],[.24,.1],[[1,0,0],[0,1,0]])
    assert result['active_rms'].tolist()==[True,False]
    assert np.allclose(result['direction'],[0,2,0],atol=1e-12)
    # Pinning face0 exposes an outward face1 component; both must be added.
    box=active_tangent_numpy([[1,1,1]],[1,-.1,-.9],[np.pi/4,np.pi/4,0],[0,0],np.zeros((2,3)))
    assert box['active_box'].tolist()==[True,True,False]
    assert box['box_satisfied'] and np.linalg.norm(box['direction'])<1e-12
    with pytest.raises(ValueError):
        active_tangent_numpy([[1,0]],[np.nan,0],[0,0],[0,0],np.zeros((2,2)))


@pytest.fixture
def torch():
    return pytest.importorskip('torch')


def zeros_ratio(torch):
    return lambda angle:angle.new_zeros((len(angle),2))


def run(torch,feature,target,kicks,signs=None,ratio=None,**kw):
    kicks=np.asarray(kicks,dtype=float)
    return projected_batch(feature,target,ratio or zeros_ratio(torch),kicks,
        np.ones(len(kicks)) if signs is None else signs,device='cpu',
        max_iterations=kw.pop('max_iterations',12),max_stalls=kw.pop('max_stalls',3),**kw)


def test_exact_closed_identity_and_nonclosed_control(torch):
    feature=lambda angle:angle[:,:1]
    closed=run(torch,feature,lambda angle:angle[:,0],np.zeros((2,2)),[1,-1])
    assert closed['status'].tolist()==['stalled','stalled']
    assert np.max(np.abs(closed['gap']))==0
    assert set(closed['stall_reason'])=={'projected_gradient_zero'}
    nonclosed=run(torch,feature,lambda angle:angle[:,1],np.zeros((2,2)),[1,-1],max_iterations=20)
    assert np.min(np.abs(nonclosed['gap']))>.7
    assert np.max(np.abs(nonclosed['theta']))<=np.pi/4
    assert np.max(nonclosed['center_residual'])<=1e-6
    assert nonclosed['feasible'].all()


def test_native_active_faces_match_independent_oracle(torch):
    j=np.array([[[1.,1.,1.]],[[1.,0.,0.]]]);g=np.array([[1.,-.1,-.9],[1.,2.,3.]])
    angles=np.array([[np.pi/4,np.pi/4,0],[0,0,0]])
    rms=np.array([[0.,0.],[.24,.1]]);q=np.array([np.zeros((2,3)),[[0,1,0],[0,0,1]]])
    actual=active_tangent(*(torch.tensor(v,dtype=torch.float64) for v in (j,g,angles,rms,q)))
    for row in range(2):
        expected=active_tangent_numpy(j[row],g[row],angles[row],rms[row],q[row])
        assert np.allclose(actual['direction'][row].numpy(),expected['direction'],atol=1e-12)
        assert np.array_equal(actual['active_box'][row].numpy(),expected['active_box'])
        assert actual['box_satisfied'][row]


def test_nonlinear_center_restoration_and_safe_bin_gate_are_distinct(torch):
    feature=lambda angle:(angle[:,0]+angle[:,1].square())[:,None]
    nonlinear=run(torch,feature,lambda angle:angle[:,1],[[0,.1]])
    assert nonlinear['retraction_corrections'][0]>0
    assert nonlinear['center_reached'][0]
    assert nonlinear['maximum_tracking_residual'][0]<=1e-6
    assert nonlinear['minimum_margin'][0]>=1e-5
    # Zero derivative cannot restore this discontinuous synthetic feature. Its
    # safe recorded bin still passes; 1e-6 is a strategy target, not a new gate.
    discrete=lambda angle:torch.where(angle[:,1:2]>0,angle.new_tensor(.0002),angle.new_tensor(0.))
    safe=run(torch,discrete,lambda angle:angle[:,1],[[0,.1]],max_iterations=2)
    assert safe['target'][0]>.1 and safe['feasible'][0]
    assert not safe['center_reached'][0]
    assert safe['center_restoration_failures'][0]>0
    assert safe['center_residual'][0]==pytest.approx(.0002)


def test_active_rms_constraint_positive_control(torch):
    feature=lambda angle:angle[:,2:3]
    ratio=lambda angle:torch.stack((torch.linalg.vector_norm(angle[:,:2]+angle.new_tensor([.24,0.]),dim=-1),angle[:,2]*0),dim=-1)
    result=run(torch,feature,lambda angle:angle[:,1],np.zeros((2,3)),[1,-1],ratio=ratio,max_iterations=18)
    assert result['maximum_active_rms_count'].min()==1
    assert np.max(result['local_rms_ratio'])<=.25
    assert np.min(np.abs(result['target']))>.05
    assert result['active_local_rms'].all() and not result['active_global_rms'].any()
    assert result['feasible'].all()


def test_curved_hard_rms_boundary_is_restored_without_relaxation(torch):
    feature=lambda angle:angle[:,2:3]
    ratio=lambda angle:torch.stack((torch.linalg.vector_norm(angle[:,:2]+angle.new_tensor([.25,0.]),dim=-1),angle[:,2]*0),dim=-1)
    result=run(torch,feature,lambda angle:angle[:,1],np.zeros((2,3)),[1,-1],ratio=ratio,max_iterations=8)
    assert result['rms_restoration_corrections'].min()>0
    assert np.min(np.abs(result['target']))>.05
    assert np.max(result['local_rms_ratio'])<=.25
    assert result['feasible'].all()
    assert result['maximum_tracking_residual'].max()<=1e-6
    with pytest.raises(ValueError,match='exact SVD'):
        run(torch,feature,lambda angle:angle[:,1],np.zeros((1,3)),svd_driver='gesvda')


def test_native_128_64_batches_match_single_rows_and_hooks(torch):
    feature=lambda angle:(angle[:,0]+.1*angle[:,1].square())[:,None]
    target=lambda angle:angle[:,2]
    kicks,signs=initial_kicks(24,'fresh',count=128,planes=3)
    seen=[]
    def jac(angle):
        seen.append(len(angle))
        return torch.stack((torch.ones_like(angle[:,0]),.2*angle[:,1],torch.zeros_like(angle[:,0])),dim=-1)[:,None,:]
    gradient=lambda angle:torch.stack((angle[:,0]*0,angle[:,0]*0,angle[:,0]*0+1),dim=-1)
    result=run(torch,feature,target,kicks,signs,max_iterations=3,jacobian_function=jac,target_gradient_function=gradient)
    half=run(torch,feature,target,kicks[:64],signs[:64],max_iterations=3,jacobian_function=jac,target_gradient_function=gradient)
    assert 128 in seen and 64 in seen
    assert np.allclose(result['theta'][:64],half['theta'],atol=1e-12,rtol=0)
    for row in (0,1,64,65):
        single=run(torch,feature,target,kicks[row:row+1],signs[row:row+1],max_iterations=3)
        assert np.allclose(result['theta'][row],single['theta'][0],atol=1e-12,rtol=0)
    assert np.array_equal(result['raw_kicks'],kicks)
    assert np.array_equal(result['signs'],signs)


def test_invalid_proposal_isolation_artifacts_and_checkpoints(torch):
    def feature(angle):
        return torch.where(angle[:,1:2]<-.02,torch.full_like(angle[:,:1],float('nan')),angle[:,:1])
    target=lambda angle:angle[:,2]
    snapshots=[];kicks=np.array([[0,.05,0],[0,-.05,0]])
    result=run(torch,feature,target,kicks,max_iterations=3,checkpoint=snapshots.append)
    peer=run(torch,feature,target,kicks[:1],max_iterations=3)
    assert np.allclose(result['theta'][0],peer['theta'][0],atol=1e-12)
    assert result['nonfinite_constraint_trials'][1]>0 and result['nonfinite_constraint_trials'][0]==0
    assert 1 in result['failure_start_id']
    assert np.isnan(result['failure_values']).any()
    assert np.any(result['failure_theta'][:,1]<-.02)
    assert snapshots[0]['checkpoint_phase']=='before_initialization'
    assert snapshots[-1]['checkpoint_phase']=='complete'
    assert all('minimum_margin' in snap and 'stall_reason' in snap for snap in snapshots)
    json.dumps(result['history'],allow_nan=False)


def test_nonfinite_jacobian_does_not_terminate_finite_peer(torch):
    feature=lambda angle:angle[:,:1]
    target=lambda angle:angle[:,2]
    def jac(angle):
        row=torch.stack((torch.ones_like(angle[:,0]),angle[:,0]*0,angle[:,0]*0),dim=-1)[:,None,:]
        return torch.where((angle[:,1]>.02)[:,None,None],torch.full_like(row,float('nan')),row)
    result=run(torch,feature,target,[[0,.04,0],[0,-.04,0]],jacobian_function=jac,max_iterations=3)
    # The bad initial derivative is an initialization proposal rejection; the
    # accepted state remains safe, and any later invalid current row is explicit.
    assert result['nonfinite_jacobian_evaluations'][0]>0
    assert result['target'][1]>.2 and result['feasible'][1]
    assert 'direction_jacobian' in result['failure_kind'] or 'restoration_jacobian' in result['failure_kind']


def test_timeout_checkpoint_preserves_initialization_state(torch):
    snapshots=[]
    def timeout():raise TimeoutError('synthetic cap')
    with pytest.raises(TimeoutError):
        run(torch,lambda angle:angle[:,:1],lambda angle:angle[:,1],np.zeros((2,2)),budget=timeout,checkpoint=snapshots.append)
    final=snapshots[-1]
    assert final['checkpoint_phase']=='interrupted'
    assert final['status'].tolist()==['incomplete','incomplete']
    assert not final['initialization_complete']
    assert np.array_equal(final['raw_kicks'],np.zeros((2,2)))
    assert final['failure_theta'].shape==(0,2)


def test_timeout_after_initialization_preserves_projected_state(torch):
    snapshots=[];ready=[False]
    def save(value):
        snapshots.append(value)
        if value['checkpoint_phase']=='initialized':ready[0]=True
    def timeout():
        if ready[0]:raise TimeoutError('synthetic post-initialization cap')
    with pytest.raises(TimeoutError):
        run(torch,lambda angle:(angle[:,0]+angle[:,1].square())[:,None],lambda angle:angle[:,1],
            [[0,.1]],budget=timeout,checkpoint=save)
    final=snapshots[-1]
    assert final['initialization_complete'] and final['status'][0]=='incomplete'
    assert final['initial_theta'].shape==(1,2)
    assert np.allclose(final['theta'],final['initial_theta'])
    assert final['target_available'] and final['minimum_margin'][0]>=1e-5
    assert final['retraction_corrections'][0]>0


def test_cpu_svd_backend_matches_device_and_holds_shared_lock(torch,monkeypatch):
    class SharedLock:
        def __init__(self):self.held=False;self.entries=0
        def __enter__(self):
            assert not self.held
            self.held=True;self.entries+=1
        def __exit__(self,*args):self.held=False
    generator=torch.Generator().manual_seed(87123)
    matrix=torch.randn((64,4,7),generator=generator,dtype=torch.float64)
    expected=batched_svd(matrix,svd_backend='device')
    lock=SharedLock();original=torch.linalg.svd;seen=[]
    def checked(value,**kwargs):
        assert lock.held and value.device.type=='cpu' and value.dtype==torch.float64
        seen.append(tuple(value.shape));return original(value,**kwargs)
    monkeypatch.setattr(torch.linalg,'svd',checked)
    actual=batched_svd(matrix,svd_backend='cpu',cpu_lock=lock)
    assert lock.entries==1 and not lock.held and seen==[(64,4,7)]
    for left,right in zip(expected,actual):assert torch.equal(left,right)
    reconstructed=actual[0]@torch.diag_embed(actual[1])@actual[2]
    assert torch.allclose(reconstructed,matrix,atol=1e-12,rtol=0)
    with pytest.raises(ValueError,match='backend'):
        batched_svd(matrix,svd_backend='approximate')
    with pytest.raises(TypeError,match='cpu_lock'):
        batched_svd(matrix,svd_backend='cpu',cpu_lock=object())


def test_hybrid_backend_covers_initialization_restoration_and_tangent(torch):
    class CountingLock:
        def __init__(self):self.entries=0
        def __enter__(self):self.entries+=1
        def __exit__(self,*args):pass
    feature=lambda angle:(angle[:,0]+angle[:,1].square())[:,None]
    target=lambda angle:angle[:,1]
    kicks=np.array([[0,.1],[0,-.1]])
    device=run(torch,feature,target,kicks,[1,-1],max_iterations=4)
    lock=CountingLock()
    cpu=run(torch,feature,target,kicks,[1,-1],max_iterations=4,svd_backend='cpu',cpu_lock=lock)
    assert lock.entries>4  # initial SVD, Newton restoration and search directions
    assert cpu['svd_backend']=='cpu' and cpu['svd_execution']=='cpu_float64'
    assert np.allclose(cpu['theta'],device['theta'],atol=1e-12,rtol=0)
    assert np.array_equal(cpu['rank'],device['rank'])
    j=torch.tensor([[[1.,1.,1.]]],dtype=torch.float64)
    g=torch.tensor([[1.,-.1,-.9]],dtype=torch.float64)
    x=torch.tensor([[np.pi/4,np.pi/4,0]],dtype=torch.float64)
    r=torch.zeros((1,2),dtype=torch.float64);q=torch.zeros((1,2,3),dtype=torch.float64)
    before=lock.entries
    tangent=active_tangent(j,g,x,r,q,svd_backend='cpu',cpu_lock=lock)
    assert lock.entries>=before+2 and tangent['active_box'][0,:2].all()
