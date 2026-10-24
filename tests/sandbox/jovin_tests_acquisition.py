import numpy as np
from solution import instrument_design,decoder_reference,batch_reference,reference_models,frequency_grid


def test_noiseless_geometry_recovers_known_sky_and_roll_hides_in_amplitude():
    rng = np.random.default_rng(625980)
    white = rng.normal(size=(2,3,32,32))
    noise = rng.normal(size=(2,8,32,32))
    result = batch_reference(white,noise,sigma=0.)
    assert np.max(np.sqrt(result['mse'][:,0,:3]))<1e-10
    assert result['roll_amplitude_error']<1e-10
    assert np.mean(result['mse'][:,1,1:3])>1
    assert np.max(result['mse'][:,1,3])<1e-20


def test_pointing_and_beam_errors_can_create_false_polarization():
    rng = np.random.default_rng(625981)
    white = rng.normal(size=(2,3,32,32))
    result = batch_reference(white,np.zeros((2,8,32,32)),sigma=0.,i_only=True)
    assert np.max(result['mse'][:,0,1:3])<1e-20
    assert np.mean(result['mse'][:,2,1:3])>1e-4
    assert np.mean(result['mse'][:,3,1:3])>1e-4


def test_poor_angle_coverage_is_rejected_without_pseudoinverse():
    a,target,mask = instrument_design(32,'poor_angles')
    try:
        decoder_reference(a,target,mask)
    except ValueError:
        pass
    else:
        raise AssertionError('singular geometry accepted')


def test_white_noise_variance_matches_fourier_prediction():
    rng = np.random.default_rng(625982)
    white = rng.normal(size=(128,3,16,16))
    noise = rng.normal(size=(128,8,16,16))
    result = batch_reference(white,noise)
    model = reference_models(16)[0]['correct']
    weight = frequency_grid(16)[5].ravel()
    expected = .25*np.sum(np.abs(model)**2*weight[:,None,None],axis=(0,2))/16**2
    assert np.all(np.abs(result['mse'][:,0,:3].mean(0)/expected-1)<.08)


def test_actual_recorded_cuda_outputs_match_cpu_reference():
    fixture = GPU_ATTESTATION
    rng = np.random.default_rng(fixture['seed'])
    result = batch_reference(rng.normal(size=(2,3,32,32)),rng.normal(size=(2,8,32,32)),return_maps=False)
    for device in fixture['devices']:
        assert np.allclose(result['mse'],device['mse'],rtol=1e-10,atol=1e-10)
        assert device['roll_amplitude_error']<1e-10


GPU_ATTESTATION = {'seed': 625983, 'devices': [{'device': 'cuda:0', 'name': 'NVIDIA GeForce RTX 5070', 'mse': [[[0.00022897213955759777, 0.0010466689350708616, 0.0006481946014598517, 0.0005675314040826868], [0.00022897213955761696, 1.6187804177388, 1.4566830400847999, 0.0005675314040825804], [0.49843377138012, 76.08470864584272, 88.30310129643712, 82.87406910116559], [2.4005699154286058, 2.313280111079564, 3.0438034835454477, 3.14947826983076]], [[0.0004096182654924077, 0.0009449863849587289, 0.0008319407013503706, 0.0008202653787853806], [0.0004096182654926185, 1.3073276624382721, 1.1518846482588683, 0.0008202653787853613], [0.47818339289751066, 72.58638177546459, 108.12942325274307, 119.28028521136152], [2.031171171531649, 2.133944813394981, 2.456110500936462, 2.4633755250220037]]], 'roll_amplitude_error': 1.70102020308519e-14, 'scaled_cpu_error': 1.3837279845273122e-14}, {'device': 'cuda:1', 'name': 'NVIDIA GeForce GTX 1660 Ti', 'mse': [[[0.00022897213955756546, 0.0010466689350706387, 0.0006481946014599689, 0.0005675314040825938], [0.000228972139557535, 1.6187804177388019, 1.4566830400848128, 0.0005675314040826031], [0.4984337713801178, 76.08470864584274, 88.30310129643709, 82.87406910116556], [2.4005699154286058, 2.3132801110795596, 3.043803483545438, 3.1494782698307544]], [[0.00040961826549251734, 0.0009449863849587234, 0.0008319407013503356, 0.0008202653787853541], [0.0004096182654924143, 1.3073276624382695, 1.1518846482588718, 0.0008202653787853033], [0.47818339289750994, 72.5863817754646, 108.12942325274304, 119.28028521136154], [2.0311711715316645, 2.133944813394972, 2.4561105009364614, 2.463375525022001]]], 'roll_amplitude_error': 1.4066224698963981e-14, 'scaled_cpu_error': 1.142124685641591e-14}]}
