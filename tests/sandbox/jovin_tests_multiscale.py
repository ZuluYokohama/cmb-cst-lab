import numpy as np
from solution import multiscale_reference, features_from_curves, LAGS


def test_affine_brightness_invariance_and_sign_reversal():
    x = np.random.default_rng(651123).normal(size=(2, 32, 33))
    c, f = multiscale_reference(x)
    for a,b in ((3.,7.), (-2.,11.)):
        d,g = multiscale_reference(a*x+b)
        assert np.allclose(g, f, rtol=1e-12, atol=1e-12)
        for col,power in enumerate((2,4,6)):
            assert np.allclose(d[:,:,col], c[:,:,col]*a**power, rtol=1e-12, atol=1e-12)


def test_open_boundary_linear_ramp_has_exact_ratios():
    row,col = np.mgrid[:32,:32]
    curves, features = multiscale_reference(row+col)
    assert np.allclose(curves[0,:,0], np.asarray(LAGS)**2)
    assert np.allclose(curves[0,:,3], -2)
    assert np.allclose(curves[0,:,4], -14)
    assert np.allclose(features, [[2,-2,-14,0,0,0]], atol=1e-12)


def test_transposition_flips_only_directional_contrast():
    x = np.random.default_rng(83924).normal(size=(32,40))
    _, a = multiscale_reference(x)
    _, b = multiscale_reference(x.T)
    assert np.allclose(a[:,:5], b[:,:5], atol=1e-12)
    assert np.allclose(a[:,5], -b[:,5], atol=1e-12)


def test_constant_small_nonfinite_rejected():
    for x in (np.ones((32,32)), np.ones((8,8)), np.full((32,32), np.nan)):
        try:
            multiscale_reference(x)
        except ValueError:
            continue
        raise AssertionError('invalid patch accepted')


def test_actual_cuda_recorded_increment_moments():
    fixture = GPU_ATTESTATION
    x = np.random.default_rng(fixture['seed']).normal(size=(2,32,33))
    curves,features = multiscale_reference(x)
    assert np.allclose(curves, fixture['curves'], rtol=1e-11, atol=1e-11)
    assert np.allclose(features, fixture['features'], rtol=1e-11, atol=1e-11)


GPU_ATTESTATION = {'seed': 764512, 'curves': [[[1.977251888865113, 11.553991007967333, 104.68253869730854, -0.044656086523227057, -1.45783038353564, 0.01208196569687828], [2.030522706071188, 12.800172899827546, 137.697076028429, 0.10456055585714363, 1.4475481816620217, 0.007679455500488207], [1.9360843183404886, 11.115022846747847, 101.71112380146161, -0.03474651452726185, -0.984917749483726, -0.01441901790572923], [1.9752588040433259, 11.222653059938105, 112.77078618521773, -0.12361162517296309, -0.36729593010574924, 0.03202481396252387], [2.105605773449604, 13.407196840762806, 132.20806107657742, 0.024014436440964015, -0.8379296370941614, -0.026529374788588526]], [[1.9038252912148748, 11.076021793081079, 106.28325784626111, 0.05583300916970879, 0.4022292250826087, -0.010043920854445632], [1.861028453823778, 10.337409737797593, 93.30478582245426, -0.015265510566101259, -0.5241231237663246, -0.008896379760065682], [1.804538904966943, 9.310773386255605, 75.94111469451968, -0.14074257745980656, -2.076557389262737, -0.004126177042705528], [1.8847808646331463, 10.391851470563147, 90.05266462645164, -0.07469466672914082, -1.5502572989542198, -0.015618561259211482], [1.8724321848734387, 10.824053949395214, 104.1686624334182, 0.08729232427617895, 0.8678758943380132, 0.0230763804797695]]], 'features': [[0.014166788629862084, -0.01488784678506887, -0.440085103711451, -0.0043852587816928315, -0.003952954991187168, 0.002167568493114521], [-0.002967856407471655, -0.017515484261832182, -0.5761665385125319, 5.373869118552443e-05, -0.002013247619709724, -0.0031217316873317657]]}
