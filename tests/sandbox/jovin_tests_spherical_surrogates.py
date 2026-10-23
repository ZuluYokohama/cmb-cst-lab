import numpy as np
from solution import fixed_power_reference, checked_harmonic_inputs, patch_observables, alm_ell_m, cross_cl


def test_exact_shell_power_real_mzero_and_zero_shells():
    rng = np.random.default_rng(762304)
    cl = np.array([0., 0., 1.5, .7, .3, .1])
    alms = fixed_power_reference(cl, rng.normal(size=(12, 2, 21)))
    ell, m = alm_ell_m(5)
    assert np.all(alms[:, m == 0].imag == 0)
    assert np.all(alms[:, ell < 2] == 0)
    for a in alms:
        assert np.allclose(cross_cl(a, a, 5), cl, rtol=2e-14, atol=1e-14)


def test_gaussian_shell_direction_second_moments():
    rng = np.random.default_rng(91123)
    cl = np.ones(5)
    a = fixed_power_reference(cl, rng.normal(size=(6000, 2, 15)))
    ell, m = alm_ell_m(4)
    positions = np.flatnonzero(ell == 4)
    real_coordinates = np.column_stack([a[:, positions[0]].real] +
        [part for idx in positions[1:] for part in (np.sqrt(2)*a[:, idx].real, np.sqrt(2)*a[:, idx].imag)])
    assert np.max(np.abs(real_coordinates.mean(0))) < .05
    assert np.max(np.abs(real_coordinates.T @ real_coordinates / len(a) - np.eye(9))) < .065


def test_invalid_shapes_negative_spectra_and_zero_directions_rejected():
    bad = [(np.array([1., -1.]), np.ones((1, 2, 3))),
           (np.ones(3), np.ones((2, 2, 5))),
           (np.ones(2), np.zeros((1, 2, 3))),
           (np.array([1., np.nan]), np.ones((1, 2, 3)))]
    for cl, normals in bad:
        try:
            fixed_power_reference(cl, normals)
        except ValueError:
            continue
        raise AssertionError("invalid harmonic inputs accepted")


def test_topology_invariance_and_temperature_moment_scaling():
    rng = np.random.default_rng(22451)
    x = rng.normal(size=(10, 11))
    a, b = patch_observables(x), patch_observables(3*x+7)
    assert np.array_equal(a[:2], b[:2])
    assert np.isclose(b[2], 3*a[2])
    assert np.allclose(a[3:], b[3:], atol=1e-13)


def test_recorded_cuda_shell_projection():
    # An attestation fixture is added from the actual CUDA run before this suite is frozen.
    assert "GPU_ATTESTATION" in globals(), "actual GPU output has not been attached"
    fixture = GPU_ATTESTATION
    rng = np.random.default_rng(fixture["seed"])
    cl = np.asarray(fixture["cl"])
    ref = fixed_power_reference(cl, rng.normal(size=(2, 2, 15)))
    observed = np.asarray(fixture["real"]) + 1j*np.asarray(fixture["imag"])
    assert np.allclose(observed, ref, rtol=1e-12, atol=1e-12)


GPU_ATTESTATION = {'seed': 481223, 'cl': [0.0, 0.0, 1.2, 0.7, 0.4], 'real': [[-0.0, 0.0, 0.7259541955007962, -0.7097266293101969, 0.28147704485061914, -0.0, -0.8587286075674239, -0.4574483093224874, 0.28970254743420437, -0.12652870102107777, 0.2327890628076777, -0.34042890258124703, -0.6803666725472299, -0.07181558921838366, -0.370092341100045], [0.0, -0.0, -0.2063615257304614, -1.4176540924555423, -1.3592196112405377, 0.0, -0.5134980937342274, -0.026567655387558356, -0.11882191212843331, -0.38729238948803035, 0.2785691662407311, 0.3540986858891502, -0.05347787084407604, 0.27243505348510544, -0.6939142367132859]], 'imag': [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.948805014962226, -0.7329087288736352, 1.115671222855657, -1.0405959652829522, -0.6680424398782153, -0.11962356658426022, 0.6988273298236964, -0.2538612719735979, 0.3081598489651384], [0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.6402859552256225, 0.023441933726323194, -0.37804996497518567, -1.4680141922562862, 0.4527945470659479, 0.07299189291031824, 1.0762850494175211, -0.17972686101312854, -0.02161792683083921]]}
