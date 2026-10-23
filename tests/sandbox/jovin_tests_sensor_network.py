"""JoVin checks for the experimental complete-graph scalar sensor transfer.

The source functions are extracted from the shipped network and sensor_network
modules by the existing AST bundle builder. No astronomy input data are used.
"""
import numpy as np

from solution import complete_complex, fit_scalar_offsets


def test_scalar_sensor_exact_gauge():
    edges, triangles, d0, d1 = complete_complex(5)
    assert len(edges) == 10 and len(triangles) == 10
    truth = np.array([0.2, -0.1, 0.3, 0.4, -0.2])
    out = fit_scalar_offsets(d0 @ truth, np.eye(10) * 0.1, 5, 2)
    assert np.max(np.abs(out["offsets"] - (truth - truth[2]))) < 1e-10
    assert np.max(np.abs(d1 @ d0)) == 0
    assert out["residual_chi2"] < 1e-18


def test_scalar_sensor_fault_has_closure_signal():
    edges, _, d0, _ = complete_complex(4)
    y = d0 @ np.array([0.2, 0.5, -0.3, 0.1])
    y[edges.index((0, 2))] += 1.0
    out = fit_scalar_offsets(y, np.eye(6) * 0.01, 4, 0)
    assert out["closure_dof"] == 3
    assert out["residual_chi2"] > 10
    assert abs(out["closure_chi2"] - out["residual_chi2"]) < 1e-7


def test_scalar_sensor_rejects_invalid_covariance():
    try:
        fit_scalar_offsets(np.zeros(6), np.zeros((6, 6)), 4, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("singular covariance must be rejected")
