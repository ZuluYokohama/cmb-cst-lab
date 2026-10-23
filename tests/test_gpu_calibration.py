"""Run the same experiment-statistics checks locally and protect immutable run records."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_gpu_calibration.py"
SPEC = importlib.util.spec_from_file_location("gpu_calibration", SCRIPT)
calibration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(calibration)


def test_summary_checks_shared_with_jovin():
    source = (SCRIPT.parents[1] / "tests/sandbox/jovin_tests_gpu_calibration.py").read_text(encoding="utf-8")
    source = source.replace("from solution import _estimate, summarize_chi2, summarize_pulls, synthetic_sigma", "")
    namespace = {name: getattr(calibration, name) for name in ("_estimate", "summarize_chi2", "summarize_pulls", "synthetic_sigma")}
    exec(compile(source, "jovin_tests_gpu_calibration.py", "exec"), namespace)
    for name, fn in namespace.items():
        if name.startswith("test_"):
            fn()


def test_refuses_cpu_without_explicit_opt_in(monkeypatch, tmp_path):
    monkeypatch.setattr(calibration.accel, "_device", lambda: (None, None))
    args = calibration.argparse.Namespace(output=tmp_path / "run.json", allow_cpu=False)
    with pytest.raises(RuntimeError, match="No accelerator"):
        calibration.run(args)
    assert not list(tmp_path.iterdir())


def test_existing_results_and_prespec_are_never_overwritten(tmp_path):
    output = tmp_path / "run.json"
    calibration._write_new_json(output, {"keep": True})
    with pytest.raises(FileExistsError):
        calibration._write_new_json(output, {"keep": False})
    assert calibration.json.loads(output.read_text())["keep"] is True
    args = calibration.argparse.Namespace(output=output, allow_cpu=False)
    with pytest.raises(FileExistsError):
        calibration.run(args)


def test_injected_bandpower_mean_recovers_known_differential_angles():
    alpha = np.array([0.19, -0.31, 0, 0.04, -0.06, -0.06, -0.08, -0.10]) * np.pi / 60
    cl, nu = calibration.synthetic_bandpowers(alpha, beta=0.3 * np.pi / 180, eb=0.03)
    edges, _, d0, _ = calibration.complete_complex(8)
    assert cl.shape == (14, 16, 16)
    assert np.array_equal(nu, np.round(nu))
    assert np.max(np.abs(calibration.pair_angles(cl, nu, edges)["delta"] - d0 @ alpha)) < 1e-12
