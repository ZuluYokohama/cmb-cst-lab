"""Run a reproducible, data-free calibration ensemble through the shipped estimators.

The synthetic sky follows tests/sandbox/jovin_tests_differential.py. Each bin is an
exact Wishart draw at its weighted mean covariance; this is a *bin-level* model,
not an exact sum of Wisharts with ell-dependent covariance and not Planck data.

Example (requires an accelerator unless --allow-cpu is explicitly supplied):
    CMB_CST_DEVICE=cuda:0 python scripts/run_gpu_calibration.py --n-sims 2000 --batch 32

An immutable .prespec.json is written before drawing any random samples. Existing
outputs are never replaced. Statistical decisions use fixed five-MCSE gates;
correlated map pulls are grouped by independent simulation when computing MCSEs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cmbcst import accel  # noqa: E402
from cmbcst.differential import pair_angles, rotation_matrix  # noqa: E402
from cmbcst.network import complete_complex, solve_network  # noqa: E402
from cmbcst.simulator import bartlett_draws  # noqa: E402


def synthetic_sigma(ell: float, alpha: np.ndarray, beta: float = 0.0, eb: float = 0.0) -> np.ndarray:
    """The eight-map synthetic sky/noise covariance from the JoVin differential suite."""
    ee = 2.0 * (ell / 80.0) ** -0.6
    sky = np.array([[ee, eb * ee], [eb * ee, 0.15 * ee]])
    signal = np.kron(np.ones((8, 8)), sky)
    rotate = rotation_matrix(np.asarray(alpha) + beta)
    noise = np.array([6.0, 6.0, 3.0, 3.0, 4.0, 4.0, 12.0, 12.0]) * 1e-2
    return rotate @ signal @ rotate.T + np.diag(np.repeat(noise, 2))


def synthetic_bandpowers(alpha: np.ndarray, beta: float, eb: float) -> tuple[np.ndarray, np.ndarray]:
    """Fixed 14 bins [51, 71), ..., [311, 331), with exact integer mode counts."""
    boundaries = np.arange(51, 332, 20)
    cl = np.zeros((len(boundaries) - 1, 16, 16))
    nu = np.zeros(len(cl))
    for b, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        for ell in range(lo, hi):
            modes = 2 * ell + 1
            cl[b] += modes * synthetic_sigma(ell, alpha, beta, eb)
            nu[b] += modes
        cl[b] /= nu[b]
    return cl, nu


def _estimate(value: float, target: float, contributions: np.ndarray, threshold: float = 5.0) -> dict:
    """Estimate plus MCSE from independent draws (one contribution per draw)."""
    contributions = np.asarray(contributions, dtype=float)
    if contributions.ndim != 1 or len(contributions) < 2 or not np.all(np.isfinite(contributions)):
        raise ValueError("MCSE needs at least two finite, independent draw contributions")
    if not np.isfinite(value) or not np.isfinite(target) or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("estimate, target, and positive threshold must be finite")
    mcse = float(np.std(contributions, ddof=1) / np.sqrt(len(contributions)))
    deviation = abs(float(value) - target)
    # A degenerate empirical MCSE is not evidence of calibrated uncertainty.
    passed = bool(mcse > 0 and deviation <= threshold * mcse)
    return {"value": float(value), "target": float(target), "mcse": mcse,
            "deviation_mcse": deviation / mcse if mcse > 0 else None,
            "acceptance_half_width": threshold * mcse, "passed": passed}


def summarize_pulls(pulls: np.ndarray, threshold: float = 5.0) -> dict:
    """Pooled pull summaries using cluster-by-simulation delta-method MCSEs.

    Maps within each draw may be arbitrarily correlated. The width is the pooled
    centered standard deviation; its influence contribution is each draw's mean
    squared centered pull divided by twice that width. Coverage contributions
    are per-draw fractions, never a fictitious n_draws*n_maps binomial sample.
    """
    pulls = np.asarray(pulls, dtype=float)
    if pulls.ndim != 2 or len(pulls) < 2 or pulls.shape[1] == 0 or not np.all(np.isfinite(pulls)):
        raise ValueError("pulls must be a finite (n_draws >= 2, n_maps >= 1) array")
    mean = float(pulls.mean())
    squared = np.mean((pulls - mean) ** 2, axis=1)
    width = float(np.sqrt(np.mean(squared)))
    width_contributions = squared / (2 * width) if width > 0 else np.zeros(len(pulls))
    return {
        "mean": _estimate(mean, 0.0, pulls.mean(axis=1), threshold),
        "width": _estimate(width, 1.0, width_contributions, threshold),
        "coverage_1sigma": _estimate(float(np.mean(np.abs(pulls) <= 1)), math.erf(1 / math.sqrt(2)),
                                     np.mean(np.abs(pulls) <= 1, axis=1), threshold),
        "coverage_2sigma": _estimate(float(np.mean(np.abs(pulls) <= 2)), math.erf(2 / math.sqrt(2)),
                                     np.mean(np.abs(pulls) <= 2, axis=1), threshold),
    }


def summarize_chi2(values: np.ndarray, dof: int, threshold: float = 5.0) -> dict:
    """Mean chi-square gate with empirical MCSE, plus its ideal chi-square MCSE."""
    values = np.asarray(values, dtype=float)
    if dof <= 0 or np.any(values < 0):
        raise ValueError("chi-square degrees of freedom must be positive and observations nonnegative")
    result = _estimate(float(values.mean()), float(dof), values, threshold)
    result["ideal_chi2_mcse"] = math.sqrt(2 * dof / len(values))
    result["dof"] = int(dof)
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype="<f8").tobytes()).hexdigest()


def _git_metadata() -> dict:
    out = {}
    for label, arguments in (("commit", ["rev-parse", "HEAD"]), ("working_tree", ["status", "--porcelain"])):
        try:
            out[label] = subprocess.check_output(["git", "-C", str(REPO), *arguments],
                                                text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            out[label] = None
    return out


def _write_new_json(path: Path, payload: dict) -> None:
    encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)


def run(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    prespec_path = output.with_suffix(".prespec.json")
    if output.exists() or prespec_path.exists():
        raise FileExistsError(f"Use a new --output path; preserving {output} and {prespec_path}")
    torch, device = accel._device()
    if device is None and not args.allow_cpu:
        raise RuntimeError("No accelerator selected. Configure CUDA/XPU or explicitly use --allow-cpu for a CPU smoke run.")
    device_type = device.type if device is not None else "cpu"
    degree = np.pi / 180
    alpha = np.array([+0.19, -0.31, 0.0, 0.04, -0.06, -0.06, -0.08, -0.10]) * degree * 3
    beta, intrinsic_eb, gauge = 0.3 * degree, 0.03, 2
    cl, nu = synthetic_bandpowers(alpha, beta, intrinsic_eb)
    edges, _, d0, _ = complete_complex(8)
    keep = [i for i in range(8) if i != gauge]
    source_paths = [Path(__file__).resolve(), REPO / "src/cmbcst/accel.py",
                    REPO / "src/cmbcst/differential.py", REPO / "src/cmbcst/network.py", REPO / "src/cmbcst/simulator.py",
                    REPO / "tests/sandbox/jovin_tests_differential.py"]
    device_name = getattr(torch, device_type).get_device_name(device) if device is not None else platform.processor()
    prespec = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "Synthetic eight-map differential CMB calibration; bin-level Wishart ensemble",
        "scope": "No measured sky data. Exact Wishart draws at weighted bin-mean covariance; not an exact ell-level simulation.",
        "n_sims": args.n_sims, "batch": args.batch, "seed": args.seed,
        "generation_benchmark": {"n_sims": min(50, args.n_sims), "seed": args.seed + 2,
                                 "scope": "CPU Bartlett and selected-device Gram produce the same bin-level Wishart distribution and output shape using different algorithms and random streams; no hardware speedup claim."},
        "model": {"alpha_degrees": (alpha / degree).tolist(), "common_beta_degrees": beta / degree,
                  "intrinsic_eb_over_ee": intrinsic_eb, "gauge_map": gauge, "fitted_maps": keep,
                  "ell_min_inclusive": 51, "ell_max_exclusive": 331, "bin_width": 20,
                  "nu": nu.astype(int).tolist(), "cl_sha256": _array_sha256(cl)},
        "acceptance": {"threshold_mcse": 5.0, "all_pooled_and_per_map_checks_required": True,
                       "pair_chi2_dof": 28, "network_residual_chi2_dof": 21,
                       "minimum_draws_for_full_run": 2000, "float32_relative_tolerance": accel.REL_TOL,
                       "exact_mean_angle_tolerance_radians": 1e-12,
                       "no_post_result_threshold_or_seed_tuning": True,
                       "mcse_method": "Independent simulation clusters; delta method for centered pull width; empirical SEM for chi-square means."},
        "device": {"type": device_type, "name": device_name, "backend": accel.backend_name(),
                   "requested": os.environ.get("CMB_CST_DEVICE", "auto"), "allow_cpu": args.allow_cpu},
        "versions": {"python": sys.version, "numpy": np.__version__,
                     "torch": str(torch.__version__) if torch is not None else None,
                     "cuda_runtime": getattr(torch.version, "cuda", None) if torch is not None else None,
                     "platform": platform.platform()},
        "git": _git_metadata(),
        "source_sha256": {str(path.relative_to(REPO)).replace("\\", "/"): _sha256(path) for path in source_paths},
        "command": sys.argv,
    }
    _write_new_json(prespec_path, prespec)
    print(f"Prespecification saved before sampling: {prespec_path}", flush=True)
    print(f"Backend: {prespec['device']['backend']}", flush=True)

    started = time.perf_counter()
    check_started = time.perf_counter()
    precision = accel.check_float32(cl, nu, seed=args.seed + 1, batch=min(args.batch, 4))
    precision_seconds = time.perf_counter() - check_started
    precision_error = precision.get("max_rel_err")
    precision_passed = bool(precision.get("applicable") and precision_error is not None
                            and np.isfinite(precision_error) and precision_error <= accel.REL_TOL)
    if device is not None and not precision_passed:
        raise RuntimeError(f"Accelerator precision check failed: {precision}")
    exact_pairs = pair_angles(cl, nu, edges)
    mean_angle_error = float(np.max(np.abs(exact_pairs["delta"] - d0 @ alpha)))
    if mean_angle_error > 1e-12:
        raise RuntimeError(f"Synthetic expectation failed angle recovery: {mean_angle_error}")

    benchmark_n = prespec["generation_benchmark"]["n_sims"]
    benchmark_seed = prespec["generation_benchmark"]["seed"]
    tick = time.perf_counter()
    bartlett_output = bartlett_draws(cl, nu, benchmark_n, np.random.default_rng(benchmark_seed))
    bartlett_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    gram_output = np.array(list(accel.wishart_draws(cl, nu, benchmark_n, seed=benchmark_seed, batch=args.batch)))
    gram_seconds = time.perf_counter() - tick
    if bartlett_output.shape != gram_output.shape or not (np.all(np.isfinite(bartlett_output)) and np.all(np.isfinite(gram_output))):
        raise RuntimeError("Generation benchmark returned invalid outputs")
    benchmark = {
        "n_sims": benchmark_n, "seed": benchmark_seed, "output_shape": list(gram_output.shape),
        "cpu_bartlett_seconds": bartlett_seconds, "selected_device_gram_seconds": gram_seconds,
        "cpu_bartlett_sha256": _array_sha256(bartlett_output), "selected_device_gram_sha256": _array_sha256(gram_output),
        "scope": prespec["generation_benchmark"]["scope"],
        "timing_notes": "End-to-end generation, including allocations, RNG and covariance transforms; Gram includes the production precision check and device transfers. Both follow the numerical preflight; neither uses the main ensemble RNG stream.",
    }
    del bartlett_output, gram_output
    print(f"Generation benchmark ({benchmark_n} draws): CPU Bartlett {bartlett_seconds:.3f}s; {device_type.upper()} Gram {gram_seconds:.3f}s (different algorithms).", flush=True)

    pulls = np.empty((args.n_sims, len(keep)))
    pair_chi2 = np.empty(args.n_sims)
    network_chi2 = np.empty(args.n_sims)
    generation_seconds, fitting_seconds = 0.0, 0.0
    draws = iter(accel.wishart_draws(cl, nu, args.n_sims, seed=args.seed, batch=args.batch))
    draws_hash = hashlib.sha256()
    for draw_id in range(args.n_sims):
        tick = time.perf_counter()
        draw = next(draws)
        generation_seconds += time.perf_counter() - tick
        draws_hash.update(np.ascontiguousarray(draw, dtype="<f8").tobytes())
        tick = time.perf_counter()
        pairs = pair_angles(draw, nu, edges)
        error = pairs["delta"] - d0 @ alpha
        pair_chi2[draw_id] = error @ np.linalg.solve(pairs["cov"], error)
        network = solve_network(pairs["delta"], pairs["cov"], d0, gauge=gauge)
        pulls[draw_id] = (network["alpha"] - (alpha - alpha[gauge]))[keep] / np.sqrt(np.diag(network["cov"])[keep])
        network_chi2[draw_id] = network["chi2"]
        fitting_seconds += time.perf_counter() - tick
        if (draw_id + 1) % max(1, args.n_sims // 10) == 0:
            print(f"{draw_id + 1}/{args.n_sims} draws; generation {generation_seconds:.2f}s, fitting {fitting_seconds:.2f}s", flush=True)

    pooled = summarize_pulls(pulls)
    per_map = {str(m): summarize_pulls(pulls[:, k:k + 1]) for k, m in enumerate(keep)}
    pair_summary, network_summary = summarize_chi2(pair_chi2, 28), summarize_chi2(network_chi2, 21)
    checks = [*pooled.values(), pair_summary, network_summary]
    checks.extend(check for summaries in per_map.values() for check in summaries.values())
    statistical_passed = all(check["passed"] for check in checks)
    sufficient_draws = args.n_sims >= 2000
    qualified = device is not None and precision_passed and sufficient_draws
    status = "passed" if statistical_passed and qualified else ("failed" if qualified else "exploratory")
    failed_checks = [f"pooled.{name}" for name, check in pooled.items() if not check["passed"]]
    failed_checks += [f"map_{m}.{name}" for m, summaries in per_map.items() for name, check in summaries.items() if not check["passed"]]
    failed_checks += [name for name, check in (("pair_chi2_mean", pair_summary), ("network_chi2_mean", network_summary)) if not check["passed"]]
    summary = (
        f"{args.n_sims} synthetic skies were generated with {device_type.upper()} Wishart sampling and fitted with the shipped pair/network estimators. "
        f"Pooled pull mean {pooled['mean']['value']:.4f}, width {pooled['width']['value']:.4f}; "
        f"one/two-sigma coverage {pooled['coverage_1sigma']['value']:.2%}/{pooled['coverage_2sigma']['value']:.2%}. "
        f"Mean pair chi-square {pair_summary['value']:.3f} versus 28; residual chi-square {network_summary['value']:.3f} versus 21. "
        f"All fixed five-MCSE statistical checks {'passed' if statistical_passed else 'did not pass'}. "
        "MCSEs account for correlations between maps within each sky. "
        + ("The accelerator precision gate passed. " if precision_passed else "No accelerator precision check applies to this CPU run. ")
        + ("This is an exploratory run, not the full accelerator calibration. " if not qualified else "")
        + "This validates only the stated synthetic bin-level model; common beta and intrinsic EB are injected nuisance parameters, not inferred quantities."
    )
    result = {
        "schema_version": 1, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": status, "summary": summary, "prespec_path": str(prespec_path),
        "prespec_sha256": _sha256(prespec_path), "prespec": prespec,
        "acceptance": {"statistical_passed": statistical_passed, "qualified_full_accelerator_run": qualified,
                       "sufficient_draws": sufficient_draws, "precision_passed": precision_passed,
                       "failed_checks": failed_checks, "n_statistical_checks": len(checks)},
        "precision_check": precision, "exact_mean_angle_max_error_radians": mean_angle_error,
        "pulls_pooled": pooled, "pulls_by_map": per_map,
        "pair_chi2": pair_summary, "network_residual_chi2": network_summary,
        "generation_benchmark": benchmark,
        "timing_seconds": {"precision_preflight": precision_seconds, "generation": generation_seconds,
                           "fitting": fitting_seconds, "total_after_prespec": time.perf_counter() - started},
        "timing_notes": "Generation includes RNG, Gram, CPU transfers/transforms, iterator setup and its built-in precision check. Fitting is CPU pair_angles plus solve_network. No CPU/GPU speedup is claimed.",
        "draws_sha256": draws_hash.hexdigest(),
        "draw_records": {"fitted_maps": keep, "pulls": pulls.tolist(), "pair_chi2": pair_chi2.tolist(),
                         "network_residual_chi2": network_chi2.tolist()},
    }
    _write_new_json(output, result)
    print(summary, flush=True)
    print(f"Saved {output}", flush=True)
    return 0 if status in ("passed", "exploratory") else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-sims", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--output", type=Path, default=REPO / "results/gpu_calibration.json")
    parser.add_argument("--allow-cpu", action="store_true", help="Explicitly permit an exploratory CPU fallback.")
    args = parser.parse_args()
    if args.n_sims < 2 or args.batch < 1 or not 0 <= args.seed < 2**63 - 1:
        parser.error("--n-sims must be >= 2, --batch >= 1, and --seed in [0, 2**63 - 2]")
    try:
        return run(args)
    except (OSError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        print(f"Calibration stopped: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
