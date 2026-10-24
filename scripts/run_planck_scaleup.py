"""Observe the frozen full-resolution reproduction without changing its numerical work.

The driver requires completed acquisition and a passed scientific input audit. It
calls run_reproduction.main with the saved plan, records stages and every returned
MK fit, and validates the resulting artifact. Wrappers forward every argument and
return value unchanged. There is no polling, retry, parameter tuning or new analysis.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import functools
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    def invalid_constant(value):
        raise ValueError(f"Nonfinite JSON constant in {path}: {value}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid_constant)


def planned_directory(value: str, base: Path) -> Path:
    path = (REPO / value).resolve()
    if base.resolve() not in path.parents:
        raise ValueError(f"Plan requires a dedicated directory beneath {base}: {path}")
    return path


def validate_inputs(plan: dict, inventory_path: Path, audit_path: Path) -> dict:
    from cmbcst import data

    inventory, audit = read_json(inventory_path), read_json(audit_path)
    if inventory.get("status") != "complete" or inventory.get("failures"):
        raise ValueError("The dataset inventory is not complete and failure-free")
    if audit.get("status") != "passed" or audit.get("failures"):
        raise ValueError("The scientific input audit has not passed")
    expected = {p.key: p for p in data.MANIFEST}
    downloaded = {r["key"]: r for r in inventory["products"]}
    audited = {r["key"]: r for r in audit["products"]}
    if set(downloaded) != set(expected) or set(audited) != set(expected):
        raise ValueError("Inventory and input audit must cover every manifest product")
    if len(inventory["products"]) != len(expected) or len(audit["products"]) != len(expected):
        raise ValueError("Duplicate product records are not accepted")
    raw = (REPO / plan["raw_directory"]).resolve()
    if data.raw_directory() != raw:
        raise ValueError(f"Configured raw directory {data.raw_directory()} differs from plan {raw}")
    for key, product in expected.items():
        record, checked = downloaded[key], audited[key]
        path = raw / product.filename
        if not checked.get("passed") or checked.get("failures"):
            raise ValueError(f"Scientific input audit failed for {key}")
        if record["filename"] != product.filename or checked["filename"] != product.filename:
            raise ValueError(f"Product identity mismatch for {key}")
        if record["sha256"] != checked["sha256"] or record["bytes"] != checked["bytes"]:
            raise ValueError(f"Inventory and scientific input audit describe different bytes for {key}")
        if path.stat().st_size != record["bytes"]:
            raise ValueError(f"Input size changed after audit for {key}")
    return {"inventory_path": str(inventory_path), "inventory_sha256": sha256(inventory_path),
            "input_audit_path": str(audit_path), "input_audit_sha256": sha256(audit_path),
            "products": [{"key": key, "sha256": downloaded[key]["sha256"], "bytes": downloaded[key]["bytes"]}
                         for key in expected],
            "scope": "The scientific audit rehashed input bytes; this driver matches its hashes to acquisition records and rechecks current file sizes."}


def validate_result(path: Path, plan: dict) -> dict:
    import numpy as np

    result = read_json(path)
    def finite_tree(value, location="result"):
        if isinstance(value, dict):
            for key, item in value.items():
                finite_tree(item, f"{location}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                finite_tree(item, f"{location}[{index}]")
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(value):
            raise ValueError(f"Nonfinite reported number: {location}")
    finite_tree(result)
    if set(result["per_mask"]) != {"0", "30"} or len(result["detectors"]) != 8 or result["gauge"] != "143A":
        raise ValueError("Result has unexpected map, mask, or gauge dimensions")
    if result["binning"] != plan["binning"] or not result["beam_source"].startswith("NPIPE QuickPol"):
        raise ValueError("Result binning or QuickPol beam provenance does not match the plan")
    for mask, record in result["per_mask"].items():
        if len(record["pair_angles_deg"]) != 28 or len(record["closures"]["z"]) != 56 or len(record["ell"]) != 72:
            raise ValueError(f"Unexpected pair/closure/bin dimensions for mask {mask}")
        for covariance_name, covariance in (("pair", record["pair_cov_deg2"]), ("MK", record["mk"]["cov_alpha_deg2"])):
            covariance = np.asarray(covariance)
            if not np.allclose(covariance, covariance.T, rtol=1e-8, atol=1e-12) or np.linalg.eigvalsh(covariance).min() <= 0:
                raise ValueError(f"{covariance_name} covariance is not symmetric positive definite for mask {mask}")
        network, closure, mk = record["network"], record["closures"], record["mk"]
        if network["dof"] != 21 or closure["rank"] != 21 or network["chi2"] < 0 or not 0 <= network["pte"] <= 1:
            raise ValueError(f"Invalid network degrees of freedom/statistic for mask {mask}")
        if not math.isclose(network["chi2"], closure["chi2"], rel_tol=1e-7, abs_tol=1e-8):
            raise ValueError(f"Network and closure chi-square disagree for mask {mask}")
        if not (0 < record["fsky_eff"] <= 1 and 0 < record["fsky_w2"] <= 1):
            raise ValueError(f"Invalid sky fraction for mask {mask}")
        uncertainty_arrays = [record["pair_sigma_deg"], mk["sigma_alpha_deg"], record["closures"]["sigma_deg"]]
        uncertainties = [mk["sigma_beta_deg"], mk["beta_profile"]["quadratic_sigma_deg"], record["anchored_sigma_deg"]]
        uncertainties.extend(x for values in uncertainty_arrays for x in values)
        if any(x <= 0 for x in uncertainties):
            raise ValueError(f"Nonpositive reported uncertainty for mask {mask}")
        gauge = result["detectors"].index(result["gauge"])
        if network["alpha_deg"][gauge] != 0 or network["sigma_deg"][gauge] != 0:
            raise ValueError("Gauge-fixed angle and uncertainty must be zero")
        if any(x <= 0 for index, x in enumerate(network["sigma_deg"]) if index != gauge):
            raise ValueError(f"Nonpositive network uncertainty for mask {mask}")
    baseline = result["per_mask"]["0"]
    bootstrap = baseline["bootstrap"]
    if bootstrap["n_draws"] != plan["bootstrap_draws"] or bootstrap["unconverged_mk_fits"] != 0:
        raise ValueError("Bootstrap count or convergence does not meet the plan")
    if not bootstrap["float32_check_applicable"] or not 0 <= bootstrap["float32_max_rel_err"] <= bootstrap["float32_tolerance"]:
        raise ValueError("The actual accelerator precision check did not pass")
    if baseline["chi2_diff"]["dof"] != 7 or baseline["anchored_with_cross"]["sigma_deg"] <= 0:
        raise ValueError("Invalid differential comparison rank or anchored cross-covariance uncertainty")
    return {"status": "passed", "path": str(path), "sha256": sha256(path),
            "scope": "Finite outputs, dimensions, positive uncertainties/covariances, QuickPol provenance, topology consistency, bootstrap count/convergence and accelerator precision. This is execution validation, not a discovery claim."}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=REPO / "data/evidence/planck_full_resolution_plan.json")
    parser.add_argument("--inventory", type=Path, default=REPO / "data/evidence/planck_dataset.json")
    parser.add_argument("--input-audit", type=Path, default=REPO / "data/evidence/planck_input_audit.json")
    args = parser.parse_args(argv)
    try:
        plan = read_json(args.plan)
        results = planned_directory(plan["results_directory"], REPO / "results")
        work = planned_directory(plan["work_directory"], REPO / "data/work")
        results.mkdir(parents=True, exist_ok=True)
        if any((results / name).exists() for name in ("reproduction.json", "fit_audit.json", "stage_audit.json")):
            raise ValueError("Existing reproduction or observer artifacts are preserved; inspect them before choosing a fresh planned run")
    except (OSError, ValueError, KeyError) as error:
        print(f"Observer preflight stopped: {error}", file=sys.stderr)
        return 2

    state = {"schema": "cmbcst.planck-stage-audit.v1", "status": "running", "started_utc": utc(),
             "plan_path": str(args.plan.resolve()), "plan_sha256": sha256(args.plan),
             "observer_sha256": sha256(Path(__file__)), "stages": [], "fits_completed": 0, "failures": []}
    fit_records, patches, runner = [], [], None
    old_device = os.environ.get("CMB_CST_DEVICE")
    started = time.perf_counter()
    write_json(results / "stage_audit.json", state)
    write_json(results / "fit_audit.json", {"status": "running", "fits": []})
    exit_code = 1
    try:
        if plan["stages"] != ["masks", "alms", "spectra", "transfer", "analyse"]:
            raise ValueError("Observer expects the prespecified five-stage reproduction")
        state["inputs"] = validate_inputs(plan, args.inventory, args.input_audit)
        for relative, expected in plan["numerical_source_sha256"].items():
            source = (REPO / relative).resolve()
            if REPO not in source.parents or sha256(source) != expected:
                raise ValueError(f"Numerical source no longer matches the saved plan: {relative}")
        state["supplemental_source_sha256"] = {str(p.relative_to(REPO)): sha256(p) for p in sorted((REPO / "src/cmbcst").glob("*.py"))}
        specification = importlib.util.spec_from_file_location("observed_reproduction", REPO / "scripts/run_reproduction.py")
        runner = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(runner)
        import numpy as np

        original_fit = runner.MKLikelihood.fit
        @functools.wraps(original_fit)
        def observed_fit(likelihood, *fit_args, **fit_kwargs):
            tick = time.perf_counter()
            record = {"index": len(fit_records) + 1, "stage": state.get("current_stage"),
                      "n_maps": likelihood.n, "n_dust_amplitudes": likelihood.n_a}
            try:
                returned = original_fit(likelihood, *fit_args, **fit_kwargs)
                finite_likelihood = bool(np.isfinite(returned["neg2lnl"]))
                free = fit_kwargs.get("free", fit_args[1] if len(fit_args) > 1 else None)
                record.update(converged=bool(returned["converged"]), iterations=int(returned["iterations"]),
                              n_params=len(returned["theta"]), n_free=len(returned["theta"]) if free is None else len(free),
                              theta_finite=bool(np.all(np.isfinite(returned["theta"]))),
                              cov_finite=bool(np.all(np.isfinite(returned["cov"]))), neg2lnl_finite=finite_likelihood,
                              neg2lnl=float(returned["neg2lnl"]) if finite_likelihood else None)
                return returned
            except BaseException as error:
                record["error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                record["seconds"] = time.perf_counter() - tick
                fit_records.append(record)
                state["fits_completed"] = len(fit_records)
                if len(fit_records) % 50 == 0:
                    write_json(results / "stage_audit.json", state)

        patches.append((runner.MKLikelihood, "fit", original_fit))
        runner.MKLikelihood.fit = observed_fit
        for name in plan["stages"]:
            attribute = "stage_" + name
            original = getattr(runner, attribute)
            def observe_stage(function, label):
                @functools.wraps(function)
                def observed(*stage_args, **stage_kwargs):
                    record = {"name": label, "status": "running", "started_utc": utc()}
                    state["stages"].append(record)
                    state["current_stage"] = label
                    write_json(results / "stage_audit.json", state)
                    tick = time.perf_counter()
                    try:
                        value = function(*stage_args, **stage_kwargs)
                        record["status"] = "completed"
                        return value
                    except BaseException as error:
                        record.update(status="failed", error=f"{type(error).__name__}: {error}")
                        raise
                    finally:
                        record.update(finished_utc=utc(), seconds=time.perf_counter() - tick)
                        state.pop("current_stage", None)
                        write_json(results / "stage_audit.json", state)
                return observed
            patches.append((runner, attribute, original))
            setattr(runner, attribute, observe_stage(original, name))

        command = ["--stages", ",".join(plan["stages"]), "--work-dir", str(work), "--results-dir", str(results),
                   "--n-sims", str(plan["bootstrap_draws"]), "--n-transfer-sims", str(plan["transfer_skies_per_mask"]),
                   "--threads", str(plan["threads"])]
        state["runner_arguments"] = command
        os.environ["CMB_CST_DEVICE"] = plan["device"]
        if runner.main(command) != 0:
            raise RuntimeError("Reproduction runner returned a nonzero status")
        state["result_validation"] = validate_result(results / "reproduction.json", plan)
        rejected = [r["index"] for r in fit_records if r.get("error") or not all(r.get(key, False) for key in
                    ("converged", "theta_finite", "cov_finite", "neg2lnl_finite"))]
        if rejected:
            raise ValueError(f"{len(rejected)} returned fits failed convergence/finite checks; inspect fit_audit.json (first indices {rejected[:10]})")
        state["status"] = "passed"
        exit_code = 0
    except BaseException as error:
        state["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        state["failures"].append(f"{type(error).__name__}: {error}")
        print(f"Observed reproduction stopped: {state['failures'][-1]}", file=sys.stderr, flush=True)
        exit_code = 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        for target, attribute, original in reversed(patches):
            setattr(target, attribute, original)
        if old_device is None:
            os.environ.pop("CMB_CST_DEVICE", None)
        else:
            os.environ["CMB_CST_DEVICE"] = old_device
        rejected = [r["index"] for r in fit_records if r.get("error") or not all(r.get(key, False) for key in
                    ("converged", "theta_finite", "cov_finite", "neg2lnl_finite"))]
        state.update(finished_utc=utc(), seconds=time.perf_counter() - started, fits_completed=len(fit_records), rejected_fit_indices=rejected)
        # Publish stage status first, so a fit-audit write failure cannot hide the run outcome.
        write_json(results / "stage_audit.json", state)
        write_json(results / "fit_audit.json", {"schema": "cmbcst.planck-fit-audit.v1", "status": state["status"],
                   "plan_sha256": state["plan_sha256"], "observer_sha256": state["observer_sha256"],
                   "completed_utc": state["finished_utc"], "n_fits": len(fit_records),
                   "rejected_fit_indices": rejected, "fits": fit_records,
                   "scope": "Observer forwards fit arguments and results unchanged. Any nonconverged/nonfinite fit fails readiness; no numerical retries were performed."})
    print(f"Observed reproduction {state['status']}; {len(fit_records)} fits; {results / 'stage_audit.json'}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
