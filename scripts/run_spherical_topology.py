"""Frozen whole-sphere conditional-power experiment; preserve all earlier runs."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cmbcst import persistence as p, spectra, theory, data
from cmbcst import spherical_surrogates as ss

OUT = ROOT / "results/spherical_topology_v1"
WORK = ROOT / "data/work/spherical_topology_v1"
SEED = 66510322026
NREF, NCONTROL, BATCH = 999, 64, 8
LMAX, NSIDE, THREADS = 1000, 512, 8
PATCHES = {"NGP": (0., 0.), "SGP": (np.pi, 0.),
           "ColdSpot": (np.radians(147.), np.radians(209.))}
STATISTICS = ["rank_PD0_count", "rank_total_PD0_lifetime", "temperature_std_uK",
              "temperature_skewness", "temperature_excess_kurtosis"]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, record):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def freeze():
    import torch
    import ducc0
    from jovin_verify import SUITES, build_bundle

    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    if (OUT / "plan.json").exists():
        raise FileExistsError("preserving frozen evaluation; do not launch a duplicate")
    input_audit = ROOT / "data/evidence/planck_input_audit.json"
    package_audit = ROOT / "data/evidence/cst_package_audit.json"
    if any(json.loads(path.read_text())["status"] != "passed" for path in (input_audit, package_audit)):
        raise ValueError("input/package audit has not passed")
    history = json.loads((ROOT / "data/evidence/jovin_receipts.json").read_text())
    receipt = [r for r in history if r["suite"] == "spherical_surrogates"][-1]
    if (not receipt["passed"] or not receipt["receipt_verified"] or not receipt["chain_valid"]
            or receipt["bundle_sha256"] != hashlib.sha256(build_bundle("spherical_surrogates")[0].encode()).hexdigest()
            or receipt["tests_sha256"] != sha(ROOT / SUITES["spherical_surrogates"][1])):
        raise ValueError("current spherical kernels/tests lack a matching JoVin receipt")
    paths = [Path(__file__), ROOT / "docs/SPHERICAL_TOPOLOGY_SPEC.md"]
    paths += [ROOT / "src/cmbcst" / name for name in
              ("spherical_surrogates.py", "spectra.py", "theory.py", "persistence.py",
               "surrogate_topology.py", "copula_topology.py")]
    plan = {"created_utc": now(), "seed": SEED, "reference_skies": NREF, "control_skies": NCONTROL,
            "batch_size": BATCH, "lmax": LMAX, "nside": NSIDE, "threads": THREADS,
            "smoothing_arcmin": 10., "removed_multipoles": [0, 1],
            "patches_radians": PATCHES, "patch_size": 128, "patch_degrees": 20,
            "statistics": STATISTICS, "holm_family_size": 15, "candidate_p": .05,
            "max_control_flags": 8, "power_relative_tolerance": 1e-10,
            "null": "Uniform independent real harmonic shell directions conditional on full-sphere C_ell; isotropic Gaussian conditional model",
            "scope": "Previously inspected uncleaned temperature; no cosmological or novelty claim",
            "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path) for path in paths},
            "jovin_receipt_id": receipt["receipt_id"], "jovin_bundle_sha256": receipt["bundle_sha256"],
            "jovin_tests_sha256": receipt["tests_sha256"],
            "numpy": np.__version__, "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "ducc0": ducc0.__version__, "device": torch.cuda.get_device_name(0),
            "input_audit_sha256": sha(input_audit), "package_audit_sha256": sha(package_audit),
            "input_inventory_sha256": sha(ROOT / "data/evidence/planck_dataset.json"),
            "mask_sha256": sha(ROOT / "data/work/rtx5070_real/mask_30.npy")}
    with (OUT / "plan.json").open("x") as stream:
        json.dump(plan, stream, indent=2)
    return plan


def prepare_coefficients():
    import ducc0
    from astropy.io import fits

    mean = np.zeros(12*2048**2)
    inventory = json.loads((ROOT / "data/evidence/planck_dataset.json").read_text())
    provenance = []
    for key in ("143A", "143B"):
        print(f"load audited temperature {key}", flush=True)
        path = data.path_of(key)
        record = next(r for r in inventory["products"] if r["key"] == key)
        # Bind this run to the actual bytes, not just the earlier inventory.
        measured_hash = sha(path)
        if measured_hash != record["sha256"]:
            raise ValueError(f"input bytes changed: {key}")
        provenance.append({"key": key, "bytes": Path(path).stat().st_size, "sha256": measured_hash})
        with fits.open(path, memmap=True) as hdus:
            header = hdus[1].header
            if header["NSIDE"] != 2048 or header["ORDERING"].strip() != "NESTED":
                raise ValueError("unexpected temperature resolution/order")
            t = np.asarray(hdus[1].data["I_STOKES"], dtype=np.float64).ravel()
        mean += .5 * spectra.nest_to_ring(np.where(np.abs(t) < 1e20, t*1e6, 0.), 2048)
        del t
    mask = np.load(ROOT / "data/work/rtx5070_real/mask_30.npy").astype(np.float64)
    mean -= np.sum(mean*mask)/mask.sum()
    del mask
    alm = ducc0.sht.experimental.adjoint_synthesis(map=mean[None], spin=0, lmax=LMAX, mmax=LMAX,
            nthreads=THREADS, **ducc0.healpix.Healpix_Base(2048, "RING").sht_info())[0] * (4*np.pi/mean.size)
    ell, _ = spectra.alm_ell_m(LMAX)
    alm[ell < 2] = 0
    cl = spectra.cross_cl(alm, alm, LMAX)
    if not np.all(np.isfinite(alm)) or not np.all(np.isfinite(cl)) or np.any(cl[2:] <= 0):
        raise ValueError("invalid observed harmonics/power")
    np.save(WORK / "observed_alm.npy", alm)
    np.save(WORK / "conditional_cl.npy", cl)
    save(OUT / "conditioning.json", {"created_utc": now(), "cl": cl.tolist(),
         "observed_alm_sha256": sha(WORK / "observed_alm.npy"), "cl_sha256": sha(WORK / "conditional_cl.npy"),
         "inputs": provenance, "scope": "Full-sphere empirical temperature power including foregrounds and noise"})
    return alm, cl


def project(alm):
    import ducc0

    ell, _ = spectra.alm_ell_m(LMAX)
    sky = ducc0.sht.experimental.synthesis(alm=(alm*theory.gaussian_beam(10., LMAX)[ell])[None],
            spin=0, lmax=LMAX, mmax=LMAX, nthreads=THREADS,
            **ducc0.healpix.Healpix_Base(NSIDE, "RING").sht_info())[0]
    patches = {name: p.project_patch(sky, *center) for name, center in PATCHES.items()}
    statistics = np.stack([ss.patch_observables(field) for field in patches.values()])
    if not np.all(np.isfinite(statistics)):
        raise ValueError("non-finite projected statistic")
    return statistics, patches


def quality(alm, cl):
    _, m = spectra.alm_ell_m(LMAX)
    power = spectra.cross_cl(alm, alm, LMAX)
    good = cl > 0
    error = float(np.max(np.abs(power[good]/cl[good]-1)))
    q = {"maximum_relative_shell_error": error, "finite": bool(np.all(np.isfinite(alm))),
         "zero_shells_exact": bool(np.all(power[~good] == 0)),
         "mzero_real": bool(np.all(alm[m == 0].imag == 0))}
    q["passed"] = bool(q["finite"] and q["zero_shells_exact"] and q["mzero_real"] and error <= 1e-10)
    return q


def compare(stats, references):
    raw = np.array([[p.mc_pvalue(references[:, row, col], stats[row, col])
                     for col in range(5)] for row in range(3)])
    return raw, p.holm(raw.ravel()).reshape(3, 5)


def sample(cl):
    nalm = (LMAX+1)*(LMAX+2)//2
    statistics, qualities = [], []
    t0 = time.perf_counter()
    for start in range(0, NREF+NCONTROL, BATCH):
        indices = range(start, min(start+BATCH, NREF+NCONTROL))
        normals = np.stack([np.random.default_rng(SEED+i).normal(size=(2, nalm)) for i in indices])
        alms = ss.fixed_power_cuda(cl, normals)
        for index, alm in zip(indices, alms):
            q = quality(alm, cl)
            q.update(index=index, seed=SEED+index, role="reference" if index < NREF else "control")
            qualities.append(q)
            if not q["passed"]:
                np.save(WORK / f"failed_alm_{index}.npy", alm)
                save(OUT / "quality_failure.json", q)
                raise ValueError(f"harmonic precision failure in draw {index}")
            statistics.append(project(alm)[0])
        np.save(WORK / "statistics_progress.npy", np.asarray(statistics))
        save(OUT / "progress.json", {"completed": len(statistics), "total": NREF+NCONTROL,
             "elapsed_s": time.perf_counter()-t0,
             "max_relative_shell_error": max(q["maximum_relative_shell_error"] for q in qualities)})
        print(f"complete skies {len(statistics)}/{NREF+NCONTROL} ({time.perf_counter()-t0:.1f}s)", flush=True)
    values = np.asarray(statistics)
    save(OUT / "ensemble.json", {"completed_utc": now(), "statistics": values.tolist(), "quality": qualities,
         "reference_skies": NREF, "control_skies": NCONTROL, "seconds": time.perf_counter()-t0})
    return values[:NREF], values[NREF:], qualities


def run():
    plan = freeze()
    start = time.perf_counter()
    stage = {"status": "running", "started_utc": now(), "plan_sha256": sha(OUT / "plan.json"), "stages": {}}
    save(OUT / "stage_audit.json", stage)
    try:
        t0 = time.perf_counter()
        observed_alm, cl = prepare_coefficients()
        stage["stages"]["prepare_seconds"] = time.perf_counter()-t0
        references, controls, qualities = sample(cl)
        control_records = []
        for i, stats in enumerate(controls):
            raw, adjusted = compare(stats, references)
            control_records.append({"index": i, "statistics": stats.tolist(), "raw_p": raw.tolist(),
                 "p_holm": adjusted.tolist(), "flag": bool(np.min(adjusted) <= .05)})
        flags = sum(r["flag"] for r in control_records)
        validation = {"completed_utc": now(), "status": "passed" if flags <= 8 else "failed",
             "familywise_flags": flags, "fields": NCONTROL, "max_allowed": 8, "holm_family_size": 15,
             "quality_failures": sum(not q["passed"] for q in qualities), "records": control_records,
             "scope": "Held-out whole-sky conditional Gaussian controls with shared reference; diagnostic gate, not a proof of physical adequacy"}
        save(OUT / "validation.json", validation)
        coherent = ss.fixed_power_cuda(cl, np.ones((1, 2, observed_alm.size)))[0]
        coherent_quality = quality(coherent, cl)
        if not coherent_quality["passed"]:
            save(OUT / "positive_control_quality_failure.json", coherent_quality)
            raise ValueError("positive-control numerical precision failure")
        coherent_stats, _ = project(coherent)
        raw, adjusted = compare(coherent_stats, references)
        save(OUT / "positive_control.json", {"statistics": coherent_stats.tolist(), "raw_p": raw.tolist(),
             "p_holm": adjusted.tolist(), "flag": bool(np.min(adjusted) <= .05), "quality": coherent_quality,
             "scope": "Deliberately coherent harmonic phases, identical C_ell; sensitivity example only"})
        result = {"completed_utc": now(), "status": "calibration_failed", "candidate_flags": [],
             "statistics": STATISTICS, "patches": list(PATCHES), "plan_sha256": sha(OUT / "plan.json"),
             "conditioning_sha256": sha(OUT / "conditioning.json"), "ensemble_sha256": sha(OUT / "ensemble.json"),
             "validation_sha256": sha(OUT / "validation.json"), "jovin_receipt_id": plan["jovin_receipt_id"],
             "accepted_cosmological_claims": []}
        if validation["status"] == "passed":
            observed, patches = project(observed_alm)
            np.savez(WORK / "observed_patches.npz", **patches)
            raw, adjusted = compare(observed, references)
            result.update(status="exploratory_complete", observed=observed.tolist(), raw_p=raw.tolist(),
                 p_holm=adjusted.tolist(), reference_mean=references.mean(0).tolist(),
                 reference_sd=references.std(0, ddof=1).tolist(),
                 candidate_flags=[{"patch": name, "statistic": STATISTICS[col], "p_holm": float(adjusted[row,col])}
                     for row,name in enumerate(PATCHES) for col in range(5) if adjusted[row,col] <= .05],
                 patches_sha256=sha(WORK / "observed_patches.npz"))
        result["limitations"] = "Uncleaned temperature and previously inspected patches. Isotropic Gaussian conditional null excludes foreground coherence, anisotropic/correlated noise, beams/systematics. Higher-order or anisotropy discrepancies are not new cosmology or unique topology information."
        save(OUT / "result.json", result)
        stage.update(status=result["status"], completed_utc=now(), elapsed_s=time.perf_counter()-start,
             control_flags=flags, quality_failures=validation["quality_failures"],
             max_relative_shell_error=max(q["maximum_relative_shell_error"] for q in qualities),
             result_sha256=sha(OUT / "result.json"))
        save(OUT / "stage_audit.json", stage)
        print(f"finished {result['status']}; held-out flags {flags}/64; candidate comparisons {len(result['candidate_flags'])}", flush=True)
    except Exception:
        stage.update(status="execution_failed", completed_utc=now(), elapsed_s=time.perf_counter()-start,
                     error=traceback.format_exc())
        save(OUT / "stage_audit.json", stage)
        raise


if __name__ == "__main__":
    run()
