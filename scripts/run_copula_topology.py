"""Fixed exploratory IAAFT topology experiment; validate first, then evaluate saved patches."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from cmbcst import persistence as p, surrogate_topology as st
from cmbcst.copula_topology import normal_scores

OUT = REPO / "results/copula_topology_v2"
WORK = REPO / "data/work/copula_topology_v2"
SEED = 49310702026
ITERATIONS = 500
PATCHES = {"NGP": (0., 0.), "SGP": (np.pi, 0.), "ColdSpot": (np.radians(147.), np.radians(209.))}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def quality_pass(q):
    return (q["spectrum_stage_relative_error"] <= 1e-10 and q["histogram_stage_exact"]
            and q["histogram_stage_spectrum_error"] <= .01 and q["spectrum_stage_histogram_ks"] <= .01)


def ensemble(target, count, seed, batch):
    records, quality = [], []
    for start in range(0, count, batch):
        initials = np.stack([np.random.default_rng(seed + i).permutation(target.ravel()).reshape(target.shape)
                             for i in range(start, min(start + batch, count))])
        spectrum, histogram = st.iaaft_gpu(target, initials, ITERATIONS)
        for s, h in zip(spectrum, histogram):
            qs = st.surrogate_quality(s, h, target)
            ts, th = st.pd0_summary(s), st.pd0_summary(h)
            qs["count_equal"] = bool(ts[0] == th[0])
            qs["passed"] = bool(quality_pass(qs) and qs["count_equal"])
            records.append([th[0], th[1], ts[1]])
            quality.append(qs)
        print(f"surrogates {min(start + batch, count)}/{count}", flush=True)
    return np.asarray(records), quality


def compare(target, simulations):
    top = st.pd0_summary(target)
    observed = np.array([top[0], top[1], top[1]])
    return observed, [p.mc_pvalue(simulations[:, k], observed[k]) for k in range(3)]


def freeze():
    import torch
    from jovin_verify import build_bundle, SUITES

    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    audit = json.loads((REPO / "data/evidence/planck_input_audit.json").read_text())
    if audit["status"] != "passed":
        raise ValueError("scientific input audit has not passed")
    receipts = json.loads((REPO / "data/evidence/jovin_receipts.json").read_text())
    receipt = [r for r in receipts if r["suite"] == "copula_topology"][-1]
    if (not receipt["passed"] or not receipt["chain_valid"]
            or receipt["bundle_sha256"] != hashlib.sha256(build_bundle("copula_topology")[0].encode()).hexdigest()
            or receipt["tests_sha256"] != sha(REPO / SUITES["copula_topology"][1])):
        raise ValueError("JoVin receipt does not match the shipped kernels/attestation tests")
    plan_file = OUT / "plan.json"
    sources = [Path(__file__), REPO / "src/cmbcst/surrogate_topology.py", REPO / "src/cmbcst/persistence.py",
               REPO / "src/cmbcst/spectra.py", REPO / "src/cmbcst/theory.py", REPO / "src/cmbcst/copula_topology.py", REPO / "docs/COPULA_TOPOLOGY_SPEC.md", REPO / "docs/SURROGATE_TOPOLOGY_SPEC.md"]
    hashes = {str(x.relative_to(REPO)).replace("\\", "/"): sha(x) for x in sources}
    if plan_file.exists():
        plan = json.loads(plan_file.read_text())
        if plan["source_sha256"] != hashes:
            raise ValueError("source differs from immutable plan; preserve this run and define a new experiment")
        return plan
    plan = {"created_utc": now(), "scope": "Exploratory algorithmic surrogate test on previously examined sky patches; no blind discovery claim",
            "source_sha256": hashes, "seed": SEED, "iterations": ITERATIONS,
            "validation_fields": 64, "validation_surrogates": 199, "validation_size": 32,
            "real_surrogates": 999, "real_size": 128, "patches_radians": PATCHES,
            "transform": "empirical midrank normal scores before surrogate generation and topology",
            "statistics": ["finite_PD0_count", "total_persistence_H", "total_persistence_S"],
            "holm_family_size": 9, "candidate_adjusted_p": .05, "validation_max_false_flags": 8,
            "quality": {"exact_spectrum_error": 1e-10, "H_spectrum_RMS": .01, "S_histogram_KS": .01},
            "numpy": np.__version__, "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "jovin_receipt_id": receipt["receipt_id"], "jovin_bundle_sha256": receipt["bundle_sha256"],
            "input_inventory_sha256": sha(REPO / "data/evidence/planck_dataset.json"),
            "input_audit_sha256": sha(REPO / "data/evidence/planck_input_audit.json"),
            "monopole_mask_sha256": sha(REPO / "data/work/rtx5070_real/mask_30.npy")}
    with plan_file.open("x") as stream:
        json.dump(plan, stream, indent=2)
    return plan


def validate():
    if (OUT / "validation.json").exists():
        raise FileExistsError("preserving existing validation; no reruns of the same planned evaluation")
    records = []
    for i in range(64):
        rng = np.random.default_rng(SEED + 100000 + i)
        axis = np.fft.fftfreq(32)
        kx, ky = np.meshgrid(axis, axis)
        x = np.fft.ifft2(np.fft.fft2(rng.normal(size=(32, 32))) * np.exp(-20 * (kx*kx + ky*ky))).real
        x = p.standardise(x)
        if i % 2:
            x = p.standardise(np.exp(.5 * x))
        x = normal_scores(x)
        sims, quality = ensemble(x, 199, SEED + 200000 + 1000*i, 199)
        observed, raw_p = compare(x, sims)
        adjusted = p.holm(raw_p)
        records.append({"index": i, "model": "monotonic_gaussian" if i % 2 else "gaussian",
                        "observed": observed.tolist(), "surrogate_statistics": sims.tolist(),
                        "raw_p": raw_p, "p_holm": adjusted.tolist(), "flag": bool(min(adjusted) <= .05),
                        "quality_failures": sum(not q["passed"] for q in quality),
                        "max_H_spectrum_error": max(q["histogram_stage_spectrum_error"] for q in quality),
                        "max_S_histogram_KS": max(q["spectrum_stage_histogram_ks"] for q in quality)})
        save(OUT / "validation_progress.json", {"completed": i+1, "records": records})
        print(f"validation field {i+1}/64: flags {sum(r['flag'] for r in records)}", flush=True)
    flags = sum(r["flag"] for r in records)
    result = {"status": "passed" if flags <= 8 else "failed", "completed_utc": now(), "fields": 64,
              "familywise_flags": flags, "cutoff": 8, "records": records,
              "scope": "Gross-calibration diagnostic on small synthetic fields; not a proof of exact real-sky significance"}
    save(OUT / "validation.json", result)
    return result


def extract_patches():
    import ducc0
    from astropy.io import fits
    from cmbcst import spectra, theory, data

    cache = WORK / "patches.npz"
    if cache.exists():
        raise FileExistsError("patch cache already exists without completed analysis; preserve it for diagnosis")
    mean = np.zeros(12*2048**2)
    for key in ("143A", "143B"):
        print(f"load real temperature {key}", flush=True)
        with fits.open(data.path_of(key), memmap=True) as hdus:
            h = hdus[1].header
            if h["NSIDE"] != 2048 or h["ORDERING"].strip() != "NESTED":
                raise ValueError("unexpected map ordering/resolution")
            t = np.asarray(hdus[1].data["I_STOKES"], dtype=np.float64).ravel()
        t = spectra.nest_to_ring(np.where(np.abs(t) < 1e20, t*1e6, 0.), 2048)
        mean += .5*t
        del t
    mask = np.load(REPO / "data/work/rtx5070_real/mask_30.npy").astype(np.float64)
    mean -= np.sum(mean * mask) / mask.sum()
    del mask
    geom = ducc0.healpix.Healpix_Base(2048, "RING").sht_info()
    alm = ducc0.sht.experimental.adjoint_synthesis(map=mean[None], spin=0, lmax=1000, mmax=1000,
                                                  nthreads=8, **geom)[0] * (4*np.pi/mean.size)
    del mean
    ell, _ = spectra.alm_ell_m(1000)
    low = ducc0.sht.experimental.synthesis(alm=(alm * theory.gaussian_beam(10., 1000)[ell])[None],
                                          spin=0, lmax=1000, mmax=1000, nthreads=8,
                                          **ducc0.healpix.Healpix_Base(512, "RING").sht_info())[0]
    patches = {name: p.standardise(p.project_patch(low, *center)) for name, center in PATCHES.items()}
    np.savez(WORK / "raw_patches.npz", **patches)
    patches = {name: normal_scores(field) for name, field in patches.items()}
    np.savez(cache, **patches)
    return patches


def analyse(validation):
    if validation["status"] != "passed":
        raise RuntimeError("synthetic gross-calibration gate failed; real evaluation blocked")
    patches = extract_patches()
    records, flat_p = {}, []
    for i, (name, target) in enumerate(patches.items()):
        print(f"evaluate real patch {name}", flush=True)
        sims, quality = ensemble(target, 999, SEED + 1000000 + 10000*i, 64)
        observed, raw_p = compare(target, sims)
        flat_p.extend(raw_p)
        records[name] = {"observed": observed.tolist(), "raw_p": raw_p, "surrogate_statistics": sims.tolist(),
                         "surrogate_mean": sims.mean(0).tolist(), "surrogate_sd": sims.std(0, ddof=1).tolist(),
                         "quality": quality, "quality_failures": sum(not q["passed"] for q in quality),
                         "moments": {"mean": float(target.mean()), "std": float(target.std()),
                                     "skewness": float(np.mean(target**3)), "excess_kurtosis": float(np.mean(target**4)-3)}}
        save(OUT / "analysis_progress.json", {"completed_patches": list(records), "records": records})
    adjusted = p.holm(flat_p).reshape(3, 3)
    for i, name in enumerate(records):
        records[name]["p_holm"] = adjusted[i].tolist()
    quality_ok = all(not r["quality_failures"] for r in records.values())
    result = {"completed_utc": now(), "status": "completed" if quality_ok else "quality_failed",
              "plan_sha256": sha(OUT / "plan.json"), "validation_sha256": sha(OUT / "validation.json"),
              "patches_sha256": sha(WORK / "patches.npz"), "records": records,
              "candidate_flags": [{"patch": name, "statistic": k, "p_holm": r["p_holm"][k]}
                                  for name, r in records.items() for k in range(3) if r["p_holm"][k] <= .05 and quality_ok],
              "limitations": "Exploratory, previously seen patches; algorithmic IAAFT distribution, finite mismatches, flat-patch power, boundary/nonstationarity, foregrounds and instrument noise prevent a cosmological attribution."}
    save(OUT / "result.json", result)
    print(f"real analysis {result['status']}; candidates {result['candidate_flags']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("validate", "analyse", "all"), default="all")
    args = parser.parse_args()
    freeze()
    if args.stage in ("validate", "all"):
        validation = validate()
    else:
        validation = json.loads((OUT / "validation.json").read_text())
    if args.stage in ("analyse", "all"):
        analyse(validation)


if __name__ == "__main__":
    main()
