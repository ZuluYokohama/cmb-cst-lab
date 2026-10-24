"""Post-hoc robustness audit: crop Gaussian fields before periodic IAAFT matching.

This audit was motivated by the real-patch results. It cannot confirm a discovery.
It preserves both original runs, their thresholds, and every audit control.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

import run_copula_topology as experiment

ROOT = experiment.REPO
OUT = ROOT / "results/copula_boundary_audit_v1"
SEED = 87391232026


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((ROOT / "results/copula_topology_v2/plan.json").read_text())
    for path, digest in frozen["source_sha256"].items():
        if experiment.sha(ROOT / path) != digest:
            raise ValueError(f"original numerical source changed: {path}")
    plan = {"created_utc": experiment.now(), "scope": "Post-hoc boundary robustness check, not confirmation",
            "motivation": "Periodic 32x32 validation does not represent cropped open sky patches",
            "source_sha256": {**frozen["source_sha256"],
                              "scripts/audit_copula_boundaries.py": experiment.sha(__file__)},
            "seed": SEED, "fields": 64, "parent_size": 64, "center_crop_size": 32,
            "surrogates_per_field": 199, "iterations": 500,
            "construction": "Same smooth Gaussian torus spectrum as V2 controls; 64x64 parent cropped to central 32x32, then normal scores",
            "statistics": frozen["statistics"], "holm_family_size": 3,
            "flag_threshold": .05, "gross_rejection_if_flags_exceed": 8,
            "jovin_receipt_id": frozen["jovin_receipt_id"]}
    with (OUT / "plan.json").open("x") as stream:
        json.dump(plan, stream, indent=2)
    records = []
    for i in range(64):
        rng = np.random.default_rng(SEED + i)
        axis = np.fft.fftfreq(64)
        kx, ky = np.meshgrid(axis, axis)
        parent = np.fft.ifft2(np.fft.fft2(rng.normal(size=(64, 64))) * np.exp(-20*(kx*kx + ky*ky))).real
        field = experiment.normal_scores(parent[16:48, 16:48])
        sims, quality = experiment.ensemble(field, 199, SEED + 100000 + 1000*i, 199)
        observed, raw_p = experiment.compare(field, sims)
        adjusted = experiment.p.holm(raw_p)
        records.append({"index": i, "observed": observed.tolist(), "raw_p": raw_p,
                        "p_holm": adjusted.tolist(), "flag": bool(min(adjusted) <= .05),
                        "surrogate_statistics": sims.tolist(),
                        "quality_failures": sum(not q["passed"] for q in quality)})
        experiment.save(OUT / "progress.json", {"completed": i+1, "records": records})
        print(f"cropped Gaussian control {i+1}/64: flags {sum(r['flag'] for r in records)}", flush=True)
    flags = sum(r["flag"] for r in records)
    experiment.save(OUT / "result.json", {"completed_utc": experiment.now(),
                    "status": "failed" if flags > 8 else "passed", "familywise_flags": flags,
                    "fields": 64, "cutoff": 8, "plan_sha256": experiment.sha(OUT / "plan.json"),
                    "records": records,
                    "interpretation": "A failure shows the periodic-control calibration does not extend to cropped Gaussian fields; a pass alone does not establish real-sky calibration."})


if __name__ == "__main__":
    main()
