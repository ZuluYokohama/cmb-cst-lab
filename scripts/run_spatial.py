"""Stage A: region-resolved calibration network on Planck PR4, run blinded.

Stages (cached under data/work/spatial):
    spectra   per-region 16x16 field spectra (+ 353 TE/TB and a dust tracer) for both partitions.
              Pair angles are NOT computed here.
    validate  null and injection simulations built from the regional spectra with the
              antisymmetric part removed and the already-public global angles imposed, so no
              real regional pair angle is ever formed -> results/spatial_validation.json
    leakage   map-level synthetic skies (N_side 256) through the base12 regions: cross-region
              correlation of pair angles -> results/spatial_leakage.json
    freeze    write data/evidence/analysis_freeze.json (code hashes, statistics, thresholds)
    unblind   requires the freeze file and unchanged code; computes the real statistics
              -> results/spatial.json

Usage:
    python scripts/run_spatial.py --stages spectra,validate,leakage
    python scripts/run_spatial.py --stages freeze      # then commit the freeze file
    python scripts/run_spatial.py --stages unblind
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cmbcst import DETECTORS, accel, network, spatial, spectra  # noqa: E402
from cmbcst.data import path_of  # noqa: E402
from cmbcst.differential import pair_angles  # noqa: E402

WORK = REPO / "data" / "work" / "spatial"
RESULTS = REPO / "results"
FREEZE = REPO / "data" / "evidence" / "analysis_freeze.json"
LMAX, LMIN_BIN, LMAX_BIN, WIDTH = 1520, 51, 1491, 20
SCHEMES = ("base12", "lat8")
GAUGE = DETECTORS.index("143A")
DEG = np.pi / 180
EDGES, TRIS, D0, D1 = network.complete_complex(8)
SAME_FREQ = spatial.same_frequency_edges(list(DETECTORS), EDGES)
DETECTION_PTE = 0.01  # pre-registered: chi2_spatial PTE below this counts as non-stationary
FROZEN_FILES = ["src/cmbcst/spatial.py", "src/cmbcst/regions.py", "src/cmbcst/differential.py",
                "src/cmbcst/network.py", "src/cmbcst/spectra.py", "scripts/run_spatial.py"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------- spectra
def stage_spectra(force: bool) -> None:
    from astropy.io import fits

    from cmbcst.regions import region_ids, region_weight

    WORK.mkdir(parents=True, exist_ok=True)
    todo = [s for s in SCHEMES if force or not (WORK / f"{s}_meta.json").exists()]
    if not todo:
        return
    mask = np.load(REPO / "data" / "work" / "mask_0.npy").astype(np.float64)
    qu, good = [], np.ones(12 * 2048**2, dtype=bool)
    for det in DETECTORS:
        q, u, g = spectra.load_qu(str(path_of(det)))
        qu.append((q.astype(np.float32), u.astype(np.float32)))
        good &= g
    temps = {}
    for det in ("353A", "353B"):
        with fits.open(path_of(det), memmap=True) as h:
            t = np.asarray(h[1].data["I_STOKES"], dtype=np.float64).ravel()
        temps[det] = spectra.nest_to_ring(np.where(np.abs(t) < 1e20, t * 1e6, 0.0), 2048).astype(np.float32)
    log("maps loaded")
    import ducc0

    geom = ducc0.healpix.Healpix_Base(2048, "RING").sht_info()
    for scheme in todo:
        ids, labels = region_ids(scheme)
        meta = {"labels": labels, "regions": []}
        for k, lab in enumerate(labels):
            w = region_weight(ids, k, mask) * good
            w2, fsky_eff = spectra.mask_moments(w)
            alms = [spectra.map2alm_pol(q.astype(np.float64), u.astype(np.float64), w, LMAX) for q, u in qu]
            cl = spectra.spectra_matrix(alms, LMAX)
            b = spectra.bin_spectra(cl, LMIN_BIN, LMAX_BIN, WIDTH, w2, fsky_eff)
            talm = {}
            for det, t in temps.items():
                tw = (t.astype(np.float64) - np.sum(t * w) / np.sum(w)) * w
                talm[det] = ducc0.sht.experimental.adjoint_synthesis(
                    map=tw[None], spin=0, lmax=LMAX, mmax=LMAX, nthreads=spectra.NTHREADS, **geom)[0] * (4 * np.pi / tw.size)
            ea, eb = alms[DETECTORS.index("353A")], alms[DETECTORS.index("353B")]
            te = 0.5 * (spectra.cross_cl(talm["353A"], eb[0], LMAX) + spectra.cross_cl(talm["353B"], ea[0], LMAX)) / w2
            tb = 0.5 * (spectra.cross_cl(talm["353A"], eb[1], LMAX) + spectra.cross_cl(talm["353B"], ea[1], LMAX)) / w2
            i_a, i_b = DETECTORS.index("353A"), DETECTORS.index("353B")
            ell = np.arange(LMAX + 1)
            sel = (ell >= 51) & (ell < 151)
            dust = float(np.mean((cl[sel, 2 * i_a, 2 * i_b] + cl[sel, 2 * i_a + 1, 2 * i_b + 1]) / w2))
            np.savez(WORK / f"{scheme}_{k:02d}.npz", cl=b["cl"], nu=b["nu"], ell=b["ell"], te=te, tb=tb)
            meta["regions"].append({"label": lab, "fsky_w2": w2, "fsky_eff": fsky_eff, "dust_353_l51_150": dust})
            log(f"{scheme} {lab}: f_sky {w2:.4f}, dust {dust:.3g}")
        (WORK / f"{scheme}_meta.json").write_text(json.dumps(meta, indent=2))


def load_regions(scheme: str) -> tuple[list[dict], dict]:
    meta = json.loads((WORK / f"{scheme}_meta.json").read_text())
    regs = [dict(np.load(WORK / f"{scheme}_{k:02d}.npz")) for k in range(len(meta["labels"]))]
    return regs, meta


# ---------------------------------------------------------------------------- blinded simulations
def region_stats(region_cls: list[np.ndarray], nus: list[np.ndarray]) -> dict:
    deltas, covs = [], []
    for cl, nu in zip(region_cls, nus):
        pa = pair_angles(cl, nu, EDGES)
        deltas.append(pa["delta"] / DEG)
        covs.append(pa["cov"] / DEG**2)
    deltas, covs = np.array(deltas), np.array(covs)
    s = spatial.stationarity_split(deltas, covs, D0, GAUGE)
    sig = np.sqrt(np.einsum("ree->re", covs))
    ab = spatial.edge_constancy(deltas, sig, SAME_FREQ)
    return {"split": s, "ab": ab, "deltas": deltas, "covs": covs}


def stage_validate(n_null: int, n_inj: int) -> None:
    alpha_g = np.array(json.loads((RESULTS / "reproduction.json").read_text())["per_mask"]["0"]["network"]["alpha_deg"])
    out = {"alpha_global_deg": alpha_g.tolist(), "schemes": {}}
    for scheme in SCHEMES:
        regs, meta = load_regions(scheme)
        nulls = [spatial.null_spectra(r["cl"], alpha_g) for r in regs]
        nus = [r["nu"] for r in regs]
        dust = np.array([m["dust_353_l51_150"] for m in meta["regions"]])
        gens = [accel.wishart_draws(c, n, n_null + 4 * n_inj, seed=300 + k) for k, (c, n) in enumerate(zip(nulls, nus))]
        t0 = time.perf_counter()
        sp, cl_, ab = [], [], []
        for s in range(n_null):
            st = region_stats([next(g) for g in gens], nus)
            sp.append(st["split"]["chi2_spatial"])
            cl_.append(st["split"]["chi2_closure"])
            ab.append(st["ab"]["chi2"])
        r_count = len(regs)
        rec = {"n_regions": r_count, "labels": meta["labels"], "n_null": n_null,
               "dof_spatial": 7 * (r_count - 1), "dof_closure": 21 * r_count, "dof_ab": 4 * (r_count - 1),
               "null_mean": {"spatial": float(np.mean(sp)), "closure": float(np.mean(cl_)), "ab": float(np.mean(ab))},
               "null_pte_spatial": [network.chi2_sf(x, 7 * (r_count - 1)) for x in sp],
               "null_seconds": round(time.perf_counter() - t0, 1)}
        # power: dust-like (353 and 217 drift with the regional dust tracer) and instrument-like
        tr = (dust - dust.mean()) / dust.std()
        power = {}
        for kind in ("dust_like", "instrument_like"):
            curve = []
            for amp in (0.05, 0.1, 0.2, 0.4):
                hits = 0
                rng = np.random.default_rng(int(amp * 1000) + (0 if kind == "dust_like" else 7))
                for s in range(n_inj // 4 if n_inj >= 4 else 1):
                    if kind == "dust_like":
                        weights = np.array([0, 0, 0, 0, 0.3, 0.3, 1, 1])
                        gam = np.outer(tr, weights) * amp
                    else:
                        gam = rng.normal(0, amp, (r_count, 8))
                    cls = [spatial.inject(next(g), gam[k]) for k, g in enumerate(gens)]
                    st = region_stats(cls, nus)
                    hits += network.chi2_sf(st["split"]["chi2_spatial"], 7 * (r_count - 1)) < DETECTION_PTE
                curve.append({"amp_deg": amp, "detection_rate": hits / max(1, n_inj // 4)})
            power[kind] = curve
        rec["power"] = power
        out["schemes"][scheme] = rec
        log(f"validate {scheme}: null mean chi2_spatial {rec['null_mean']['spatial']:.1f} (dof {rec['dof_spatial']}), "
            f"closure {rec['null_mean']['closure']:.1f} (dof {rec['dof_closure']}); power {power}")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "spatial_validation.json").write_text(json.dumps(out, indent=2))


# ---------------------------------------------------------------------------- map-level leakage
def stage_leakage(n_sims: int) -> None:
    """Cross-region correlation of pair angles from map-level synthetic skies (N_side 256)."""
    import ducc0

    from cmbcst import theory

    nside, lmax = 256, 600
    ee, bb = theory.lcdm_ee_bb(lmax)
    hb = ducc0.healpix.Healpix_Base(nside, "RING")
    ang = hb.pix2ang(np.arange(12 * nside**2))
    ids = ducc0.healpix.Healpix_Base(1, "RING").ang2pix(ang)
    noise = np.array([2.0, 2.0, 1.0, 1.0, 1.5, 1.5, 4.0, 4.0]) * 0.02
    rng = np.random.default_rng(11)
    alpha = np.array(json.loads((RESULTS / "reproduction.json").read_text())["per_mask"]["0"]["network"]["alpha_deg"]) * DEG
    deltas = []
    for s in range(n_sims):
        e_alm = spectra.random_alm(ee, lmax, rng)
        b_alm = spectra.random_alm(bb, lmax, rng)
        q0, u0 = spectra.alm2map_pol(np.stack([e_alm, b_alm]), nside, lmax)
        maps = []
        for i, a in enumerate(alpha):
            q, u = spectra.rotate_qu(q0, u0, a)
            sig = noise[i] * np.sqrt(12 * nside**2 / (4 * np.pi))
            maps.append((q + sig * rng.standard_normal(q.size), u + sig * rng.standard_normal(u.size)))
        per_region = []
        for k in range(12):
            w = (ids == k).astype(float)
            alms = [spectra.map2alm_pol(q, u, w, lmax) for q, u in maps]
            w2, fe = spectra.mask_moments(w)
            b = spectra.bin_spectra(spectra.spectra_matrix(alms, lmax), 51, 551, 20, w2, fe)
            per_region.append(pair_angles(b["cl"], b["nu"], EDGES)["delta"] / DEG)
        deltas.append(np.array(per_region))
        log(f"leakage sim {s + 1}/{n_sims}")
    d = np.array(deltas)  # (sims, regions, edges)
    corr = np.zeros((12, 12))
    for e in range(28):
        corr += np.corrcoef(d[:, :, e].T) / 28
    off = corr[~np.eye(12, dtype=bool)]
    adj = [(i, j) for i in range(12) for j in range(i + 1, 12)]
    out = {"n_sims": n_sims, "nside": nside, "mean_corr_offdiag": float(off.mean()),
           "max_abs_corr_offdiag": float(np.max(np.abs(off))), "sd_expected_noise": float(1 / np.sqrt(n_sims)),
           "corr_matrix_edge_averaged": corr.tolist(), "n_pairs": len(adj)}
    (RESULTS / "spatial_leakage.json").write_text(json.dumps(out, indent=2))
    log(f"leakage: mean off-diagonal corr {out['mean_corr_offdiag']:+.3f}, max |corr| {out['max_abs_corr_offdiag']:.3f} "
        f"(noise floor ~{out['sd_expected_noise']:.2f})")


# ---------------------------------------------------------------------------- freeze / unblind
def _hashes() -> dict[str, str]:
    return {f: hashlib.sha256((REPO / f).read_bytes()).hexdigest() for f in FROZEN_FILES}


def stage_freeze() -> None:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    freeze = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(), "git_head_at_freeze": head,
        "file_sha256": _hashes(),
        "statistics": {
            "chi2_spatial": "stationarity of the per-map calibration pattern across regions, dof 7(R-1)",
            "chi2_closure": "triangle closure within regions, dof 21R",
            "chi2_ab": "constancy of the 4 same-frequency A/B pair angles across regions, dof 4(R-1)",
            "tracer_regression": "per-map slope of gamma_r against the regional 353 GHz dust tracer, dof 7",
        },
        "schemes": list(SCHEMES), "detection_pte": DETECTION_PTE,
        "primary": "base12 chi2_spatial; lat8 is a robustness partition",
        "validation_files": ["results/spatial_validation.json", "results/spatial_leakage.json"],
    }
    FREEZE.write_text(json.dumps(freeze, indent=2))
    log(f"wrote {FREEZE.relative_to(REPO)}; commit it before --stages unblind")


def stage_unblind() -> None:
    if not FREEZE.exists():
        raise SystemExit("no analysis_freeze.json: run --stages freeze and commit it first")
    freeze = json.loads(FREEZE.read_text())
    changed = [f for f, h in _hashes().items() if freeze["file_sha256"].get(f) != h]
    if changed:
        raise SystemExit(f"code changed since freeze: {changed}")
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(FREEZE.relative_to(REPO))], cwd=REPO,
                             capture_output=True, text=True).returncode == 0
    if not tracked:
        raise SystemExit("analysis_freeze.json is not committed")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    out = {"unblinded_utc": datetime.now(timezone.utc).isoformat(), "git_head": head,
           "freeze_utc": freeze["frozen_utc"], "detection_pte": DETECTION_PTE, "schemes": {}}
    for scheme in SCHEMES:
        regs, meta = load_regions(scheme)
        st = region_stats([r["cl"] for r in regs], [r["nu"] for r in regs])
        s, ab = st["split"], st["ab"]
        dust = np.array([m["dust_353_l51_150"] for m in meta["regions"]])
        reg = spatial.tracer_regression(s["gamma"], s["fisher_region"], np.log(dust), GAUGE)
        r_count = len(regs)
        out["schemes"][scheme] = {
            "labels": meta["labels"], "fsky": [m["fsky_w2"] for m in meta["regions"]], "dust_tracer": dust.tolist(),
            "chi2_spatial": s["chi2_spatial"], "dof_spatial": s["dof_spatial"], "pte_spatial": network.chi2_sf(s["chi2_spatial"], s["dof_spatial"]),
            "chi2_closure": s["chi2_closure"], "dof_closure": s["dof_closure"], "pte_closure": network.chi2_sf(s["chi2_closure"], s["dof_closure"]),
            "chi2_ab": ab["chi2"], "dof_ab": ab["dof"], "pte_ab": network.chi2_sf(ab["chi2"], ab["dof"]),
            "chi2_spatial_region": s["chi2_spatial_region"].tolist(), "chi2_closure_region": s["chi2_closure_region"].tolist(),
            "alpha_global_deg": s["alpha_global"].tolist(), "alpha_region_deg": s["alpha_region"].tolist(),
            "gamma_deg": s["gamma"].tolist(),
            "gamma_sigma_deg": [np.insert(np.sqrt(np.diag(np.linalg.inv(f))), GAUGE, 0.0).tolist() for f in s["fisher_region"]],
            "tracer_slope_deg_per_ln": reg["slope"].tolist(), "tracer_slope_sigma": np.insert(np.sqrt(np.diag(reg["slope_cov"])), GAUGE, 0.0).tolist(),
            "tracer_delta_chi2": reg["delta_chi2"], "tracer_dof": reg["dof"], "tracer_pte": network.chi2_sf(reg["delta_chi2"], reg["dof"]),
            "ab_mean_per_edge_deg": ab["mean_per_edge"], "ab_chi2_per_edge": ab["chi2_per_edge"],
            "n_regions": r_count,
        }
        o = out["schemes"][scheme]
        log(f"UNBLIND {scheme}: spatial {o['chi2_spatial']:.1f}/{o['dof_spatial']} PTE {o['pte_spatial']:.3g}; "
            f"closure {o['chi2_closure']:.1f}/{o['dof_closure']} PTE {o['pte_closure']:.3g}; AB {o['chi2_ab']:.1f}/{o['dof_ab']} "
            f"PTE {o['pte_ab']:.3g}; tracer dchi2 {o['tracer_delta_chi2']:.1f}/7 PTE {o['tracer_pte']:.3g}")
    (RESULTS / "spatial.json").write_text(json.dumps(out, indent=2))
    with open(REPO / "data" / "evidence" / "unblind_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"stage": "A", "utc": out["unblinded_utc"], "git_head": head, "freeze_utc": freeze["frozen_utc"]}) + "\n")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="spectra,validate,leakage")
    ap.add_argument("--force", default="")
    ap.add_argument("--n-null", type=int, default=500)
    ap.add_argument("--n-inj", type=int, default=200)
    ap.add_argument("--n-leak", type=int, default=40)
    args = ap.parse_args(argv)
    stages = args.stages.split(",")
    if "spectra" in stages:
        stage_spectra("spectra" in args.force)
    if "validate" in stages:
        stage_validate(args.n_null, args.n_inj)
    if "leakage" in stages:
        stage_leakage(args.n_leak)
    if "freeze" in stages:
        stage_freeze()
    if "unblind" in stages:
        stage_unblind()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
