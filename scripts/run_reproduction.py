"""Real-data reproduction of Lonappan, Keating & Arnold (2026) on public Planck PR4 maps.

Stages (each cached under data/work, rerun with --force STAGE):
    masks    build masks "0" and "30" from PLA products
    alms     E/B alms of the 8 split maps (and T of 353A/B for the dust angle psi_l) per mask
    spectra  16x16 field cross-spectra, binned 51 <= l < 1491, width 20
    transfer simulated pseudo-C_l transfer of each mask for the MK theory term
    analyse  pair angles, network, closures, MK fit, anchored beta, chi2_diff, dust EE/BB
             -> results/reproduction.json
    persistence  PD_0 statistics of 143 GHz temperature patches vs Gaussian sims
             -> results/persistence.json

Usage:
    python scripts/run_reproduction.py [--stages masks,alms,spectra,analyse] [--force STAGE] [--n-sims 200]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cmbcst import DETECTORS, FREQS, accel, network, spectra, theory  # noqa: E402
from cmbcst.data import path_of  # noqa: E402
from cmbcst.differential import pair_angles  # noqa: E402
from cmbcst.mk_likelihood import MKLikelihood, anchored_beta  # noqa: E402

WORK = REPO / "data" / "work"
RESULTS = REPO / "results"
LMAX = 1520
LMIN_BIN, LMAX_BIN, WIDTH = 51, 1491, 20
MASKS = ("0", "30")
GAUGE = DETECTORS.index("143A")
DEG = np.pi / 180


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------- stages
def stage_masks(force: bool) -> None:
    out = WORK / "masks.json"
    if out.exists() and not force:
        return
    from cmbcst.masks import build_masks

    m = build_masks(0.92)
    for k in MASKS:
        np.save(WORK / f"mask_{k}.npy", m[k].astype(np.float32))
    meta = {k: m[k] for k in ("co_column", "co_threshold", "galactic_mask_30", "fsky")}
    out.write_text(json.dumps(meta, indent=2))
    log(f"masks: {meta}")


def _mask(k: str) -> np.ndarray:
    return np.load(WORK / f"mask_{k}.npy").astype(np.float64)


def stage_alms(force: bool) -> None:
    import ducc0

    geom = ducc0.healpix.Healpix_Base(2048, "RING").sht_info()
    masks = {k: _mask(k) for k in MASKS}
    for det in DETECTORS:
        todo = [k for k in MASKS if force or not (WORK / f"alm_{det}_m{k}.npy").exists()]
        need_t = det.startswith("353") and (force or not all((WORK / f"almT_{det}_m{k}.npy").exists() for k in MASKS))
        if not todo and not need_t:
            continue
        t0 = time.time()
        q, u, good = spectra.load_qu(str(path_of(det)))
        for k in todo:
            np.save(WORK / f"alm_{det}_m{k}.npy", spectra.map2alm_pol(q, u, masks[k] * good, LMAX))
        del q, u
        if need_t:
            from astropy.io import fits

            with fits.open(path_of(det), memmap=True) as h:
                t = np.asarray(h[1].data["I_STOKES"], dtype=np.float64).ravel()
            t = spectra.nest_to_ring(np.where(np.abs(t) < 1e20, t * 1e6, 0.0), 2048)
            for k in MASKS:
                w = masks[k] * good
                t_m = (t - np.sum(t * w) / np.sum(w)) * w  # remove the masked-sky monopole first
                alm_t = ducc0.sht.experimental.adjoint_synthesis(map=t_m[None], spin=0, lmax=LMAX, mmax=LMAX,
                                                                 nthreads=spectra.NTHREADS, **geom)[0]
                np.save(WORK / f"almT_{det}_m{k}.npy", alm_t * (4 * np.pi / t.size))
        log(f"alms {det}: {time.time() - t0:.0f}s")


def stage_spectra(force: bool) -> None:
    for k in MASKS:
        out = WORK / f"spectra_m{k}.npz"
        if out.exists() and not force:
            continue
        alms = [np.load(WORK / f"alm_{d}_m{k}.npy") for d in DETECTORS]
        cl = spectra.spectra_matrix(alms, LMAX)
        w2, fsky_eff = spectra.mask_moments(_mask(k))
        b = spectra.bin_spectra(cl, LMIN_BIN, LMAX_BIN, WIDTH, w2, fsky_eff)
        ta, tb = (np.load(WORK / f"almT_353{s}_m{k}.npy") for s in "AB")
        ea, eb = alms[DETECTORS.index("353A")], alms[DETECTORS.index("353B")]
        # symmetrised 353A x 353B TE and TB (reference spectratheory.calculate_psi_ell)
        te = 0.5 * (spectra.cross_cl(ta, eb[0], LMAX) + spectra.cross_cl(tb, ea[0], LMAX))
        tbb = 0.5 * (spectra.cross_cl(ta, eb[1], LMAX) + spectra.cross_cl(tb, ea[1], LMAX))
        np.savez(out, cl=b["cl"], nu=b["nu"], ell=b["ell"], edges=b["edges"], w2=w2, fsky_eff=fsky_eff,
                 cl_ell=cl, te=te / w2, tb=tbb / w2)
        log(f"spectra mask {k}: w2={w2:.4f} fsky_eff={fsky_eff:.4f}")


# ---------------------------------------------------------------------------- analysis
def binned_mean(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.array([x[lo:hi].mean() for lo, hi in zip(edges[:-1], edges[1:])])


def gaussian_filter1d(x: np.ndarray, sigma: float) -> np.ndarray:
    """scipy.ndimage.gaussian_filter1d with mode='reflect', truncate=4 (used by the reference for psi_l)."""
    from scipy.ndimage import gaussian_filter1d as gf

    return gf(x, sigma)


_THEORY_CACHE: dict = {}


def theory_ell() -> dict[str, np.ndarray]:
    """Beam- and pixel-window-smoothed LCDM EE/BB per map pair, per ell (cached)."""
    if not _THEORY_CACHE:
        from cmbcst.beams import extract, split_windows

        extract()
        ee, bb = theory.lcdm_ee_bb(LMAX + 80)
        pw = theory.pixel_window(2048, LMAX, 1_000_000)
        wee, wbb, src = split_windows(LMAX)
        _THEORY_CACHE.update(ee=ee[: LMAX + 1] * wee * pw**2, bb=bb[: LMAX + 1] * wbb * pw**2, source=src)
    return _THEORY_CACHE


def theory_matrices(edges: np.ndarray, mask: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Binned beam-smoothed LCDM per pair; with `mask`, multiplied by that mask's simulated
    pseudo-C_l transfer (mode coupling + E->B leakage), so theory and data are both pseudo-spectra."""
    t = theory_ell()
    n = len(DETECTORS)
    th_ee = np.zeros((len(edges) - 1, n, n))
    th_bb = np.zeros_like(th_ee)
    for i in range(n):
        for j in range(n):
            th_ee[:, i, j] = binned_mean(t["ee"][i, j], edges)
            th_bb[:, i, j] = binned_mean(t["bb"][i, j], edges)
    if mask is not None:
        tr = np.load(WORK / f"transfer_m{mask}.npz")
        th_ee *= tr["t_ee"][:, None, None]
        th_bb *= tr["t_bb"][:, None, None]
    return th_ee, th_bb


def stage_transfer(force: bool, n_sims: int = 8) -> None:
    """Per-bin pseudo-C_l transfer of each mask, from LCDM skies smoothed with the 143Ax143B window.

    T^X_b = <binned pseudo C^X of masked sims, divided by w2> / binned true C^X (true = the same
    beam-smoothed input spectrum). Captures mode coupling from the point-source holes and E->B
    leakage, which the f_sky division alone does not.
    """
    import ducc0

    lsim = 2500
    ee, bb = theory.lcdm_ee_bb(lsim)
    i143a, i143b = DETECTORS.index("143A"), DETECTORS.index("143B")
    from cmbcst.beams import extract, split_windows

    extract()  # the transfer must use the same QuickPol windows as the likelihood, not the Gaussian fallback
    wee, wbb, src = split_windows(lsim)
    in_ee, in_bb = ee * wee[i143a, i143b], bb * wbb[i143a, i143b]
    geom = ducc0.healpix.Healpix_Base(2048, "RING").sht_info()
    edges = np.arange(LMIN_BIN, LMAX_BIN + 1, WIDTH)
    for k in MASKS:
        out = WORK / f"transfer_m{k}.npz"
        if out.exists() and not force:
            continue
        mask = _mask(k)
        w2, _ = spectra.mask_moments(mask)
        rng = np.random.default_rng(100 + int(k))
        acc_ee, acc_bb = np.zeros(LMAX + 1), np.zeros(LMAX + 1)
        for _ in range(n_sims):
            alm = np.stack([spectra.random_alm(in_ee, lsim, rng), spectra.random_alm(in_bb, lsim, rng)])
            q, u = ducc0.sht.experimental.synthesis(alm=alm, spin=2, lmax=lsim, mmax=lsim, nthreads=spectra.NTHREADS, **geom)
            a = spectra.map2alm_pol(q, u, mask, LMAX)
            acc_ee += spectra.cross_cl(a[0], a[0], LMAX) / w2
            acc_bb += spectra.cross_cl(a[1], a[1], LMAX) / w2
        t_ee = binned_mean(acc_ee / n_sims, edges) / binned_mean(in_ee[: LMAX + 1], edges)
        t_bb = binned_mean(acc_bb / n_sims, edges) / binned_mean(in_bb[: LMAX + 1], edges)
        np.savez(out, t_ee=t_ee, t_bb=t_bb, n_sims=n_sims, beam_source=src)
        log(f"transfer mask {k}: EE {t_ee[[0, 20, 40, 60]].round(3)} BB {t_bb[[0, 20, 40, 60]].round(3)}")


def pair_spectra_export(cl, nu, edges_k8, delta) -> dict:
    """D_ij, S_ij and the residual sigma per bin (uK^2) for the studio's per-pair panel."""
    from cmbcst.differential import pair_spectra, residual_cov

    t = np.tan(2 * delta)
    sig = np.sqrt(np.einsum("bee->be", residual_cov(cl, nu, edges_k8, t)))
    d, s = zip(*(pair_spectra(cl, i, j) for i, j in edges_k8))
    return {"D": np.array(d).tolist(), "S": np.array(s).tolist(), "sigma_r": sig.T.tolist()}


def beta_profile(lk: MKLikelihood, theta_best: np.ndarray, half_width_deg: float = 0.6, n: int = 9) -> dict:
    """Profile -2 ln L (ln det included) over beta with all other parameters refitted."""
    ib = lk.n
    free = [p for p in range(len(theta_best)) if p != ib]
    grid = theta_best[ib] + np.linspace(-half_width_deg, half_width_deg, n) * DEG
    th, vals = theta_best.copy(), []
    for b in grid:
        t0 = th.copy()
        t0[ib] = b
        f = lk.fit(theta0=t0, free=free)
        vals.append(f["neg2lnl"])
        th = f["theta"]
    vals = np.array(vals) - min(vals)
    c = np.polyfit(grid / DEG, vals, 2)
    return {"beta_deg": (grid / DEG).tolist(), "d_neg2lnl": vals.tolist(),
            "quadratic_beta_deg": float(-c[1] / (2 * c[0])), "quadratic_sigma_deg": float(1 / np.sqrt(c[0]))}


def run_estimators(cl, nu, th_ee, th_bb, psi, edges_k8, d0, d1, mk_start=None):
    """Pair angles, network and MK fit. `mk_start` warm-starts the MK fit (bootstrap draws start
    from the data solution; the converged point is the same to < 1e-9 rad)."""
    pa = pair_angles(cl, nu, edges_k8)
    net = network.solve_network(pa["delta"], pa["cov"], d0, GAUGE)
    mk = MKLikelihood(cl, nu, th_ee, th_bb, psi=psi, n_a=4).fit(theta0=mk_start)
    return pa, net, mk


def stage_analyse(n_sims: int) -> dict:
    edges_k8, tris, d0, d1 = network.complete_complex(8)
    beam_source = theory_ell()["source"]
    res: dict = {"generated_utc": datetime.now(timezone.utc).isoformat(), "detectors": list(DETECTORS),
                 "gauge": "143A", "binning": {"lmin": LMIN_BIN, "lmax_exclusive": LMAX_BIN, "width": WIDTH},
                 "beam_source": beam_source, "masks": json.loads((WORK / "masks.json").read_text()), "per_mask": {}}
    for k in MASKS:
        s = np.load(WORK / f"spectra_m{k}.npz")
        cl, nu, edges = s["cl"], s["nu"], s["edges"]
        th_ee, th_bb = theory_matrices(edges, mask=k)
        te, tb = binned_mean(s["te"], edges), binned_mean(s["tb"], edges)
        psi = 0.5 * np.arctan2(gaussian_filter1d(tb, 1.5), gaussian_filter1d(te, 1.5))
        log(f"mask {k}: estimators on data")
        pa, net, mk = run_estimators(cl, nu, th_ee, th_bb, psi, edges_k8, d0, d1)
        clo = network.closures(pa["delta"], pa["cov"], d1)
        c2, rank = network.closure_chi2(pa["delta"], pa["cov"], d1)
        q_rel = network.centering(8) @ net["alpha"]
        q_mk = network.centering(8) @ mk["alpha"]
        diff_analytic = network.chi2_diff(net["alpha"], mk["alpha"], net["cov"], mk["cov_alpha"])
        variants = {}
        for n_a in (0, 1):
            v = MKLikelihood(cl, nu, th_ee, th_bb, psi=psi, n_a=n_a).fit()
            variants[str(n_a)] = {"beta_deg": v["beta"] / DEG, "sigma_deg": v["sigma_beta"] / DEG}
        variants["4"] = {"beta_deg": mk["beta"] / DEG, "sigma_deg": mk["sigma_beta"] / DEG}
        profile = beta_profile(MKLikelihood(cl, nu, th_ee, th_bb, psi=psi, n_a=4), mk["theta"])

        # anchored beta: theta_m from an MK fit on the 100/143 maps with beta = 0, no dust term
        sub = [DETECTORS.index(d) for d in ("100A", "100B", "143A", "143B")]
        fidx = np.r_[[2 * i for i in sub], [2 * i + 1 for i in sub]]
        fidx.sort()
        lt = MKLikelihood(cl[:, fidx][:, :, fidx], nu, th_ee[:, sub][:, :, sub], th_bb[:, sub][:, :, sub])
        th_fit = lt.fit(free=np.arange(len(sub)))
        rec = {"fsky_w2": float(s["w2"]), "fsky_eff": float(s["fsky_eff"]),
               "pair_angles_deg": (pa["delta"] / DEG).tolist(),
               "pair_sigma_deg": (np.sqrt(np.diag(pa["cov"])) / DEG).tolist(),
               "pair_cov_deg2": (pa["cov"] / DEG**2).tolist(),
               "pairs": [[DETECTORS[i], DETECTORS[j]] for i, j in edges_k8],
               "network": {"alpha_deg": (net["alpha"] / DEG).tolist(),
                           "sigma_deg": (np.sqrt(np.diag(net["cov"])) / DEG).tolist(),
                           "chi2": net["chi2"], "dof": net["dof"], "pte": net["pte"],
                           "curl_sigma_deg": (np.sqrt(np.clip(np.diag(net["cov_curl"]), 0, None)) / DEG).tolist()},
               "closures": {"triangles": [[DETECTORS[a] for a in t] for t in tris],
                            "closure_deg": (clo["closure"] / DEG).tolist(), "sigma_deg": (clo["sigma"] / DEG).tolist(),
                            "z": clo["z"].tolist(), "chi2": c2, "rank": rank},
               "hodge": {**network.betti_numbers(d0, d1),
                         "gradient_norm_deg": float(np.linalg.norm(net["gradient"]) / DEG),
                         "curl_norm_deg": float(np.linalg.norm(net["curl"]) / DEG)},
               "mk": {"alpha_deg": (mk["alpha"] / DEG).tolist(),
                      "sigma_alpha_deg": (np.sqrt(np.diag(mk["cov_alpha"])) / DEG).tolist(),
                      "cov_alpha_deg2": (mk["cov_alpha"] / DEG**2).tolist(),
                      "sigma_rel_gauge_deg": (np.sqrt(np.maximum(np.diag(mk["cov_alpha"]) + mk["cov_alpha"][GAUGE, GAUGE]
                                                                  - 2 * mk["cov_alpha"][:, GAUGE], 0)) / DEG).tolist(),
                      "beta_deg": mk["beta"] / DEG, "sigma_beta_deg": mk["sigma_beta"] / DEG,
                      "A_dust": mk["A"].tolist(), "iterations": mk["iterations"],
                      "beta_by_dust_eb_amplitudes": variants, "beta_profile": profile},
               "psi_deg": (psi / DEG).tolist(), "ell": s["ell"].tolist(),
               "pearson_r_pairs": network.pearson_r(pa["delta"], d0 @ mk["alpha"]),
               "chi2_diff_no_cross": {"chi2": diff_analytic["chi2"], "dof": diff_analytic["dof"],
                                      "pte": diff_analytic["pte"]},
               "theta_100_143_deg": dict(zip(["100A", "100B", "143A", "143B"], (th_fit["alpha"] / DEG).tolist())),
               "theta_cov_deg2": (th_fit["cov_alpha"] / DEG**2).tolist(),
               "pair_spectra": pair_spectra_export(cl, nu, edges_k8, pa["delta"])}

        # Parametric bootstrap for C_rel, C_MK and their cross term (the paper keeps it)
        if k == "0" and n_sims:
            chk = accel.check_float32(cl, nu, seed=7)
            log(f"mask {k}: {n_sims} bootstrap draws on {chk['backend']}, float32 check {chk['max_rel_err']:.1e}")
            qs_rel, qs_mk, th_s, anc_s, unconverged = [], [], [], [], 0
            t_boot = time.perf_counter()
            for n_done, cls in enumerate(accel.wishart_draws(cl, nu, n_sims, seed=2026), 1):
                _, nets, mks = run_estimators(cls, nu, th_ee, th_bb, psi, edges_k8, d0, d1, mk_start=mk["theta"])
                unconverged += not mks["converged"]
                qs_rel.append(network.centering(8) @ nets["alpha"])
                qs_mk.append(network.centering(8) @ mks["alpha"])
                anc_s.append(mks["alpha"][sub])
                th_s.append(MKLikelihood(cls[:, fidx][:, :, fidx], nu, th_ee[:, sub][:, :, sub],
                                         th_bb[:, sub][:, :, sub]).fit(theta0=th_fit["theta"], free=np.arange(4))["alpha"])
                if n_done % 250 == 0:
                    log(f"  sims {n_done}/{n_sims}")
            qr, qm = np.array(qs_rel), np.array(qs_mk)
            c_rel, c_mk = np.cov(qr.T), np.cov(qm.T)
            c_x = (qr - qr.mean(0)).T @ (qm - qm.mean(0)) / (len(qr) - 1)
            diff = network.chi2_diff(q_rel, q_mk, c_rel, c_mk, c_x)
            rec["chi2_diff"] = {"chi2": diff["chi2"], "dof": diff["dof"], "pte": diff["pte"],
                                "pulls": dict(zip(DETECTORS, diff["pulls"].tolist())), "n_sims": n_sims}
            rec["bootstrap"] = {"n_draws": n_sims, "backend": chk["backend"], "float32_max_rel_err": chk["max_rel_err"],
                                "float32_tolerance": accel.REL_TOL, "unconverged_mk_fits": int(unconverged),
                                "seconds": round(time.perf_counter() - t_boot, 1)}
            rec["bootstrap_sigma_rel_deg"] = (np.sqrt(np.diag(c_rel)) / DEG).tolist()
            rec["bootstrap_sigma_mk_deg"] = (np.sqrt(np.diag(c_mk)) / DEG).tolist()
            ths, ancs = np.array(th_s), np.array(anc_s)
            # Cov(theta, alpha^MK) on shared data, for the anchored beta on the same mask
            res["_theta_anchor_cross"] = (ths - ths.mean(0)).T @ (ancs - ancs.mean(0)) / (len(ths) - 1)
        res["per_mask"][k] = rec
        res.setdefault("_mk_full", {})[k] = mk

    # anchored beta: anchor alpha^MK from mask "0" (100/143 maps), theta from each mask
    sub = [DETECTORS.index(d) for d in ("100A", "100B", "143A", "143B")]
    mk0 = res["_mk_full"]["0"]
    anchor, c_anchor = mk0["alpha"][sub], mk0["cov_alpha"][np.ix_(sub, sub)]
    c_cross0 = res.pop("_theta_anchor_cross", None)
    for k in MASKS:
        r = res["per_mask"][k]
        theta = np.array(list(r["theta_100_143_deg"].values())) * DEG
        c_theta = np.array(r["theta_cov_deg2"]) * DEG**2
        a = anchored_beta(theta, c_theta, anchor, c_anchor)
        r["anchored_beta_deg"] = a["beta"] / DEG
        r["anchored_sigma_deg"] = a["sigma"] / DEG
        r["anchored_per_map_deg"] = dict(zip(["100A", "100B", "143A", "143B"], (a["per_map"] / DEG).tolist()))
        if k == "0" and c_cross0 is not None:
            ax = anchored_beta(theta, c_theta, anchor, c_anchor, c_cross0)
            r["anchored_with_cross"] = {"beta_deg": ax["beta"] / DEG, "sigma_deg": ax["sigma"] / DEG}
    res.pop("_mk_full")
    dust_ratios(res)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "reproduction.json").write_text(json.dumps(res, indent=2))
    log("wrote results/reproduction.json")
    return res


def dust_ratios(res: dict) -> None:
    """Same-frequency A x B EE/BB at l < 300 (mask "0"), raw and with LCDM subtracted."""
    s = np.load(WORK / "spectra_m0.npz")
    t = theory_ell()
    ell = np.arange(LMAX + 1)
    sel = (ell >= LMIN_BIN) & (ell < 300)
    out = {}
    for f in FREQS:
        a, b = DETECTORS.index(f + "A"), DETECTORS.index(f + "B")
        cl = s["cl_ell"] / float(s["w2"])
        ee = cl[sel, 2 * a, 2 * b]
        bb = cl[sel, 2 * a + 1, 2 * b + 1]
        out[f] = {"raw": float(ee.sum() / bb.sum()),
                  "lcdm_subtracted": float((ee - t["ee"][a, b][sel]).sum() / (bb - t["bb"][a, b][sel]).sum())}
    res["dust_ee_over_bb_l_lt_300"] = out


PATCHES = {"NGP": (0.0, 0.0), "SGP": (np.pi, 0.0), "ColdSpot": (np.radians(147.0), np.radians(209.0))}


def stage_persistence(n_sims: int) -> dict:
    """PD_0 statistics of 20 deg patches of the 143 GHz (A+B)/2 temperature map vs Gaussian sims.

    The map is band-limited to l <= 1000 with a 10' Gaussian on top of the instrument beam and
    synthesised at N_side 512. Sims draw alm from the measured masked-sky spectrum of the same
    map (signal + noise), then go through the same smoothing, projection and statistics.
    """
    import ducc0
    from astropy.io import fits

    from cmbcst import persistence

    lmax, nside_lo = 1000, 512
    geom_hi = ducc0.healpix.Healpix_Base(2048, "RING").sht_info()
    geom_lo = ducc0.healpix.Healpix_Base(nside_lo, "RING").sht_info()
    mask = _mask("30")
    t = np.zeros(12 * 2048**2)
    for det in ("143A", "143B"):
        with fits.open(path_of(det), memmap=True) as h:
            x = np.asarray(h[1].data["I_STOKES"], dtype=np.float64).ravel()
        t += 0.5 * spectra.nest_to_ring(np.where(np.abs(x) < 1e20, x * 1e6, 0.0), 2048)
    t -= np.sum(t * mask) / mask.sum()

    def alm_of(m):
        return ducc0.sht.experimental.adjoint_synthesis(map=m[None], spin=0, lmax=lmax, mmax=lmax,
                                                        nthreads=spectra.NTHREADS, **geom_hi)[0] * (4 * np.pi / m.size)

    smooth = theory.gaussian_beam(10.0, lmax)
    ell_idx, _ = spectra.alm_ell_m(lmax)
    alm_full = alm_of(t)  # full sky: the patches sit at high latitude, the map is used as observed
    w2, _ = spectra.mask_moments(mask)
    cl_tt = spectra.cross_cl(alm_of(t * mask), alm_of(t * mask), lmax) / w2

    def summaries(alm):
        m = ducc0.sht.experimental.synthesis(alm=(alm * smooth[ell_idx])[None], spin=0, lmax=lmax, mmax=lmax,
                                             nthreads=spectra.NTHREADS, **geom_lo)[0]
        return {k: persistence.patch_summary(persistence.standardise(persistence.project_patch(m, *c)))
                for k, c in PATCHES.items()}

    real = summaries(alm_full)
    rng = np.random.default_rng(7)
    sims = [summaries(spectra.random_alm(cl_tt, lmax, rng)) for _ in range(n_sims)]

    def pval(patch, stat):
        v = np.array([s[patch][stat] for s in sims])
        r = real[patch][stat]
        return {"real": r, "sim_mean": float(v.mean()), "sim_std": float(v.std()),
                "p_two_sided": float(min(1.0, 2 * min((v <= r).mean(), (v >= r).mean())))}

    out = {"n_sims": n_sims, "map": "143 GHz (A+B)/2 temperature, 10' smoothing, l <= 1000, N_side 512",
           "patches_deg": {k: [float(np.degrees(c[0])), float(np.degrees(c[1]))] for k, c in PATCHES.items()},
           "stats": {p: {s: pval(p, s) for s in ("mean_life0", "max_life0", "n0", "min_value")} for p in PATCHES}}
    asym_real = real["NGP"]["mean_life0"] - real["SGP"]["mean_life0"]
    asym = np.array([s["NGP"]["mean_life0"] - s["SGP"]["mean_life0"] for s in sims])
    out["ngp_minus_sgp_mean_life0"] = {"real": asym_real, "sim_mean": float(asym.mean()), "sim_std": float(asym.std()),
                                       "p_two_sided": float(min(1.0, 2 * min((asym <= asym_real).mean(), (asym >= asym_real).mean())))}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "persistence.json").write_text(json.dumps(out, indent=2))
    log("wrote results/persistence.json")
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="masks,alms,spectra,transfer,analyse,persistence")
    ap.add_argument("--force", default="")
    ap.add_argument("--n-sims", type=int, default=2000)
    ap.add_argument("--n-persistence-sims", type=int, default=200)
    args = ap.parse_args(argv)
    WORK.mkdir(parents=True, exist_ok=True)
    stages, force = args.stages.split(","), set(args.force.split(","))
    if "masks" in stages:
        stage_masks("masks" in force)
    if "alms" in stages:
        stage_alms("alms" in force)
    if "spectra" in stages:
        stage_spectra("spectra" in force)
    if "transfer" in stages:
        stage_transfer("transfer" in force)
    if "analyse" in stages:
        stage_analyse(args.n_sims)
    if "persistence" in stages:
        stage_persistence(args.n_persistence_sims)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
