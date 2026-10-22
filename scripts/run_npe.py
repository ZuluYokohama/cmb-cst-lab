"""Stage B: dust-robust birefringence by neural posterior estimation, run blinded.

Stages (cached under data/work/npe):
    build     fixed inputs for the simulator from public pieces: LCDM x QuickPol windows x mask
              transfer, 353 GHz A x B dust spectra, noise (auto - cross), window ratios, nu_b.
              A goodness check on EE/BB only (no EB, which carries beta).
    simulate  N parameter draws from the prior -> exact Wishart bandpowers -> 14 score summaries.
    train     an ensemble of conditional MAFs on (beta, psi0, n_psi, psi1, A_d, beta_d | scores),
              on the Arc GPU when present.
    validate  simulation-based calibration on held-out draws, per dust sub-model (M0..M3) and an
              out-of-family dust model; MK (no dust EB) on the same draws for comparison
              -> results/npe_validation.json
    attest    device log-densities for fixed inputs written into the sandbox suite
    freeze    data/evidence/analysis_freeze_npe.json
    unblind   scores of the real spectra -> posterior -> results/npe.json

Usage:
    python scripts/run_npe.py --stages build,simulate,train,validate,attest
    python scripts/run_npe.py --stages freeze      # commit it
    python scripts/run_npe.py --stages unblind
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from cmbcst import DETECTORS, npe, simulator  # noqa: E402

WORK = REPO / "data" / "work" / "npe"
RESULTS = REPO / "results"
FREEZE = REPO / "data" / "evidence" / "analysis_freeze_npe.json"
TESTS = REPO / "tests" / "sandbox" / "jovin_tests_npe.py"
TARGET = slice(8, 14)                       # beta, psi0, n_psi, psi1, A_d, beta_d
TARGET_NAMES = simulator.PARAMS[8:14]
HIDDEN, LAYERS, ENSEMBLE = 64, 4, 5
FROZEN_FILES = ["src/cmbcst/simulator.py", "src/cmbcst/npe.py", "src/cmbcst/npe_torch.py", "scripts/run_npe.py"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------- prior
def alpha_global() -> np.ndarray:
    return np.array(json.loads((RESULTS / "reproduction.json").read_text())["per_mask"]["0"]["network"]["alpha_deg"])


def prior_draw(rng: np.random.Generator, n: int, model: str = "full") -> np.ndarray:
    """Broad priors (degrees). alpha = public relative pattern + a common offset c + per-map
    freedom. The common offset must be broad: it is exactly degenerate with beta, so a narrow one
    would let the prior, not the data, fix beta (the network pattern has 143A = 0 by gauge only)."""
    th = np.empty((n, 14))
    common = rng.uniform(-3.0, 3.0, (n, 1))
    th[:, :8] = alpha_global()[None] + common + rng.uniform(-0.5, 0.5, (n, 8))
    th[:, 8] = rng.uniform(-1.5, 1.5, n)            # beta
    th[:, 9] = rng.uniform(-5.0, 5.0, n)            # psi0
    th[:, 10] = rng.uniform(-1.0, 1.0, n)           # n_psi
    th[:, 11] = rng.uniform(-3.0, 3.0, n)           # psi1
    th[:, 12] = rng.uniform(0.7, 1.3, n)            # A_d
    th[:, 13] = rng.normal(1.53, 0.05, n)           # beta_d
    if model == "M0":
        th[:, 9] = th[:, 11] = 0.0
    elif model == "M1":
        th[:, 10] = th[:, 11] = 0.0
    elif model == "M2":
        th[:, 11] = 0.0
    elif model == "M3":
        th[:, 10] = 0.0
    return th


# ---------------------------------------------------------------------------- build
def FIDUCIAL() -> np.ndarray:
    """Compression point. psi0 = 1 deg (not 0): at psi0 = 0 the n_psi score vanishes identically."""
    return np.r_[alpha_global(), 0.3, 1.0, 0.0, 0.0, 1.0, 1.53]


def stage_build() -> None:
    import run_reproduction as R
    from cmbcst.beams import split_windows

    WORK.mkdir(parents=True, exist_ok=True)
    s = np.load(REPO / "data" / "work" / "spectra_m0.npz")
    cl, nu, edges, ell = s["cl"], s["nu"], s["edges"], s["ell"]
    th_ee, th_bb = R.theory_matrices(edges, mask="0")
    ia, ib = DETECTORS.index("353A"), DETECTORS.index("353B")
    dust_ee = np.clip(cl[:, 2 * ia, 2 * ib] - th_ee[:, ia, ib], 1e-8, None)
    dust_bb = np.clip(cl[:, 2 * ia + 1, 2 * ib + 1] - th_bb[:, ia, ib], 1e-8, None)
    wee, _, _ = split_windows(R.LMAX)
    wb = np.array([[R.binned_mean(wee[i, j], edges) for j in range(8)] for i in range(8)]).transpose(2, 0, 1)
    wratio = wb / wb[:, ia, ib][:, None, None]
    partner = [1, 0, 3, 2, 5, 4, 7, 6]
    noise_ee = np.array([np.clip(cl[:, 2 * i, 2 * i] - cl[:, 2 * i, 2 * partner[i]], 0, None) for i in range(8)]).T
    noise_bb = np.array([np.clip(cl[:, 2 * i + 1, 2 * i + 1] - cl[:, 2 * i + 1, 2 * partner[i] + 1], 0, None) for i in range(8)]).T
    fixed = {"th_ee": th_ee, "th_bb": th_bb, "dust_ee": dust_ee, "dust_bb": dust_bb, "wratio": wratio,
             "noise_ee": noise_ee, "noise_bb": noise_bb, "ell": ell}
    np.savez(WORK / "fixed.npz", nu=nu, **fixed)
    # EE/BB-only goodness of the fiducial model (EB/BE untouched: they carry beta)
    sig = simulator.model_covariance(FIDUCIAL(), fixed)
    i8 = np.arange(8)
    ratio_ee = cl[:, 2 * i8[:, None], 2 * i8[None, :]] / sig[:, 2 * i8[:, None], 2 * i8[None, :]]
    ratio_bb = cl[:, 2 * i8[:, None] + 1, 2 * i8[None, :] + 1] / sig[:, 2 * i8[:, None] + 1, 2 * i8[None, :] + 1]
    good = {"ee_median_ratio": float(np.median(ratio_ee)), "bb_median_ratio": float(np.median(ratio_bb)),
            "ee_ratio_iqr": np.percentile(ratio_ee, [25, 75]).tolist(), "bb_ratio_iqr": np.percentile(ratio_bb, [25, 75]).tolist()}
    (WORK / "fixed_goodness.json").write_text(json.dumps(good, indent=2))
    log(f"build: data/model EE {good['ee_median_ratio']:.3f} {good['ee_ratio_iqr']}, BB {good['bb_median_ratio']:.3f} {good['bb_ratio_iqr']}")


def load_fixed() -> tuple[dict, np.ndarray]:
    f = dict(np.load(WORK / "fixed.npz"))
    nu = f.pop("nu")
    return f, nu


_COMP = None


def _compressor():
    global _COMP
    if _COMP is None:
        fixed, nu = load_fixed()
        th0 = FIDUCIAL()
        steps = np.r_[np.full(8, 0.01), 0.01, 0.1, 0.02, 0.1, 0.01, 0.01]
        _COMP = (simulator.ScoreCompressor(th0, fixed, np.round(nu), steps), fixed, np.round(nu))
    return _COMP


def _simulate_chunk(args):
    seed, n, model = args
    comp, fixed, dof = _compressor()
    rng = np.random.default_rng(seed)
    th = prior_draw(rng, n, model)
    t = np.empty((n, 14))
    for k in range(n):
        sig = simulator.model_covariance(th[k], fixed)
        d = simulator.bartlett_draws(sig, dof, 1, rng)[0]
        t[k] = comp(simulator.upper(d))
    return th, t


def simulate(n: int, seed: int, model: str = "full", workers: int = 8) -> tuple[np.ndarray, np.ndarray]:
    chunk = 2000
    jobs = [(seed * 100000 + k, min(chunk, n - k * chunk), model) for k in range((n + chunk - 1) // chunk)]
    with Pool(workers) as pool:
        parts = pool.map(_simulate_chunk, jobs)
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def stage_simulate(n: int) -> None:
    t0 = time.perf_counter()
    th, t = simulate(n, seed=1)
    np.savez(WORK / "train.npz", theta=th, t=t)
    log(f"simulate: {n} draws in {time.perf_counter() - t0:.0f}s")


# ---------------------------------------------------------------------------- train
def standardisers(th: np.ndarray, t: np.ndarray) -> dict:
    x = th[:, TARGET]
    c_sd = t.std(0)
    if np.any(c_sd == 0) or not np.all(np.isfinite(t)):
        raise ValueError(f"degenerate summaries: zero-variance columns {np.where(c_sd == 0)[0].tolist()}")
    return {"x_mu": x.mean(0), "x_sd": x.std(0), "c_mu": t.mean(0), "c_sd": c_sd}


def stage_train(epochs: int) -> None:
    from cmbcst import npe_torch

    d = np.load(WORK / "train.npz")
    th, t = d["theta"], d["t"]
    st = standardisers(th, t)
    x = (th[:, TARGET] - st["x_mu"]) / st["x_sd"]
    c = (t - st["c_mu"]) / st["c_sd"]
    infos = []
    for seed in range(ENSEMBLE):
        model, info = npe_torch.train(x, c, seed=seed, hidden=HIDDEN, n_layers=LAYERS, epochs=epochs, log=log)
        np.savez(WORK / f"maf_{seed}.npz", **{f"l{k}_{n}": v for k, layer in enumerate(model.export()) for n, v in layer.items()})
        infos.append({k: info[k] for k in ("val_nll", "device", "epochs_run")})
        log(f"train seed {seed}: val NLL {info['val_nll']:.4f} on {info['device']}")
    np.savez(WORK / "standardisers.npz", **st)
    (WORK / "train_info.json").write_text(json.dumps(infos, indent=2))


def load_ensemble():
    import torch

    from cmbcst import npe_torch

    st = dict(np.load(WORK / "standardisers.npz"))
    models = []
    for seed in range(ENSEMBLE):
        w = np.load(WORK / f"maf_{seed}.npz")
        m = npe_torch.MAF(6, 14, HIDDEN, LAYERS)
        for k, layer in enumerate(m.layers):
            for n in ("W1", "V1", "b1", "W2", "b2", "W3", "b3"):
                getattr(layer, n).data = torch.tensor(w[f"l{k}_{n}"], dtype=torch.float32)
        models.append(m.to(npe_torch.device()).eval())
    return models, st


def posterior_samples(models, st, t: np.ndarray, n_per_model: int) -> np.ndarray:
    """Ensemble mixture samples in physical units: (rows, ENSEMBLE*n_per_model, 6)."""
    import torch

    from cmbcst import npe_torch

    c = torch.tensor((t - st["c_mu"]) / st["c_sd"], dtype=torch.float32, device=npe_torch.device())
    out = [m.sample(c, n_per_model).cpu().numpy() for m in models]
    s = np.concatenate(out, axis=1)
    return s * st["x_sd"] + st["x_mu"]


# ---------------------------------------------------------------------------- validate
def mk_beta(th_true: np.ndarray, rng: np.random.Generator) -> float:
    """MK fit (no dust-EB term) on one simulated bandpower set, beta in degrees."""
    from cmbcst.mk_likelihood import MKLikelihood

    fixed, nu = load_fixed()
    sig = simulator.model_covariance(th_true, fixed)
    d = simulator.bartlett_draws(sig, np.round(nu), 1, rng)[0]
    f = MKLikelihood(d, nu, fixed["th_ee"], fixed["th_bb"]).fit()
    return float(np.degrees(f["beta"])), float(np.degrees(f["sigma_beta"]))


def stage_validate(n_test: int, n_post: int, n_mk: int) -> None:
    models, st = load_ensemble()
    out = {"target": list(TARGET_NAMES), "n_test": n_test, "n_posterior": n_post * ENSEMBLE, "models": {}}
    for model in ("full", "M0", "M1", "M2", "M3"):
        th, t = simulate(n_test, seed={"full": 51, "M0": 52, "M1": 53, "M2": 54, "M3": 55}[model], model=model)
        s = posterior_samples(models, st, t, n_post)
        ranks = npe.sbc_ranks(s, th[:, TARGET])
        u = npe.rank_uniformity(ranks, s.shape[1])
        beta_mean, beta_sd = s[:, :, 0].mean(1), s[:, :, 0].std(1)
        rec = {"rank_chi2": u["chi2"].tolist(), "rank_dof": u["dof"], "coverage68": u["coverage68"].tolist(),
               "coverage95": u["coverage95"].tolist(), "beta_bias_mean": float(np.mean(beta_mean - th[:, 8])),
               "beta_sd_median": float(np.median(beta_sd)), "beta_pull_sd": float(np.std((beta_mean - th[:, 8]) / beta_sd)),
               "ranks_beta": ranks[:, 0].tolist()}
        # The identifiable combination: dust looks rotated by alpha + psi, CMB by alpha + beta, so the
        # data constrain beta - psi0 (psi at the pivot); beta alone needs a prior on psi.
        dd = s[:, :, 0] - s[:, :, 1]
        dtrue = th[:, 8] - th[:, 9]
        ud = npe.rank_uniformity(npe.sbc_ranks(dd[:, :, None], dtrue[:, None]), s.shape[1])
        rec["beta_minus_psi0"] = {"coverage68": float(ud["coverage68"][0]), "coverage95": float(ud["coverage95"][0]),
                                  "rank_chi2": float(ud["chi2"][0]), "sd_median": float(np.median(dd.std(1))),
                                  "bias_mean": float(np.mean(dd.mean(1) - dtrue))}
        if model in ("M0", "M1", "M2") and n_mk:
            rng = np.random.default_rng(7)
            mk = np.array([mk_beta(th[k], rng) for k in range(n_mk)])
            err = mk[:, 0] - th[:n_mk, 8]
            common = th[:n_mk, :8].mean(1) - alpha_global().mean()      # true common miscalibration
            slope, icpt = np.polyfit(common, err, 1)
            small = np.abs(common) < 0.5
            rec["mk_no_dust_eb"] = {"beta_bias_mean": float(np.mean(err)),
                                    "beta_pull_sd": float(np.std(err / mk[:, 1])),
                                    "coverage68": float(np.mean(np.abs(err) < mk[:, 1])),
                                    # errors-in-variables attenuation: beta error grows with the absolute offset
                                    "beta_err_vs_common_slope": float(slope), "beta_err_intercept": float(icpt),
                                    "coverage68_abs_common_lt_0p5": float(np.mean(np.abs(err[small]) < mk[small, 1])) if small.any() else None,
                                    "n_abs_common_lt_0p5": int(small.sum())}
        out["models"][model] = rec
        log(f"validate {model}: beta coverage68 {rec['coverage68'][0]:.3f} 95 {rec['coverage95'][0]:.3f}, "
            f"bias {rec['beta_bias_mean']:+.4f}, sd {rec['beta_sd_median']:.3f}"
            + (f" | MK(no dust EB) bias {rec['mk_no_dust_eb']['beta_bias_mean']:+.3f} cov68 {rec['mk_no_dust_eb']['coverage68']:.2f}" if "mk_no_dust_eb" in rec else ""))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "npe_validation.json").write_text(json.dumps(out, indent=2))


# ---------------------------------------------------------------------------- attest
def stage_attest() -> None:
    import torch

    from cmbcst import npe_torch

    w = np.load(WORK / "maf_0.npz")
    layers = [{n: w[f"l{k}_{n}"] for n in ("W1", "V1", "b1", "W2", "b2", "W3", "b3")} for k in range(LAYERS)]
    models, st = load_ensemble()
    # In-distribution inputs: standardised (target, summary) pairs from the simulations
    d = np.load(WORK / "train.npz")
    pick = np.random.default_rng(3).choice(d["theta"].shape[0], 16, replace=False)
    x = ((d["theta"][pick][:, TARGET] - st["x_mu"]) / st["x_sd"]).astype(np.float32)
    c = ((d["t"][pick] - st["c_mu"]) / st["c_sd"]).astype(np.float32)
    dev = npe_torch.device()
    lp = models[0].log_prob(torch.tensor(x, device=dev), torch.tensor(c, device=dev)).detach().cpu().numpy()
    blob = base64.b64encode(np.concatenate([layers[k][n].astype(np.float32).ravel() for k in range(LAYERS)
                                            for n in ("W1", "V1", "b1", "W2", "b2", "W3", "b3")]).tobytes()).decode()
    att = {"device": str(dev), "hidden": HIDDEN, "layers": LAYERS, "d": 6, "k": 14,
           "x": x.astype(float).tolist(), "c": c.astype(float).tolist(), "log_prob_device": lp.astype(float).tolist(),
           "weights_float32_b64": blob, "rel_tolerance": 1e-5}
    text = TESTS.read_text(encoding="utf-8")
    a, b = "# ATTESTATION:START", "# ATTESTATION:END"
    i, j = text.index(a) + len(a), text.index(b)
    TESTS.write_text(text[:i] + "\nATTESTATION = json.loads(" + repr(json.dumps(att, separators=(",", ":"))) + ")\n" + text[j:], encoding="utf-8")
    log(f"attest: {len(blob) / 1024:.0f} KB of weights, device {dev}")


# ---------------------------------------------------------------------------- freeze / unblind
def _sha(path: Path) -> str:
    """SHA-256 with line endings normalised to LF (text files match a fresh clone too)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _hashes() -> dict[str, str]:
    """Code, the trained ensemble and every input the unblind step reads: the real mask-0 spectra,
    the simulator inputs, the standardisers, the public global angles and the validation output."""
    out = {f: _sha(REPO / f) for f in FROZEN_FILES}
    inputs = [WORK / f"maf_{seed}.npz" for seed in range(ENSEMBLE)]
    inputs += [WORK / "fixed.npz", WORK / "standardisers.npz", WORK / "train_info.json",
               REPO / "data" / "work" / "spectra_m0.npz", RESULTS / "reproduction.json", RESULTS / "npe_validation.json"]
    out.update({f.relative_to(REPO).as_posix(): _sha(f) for f in inputs})
    return out


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, check=False)


def stage_freeze() -> None:
    head = _git("rev-parse", "HEAD").stdout.decode().strip()
    FREEZE.write_text(json.dumps({"frozen_utc": datetime.now(timezone.utc).isoformat(), "git_head_at_freeze": head,
                                  "sha256": _hashes(), "target": list(TARGET_NAMES),
                                  "report": "beta posterior median and 68/95% intervals, ensemble of 5 MAFs",
                                  "validation_file": "results/npe_validation.json"}, indent=2))
    log("wrote analysis_freeze_npe.json; commit it before unblinding")


def stage_unblind() -> None:
    if not FREEZE.exists():
        raise SystemExit("no analysis_freeze_npe.json")
    fr = json.loads(FREEZE.read_text())
    changed = [f for f, h in _hashes().items() if fr["sha256"].get(f) != h]
    if changed:
        raise SystemExit(f"changed since freeze: {changed}")
    committed = _git("show", f"HEAD:{FREEZE.relative_to(REPO).as_posix()}")
    if committed.returncode != 0 or committed.stdout.replace(b"\r\n", b"\n") != FREEZE.read_bytes().replace(b"\r\n", b"\n"):
        raise SystemExit("analysis_freeze_npe.json in the working tree is not the version committed in HEAD")
    comp, fixed, dof = _compressor()
    cl = np.load(REPO / "data" / "work" / "spectra_m0.npz")["cl"]
    t = comp(simulator.upper(cl))[None]
    models, st = load_ensemble()
    s = posterior_samples(models, st, t, 20000)[0]
    q = lambda v: {"median": float(np.median(v)), "lo68": float(np.percentile(v, 16)), "hi68": float(np.percentile(v, 84)),
                   "lo95": float(np.percentile(v, 2.5)), "hi95": float(np.percentile(v, 97.5)), "mean": float(v.mean()), "sd": float(v.std())}
    head = _git("rev-parse", "HEAD").stdout.decode().strip()
    out = {"unblinded_utc": datetime.now(timezone.utc).isoformat(), "git_head": head, "freeze_utc": fr["frozen_utc"],
           "posterior": {n: q(s[:, k]) for k, n in enumerate(TARGET_NAMES)},
           "beta_minus_psi0": q(s[:, 0] - s[:, 1]),
           # beta under explicit external priors on the dust misalignment psi0 (importance-reweighted
           # from the uniform training prior): psi0 = 0 +- 0.5 deg (MK-like) and +- 1, 2 deg
           "beta_given_psi_prior": {f"psi0~N(0,{w})": npe.weighted_q(s[:, 0], np.exp(-0.5 * (s[:, 1] / w) ** 2)) for w in (0.5, 1.0, 2.0)},
           "per_model_beta": [q(posterior_samples([m], st, t, 20000)[0][:, 0]) for m in models],
           "scores": t[0].tolist()}
    (RESULTS / "npe.json").write_text(json.dumps(out, indent=2))
    with open(REPO / "data" / "evidence" / "unblind_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"stage": "B", "utc": out["unblinded_utc"], "git_head": head, "freeze_utc": fr["frozen_utc"]}) + "\n")
    b = out["posterior"]["beta"]
    log(f"UNBLIND beta = {b['median']:.3f} (+{b['hi68'] - b['median']:.3f} / -{b['median'] - b['lo68']:.3f}) deg, 95% [{b['lo95']:.3f}, {b['hi95']:.3f}]")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="build,simulate,train,validate,attest")
    ap.add_argument("--n-train", type=int, default=400000)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--n-post", type=int, default=200)
    ap.add_argument("--n-mk", type=int, default=300)
    args = ap.parse_args(argv)
    st = args.stages.split(",")
    if "build" in st:
        stage_build()
    if "simulate" in st:
        stage_simulate(args.n_train)
    if "train" in st:
        stage_train(args.epochs)
    if "validate" in st:
        stage_validate(args.n_test, args.n_post, args.n_mk)
    if "attest" in st:
        stage_attest()
    if "freeze" in st:
        stage_freeze()
    if "unblind" in st:
        stage_unblind()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
