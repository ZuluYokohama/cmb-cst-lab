"""Stage P: is the PD_0 feature-count excess in the real 143 GHz patches sky or simulation mismatch?

Same map, smoothing, patches and statistics as run_reproduction.stage_persistence, compared with
three null ensembles:

    gaussian     isotropic Gaussian skies from the measured spectrum (the original null)
    noise_aniso  Gaussian CMB from the A x B cross spectrum (signal only) plus anisotropic white
                 noise with Planck's per-pixel variance (II_COV of the A and B maps), so the real
                 scan-pattern noise structure is in the null
    phase        phase-randomised surrogates of the masked map: |a_lm| kept, phases random. This
                 is the literal "structure beyond the power spectrum" null.

Null spectra and phase-surrogate amplitudes come from the masked (mask_30) map, so excluded
Galactic-plane power enters no null. The real patches use the map as observed, as in Stage 0:
mask_30 also has compact-source holes inside the patches, and multiplying by it would cut them
into the real map as artificial minima (one at the NGP reaches -10.7 sigma).

p-values: two-sided Monte Carlo with the +1 correction; Holm over every comparison reported.
-> results/persistence_validation.json

Usage:
    python scripts/run_persistence_validation.py [--n-sims 1000]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cmbcst import persistence, spectra, theory  # noqa: E402
from cmbcst.data import path_of  # noqa: E402

LMAX, NSIDE_LO = 1000, 512
PATCHES = {"NGP": (0.0, 0.0), "SGP": (np.pi, 0.0), "ColdSpot": (np.radians(147.0), np.radians(209.0))}
STATS = ("mean_life0", "max_life0", "n0", "min_value")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_t_and_var(det: str) -> tuple[np.ndarray, np.ndarray]:
    from astropy.io import fits

    with fits.open(path_of(det), memmap=True) as h:
        t = np.asarray(h[1].data["I_STOKES"], dtype=np.float64).ravel()
        v = np.asarray(h[1].data["II_COV"], dtype=np.float64).ravel()
    good = (np.abs(t) < 1e20) & (v > 0) & (v < 1e20)
    t = spectra.nest_to_ring(np.where(good, t * 1e6, 0.0), 2048)
    v = spectra.nest_to_ring(np.where(good, v * 1e12, 0.0), 2048)  # uK^2
    return t, v


def main(argv: list[str]) -> int:
    import ducc0

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sims", type=int, default=1000)
    args = ap.parse_args(argv)
    if args.n_sims < 2:
        ap.error("--n-sims must be at least 2 (z-scores need a simulated standard deviation)")
    geom_hi = ducc0.healpix.Healpix_Base(2048, "RING").sht_info()
    geom_lo = ducc0.healpix.Healpix_Base(NSIDE_LO, "RING").sht_info()
    nt = spectra.NTHREADS
    mask = np.load(REPO / "data" / "work" / "mask_30.npy").astype(np.float64)
    (ta, va), (tb, vb) = load_t_and_var("143A"), load_t_and_var("143B")
    t = 0.5 * (ta + tb)
    t -= np.sum(t * mask) / mask.sum()
    var_mean = 0.25 * (va + vb)                          # variance of (A + B) / 2 per pixel

    def alm_hi(m):
        return ducc0.sht.experimental.adjoint_synthesis(map=m[None], spin=0, lmax=LMAX, mmax=LMAX, nthreads=nt, **geom_hi)[0] * (4 * np.pi / m.size)

    smooth = theory.gaussian_beam(10.0, LMAX)
    ell_idx, m_idx = spectra.alm_ell_m(LMAX)
    w2, _ = spectra.mask_moments(mask)
    alm_data = alm_hi(t)                                 # real patches: the map as observed
    alm_masked = alm_hi(t * mask) / np.sqrt(w2)          # phase null: masked sky only
    a_m, b_m = alm_hi(ta * mask), alm_hi(tb * mask)
    cl_total = spectra.cross_cl(alm_hi(t * mask), alm_hi(t * mask), LMAX) / w2
    cl_signal = spectra.cross_cl(a_m, b_m, LMAX) / w2     # A x B: no noise bias
    # noise variance at N_side 512: mean of the 16 child pixels' variances / 16 (averaging 16 pixels)
    base_hi = ducc0.healpix.Healpix_Base(2048, "RING")
    nest_hi = base_hi.ring2nest(np.arange(var_mean.size, dtype=np.int64))
    var_nest = np.empty_like(var_mean)
    var_nest[nest_hi] = var_mean
    var_lo_nest = var_nest.reshape(-1, 16).mean(1) / 16.0
    base_lo = ducc0.healpix.Healpix_Base(NSIDE_LO, "RING")
    sd_lo = np.sqrt(var_lo_nest[base_lo.ring2nest(np.arange(12 * NSIDE_LO**2, dtype=np.int64))])

    def to_map(alm):
        return ducc0.sht.experimental.synthesis(alm=(alm * smooth[ell_idx])[None], spin=0, lmax=LMAX, mmax=LMAX, nthreads=nt, **geom_lo)[0]

    def summaries(m):
        return {k: persistence.patch_summary(persistence.standardise(persistence.project_patch(m, *c))) for k, c in PATCHES.items()}

    def smooth_lo_map(m_lo):
        alm = ducc0.sht.experimental.adjoint_synthesis(map=m_lo[None], spin=0, lmax=LMAX, mmax=LMAX, nthreads=nt, **geom_lo)[0] * (4 * np.pi / m_lo.size)
        return to_map(alm)

    real = summaries(to_map(alm_data))
    rng = np.random.default_rng(42)
    ensembles = {"gaussian": [], "noise_aniso": [], "phase": []}
    t0 = time.perf_counter()
    for s in range(args.n_sims):
        ensembles["gaussian"].append(summaries(to_map(spectra.random_alm(cl_total, LMAX, rng))))
        cmb = to_map(spectra.random_alm(cl_signal, LMAX, rng))
        noise = smooth_lo_map(sd_lo * rng.standard_normal(sd_lo.size))
        ensembles["noise_aniso"].append(summaries(cmb + noise))
        phase = np.exp(2j * np.pi * rng.random(alm_masked.size))
        phase[m_idx == 0] = np.sign(rng.standard_normal(int(np.sum(m_idx == 0))))  # m = 0 stays real
        ensembles["phase"].append(summaries(to_map(np.abs(alm_masked) * phase)))
        if (s + 1) % 100 == 0:
            log(f"{s + 1}/{args.n_sims} ({(time.perf_counter() - t0) / (s + 1):.2f} s per draw)")

    out = {"n_sims": args.n_sims, "map": "143 GHz (A+B)/2 temperature, 10' smoothing, l <= 1000, N_side 512",
           "real_source": "a_lm of the unmasked map", "phase_null_source": "a_lm of the mask_30 masked map / sqrt(w2)",
           "nulls": {}, "p_value_method": "two-sided Monte Carlo with +1 correction; Holm over all comparisons"}
    flat = []
    for null, sims in ensembles.items():
        rec = {}
        for patch in PATCHES:
            rec[patch] = {}
            for st in STATS:
                v = np.array([x[patch][st] for x in sims])
                r = real[patch][st]
                rec[patch][st] = {"real": r, "sim_mean": float(v.mean()), "sim_std": float(v.std()),
                                  # z undefined (None) when every simulation gives the same value
                                  "z": float((r - v.mean()) / v.std()) if v.std() > 0 else None,
                                  "p": persistence.mc_pvalue(v, r)}
                flat.append((null, patch, st))
        out["nulls"][null] = rec
    adj = persistence.holm([out["nulls"][n][p][s]["p"] for n, p, s in flat])
    for (n, p, s), a in zip(flat, adj):
        out["nulls"][n][p][s]["p_holm"] = float(a)
    out["n_comparisons"] = len(flat)
    (REPO / "results" / "persistence_validation.json").write_text(json.dumps(out, indent=2))
    for n in ensembles:
        zs = {p: round(out["nulls"][n][p]["n0"]["z"], 2) for p in PATCHES}
        ph = {p: round(out["nulls"][n][p]["n0"]["p_holm"], 3) for p in PATCHES}
        log(f"{n}: n0 z {zs}, Holm p {ph}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
