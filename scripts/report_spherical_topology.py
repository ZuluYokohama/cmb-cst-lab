"""Render the frozen whole-sphere experiment without modifying numerical results."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/spherical_topology_v1"


def read(name):
    return json.loads((OUT / name).read_text())


def main():
    plan, result = read("plan.json"), read("result.json")
    validation, stage = read("validation.json"), read("stage_audit.json")
    ensemble, conditioning = read("ensemble.json"), read("conditioning.json")
    positive = read("positive_control.json")
    for path, digest in plan["source_sha256"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"frozen source changed: {path}")
    values = np.asarray(ensemble["statistics"])
    references = values[:plan["reference_skies"]]
    cl = np.asarray(conditioning["cl"])
    ell = np.arange(len(cl))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False, "text.color": "#25374d",
        "axes.labelcolor": "#25374d"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    fig.subplots_adjust(left=.09, right=.97, bottom=.22, top=.72, wspace=.32)
    fig.text(.09, .93, "Whole skies, identical cuts", fontsize=24, weight="bold")
    fig.text(.09, .84, f"Held-out controls: {validation['familywise_flags']}/64 flagged | Numerical failures: {validation['quality_failures']}", fontsize=12)
    axes[0].loglog(ell[2:], (ell*(ell+1)*cl/(2*np.pi))[2:], color="#087f8c")
    axes[0].set(xlabel="Multipole ℓ", ylabel="ℓ(ℓ+1) Cℓ / 2π [µK²]", title="Same empirical spherical power in every sky")
    minima = [np.min(r["p_holm"]) for r in validation["records"]]
    axes[1].hist(minima, bins=np.linspace(0, 1, 21), color="#087f8c", alpha=.75)
    axes[1].axvline(.05, color="#cc7c24", linewidth=2, label="Flag threshold")
    axes[1].set(xlabel="Smallest Holm p across 15 tests", ylabel="Held-out control skies",
                title=f"Gross calibration: {validation['status']}")
    axes[1].legend(frameon=False)
    fig.text(.09, .11, "Conditional isotropic Gaussian model; Planck foregrounds and anisotropic noise are not removed by this test.", fontsize=9)
    fig.text(.09, .06, f"64 independent controls share 999 references; maximum relative shell-power error {stage['max_relative_shell_error']:.2e}.", fontsize=9)
    fig.savefig(OUT / "calibration.png", dpi=160, facecolor="white")
    plt.close(fig)

    rows = []
    topology_flags = conventional_flags = 0
    model_warning = ""
    if "observed" in result:
        observed = np.asarray(result["observed"])
        adjusted = np.asarray(result["p_holm"])
        raw = np.asarray(result["raw_p"])
        topology_flags = int(np.sum(adjusted[:, :2] <= .05))
        conventional_flags = int(np.sum(adjusted[:, 2:] <= .05))
        fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
        fig.subplots_adjust(left=.08, right=.97, bottom=.17, top=.77, hspace=.55, wspace=.3)
        fig.text(.08, .94, "Temperature rank geometry | Spherical controls", fontsize=21, weight="bold")
        fig.text(.08, .865, "999 conditional skies · identical power and projection · Holm correction across all 15 comparisons", fontsize=11)
        for row in range(2):
            for col, patch in enumerate(result["patches"]):
                ax = axes[row, col]
                ax.hist(references[:, col, row], bins=25, color="#087f8c", alpha=.75)
                ax.axvline(observed[col, row], color="#cc7c24", linewidth=2, label="Observed sky")
                ax.set(title=f"{patch} | Holm p = {adjusted[col, row]:.3f}", ylabel="Reference skies",
                       xlabel="Finite PD0 count" if row == 0 else "Total rank PD0 lifetime")
                ax.legend(frameon=False, fontsize=9)
        fig.text(.08, .08, "Exploratory test on previously inspected, uncleaned temperature patches; no cosmological attribution.", fontsize=10)
        fig.savefig(OUT / "topology.png", dpi=160, facecolor="white")
        plt.close(fig)
        for i, patch in enumerate(result["patches"]):
            for j, stat in enumerate(result["statistics"]):
                rows.append(f"| {patch} | {stat} | {observed[i,j]:.6g} | {np.mean(references[:,i,j]):.6g} ± {np.std(references[:,i,j],ddof=1):.4g} | {raw[i,j]:.4g} | {adjusted[i,j]:.4g} |")
        outcome = (f"{topology_flags}/6 topology comparisons and {conventional_flags}/9 conventional-moment comparisons "
                   "crossed adjusted p<=.05 within the conditional isotropic model. These are exploratory diagnostic flags, not new cosmology.")
        model_warning = (f"The ordinary spread comparison shows a large model/data mismatch: real patch standard deviations "
                         f"are {observed[:,2].min():.1f}–{observed[:,2].max():.1f} µK, versus reference means "
                         f"{references[:,:,2].mean(0).min():.1f}–{references[:,:,2].mean(0).max():.1f} µK. "
                         "The global uncleaned spectrum spread isotropically is not representative of these high-latitude patches. "
                         "Foreground concentration is a plausible contributor, not an isolated causal finding. "
                         "Rank topology is invariant under uniform amplitude rescaling, so the spread difference alone "
                         "does not mechanically explain the component counts. Local spectral shape and anisotropy "
                         "remain possible explanations; higher-order-specific or unique topological information is not established.")
        sky_section = "![Topology comparisons](../results/spherical_topology_v1/topology.png)\n\n" + "\n".join([
            "| Patch | Statistic | Observed | Reference mean ± SD | Two-sided p | Holm p (15 tests) |",
            "|---|---|---:|---:|---:|---:|", *rows])
    else:
        outcome = "The held-out control gate failed. Real-patch evaluation was blocked and no candidate claims are accepted."
        sky_section = "No new real-patch statistics were evaluated. The failed ensemble and every control are retained."
    coherent_p = np.asarray(positive["p_holm"])
    coherent_top = int(np.sum(coherent_p[:, :2] <= .05))
    coherent_moments = int(np.sum(coherent_p[:, 2:] <= .05))
    disposition = {"status": result["status"], "topology_flags": topology_flags,
        "conventional_moment_flags": conventional_flags, "control_flags": validation["familywise_flags"],
        "positive_control_topology_flags": coherent_top,
        "higher_order_specificity_established": False,
        "accepted_cosmological_claims": [], "interpretation": outcome,
        "result_sha256": hashlib.sha256((OUT / "result.json").read_bytes()).hexdigest()}
    (OUT / "assessment.json").write_text(json.dumps(disposition, indent=2)+"\n")
    elapsed = stage["elapsed_s"]
    report = f'''# Whole-sphere conditional-power topology run

Completed {result['completed_utc']}; status **{result['status']}**. {outcome}

{model_warning}

![Whole-sphere controls and fixed power](../results/spherical_topology_v1/calibration.png)

## What changed and why

The previous flat-patch test falsely flagged 20/64 cropped Gaussian controls.
We preserve [that failure](COPULA_TOPOLOGY_RUN.md). This new null generates
whole spherical skies, then applies the same 10 arcmin smoothing, NSIDE512
pixelization, gnomonic projection and 128×128 open-boundary 20° NGP/SGP/ColdSpot
cuts to data and simulations. The ell=0,1 coefficients are zeroed in both.
No patch-periodic Fourier matching or source-mask holes are introduced.

Each ell>=2 harmonic shell has the exact empirical full-sphere C_ell of the
143 GHz A/B mean temperature. Gaussian real harmonic coordinates are normalized
to those shell powers. Under an isotropic Gaussian scalar model, shell directions
are uniform conditional on the powers. The null therefore changes spatial
arrangement while preserving the spherical two-point spectrum. Individual
harmonic magnitudes and pixel histograms are not preserved. This is a different
conditioning question from the preceding Gaussianized flat-patch experiment.

The real coefficients come from the area-weighted adjoint at NSIDE2048, not an
asserted exact inverse. Every comparison starts with band-limited coefficients
and uses the identical downstream sampling operator. See the
[ducc0 operator definitions](https://mtr.pages.mpcdf.de/ducc/sht.html) and
[established persistence context](https://arxiv.org/abs/1712.08159).

## All real-sky comparisons

{sky_section}

Topology uses empirical normal scores and finite PD0 components/lifetimes.
Conventional temperature standard deviation is in microkelvin; skewness and
excess kurtosis use standardized original temperature. Rank lifetimes are
dimensionless. All 15 comparisons use one Holm family and the +1-corrected
two-sided Monte Carlo convention; the smallest attainable raw p is .002.
A difference between rejection decisions is not a validated superiority or
incremental-information test. These previously studied patches are not blind
confirmation, and the full-sphere spectrum is not every local two-point statistic.

## Calibration, precision and positive control

- Held-out controls: **{validation['familywise_flags']}/64 flagged**; preset rejection if more than eight.
- All reference and held-out skies used identical power and map/projection operators.
- Numerical/finite-value failures: **{validation['quality_failures']}**.
- Maximum relative C_ell error: **{stage['max_relative_shell_error']:.4g}**, limit1e-10.
- Zero-power shells and real m=0 coefficients were checked for every sky.
- Positive phase-coherent control: {coherent_top}/6 topology and {coherent_moments}/9 moment tests flagged; all target powers matched.

The positive control is deliberately extreme and only demonstrates sensitivity
to one alternative; it is not a realistic primordial non-Gaussian benchmark.
The 64 controls share the reference ensemble, so their flag events are dependent.
The gross cutoff is a diagnostic, not a binomial significance claim or proof
that Planck foreground/noise assumptions are correct.

## JoVin and execution evidence

The new shipped numerical bundle and actual recorded RTX CUDA projection were
checked by JoVin receipt `{plan['jovin_receipt_id']}`. JoVin checks the recorded
output on CPU; the actual CUDA computation ran separately in the existing WSL
environment. Every production sky was independently checked against its fixed
power by NumPy. Indexed CUDA reductions can vary in final rounding bits.
All **17 current JoVin suites passed (116 checks including collection)**;
the local regression suite had **177 passed, three unavailable Intel XPU skips**.

Environment: {plan['device']}; PyTorch {plan['torch']}, CUDA {plan['cuda']},
NumPy {plan['numpy']}, ducc0 {plan['ducc0']}; eight CPU transform threads.
Wall time after plan freeze: **{elapsed:.1f} s ({elapsed/60:.1f} min)**,
including {stage['stages']['prepare_seconds']:.1f} s to hash/read both input maps
and prepare harmonic coefficients. No end-to-end GPU speedup is claimed;
spherical synthesis and topology remain CPU work.

[Final authentic receipt/source/result verification](../results/spherical_topology_v1/verification.json)
records the current suites, local test result and all frozen bindings.
The source Planck files were rehashed in this run, matched to the audited
inventory, and referenced in [conditioning provenance](../results/spherical_topology_v1/conditioning.json).

## What the supplied CST package contributed

The archive's five supplied checksums matched. Its reviewed mathematical script
was replayed unchanged in a separate working folder: **27/27 checks passed**.
The package's original results were not overwritten. This was a local replay,
separate from the JoVin CMB suite; it establishes no CMB, neural-model or market
performance. [Package evidence](../data/evidence/cst_package_audit.json) records
archive/content hashes, the NumPy version, every check and the scope distinction.

We used the working paper's explicit-projection and negative-control principles
(sections7.2,8.1) to specify this test. We did not adopt embedded document commands
as user instructions or claim an exact operational quotient for these summaries.
The package provides no new physical model for this temperature-map test.

## Scientific limits

The empirical spectrum contains foregrounds and noise from an uncleaned
143 GHz map. Uniform isotropic shell directions remove their spatial coherence
and do not model anisotropic/correlated instrument noise or all beam effects.
Any difference can reflect these known ingredients, statistical anisotropy or
higher-order structure. It does not isolate primordial non-Gaussianity,
establish unique information unavailable to every conventional statistic,
improve polarization β, or establish new CST physics. A clean cosmological
claim would require justified foreground/noise modeling and independent data.

[Frozen specification](SPHERICAL_TOPOLOGY_SPEC.md),
[immutable plan](../results/spherical_topology_v1/plan.json),
[all skies and precision audits](../results/spherical_topology_v1/ensemble.json),
[held-out controls](../results/spherical_topology_v1/validation.json),
[positive control](../results/spherical_topology_v1/positive_control.json),
[raw result](../results/spherical_topology_v1/result.json),
[scientific assessment](../results/spherical_topology_v1/assessment.json).
'''
    (ROOT / "docs/SPHERICAL_TOPOLOGY_RUN.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": result["status"], "topology_flags": topology_flags,
                      "moment_flags": conventional_flags, "control_flags": validation["familywise_flags"]}, indent=2))


if __name__ == "__main__":
    main()
