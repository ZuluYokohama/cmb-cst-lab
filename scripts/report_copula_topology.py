"""Report every planned topology comparison and retain the failed pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-only", action="store_true", help="refresh text from shipped results without rendering cached maps")
    args = parser.parse_args(argv)
    out = REPO / "results/copula_topology_v2"
    result = json.loads((out / "result.json").read_text())
    validation = json.loads((out / "validation.json").read_text())
    pilot = json.loads((REPO / "results/surrogate_topology_v1/validation.json").read_text())
    plan = json.loads((out / "plan.json").read_text())
    boundary = json.loads((REPO / "results/copula_boundary_audit_v1/result.json").read_text())
    assessment = {
        "status": "calibration_failed" if boundary["status"] == "failed" else "exploratory_only",
        "sky_numerical_status": result["status"],
        "sky_algorithmic_threshold_flags": len(result["candidate_flags"]),
        "boundary_false_flags": boundary["familywise_flags"], "boundary_controls": boundary["fields"],
        "accepted_cosmological_claims": [],
        "reason": "Cropped Gaussian control calibration failed" if boundary["status"] == "failed" else "Exploratory previously inspected sky; physical null not established",
        "sky_result_sha256": hashlib.sha256((out / "result.json").read_bytes()).hexdigest(),
        "boundary_result_sha256": hashlib.sha256((REPO / "results/copula_boundary_audit_v1/result.json").read_bytes()).hexdigest(),
    }
    (out / "assessment.json").write_text(json.dumps(assessment, indent=2) + "\n")
    for path, digest in plan["source_sha256"].items():
        if hashlib.sha256((REPO / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"source changed: {path}")
    if not args.report_only:
        fields = np.load(REPO / "data/work/copula_topology_v2/patches.npz")
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                             "axes.spines.top": False, "axes.spines.right": False,
                             "text.color": "#24364b", "axes.labelcolor": "#24364b"})
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        fig.subplots_adjust(left=.07, right=.96, top=.80, bottom=.14, wspace=.3, hspace=.35)
        fig.text(.07, .95, "Planck 143 GHz | Rank geometry", fontsize=24, weight="bold")
        fig.text(.07, .90, f"999 matched surrogates per patch · 9 planned tests · matching status: {result['status']}", fontsize=12)
        fig.text(.07, .855, f"Periodic controls: {validation['familywise_flags']}/64 flags | Cropped controls: {boundary['familywise_flags']}/64 ({boundary['status']})", fontsize=11)
        for col, (name, record) in enumerate(result["records"].items()):
            axes[0, col].imshow(fields[name], cmap="RdBu_r", vmin=-3, vmax=3, origin="lower")
            axes[0, col].set_title(f"{name}: fixed 20° patch")
            axes[0, col].set_xticks([])
            axes[0, col].set_yticks([])
            simulations = np.asarray(record["surrogate_statistics"])
            axes[1, col].hist(simulations[:, 0], bins=25, color="#087f8c", alpha=.7)
            axes[1, col].axvline(record["observed"][0], color="#d88620", linewidth=2, label="Observed")
            axes[1, col].set(xlabel="Finite connected components (PD0)", ylabel="Surrogates",
                             title=f"Count: Holm p = {record['p_holm'][0]:.3f}")
            axes[1, col].legend(frameon=False)
        fig.text(.07, .075, "Histograms condition on the Gaussianized patch's spectrum and marginal distribution, within measured tolerances.", fontsize=9)
        fig.text(.07, .043, "Boundary audit determines calibration eligibility. These plots do not establish a cosmological anomaly.", fontsize=9)
        fig.savefig(out / "topology.png", dpi=160, facecolor="white")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        fig.subplots_adjust(left=.11, right=.96, bottom=.23, top=.78)
        counts = [pilot["familywise_flags"], validation["familywise_flags"], boundary["familywise_flags"]]
        bars = ax.bar(["Original pilot\n(skewed controls included)", "Rank method\n(periodic Gaussian)",
                       "Rank method\n(cropped Gaussian)"], counts, color=["#ba5c45", "#087f8c", "#ba5c45"], width=.55)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x()+bar.get_width()/2, count+.5, f"{count}/64", ha="center", weight="bold")
        ax.axhline(8, color="#586579", linestyle="--", label="Preset gross failure: more than 8")
        ax.set(ylabel="Control fields flagged (Holm across 3 tests)", ylim=(0, max(20, max(counts)+5)))
        ax.legend(loc="upper left", frameon=False, fontsize=9)
        fig.text(.11, .92, "Check the comparison before trusting the sky", fontsize=20, weight="bold")
        fig.text(.11, .85, f"Cropped-Gaussian calibration: {boundary['status'].upper()}", fontsize=13)
        fig.text(.11, .10, "All flags shown before per-draw matching gates; 199 surrogates per field.", fontsize=9)
        fig.text(.11, .055, "Cropped controls were a separately frozen post-hoc stress test, not discovery confirmation.", fontsize=9)
        fig.savefig(out / "calibration_controls.png", dpi=160, facecolor="white")
        plt.close(fig)
    names = plan["statistics"]
    rows = []
    for patch, rec in result["records"].items():
        for k, name in enumerate(names):
            rows.append(f"| {patch} | {name} | {rec['observed'][k]:.5g} | {rec['surrogate_mean'][k]:.5g} ± {rec['surrogate_sd'][k]:.4g} | {rec['raw_p'][k]:.4g} | {rec['p_holm'][k]:.4g} |")
    q = [q for rec in result["records"].values() for q in rec["quality"]]
    eligible_pilot = sum(r["flag"] and not r["quality_failures"] for r in pilot["records"])
    pilot_eligible = sum(not r["quality_failures"] for r in pilot["records"])
    quality_failures = sum(not item["passed"] for item in q)
    numerical_flags = sum(p <= .05 for rec in result["records"].values() for p in rec["p_holm"])
    if result["status"] != "completed":
        note = f"{numerical_flags} comparisons crossed the adjusted .05 threshold, but matching failures block candidate interpretation."
    elif result["candidate_flags"]:
        note = f"{len(result['candidate_flags'])} comparisons crossed the adjusted .05 threshold and passed the run's quality gate. These are exploratory leads."
    else:
        note = "No comparison crossed the adjusted .05 threshold."
    if boundary["status"] == "failed":
        note += " **The cropped-Gaussian boundary audit failed; these sky flags are not calibrated anomaly evidence.**"
    elapsed = (datetime.fromisoformat(result["completed_utc"])
               - datetime.fromisoformat(plan["created_utc"])).total_seconds()
    report = f'''# Exploratory topology beyond Gaussianized two-point structure

Completed {result['completed_utc']}. Run status: **{result['status']}**.
Scientific assessment: **{assessment['status']}** ([machine-readable assessment](../results/copula_topology_v2/assessment.json)).
{note}
Elapsed wall time from plan freeze: {elapsed:.1f} seconds ({elapsed / 60:.1f} minutes).

![Control calibration across the retained experiments](../results/copula_topology_v2/calibration_controls.png)

## Question and scope

Can the spatial arrangement of temperature-map features differ after removing
brightness-distribution effects and matching the transformed patch's two-point
structure? This is a temperature geometry test, separate from polarization β.
Persistent homology and IAAFT are established methods, not inventions of this lab.
The added experiment is a controlled application to three previously examined
patches. It is exploratory, not blind confirmation.

Sources: [Cole & Shiu, CMB persistence](https://arxiv.org/abs/1712.08159),
[Schreiber & Schmitz, IAAFT](https://arxiv.org/abs/chao-dyn/9909041), and
[TISEAN implementation caveats](https://www.pks.mpg.de/tisean/TISEAN_2.1/docs/docs_f/surrogates.html).

## Why the failed pilot matters

V1's matched-surrogate model failed its fixed calibration gate: {pilot['familywise_flags']}/64
synthetic examples flagged at the Holm-adjusted .05 level, all in the skewed
monotonic-Gaussian group (16/32), versus 0/32 Gaussian controls. These flags are
before mandatory matching-quality requirements. Only {eligible_pilot}/{pilot_eligible}
fully quality-matched controls flagged. The failed gate blocked sky evaluation.
We retain its [plan](../results/surrogate_topology_v1/plan.json) and
[all controls](../results/surrogate_topology_v1/validation.json).

V2 changes the question explicitly. It maps pixel midranks to normal quantiles;
a strictly monotonic brightness remapping then leaves the input unchanged.
It uses fresh, disjoint seeds and retains the original thresholds. Its held-out
gross-calibration check had {validation['familywise_flags']}/64 family-wise flags,
against a preset rejection cutoff of more than eight. There were
{sum(r['quality_failures'] for r in validation['records'])} control-surrogate matching failures
among 12,736 draws. This small synthetic check is not a proof of exact real-sky p-values.

## Every planned comparison

![Rank geometry and matched-surrogate component counts](../results/copula_topology_v2/topology.png)

143 GHz A/B mean temperature, 10 arcmin smoothing, ell <=1000, NSIDE512,
128×128 open-boundary 20° NGP/SGP/ColdSpot patches. Each has 999 IAAFT surrogates,
500 iterations. S has exact transformed-field Fourier amplitudes; H has the
exact transformed pixel histogram. Matching both properties exactly in one
finite field is generally impossible. No draws were discarded.

| Patch | Statistic | Observed | Surrogate mean ± SD | Two-sided p | Holm p (9 tests) |
|---|---|---:|---:|---:|---:|
{chr(10).join(rows)}

Totals are finite PD0 lifetime sums in normal-score units, not microkelvin.
The component count is invariant under strict monotonic brightness transforms.
The displayed p-values use the predeclared +1-corrected Monte Carlo convention.
They describe the algorithmic surrogate ensemble; they are not probabilities
that cosmology is true or false. The smallest available raw two-sided p is .002.

## Constraint diagnostics and implementation evidence

- Matching failures: **{quality_failures} / 2,997** real-patch surrogates.
- Largest S exact-spectrum relative error: {max(item['spectrum_stage_relative_error'] for item in q):.4g} (limit 1e-10).
- Largest H spectrum mismatch: {max(item['histogram_stage_spectrum_error'] for item in q):.4g} (limit .01).
- Largest S histogram Kolmogorov distance: {max(item['spectrum_stage_histogram_ks'] for item in q):.4g} (limit .01).
- H histograms and S/H component-count agreement are checked for every draw.
- JoVin surrogate-kernel/recorded-CUDA suite: `20261023T185704Z-58ac2ab53e53`.
- JoVin rank-invariance suite: `{plan['jovin_receipt_id']}`.
- Local CPU/GPU parity and shared numerical tests passed.
- Full local regression suite: 175 passed, three unavailable Intel XPU skips.
- [Current source, authentic receipt and frozen-run verification](../results/copula_topology_v2/verification.json).

JoVin establishes the behavior tested in the bundled functions and recomputes
recorded CUDA outputs. It does not validate the astrophysical null assumptions.

## Interpretation limits

### Post-hoc boundary robustness audit

After seeing the pole-patch discrepancies, we identified a control mismatch:
the original 32×32 controls are periodic Gaussian fields; real sky patches
are crops from a larger field. A separately frozen audit used 64 fresh Gaussian
64×64 parents with the same smoothing spectrum, cropped each to its central
32×32 pixels, applied the same normal-score transform, and ran the unchanged
199-surrogate, 500-iteration method and Holm-three test. Nothing in either sky
run was retuned or replaced.

The cropped-Gaussian audit **{boundary['status']}**, with **{boundary['familywise_flags']}/64**
controls flagged (preset gross-rejection rule: more than eight). There were
{sum(r['quality_failures'] for r in boundary['records'])} matching failures among 12,736 draws.
This is a post-hoc method stress test, not independent sky confirmation.
Even ordinary Gaussian crops can produce flags in this boundary setting;
the sky p-values remain numerical surrogate tail fractions but lose their
interpretation as calibrated evidence beyond a suitable Gaussian sky model.
Boundary handling changes Fourier conditioning and the algorithmic ensemble;
this audit does not isolate one unique mechanism or establish that it explains
the entire real-sky difference.

[Boundary audit plan](../results/copula_boundary_audit_v1/plan.json),
[all boundary controls](../results/copula_boundary_audit_v1/result.json).

The next valid experiment should generate full Gaussian skies and apply the
same smoothing, spherical pixelization, projection, patch dimensions and edge
handling as the real data. It needs fresh controls and a frozen acceptance rule
before interpreting any sky comparison. A method passing those checks can then
be challenged with detector-split and frequency comparisons. More downloaded
data alone does not repair this control mismatch.

The constraints refer to a Gaussianized finite patch's 2D Fourier spectrum,
not the original spherical C_ell. Periodic Fourier synthesis and open topology
boundaries, patch nonstationarity, foregrounds, correlated noise, finite matching
errors and IAAFT sampling bias remain possible explanations for a discrepancy.
No compact sources were masked into artificial holes. These patches were chosen
in earlier work and already have known topological results. No claim of new
cosmology or superiority over all conventional higher-order statistics follows.

[Frozen V2 specification](COPULA_TOPOLOGY_SPEC.md), [run plan](../results/copula_topology_v2/plan.json),
[held-out controls](../results/copula_topology_v2/validation.json),
[full results](../results/copula_topology_v2/result.json).
'''
    (REPO / "docs/COPULA_TOPOLOGY_RUN.md").write_text(report)
    print(out / "topology.png")


if __name__ == "__main__":
    main()
