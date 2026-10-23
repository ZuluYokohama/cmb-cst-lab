"""Render measured calibration and dust-model sensitivity from the retained real-data run."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    directory = repo / "results/rtx5070_real"
    measured = json.loads((directory / "reproduction.json").read_text())
    reference = json.loads((repo / "results/reproduction.json").read_text())
    audit = json.loads((directory / "stage_audit.json").read_text())
    inventory = json.loads((repo / "data/evidence/planck_dataset.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "text.color": "#24364b", "axes.labelcolor": "#24364b"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), gridspec_kw={"width_ratios": [1.35, 1]})
    fig.subplots_adjust(left=.08, right=.97, bottom=.22, top=.72, wspace=.28)
    fig.text(.08, .94, "Planck PR4  |  Full-resolution calibration", fontsize=23, weight="bold")
    fig.text(.08, .865, f"{inventory['total_bytes'] / 1e9:.2f} GB · 8 maps · 2,000 bootstrap draws · readiness: {audit['status']}", fontsize=13)
    fig.text(.08, .80, "Fresh measured-data run, with the same estimator and fixed analysis settings", fontsize=11, color="#526579")
    x = np.arange(8)
    for offset, source, label, color, marker in ((-.08, reference, "Repository reference", "#a3b0be", "o"),
                                                (.08, measured, "Fresh full-map run", "#087f8c", "s")):
        network = source["per_mask"]["0"]["network"]
        axes[0].errorbar(x + offset, network["alpha_deg"], yerr=network["sigma_deg"],
                         fmt=marker, color=color, capsize=3, ms=5, label=label)
    axes[0].axhline(0, color="#dce4e9", lw=1)
    axes[0].set(xticks=x, xticklabels=measured["detectors"], ylabel="Relative calibration angle α (degrees)",
                title="Differential detector pattern (mask 0)")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].tick_params(axis="x", labelrotation=35)
    for offset, mask, label, color, marker in ((-.07, "0", "Mask 0", "#087f8c", "s"),
                                             (.07, "30", "Mask 30", "#d88620", "o")):
        variants = measured["per_mask"][mask]["mk"]["beta_by_dust_eb_amplitudes"]
        axes[1].errorbar(np.arange(3) + offset, [variants[k]["beta_deg"] for k in ("0", "1", "4")],
                         yerr=[variants[k]["sigma_deg"] for k in ("0", "1", "4")],
                         fmt=marker, capsize=4, color=color, label=label)
    axes[1].axhline(0, color="#dce4e9", lw=1)
    axes[1].set(xticks=np.arange(3), xticklabels=["0", "1", "4"], xlabel="Dust EB amplitudes in the model",
                ylabel="MK β (degrees)", title="Sensitivity to the dust model")
    axes[1].legend(frameon=False, fontsize=9)
    for axis in axes:
        axis.grid(axis="y", color="#dce4e9", alpha=.5)
        axis.set_axisbelow(True)
    fig.text(.08, .105, "Map-angle gauge: α143A = 0. Error bars are the pipeline's 1σ estimates.", fontsize=10)
    fig.text(.08, .065, "Reproduction on public NPIPE maps; dust-model dependence and pseudo-Cℓ approximations remain material.", fontsize=10)
    path = directory / "calibration.png"
    fig.savefig(path, dpi=170, facecolor="white")
    print(path)


if __name__ == "__main__":
    main()
