"""Plot the retained synthetic calibration result without rerunning the experiment."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2, norm


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    result = json.loads((repo / "results/gpu_calibration.json").read_text())
    residual = np.asarray(result["draw_records"]["network_residual_chi2"])
    pulls = np.asarray(result["draw_records"]["pulls"]).ravel()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#24364b", "text.color": "#24364b"})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8))
    fig.subplots_adjust(left=.08, right=.97, bottom=.22, top=.72, wspace=.25)
    fig.text(.08, .94, "RTX 5070  |  Calibration check", fontsize=24, weight="bold")
    checks = result["acceptance"]["n_statistical_checks"]
    verdict = "passed" if result["acceptance"]["statistical_passed"] else "did not all pass"
    fig.text(.08, .87, f"{len(residual):,} synthetic skies · {checks} predeclared checks {verdict}", fontsize=14)
    fig.text(.08, .80, "CUDA float32 sampling → float64 CPU inference", color="#526579", fontsize=12)
    axes[0].hist(residual, bins=np.linspace(0, 55, 33), density=True,
                 color="#087f8c", alpha=.65, label="Simulated skies")
    x = np.linspace(0, 55, 500)
    axes[0].plot(x, chi2.pdf(x, 21), color="#d88620", lw=2.5, label="Expected χ² (21 dof)")
    axes[0].set(xlabel="Network residual χ²", ylabel="Density", xlim=(0, 55),
                title=f"Mean {residual.mean():.2f}  |  expected 21")
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].hist(pulls, bins=np.linspace(-4, 4, 49), density=True,
                 color="#087f8c", alpha=.65, label="Fitted-angle pulls")
    x = np.linspace(-4, 4, 500)
    axes[1].plot(x, norm.pdf(x), color="#d88620", lw=2.5, label="Standard normal")
    width = result["pulls_pooled"]["width"]["value"]
    axes[1].set(xlabel="(Estimated angle − injected angle) / error", ylabel="Density", xlim=(-4, 4),
                title=f"Pull width {width:.3f}  |  expected 1")
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.grid(axis="y", color="#dce4e9", alpha=.5)
        ax.set_axisbelow(True)
    fig.text(.08, .105, "Synthetic bin-level Wishart model; no measured Planck maps or new cosmological result.", fontsize=10)
    fig.text(.08, .065, "Map pulls within each sky are correlated; reported uncertainty uses independent sky clusters.", fontsize=10)
    path = repo / "results/gpu_calibration.png"
    fig.savefig(path, dpi=170, facecolor="white")
    print(path)


if __name__ == "__main__":
    main()
