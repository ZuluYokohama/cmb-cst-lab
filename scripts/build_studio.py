"""Build studio/index.html: the template with every number injected from results/ and evidence/.

Nothing numeric is typed into the page. Measured values come from results/reproduction.json and
results/persistence.json; published values are listed here once, each with its source, and are
always shown with a "Cited" badge; sandbox receipts come from data/evidence/jovin_receipts.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

CITED = {
    "paper": {
        "ref": "Lonappan, Keating & Arnold 2026, ApJL 1009, L12 (arXiv:2609.09149)",
        "url": "https://arxiv.org/abs/2609.09149",
        "chi2_diff": 8.18, "dof": 7, "pte": 0.32, "pearson_r": 0.98,
        "beta_mk": [0.37, 0.11], "beta_anchored_092": [0.37, 0.12], "beta_anchored_062": [0.40, 0.13],
        "dust_ee_bb": {"143": 1.41, "217": 1.37, "353": 1.42}, "max_pull": 1.8,
    },
    "beta_compendium": [
        {"label": "Minami & Komatsu 2020 (PR3)", "beta": 0.35, "sigma": 0.14, "url": "https://arxiv.org/abs/2011.11254"},
        {"label": "Diego-Palazuelos et al. 2022 (PR4)", "beta": 0.30, "sigma": 0.11, "url": "https://arxiv.org/abs/2201.07682"},
        {"label": "Eskilt & Komatsu 2022 (PR4 + WMAP)", "beta": 0.342, "sigma": 0.093, "url": "https://arxiv.org/abs/2205.13962"},
        {"label": "Lonappan, Keating & Arnold 2026 (MK)", "beta": 0.37, "sigma": 0.11, "url": "https://arxiv.org/abs/2609.09149"},
    ],
}


def add_curl_sigma(rep: dict) -> None:
    """Residual (curl) standard deviations per edge, from the stored pair covariance, if a results
    file predates the pipeline exporting them. Also checks the stored network fit reproduces."""
    import numpy as np

    sys.path.insert(0, str(REPO / "src"))
    from cmbcst import network

    _, _, d0, _ = network.complete_complex(8)
    g = rep["detectors"].index(rep["gauge"])
    for m in rep["per_mask"].values():
        cov = np.array(m["pair_cov_deg2"])
        net = network.solve_network(np.array(m["pair_angles_deg"]), cov, d0, g)
        assert np.allclose(net["alpha"], m["network"]["alpha_deg"], atol=1e-9), "stored network fit does not reproduce"
        m["network"].setdefault("curl_sigma_deg", np.sqrt(np.clip(np.diag(net["cov_curl"]), 0, None)).tolist())


def latest_receipts() -> dict[str, dict]:
    path = REPO / "data" / "evidence" / "jovin_receipts.json"
    out: dict[str, dict] = {}
    sys.path.insert(0, str(REPO / "scripts"))
    from jovin_verify import SUITES

    if path.exists():
        for r in json.loads(path.read_text(encoding="utf-8")):
            if r["suite"] not in SUITES:
                continue
            out[r["suite"]] = {k: r.get(k) for k in ("passed", "passed_count", "total", "receipt_id", "verified_at",
                                                      "chain_valid", "bundle_sha256")}
    return out


def _between(text: str, tag: str, body: str) -> str:
    start, end = f"<!-- {tag}:START -->", f"<!-- {tag}:END -->"
    i, j = text.index(start) + len(start), text.index(end)
    return text[:i] + "\n" + body.strip() + "\n" + text[j:]


def readme_tables(data: dict) -> None:
    rep, p = data["reproduction"], CITED["paper"]
    m0, m30 = rep["per_mask"]["0"], rep["per_mask"]["30"]
    cd = m0.get("chi2_diff") or m0["chi2_diff_no_cross"]
    pulls = cd.get("pulls", {})
    rows = [
        ("Sky fraction (masks \"0\" / \"30\")", f"{rep['masks']['fsky']['0']:.3f} / {rep['masks']['fsky']['30']:.3f}", "0.92 / 0.62"),
        ("Network residual χ² (21 dof), f_sky 0.92", f"{m0['network']['chi2']:.1f}, PTE {m0['network']['pte']:.2f}", "not reported"),
        ("Network residual χ² (21 dof), f_sky 0.62", f"{m30['network']['chi2']:.1f}, PTE {m30['network']['pte']:.2f}", "not reported"),
        ("Network vs MK, χ²_diff (7 dof)", f"{cd['chi2']:.2f}, PTE {cd['pte']:.2f}", f"{p['chi2_diff']}, PTE {p['pte']}"),
        ("Largest per-map pull", (max(pulls.items(), key=lambda kv: abs(kv[1]))[0] + f" {max(pulls.values(), key=abs):+.2f}") if pulls else "–", f"143B +{p['max_pull']}"),
        ("Pearson r over 28 pair angles", f"{m0['pearson_r_pairs']:.3f}", f"{p['pearson_r']}"),
        ("MK β, f_sky 0.92", f"{m0['mk']['beta_deg']:.2f}° ± {m0['mk']['sigma_beta_deg']:.2f}°", f"{p['beta_mk'][0]:.2f}° ± {p['beta_mk'][1]:.2f}°"),
        ("Anchored β, f_sky 0.92", f"{m0['anchored_beta_deg']:.2f}° ± {m0['anchored_sigma_deg']:.2f}°", f"{p['beta_anchored_092'][0]:.2f}° ± {p['beta_anchored_092'][1]:.2f}°"),
        ("Anchored β, f_sky 0.62", f"{m30['anchored_beta_deg']:.2f}° ± {m30['anchored_sigma_deg']:.2f}°", f"{p['beta_anchored_062'][0]:.2f}° ± {p['beta_anchored_062'][1]:.2f}°"),
        ("H¹(K₈), filled complex", f"b₁ = {m0['hodge']['b1']} (rank d₀ = {m0['hodge']['rank_d0']}, rank d₁ = {m0['hodge']['rank_d1']})", "not discussed"),
    ]
    pers = data.get("persistence")
    if pers:
        a = pers["ngp_minus_sgp_mean_life0"]
        rows.append(("NGP − SGP mean PD₀ lifetime, 143 GHz", f"{a['real']:+.3f} (sims {a['sim_mean']:+.3f} ± {a['sim_std']:.3f}, p = {a['p_two_sided']:.2f})",
                     "not in paper"))
    d = rep["dust_ee_over_bb_l_lt_300"]
    rows.append(("Dust EE/BB, ℓ < 300 (ΛCDM-subtracted) 143/217/353",
                 " / ".join(f"{d[q]['lcdm_subtracted']:.2f}" for q in ("143", "217", "353")),
                 " / ".join(str(p["dust_ee_bb"][q]) for q in ("143", "217", "353"))))
    table = "| Quantity | This repository | Paper |\n|---|---|---|\n" + "\n".join(f"| {a} | {b} | {c} |" for a, b, c in rows)
    table += "\n\nPer-map angles (network, gauge α_143A = 0, f_sky 0.92):\n\n| Map | " + " | ".join(rep["detectors"]) + " |\n|---|" + "---|" * 8
    table += "\n| α (deg) | " + " | ".join(f"{a:+.2f} ± {s:.2f}" if s > 0 else "0 (gauge)" for a, s in zip(m0["network"]["alpha_deg"], m0["network"]["sigma_deg"])) + " |"
    rc = data["receipts"]
    rtab = "| Suite | Result | Receipt |\n|---|---|---|\n" + "\n".join(
        f"| `{s}` | {'pass' if r['passed'] and r['chain_valid'] else 'FAIL'} {r['passed_count']}/{r['total']} | `{r['receipt_id']}` |" for s, r in rc.items())
    path = REPO / "README.md"
    text = path.read_text(encoding="utf-8")
    text = _between(_between(text, "RESULTS", table), "RECEIPTS", rtab)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    rep = json.loads((REPO / "results" / "reproduction.json").read_text(encoding="utf-8"))
    add_curl_sigma(rep)
    pers_path = REPO / "results" / "persistence.json"
    data = {"reproduction": rep, "persistence": json.loads(pers_path.read_text(encoding="utf-8")) if pers_path.exists() else None,
            "cited": CITED, "receipts": latest_receipts()}
    tpl = (REPO / "studio" / "template.html").read_text(encoding="utf-8")
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    out = tpl.replace("/*__DATA__*/null", payload)
    assert out != tpl, "data placeholder not found in template"
    (REPO / "studio" / "index.html").write_text(out, encoding="utf-8")
    print(f"studio/index.html: {len(out) / 1024:.0f} KB")
    readme_tables(data)
    print("README.md tables updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
