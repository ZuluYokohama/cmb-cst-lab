"""Post-hoc covariance audit of previously unblinded Planck spatial tests.

Reads locally cached regional pseudo-C_l spectra. It does NOT modify
results/spatial.json, the preregistered freeze, or the public dashboard.
Output is REQUIRED to be outside the public repository, never silently
overwritten, and is explicitly labeled as a post-hoc diagnostic.

Usage:
 python scripts/audit_spatial_ab.py --out ../private-audits/ab_joint_v2.json
Requires cached data/work/spatial/{base12,lat8}_meta.json and region NPZ files.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cmbcst import DETECTORS, network, spatial
from cmbcst.differential import pair_angles
from cmbcst.spatial_ab_audit import same_frequency_joint_gls

DEG = np.pi / 180
WORK = REPO / "data" / "work" / "spatial"
RESULTS = REPO / "results" / "spatial.json"
SCHEMES = ("base12", "lat8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def scheme_audit(scheme: str, expected: dict) -> dict:
    meta_file = WORK / f"{scheme}_meta.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    ed, _, _, _ = network.complete_complex(8)
    ab_edges = spatial.same_frequency_edges(list(DETECTORS), ed)
    covariance_list, delta_list = [], []
    hashes = {str(meta_file.relative_to(REPO)): sha256_file(meta_file)}
    for i, label in enumerate(meta["labels"]):
        file = WORK / f"{scheme}_{i:02d}.npz"
        blob = file.read_bytes()                       # hash and load the same bytes
        hashes[str(file.relative_to(REPO))] = hashlib.sha256(blob).hexdigest()
        with np.load(io.BytesIO(blob), allow_pickle=False) as bundle:
            pa = pair_angles(bundle["cl"], bundle["nu"], ed)
        delta_list.append(pa["delta"] / DEG)
        covariance_list.append(pa["cov"] / DEG**2)
    delta, covariance = np.array(delta_list), np.array(covariance_list)
    sigma = np.sqrt(np.diagonal(covariance, axis1=1, axis2=2))
    baseline = spatial.edge_constancy(delta, sigma, ab_edges)
    original = float(expected["chi2_ab"])
    if not np.isclose(baseline["chi2"], original, atol=1e-5, rtol=1e-6):
        raise RuntimeError(f"{scheme}: local caches do NOT reproduce committed original A/B statistic")
    joint = same_frequency_joint_gls(delta, covariance, ab_edges)
    return {
        "n_regions": len(meta["labels"]),
        "edges": [[DETECTORS[i], DETECTORS[j]] for k,(i,j) in enumerate(ed) if k in ab_edges],
        "original_reported_diagonal_chi2": original,
        "diagonal_recomputed_chi2": baseline["chi2"],
        "degrees_of_freedom": baseline["dof"],
        "joint_chi2": joint["chi2"],
        "joint_pte_nominal": network.chi2_sf(joint["chi2"], joint["dof"]),
        "max_abs_within_region_edge_correlation": joint["max_abs_within_region_edge_correlation"],
        "source_cache_sha256": hashes,
        "limitation": "Within-region edge correlations modeled; between-region covariance and covariance-estimation uncertainty NOT modeled.",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True, help="new PRIVATE output file outside public repository")
    args = ap.parse_args()
    out = args.out.expanduser().resolve()
    if out == REPO.resolve() or REPO.resolve() in out.parents:
        print("REFUSED: output cannot be inside public repository", file=sys.stderr)
        return 2
    if out.exists():
        print("REFUSED: output exists (no overwrites)", file=sys.stderr)
        return 2
    try:
        original = json.loads(RESULTS.read_text(encoding="utf-8"))
        data = {
            "status": "post-hoc, exploratory: not part of the original blind freeze",
            "original_unblinded_utc": original.get("unblinded_utc"),
            "original_freeze_utc": original.get("freeze_utc"),
            "original_results_sha256": sha256_file(RESULTS),
            "schemes": {s: scheme_audit(s, original["schemes"][s]) for s in SCHEMES},
            "release_gate": "external review before interpreting results; NOT a discovery claim",
        }
        out.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with out.open("x", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=2) + "\n")
    except (OSError, ValueError, KeyError, RuntimeError, np.linalg.LinAlgError) as exc:
        print("COVARIANCE_AUDIT_FAILED:", type(exc).__name__, str(exc)[:220], file=sys.stderr)
        return 2
    print("PASS: post-hoc audit written privately; no original or public result overwritten.")
    print("Please review original-vs-joint differences and simulation assumptions before publication.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
