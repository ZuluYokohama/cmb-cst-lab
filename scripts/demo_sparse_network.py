"""Synthetic sparse network experiment: python scripts/demo_sparse_network.py."""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cmbcst.sparse_network import graph_design, fit_sparse_offsets


def main():
    rng = np.random.default_rng(313)
    truth = np.array([0., .1, -.2, .4, .3])
    topologies = {
        "tree": [(0,1),(1,2),(2,3),(3,4)],
        "two_cycles": [(0,1),(1,2),(2,3),(3,4),(0,4),(1,4)],
    }
    results = {}
    for name, edges in topologies.items():
        a, _ = graph_design(5,edges)
        y = a @ truth + rng.normal(0,.01,len(edges))
        y[0] += .2
        out = fit_sparse_offsets(y,np.eye(len(edges))*.01**2,5,edges)
        results[name] = {
            "edges":len(edges),"cycle_dof":out["dof"],"chi2":out["residual_chi2"],
            "closure_chi2":out["closure_chi2"],"pte":out["pte"],
            "limitation":"no redundant constraints" if not out["dof"] else "does not uniquely identify faults",
        }
    print(json.dumps({"synthetic_only":True,"results":results},indent=2))


if __name__ == "__main__":
    main()
