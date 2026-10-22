"""Exploratory synthetic single-edge GLR pilot, not a Planck map analysis.

Run: python scripts/run_edge_fault_pilot.py
Fixed seeds and complete simulation configuration are recorded in results/edge_fault_pilot.json.
The GLR is an established statistical baseline, not a claimed novel cosmology detection.
"""
from __future__ import annotations
from itertools import combinations
from pathlib import Path
import json, sys
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/"src"))
from cmbcst.edge_diagnostics import edge_glrt,null_max_scores,monte_carlo_pvalue

n=8
edges=list(combinations(range(n),2))
a=np.zeros((len(edges),n))
for k,(i,j) in enumerate(edges):
    a[k,i],a[k,j]=1.0,-1.0
rng=np.random.default_rng(20261022)
z=rng.normal(size=(28,28))
sd=np.linspace(0.65,1.45,28)
cov=np.diag(sd)@(0.35*z@z.T/28+0.65*np.eye(28))@np.diag(sd)
cal=null_max_scores(cov,a,4000,seed=23)
threshold=float(np.quantile(cal,0.95,method="higher"))
holdout=null_max_scores(cov,a,4000,seed=24)
edge=9
baseline=edge_glrt(np.zeros(28),cov,a)
amplitude=4.0*baseline["sigma"][edge]
noise=rng.standard_normal((4000,28))@np.linalg.cholesky(cov).T
scores=[];located=[]
for y in noise:
    y[edge]+=amplitude
    result=edge_glrt(y,cov,a)
    scores.append(float(np.max(result["delta_chi2"])))
    located.append(result["index"]==edge)
scores=np.asarray(scores)
located=np.asarray(located,dtype=bool)
out={
    "status":"exploratory synthetic pilot, not observational or preregistered",
    "seed":20261022,"n_maps":8,"n_edges":28,
    "null_calibration_seed":23,"null_holdout_seed":24,
    "n_calibration":4000,"n_holdout_null":4000,"n_injected":4000,
    "injected_edge_zero_based":edge,"injected_edge_vertices":list(edges[edge]),
    "amplitude_in_profile_sigma":4.0,"nominal_familywise_alpha":0.05,
    "calibrated_max_glrt_threshold":threshold,
    "heldout_null_fwer":float(np.mean(holdout>threshold)),
    "injected_detection_power":float(np.mean(scores>threshold)),
    "correct_localization_unconditional":float(np.mean(located)),
    "joint_detection_correct_localization":float(np.mean(located&(scores>threshold))),
    "first_injection_empirical_p":monte_carlo_pvalue(float(scores[0]),cal)
}
dest=REPO/"results"/"edge_fault_pilot.json"
dest.parent.mkdir(exist_ok=True)
dest.write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
