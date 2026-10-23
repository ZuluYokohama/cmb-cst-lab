"""Experimental connected sparse-graph scalar offset calibration.

Each oriented edge (i,j) measures y_ij = x_i - x_j + noise.
Spanning-tree fundamental cycles provide independent closure diagnostics.
The graph is sparse; the covariance and GLS linear algebra are dense.
No rigid-body rotations, clock wrapping, robust outlier modeling, or
production sensor/security claims are implemented.
"""
from __future__ import annotations

from collections import deque
import numpy as np
from .network import chi2_sf


def graph_design(n: int, edges: list[tuple[int, int]],
                 root: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return edge incidence A and full-row-rank fundamental-cycle matrix Z.

    For a connected graph, Z @ A == 0 and rank(Z) = m - n + 1.
    For a tree, Z has zero rows: no redundant measurements.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ValueError("n must be an integer >= 2")
    if isinstance(root, bool) or not isinstance(root, int) or not 0 <= root < n:
        raise ValueError("invalid root")
    edge_list = list(edges)
    if len(edge_list) < n - 1:
        raise ValueError("not enough edges for a connected graph")
    seen = set()
    adj = [[] for _ in range(n)]
    a = np.zeros((len(edge_list), n))
    for e, pair in enumerate(edge_list):
        if (not isinstance(pair, (tuple, list)) or len(pair) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in pair)):
            raise ValueError("each edge must contain two integer vertex indices")
        i, j = map(int, pair)
        if not (0 <= i < n and 0 <= j < n) or i == j:
            raise ValueError("invalid edge endpoints")
        key = (min(i, j), max(i, j))
        if key in seen:
            raise ValueError("duplicate undirected edge")
        seen.add(key)
        a[e, i], a[e, j] = 1.0, -1.0
        adj[i].append((j, e, +1))
        adj[j].append((i, e, -1))

    parent = [-1] * n
    parent_edge = [-1] * n
    parent_dir = [0] * n
    parent[root] = root
    queue = deque([root])
    tree = set()
    while queue:
        u = queue.popleft()
        for v, e, direction in adj[u]:
            if parent[v] != -1:
                continue
            parent[v], parent_edge[v], parent_dir[v] = u, e, direction
            tree.add(e)
            queue.append(v)
    if any(p == -1 for p in parent):
        raise ValueError("graph must be connected")

    rows = []
    for e, (i, j) in enumerate(edge_list):
        if e in tree:
            continue
        row = np.zeros(len(edge_list))
        row[e] = 1.0
        u = i
        while u != root:
            row[parent_edge[u]] += parent_dir[u]
            u = parent[u]
        u = j
        while u != root:
            row[parent_edge[u]] -= parent_dir[u]
            u = parent[u]
        rows.append(row)
    z = np.stack(rows) if rows else np.zeros((0, len(edge_list)))
    return a, z


def fit_sparse_offsets(measurements: np.ndarray, covariance: np.ndarray,
                       n_sensors: int, edges: list[tuple[int, int]],
                       gauge: int = 0) -> dict[str, object]:
    """Gauge-fixed GLS and covariance-aware cycle closure on a connected graph.

    The nominal chi-square PTE requires correct known Gaussian edge covariance.
    A tree has no closure redundancy: its PTE is None.
    """
    a, z = graph_design(n_sensors, edges, root=gauge)
    m, n = a.shape
    y = np.asarray(measurements, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    if y.shape != (m,) or cov.shape != (m, m):
        raise ValueError("measurements/covariance shape mismatch")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(cov))):
        raise ValueError("measurements and covariance must be finite")
    atol = 1e-12 * float(np.max(np.abs(cov)))
    if not np.allclose(cov, cov.T, atol=atol, rtol=1e-10):
        raise ValueError("covariance must be symmetric")
    cov = (cov + cov.T) / 2.0
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be positive definite") from exc
    keep = [i for i in range(n) if i != gauge]
    ar = a[:, keep]
    aw = np.linalg.solve(chol, ar)
    yw = np.linalg.solve(chol, y)
    estimate, _, rank, _ = np.linalg.lstsq(aw, yw, rcond=1e-12)
    if rank != n - 1:
        raise ValueError("numerically rank-deficient network")
    x = np.zeros(n)
    x[keep] = estimate
    cov_red = np.linalg.inv(aw.T @ aw)
    cov_x = np.zeros((n, n))
    cov_x[np.ix_(keep, keep)] = cov_red
    residual = y - a @ x
    rw = yw - aw @ estimate
    chi2 = float(rw @ rw)
    c = z @ y
    dof = m - n + 1
    if dof:
        c_cov = z @ cov @ z.T
        uc = np.linalg.solve(np.linalg.cholesky(c_cov), c)
        chi2_cl = float(uc @ uc)
    else:
        c_cov = np.zeros((0, 0))
        chi2_cl = 0.0
    if abs(chi2 - chi2_cl) > 1e-7 * max(1.0, chi2, chi2_cl):
        raise ValueError("residual and closure disagree numerically")
    return {
        "offsets": x, "offset_covariance": cov_x,
        "fitted_edges": a @ x, "residual": residual,
        "cycle_basis": z, "closure": c, "closure_covariance": c_cov,
        "residual_chi2": chi2, "closure_chi2": chi2_cl, "dof": dof,
        "pte": chi2_sf(chi2, dof) if dof else None,
        "cycles_available": bool(dof),
    }
