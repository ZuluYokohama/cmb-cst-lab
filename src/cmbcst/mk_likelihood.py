"""Minami-Komatsu likelihood for per-map angles alpha_i and birefringence beta (numpy only).

A numpy port of the model in the authors' pipeline (github.com/antolonappan/cosmic_birefringence,
src/cosmic_bire/tools_fast.py, MIT licence, itself from J. R. Eskilt's code): for every ordered
pair of distinct maps (i, j) and each bin,

    v_ij = m_ij . (C^EE_o, C^BB_o)_ij + C^{E_i B_j}_o - b_ij . (C^EE_th, C^BB_th)_ij,

with m_ij = -lambda_vec lambda_mat^{-1} (dust EB amplitude A_k sin 4 psi_l folded in) and
b_ij = R_vec(2a+2b) + m_ij R_mat(2a+2b). Observables (EE, BB, EB)_ij get the same simplified
Gaussian covariance as the reference code, and -2 ln L = sum_b v^T C^{-1} v + ln det C.

Differences from the reference: bins carry nu_b = f_sky sum(2l+1) modes instead of a per-ell sum;
the point estimate and covariance come from Fisher scoring rather than MCMC.
"""
from __future__ import annotations

import numpy as np

# Upper bin indices for 4 dust-EB amplitudes (reference get_A_ell): l in [51,131), [131,211), [211,511), [511,...)
A_BIN_EDGES = (4, 8, 23)


def ordered_pairs(n: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(n) if i != j]


def _rvec(ti, tj):
    return np.stack([np.cos(ti) * np.sin(tj), -np.sin(ti) * np.cos(tj)], axis=-1)


def _rmat(ti, tj):
    d, a = np.cos(ti) * np.cos(tj), np.sin(ti) * np.sin(tj)
    return np.stack([np.stack([d, a], -1), np.stack([a, d], -1)], -2)


def _dvec(ti, tj):
    return np.stack([np.cos(ti) * np.cos(tj), -np.sin(ti) * np.sin(tj)], axis=-1)


def _dmat(ti, tj):
    d, a = np.cos(ti) * np.sin(tj), np.cos(tj) * np.sin(ti)
    return np.stack([np.stack([-d, -a], -1), np.stack([a, d], -1)], -2)


def model_rows(alpha: np.ndarray, beta: float, a_sin4psi: float, pairs: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """m (P, 2) acting on observed (EE, BB) and b (P, 2) acting on theory (EE, BB), one bin."""
    i = np.array([p[0] for p in pairs])
    j = np.array([p[1] for p in pairs])
    ti, tj = 2 * alpha[i], 2 * alpha[j]
    f = a_sin4psi * np.array([[1.0, 0.0], [1.0, 0.0]])
    lam_vec = _rvec(ti, tj) + _dvec(ti, tj) @ f
    lam_mat = _rmat(ti, tj) + _dmat(ti, tj) @ f
    m = -np.einsum("pk,pkl->pl", lam_vec, np.linalg.inv(lam_mat))
    tb_i, tb_j = ti + 2 * beta, tj + 2 * beta
    b = _rvec(tb_i, tb_j) + np.einsum("pk,pkl->pl", m, _rmat(tb_i, tb_j))
    return m, b


def observables(cl: np.ndarray, pairs: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Observed EE_ij, BB_ij, EB_ij (= C^{E_i B_j}) per bin, shape (nb, P) each."""
    i = np.array([p[0] for p in pairs])
    j = np.array([p[1] for p in pairs])
    return cl[:, 2 * i, 2 * j], cl[:, 2 * i + 1, 2 * j + 1], cl[:, 2 * i, 2 * j + 1]


def obs_covariance(cl: np.ndarray, nu: np.ndarray, pairs: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gaussian covariance blocks (EE-EE, BB-BB, EB-EB) per bin, as in the reference code."""
    # v = (i, j) indexes rows, h = (p, q) columns:
    #   Cov(EE_ij, EE_pq) = EE_ip EE_jq + EE_iq EE_pj,  same for BB,  Cov(EB_ij, EB_pq) = EE_ip BB_jq
    i = np.array([p[0] for p in pairs])
    j = np.array([p[1] for p in pairs])
    rv_i, rv_j = i[:, None], j[:, None]
    ch_p, ch_q = i[None, :], j[None, :]
    ee = cl[:, 0::2, 0::2]
    bb = cl[:, 1::2, 1::2]
    s_ee = ee[:, rv_i, ch_p] * ee[:, rv_j, ch_q] + ee[:, rv_i, ch_q] * ee[:, ch_p, rv_j]
    s_bb = bb[:, rv_i, ch_p] * bb[:, rv_j, ch_q] + bb[:, rv_i, ch_q] * bb[:, ch_p, rv_j]
    s_eb = ee[:, rv_i, ch_p] * bb[:, rv_j, ch_q]
    k = nu[:, None, None]
    return s_ee / k, s_bb / k, s_eb / k


def a_index(nb: int, n_a: int) -> np.ndarray:
    """Which dust amplitude applies in each bin (4-amplitude scheme of the reference)."""
    if n_a == 0:
        return np.full(nb, -1)
    if n_a == 4:
        if nb <= A_BIN_EDGES[-1]:
            raise ValueError(f"4 dust amplitudes need more than {A_BIN_EDGES[-1]} bins, got {nb}")
        return np.searchsorted(np.array(A_BIN_EDGES), np.arange(nb), side="right")
    if n_a == 1:
        return np.zeros(nb, dtype=int)
    raise ValueError("n_a must be 0, 1 or 4")


class MKLikelihood:
    def __init__(self, cl, nu, th_ee, th_bb, psi=None, n_a: int = 0, ln_det: bool = True):
        """cl: (nb, 2n, 2n) observed; th_ee/th_bb: (nb, n, n) beam-smoothed theory; psi: (nb,) radians."""
        self.n = cl.shape[1] // 2
        self.pairs = ordered_pairs(self.n)
        self.nb = cl.shape[0]
        self.ee, self.bb, self.eb = observables(cl, self.pairs)
        i = np.array([p[0] for p in self.pairs])
        j = np.array([p[1] for p in self.pairs])
        self.th = np.stack([th_ee[:, i, j], th_bb[:, i, j]], axis=-1)  # (nb, P, 2)
        self.s_ee, self.s_bb, self.s_eb = obs_covariance(cl, nu, self.pairs)
        self.psi = np.zeros(self.nb) if psi is None else np.asarray(psi)
        self.n_a = n_a
        self.aidx = a_index(self.nb, n_a)
        self.ln_det = ln_det

    def unpack(self, theta):
        return theta[: self.n], theta[self.n], theta[self.n + 1:]

    def residuals(self, theta) -> tuple[list[np.ndarray], list[np.ndarray]]:
        alpha, beta, amps = self.unpack(theta)
        vs, cs = [], []
        for b in range(self.nb):
            a = amps[self.aidx[b]] * np.sin(4 * self.psi[b]) if self.n_a else 0.0
            m, bt = model_rows(alpha, beta, a, self.pairs)
            v = m[:, 0] * self.ee[b] + m[:, 1] * self.bb[b] + self.eb[b] - np.sum(bt * self.th[b], axis=1)
            c = np.outer(m[:, 0], m[:, 0]) * self.s_ee[b] + np.outer(m[:, 1], m[:, 1]) * self.s_bb[b] + self.s_eb[b]
            vs.append(v)
            cs.append(c)
        return vs, cs

    def neg2lnl(self, theta) -> float:
        vs, cs = self.residuals(theta)
        out = 0.0
        for v, c in zip(vs, cs):
            out += float(v @ np.linalg.solve(c, v))
            if self.ln_det:
                out += float(np.linalg.slogdet(c)[1])
        return out

    def fit(self, theta0=None, free=None, n_iter: int = 20, h: float = 1e-6, tol: float = 1e-10) -> dict[str, object]:
        """Fisher scoring: linearise v in theta with C held at the current point, step, repeat.

        `free` lists the parameter indices to fit (default all); the rest stay at theta0. With
        beta fixed at 0 and no dust term, the fitted alphas are total angles theta_m = alpha_m + beta.
        """
        k = self.n + 1 + self.n_a
        theta = np.zeros(k) if theta0 is None else np.array(theta0, dtype=float)
        free = np.arange(k) if free is None else np.asarray(free)
        for it in range(n_iter):
            vs, cs = self.residuals(theta)
            jac = []
            for p in free:
                tp = theta.copy()
                tp[p] += h
                vp, _ = self.residuals(tp)
                jac.append([(a - b) / h for a, b in zip(vp, vs)])
            nf = len(free)
            fisher = np.zeros((nf, nf))
            grad = np.zeros(nf)
            for b in range(self.nb):
                jb = np.stack([jac[p][b] for p in range(nf)], axis=1)  # (P, nf)
                cinv_j = np.linalg.solve(cs[b], jb)
                fisher += jb.T @ cinv_j
                grad += cinv_j.T @ vs[b]
            step = -np.linalg.solve(fisher, grad)
            theta[free] = theta[free] + step
            if np.max(np.abs(step)) < tol:
                break
        cov = np.zeros((k, k))
        cov[np.ix_(free, free)] = np.linalg.inv(fisher)
        alpha, beta, amps = self.unpack(theta)
        return {"theta": theta, "cov": cov, "alpha": alpha, "beta": beta, "A": amps,
                "cov_alpha": cov[: self.n, : self.n], "sigma_beta": float(np.sqrt(cov[self.n, self.n])),
                "neg2lnl": self.neg2lnl(theta), "iterations": it + 1}


def anchored_beta(theta: np.ndarray, cov_theta: np.ndarray, anchor: np.ndarray, cov_anchor: np.ndarray,
                  cov_cross: np.ndarray | None = None) -> dict[str, object]:
    """GLS beta from per-map beta_m = theta_m - alpha_m^MK (reference anchored.beta_profile).

    C = C_theta + C_anchor - C_x - C_x^T; with C_x omitted (zero) this double-counts shared data,
    which the reference also notes.
    """
    b = theta - anchor
    cx = np.zeros_like(cov_theta) if cov_cross is None else cov_cross
    c = cov_theta + cov_anchor - cx - cx.T
    cinv = np.linalg.inv(c)
    one = np.ones(len(b))
    var = 1.0 / float(one @ cinv @ one)
    return {"beta": var * float(one @ cinv @ b), "sigma": float(np.sqrt(var)), "per_map": b,
            "per_map_sigma": np.sqrt(np.diag(c))}
