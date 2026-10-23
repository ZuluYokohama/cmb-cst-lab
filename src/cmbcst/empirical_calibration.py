"""Finite-sample rank calibration against independent reference sessions.

Only *whole-session exchangeability* supports the reference-rank p-value.
Within each session, observations may be temporally dependent. The reference
and tested scores must nevertheless have the same null law, must be computed
using a fixed scoring algorithm, and must come from independent/exchangeable
sessions. Shared/calibration-contaminated observations invalidate this claim.

This utility does not identify a faulty sensor or validate operational security.
"""
from __future__ import annotations

import numpy as np


def upper_tail_rank_pvalues(scores: np.ndarray, reference_scores: np.ndarray) -> np.ndarray:
    """Conservative upper-tail empirical p-values from a separate reference bank.

    p(t) = (1 + number of reference scores >= t) / (N_reference + 1).

    For a candidate null score exchangeable with the N reference scores and
    a pre-specified score, the p-value is superuniform, with ties handled
    conservatively. Reference scores must NOT include a candidate evaluated
    here. Returns a 1-D vector; scalar candidates must be supplied as [value].
    """
    candidate = np.asarray(scores, dtype=float)
    reference = np.asarray(reference_scores, dtype=float)
    if candidate.ndim != 1 or reference.ndim != 1 or not len(candidate) or not len(reference):
        raise ValueError("scores and reference_scores must be nonempty 1-D arrays")
    if not (np.all(np.isfinite(candidate)) and np.all(np.isfinite(reference))):
        raise ValueError("scores and reference_scores must be finite")
    sorted_ref = np.sort(reference)
    # side='left' counts all tied reference values as >= candidate
    ge_count = len(reference) - np.searchsorted(sorted_ref, candidate, side="left")
    return (1.0 + ge_count) / (len(reference) + 1.0)
