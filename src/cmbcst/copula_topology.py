"""Marginal-invariant normal scores for an exploratory Gaussian-copula topology null."""
from __future__ import annotations

import numpy as np

from .surrogate_topology import checked_field


def normal_scores(field: np.ndarray) -> np.ndarray:
    """Map midranks to normal quantiles; preserve ties and strict ordering.

    Strictly monotonic changes to brightness leave this field unchanged. This
    changes the question to the geometry of ranks, not the original CMB spectrum.
    """
    from statistics import NormalDist

    x = checked_field(field)
    unique, inverse, counts = np.unique(x.ravel(), return_inverse=True, return_counts=True)
    centers = np.cumsum(counts) - .5*counts
    nd = NormalDist()
    scores = np.array([nd.inv_cdf(float(c / x.size)) for c in centers])
    out = scores[inverse].reshape(x.shape)
    return (out - out.mean()) / out.std()
