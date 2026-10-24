"""Explicit simulated inputs for integration qualification, not observed skies."""
from __future__ import annotations

import numpy as np

from .family import harmonic_layout


def simulated_split_alms(lmax=64, seed=82024102027):
    """Common Gaussian E/B sky plus independent isotropic Gaussian split noise.

    All amplitudes are declared in uK_CMB. This fixture has no foreground,
    correlated noise, scan filtering or instrumental systematics model.
    """
    ell, m = harmonic_layout(lmax)
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**63:
        raise ValueError('a nonnegative explicit int64 seed is required')
    generator = np.random.default_rng(seed)
    def normal():
        x = (generator.normal(size=(2, len(ell)))+1j*generator.normal(size=(2, len(ell))))/np.sqrt(2)
        x[:, m == 0] = generator.normal(size=(2, np.count_nonzero(m == 0)))
        x[:, ell < 2] = 0
        return x
    sky = normal()*np.array([8., 1.6])[:, None]/(ell+1)
    noise_a, noise_b = [normal()*.35/(ell+1) for _ in range(2)]
    return np.stack([sky+noise_a, sky+noise_b])
