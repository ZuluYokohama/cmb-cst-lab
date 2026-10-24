"""Map-facing preparation without claiming that an adjoint inverts the sky."""
from __future__ import annotations

import numpy as np

from .family import RotationFamily
from .models import SplitPair, array_sha256
from .operators import DuccHealpixOperator, HealpyHealpixOperator


class PreparedCounterfactualFamily:
    """A retained-harmonic family and the original observed-map residual.

    A candidate map adds the same synthesized coadd increment to both original
    splits. The original residual is carried unchanged, including information
    not represented by the chosen band limit. Full-sky Gram invariants concern
    the canonical retained alms, not a newly claimed inverse of pixel data.
    """
    def __init__(self, pair, family, operator):
        self.pair, self.family, self.operator = pair, family, operator
        self.reference_maps = np.stack([pair.a, pair.b])
        self.reference_maps.setflags(write=False)
        modeled = np.stack([operator.synthesize(x) for x in family.reference_alms])
        self.residual_maps = self.reference_maps-modeled
        self.residual_maps.setflags(write=False)

    def candidate_maps(self, theta):
        if np.asarray(theta).dtype.kind not in 'iuf':
            raise ValueError('rotation angles must be real')
        value = np.asarray(theta, dtype=float)
        if value.shape != (self.family.nplanes,):
            raise ValueError('one explicit angle vector required for map reconstruction')
        candidate = self.family.candidate_alms(value)
        delta = self.operator.synthesize(candidate.mean(axis=0)-self.family.coadd_alms)
        return self.reference_maps+delta[None]

    def candidate_pair(self, theta):
        a, b = self.candidate_maps(theta)
        return SplitPair(a, b, self.pair.mask_a, self.pair.mask_b,
                         self.pair.metadata, self.pair.input_sha256)

    def provenance(self):
        return dict(format='cmbcst.prepared_counterfactual_family.v1',
                    input=self.pair.provenance(), family=self.family.provenance(),
                    operator=self.operator.provenance(),
                    residual_maps_sha256=array_sha256(self.residual_maps),
                    reference_maps_sha256=array_sha256(self.reference_maps),
                    analysis='area-weighted synthesis adjoint of explicit zero-filled validity view',
                    reconstruction='original observed maps plus common synthesized coadd increment',
                    spectral_invariant_scope='canonical retained alms only',
                    physical_noise_model_validated=False)


def generate_counterfactual_family(pair, *, n_planes=128, shells=None,
                                   planes_per_shell=4, seed=82024102027,
                                   backend='ducc0', nthreads=8):
    """Prepare a bounded, empirical-difference-preserving map intervention.

    Explicit common known response and validity are required in this first
    profile. Metadata declarations are bound, not treated as instrument proofs.
    A requested plane count that cannot fit the declared shells fails clearly.
    """
    if not isinstance(pair, SplitPair):
        raise TypeError('a validated SplitPair is required')
    p = pair.metadata
    if not p.same_observing_response or p.response_a.kind == 'unknown':
        raise ValueError('this family requires a declared common known observing response')
    if not np.array_equal(pair.mask_a, pair.mask_b):
        raise ValueError('this first profile requires common split validity')
    if (not isinstance(n_planes, (int, np.integer)) or isinstance(n_planes, bool)
            or n_planes < 1 or not isinstance(planes_per_shell, (int, np.integer))
            or isinstance(planes_per_shell, bool) or planes_per_shell < 1
            or n_planes % planes_per_shell):
        raise ValueError('plane count must be a positive multiple of planes_per_shell')
    canonical = pair.converted(units='uK_CMB', polarization_convention='COSMO', ordering='RING')
    if shells is None:
        count = int(n_planes)//int(planes_per_shell)
        # Guarantee enough ambient dimensions even for rank-two differences;
        # actual rank and distinct-shell feasibility are checked by the family.
        minimum_ell = max(2, int(planes_per_shell)+1)
        if count > p.lmax-minimum_ell+1:
            raise ValueError('requested plane count exceeds the supported shell domain')
        shells = np.rint(np.linspace(minimum_ell, p.lmax, count)).astype(np.int64)
    if len(shells)*planes_per_shell != n_planes:
        raise ValueError('shell manifest and requested plane count disagree')
    classes = {'ducc0': DuccHealpixOperator, 'healpy': HealpyHealpixOperator}
    if backend not in classes:
        raise ValueError('supported first-profile backends are ducc0 and healpy')
    operator = classes[backend](p.nside, p.lmax, nthreads=nthreads)
    alms = np.stack([operator.adjoint(canonical.operator_maps(split)) for split in ('A', 'B')])
    family = RotationFamily(alms, shells, planes_per_shell=planes_per_shell, seed=seed)
    return PreparedCounterfactualFamily(canonical, family, operator)
