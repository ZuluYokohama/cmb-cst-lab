"""Explicit contracts and separately qualified CMB integration prototypes.

These adapters do not change the frozen Planck analyses. A feasible family is
not a physical noise distribution or a universal closure certificate.
"""

from .models import SplitPair, SplitMetadata, ObservingResponse, MissingOptionalDependency
from .io import (read_split_pair_fits, write_split_pair_fits,
                 read_split_pair_hdf5, write_split_pair_hdf5)
from .operators import DuccHealpixOperator, HealpyHealpixOperator
from .api import generate_counterfactual_family, PreparedCounterfactualFamily
from .family import RotationFamily
from .quadratic import SymmetricQuadraticModel
from .certificates import certify_candidate, evidence_status

__all__ = ['SplitPair', 'SplitMetadata', 'ObservingResponse', 'MissingOptionalDependency',
           'read_split_pair_fits', 'write_split_pair_fits',
           'read_split_pair_hdf5', 'write_split_pair_hdf5',
           'DuccHealpixOperator', 'HealpyHealpixOperator',
           'generate_counterfactual_family', 'PreparedCounterfactualFamily',
           'RotationFamily', 'SymmetricQuadraticModel', 'certify_candidate', 'evidence_status']
