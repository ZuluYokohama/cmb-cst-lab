"""Realized, binned coupled pseudo-C_ell constraints for a fixed HEALPix mask.

NaMaster is used to build the linear masked/purified response and independently
evaluate reconstructed candidates. Its ensemble coupling matrix is not used as
a surrogate for the spectra of an individual realization. This initial profile
supports a common scalar mask, no templates and a declared common response.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from .family import harmonic_layout
from .operators import DuccHealpixOperator
from .quadratic import SymmetricQuadraticModel


def array_digest(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256(array.dtype.str.encode()+str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _real_values(value, label, allow_boolean=False):
    raw = np.asarray(value)
    if raw.dtype.kind not in ('biuf' if allow_boolean else 'iuf'):
        raise ValueError(f'{label} must contain native real numeric values')
    return np.asarray(raw, dtype=np.float64)


@dataclass(frozen=True)
class CutSkyProfile:
    nside: int
    lmax: int
    band_edges: tuple[int, ...]
    purify_e: bool = False
    purify_b: bool = True
    n_iter: int = 3
    n_iter_mask: int = 3
    nthreads: int = 8
    mask_lmax: int | None = None

    def __post_init__(self):
        integer = lambda x: isinstance(x, (int, np.integer)) and not isinstance(x, bool)
        if not integer(self.nside) or self.nside < 1 or self.nside & (self.nside-1):
            raise ValueError('positive power-of-two HEALPix NSIDE required')
        if not integer(self.lmax) or not 2 <= self.lmax <= 3*self.nside-1:
            raise ValueError('supported spin-2 HEALPix band limit required')
        edges = tuple(self.band_edges)
        if (len(edges) < 2 or not all(integer(x) for x in edges)
                or edges[0] < 2 or edges[-1] > self.lmax+1
                or any(a >= b for a, b in zip(edges, edges[1:]))):
            raise ValueError('strictly increasing, in-range band edges required')
        if type(self.purify_e) is not bool or type(self.purify_b) is not bool:
            raise ValueError('purification choices must be explicit booleans')
        if (not integer(self.n_iter) or not integer(self.n_iter_mask)
                or not 0 <= self.n_iter <= 10 or not 0 <= self.n_iter_mask <= 10
                or not integer(self.nthreads) or not 1 <= self.nthreads <= 8):
            raise ValueError('bounded iterations and one to eight threads required')
        mask_lmax = self.lmax if self.mask_lmax is None else self.mask_lmax
        if not integer(mask_lmax) or not 2 <= mask_lmax <= 3*self.nside-1:
            raise ValueError('supported mask harmonic band limit required')
        if (self.purify_e or self.purify_b) and mask_lmax != self.lmax:
            raise ValueError('this qualified purification profile requires matching field and mask band limits')
        object.__setattr__(self, 'mask_lmax', int(mask_lmax))
        object.__setattr__(self, 'band_edges', tuple(int(x) for x in edges))

    def manifest(self):
        return dict(nside=int(self.nside), lmax=int(self.lmax),
                    band_edges=list(self.band_edges), purify_e=self.purify_e,
                    purify_b=self.purify_b, n_iter=int(self.n_iter),
                    n_iter_mask=int(self.n_iter_mask), nthreads=int(self.nthreads),
                    mask_lmax=int(self.mask_lmax),
                    representation='coupled_binned_pseudo_cl',
                    pairs=['AA', 'BB', 'AB'], channels=['EE', 'EB', 'BE', 'BB'],
                    polarization_convention='COSMO', ordering='RING',
                    templates=None, anisotropic_weights=False,
                    beam_deconvolution=False, pixel_window_deconvolution=False)


class NaMasterCutSky:
    """One fixed observing profile; cache invalidation is explicit by hashes.

    The common harmonic response is applied before masking. A unit response is
    an explicit supported model, not an inference of an instrument's beam.
    Returned features are AA/BB/AB x EE/EB/BE/BB x band, in that order.
    """
    def __init__(self, profile, mask, response_window=None):
        if not isinstance(profile, CutSkyProfile):
            raise TypeError('CutSkyProfile required')
        try:
            import pymaster as nmt
        except ImportError as exc:
            raise ImportError('NaMaster is optional; install the cutsky integration extra') from exc
        self.profile, self.nmt = profile, nmt
        value = _real_values(mask, 'mask', allow_boolean=True)
        if (value.shape != (12*profile.nside**2,) or not np.isfinite(value).all()
                or np.any(value < 0) or np.any(value > 1) or not np.any(value > 0)):
            raise ValueError('finite nonempty scalar HEALPix mask in [0,1] required')
        self.mask = value.copy()
        self.mask.setflags(write=False)
        window = np.ones(profile.lmax+1) if response_window is None else _real_values(response_window, 'response window', allow_boolean=True)
        if window.shape != (profile.lmax+1,) or not np.isfinite(window).all() or np.any(window < 0):
            raise ValueError('explicit finite nonnegative common harmonic response required')
        self.response_window = window.copy()
        self.response_window.setflags(write=False)
        self.operator = DuccHealpixOperator(profile.nside, profile.lmax, nthreads=profile.nthreads)
        self.ell, self.m = harmonic_layout(profile.lmax)
        self.binning = nmt.NmtBin.from_edges(np.asarray(profile.band_edges[:-1]),
                                           np.asarray(profile.band_edges[1:]), is_Dell=False)
        self.bin_weights = np.asarray(self.binning.bin_cell(np.eye(profile.lmax+1))).T
        self.labels = [f'{pair}:{channel}:{lo}-{hi-1}' for pair in ('AA', 'BB', 'AB')
                       for channel in ('EE', 'EB', 'BE', 'BB')
                       for lo, hi in zip(profile.band_edges, profile.band_edges[1:])]

    def _field_from_map(self, maps):
        value = _real_values(maps, 'observed Q/U maps')
        if value.shape != (2, len(self.mask)) or not np.isfinite(value).all():
            raise ValueError('finite COSMO Q/U map required')
        p = self.profile
        return self.nmt.NmtField(self.mask, value, spin=2, lmax=p.lmax,
                                lmax_mask=p.mask_lmax, purify_e=p.purify_e,
                                purify_b=p.purify_b, n_iter=p.n_iter,
                                n_iter_mask=p.n_iter_mask, masked_on_input=False,
                                templates=None, beam=None)

    def observe(self, alms):
        value = np.asarray(alms, dtype=np.complex128)
        if value.shape != (2, len(self.ell)) or not np.isfinite(value).all():
            raise ValueError('finite [E/B,nalm] coefficients required')
        return self.operator.synthesize(value*self.response_window[self.ell])

    def _field(self, alms):
        return self._field_from_map(self.observe(alms))

    def direct_features_from_maps(self, maps):
        value = _real_values(maps, 'observed split maps')
        if value.shape != (2, 2, len(self.mask)) or not np.isfinite(value).all():
            raise ValueError('finite [split,Q/U,pixel] maps required')
        a, b = [self._field_from_map(x) for x in value]
        return np.stack([self.binning.bin_cell(self.nmt.compute_coupled_cell(x, y))
                         for x, y in ((a, a), (b, b), (a, b))]).ravel()

    def direct_features(self, split_alms):
        value = np.asarray(split_alms)
        if value.shape != (2, 2, len(self.ell)):
            raise ValueError('complete [split,E/B,nalm] state required')
        return self.direct_features_from_maps(np.stack([self.observe(x) for x in value]))

    def candidate_maps(self, family, theta, reference_maps=None):
        candidate = family.candidate_alms(theta)
        if np.shape(theta) != (family.nplanes,):
            raise ValueError('one angle vector required for direct map reconstruction')
        if reference_maps is None:
            return np.stack([self.observe(x) for x in candidate])
        reference = self._observed_reference(reference_maps)
        delta = self.observe(candidate.mean(axis=0)-family.coadd_alms)
        return reference+delta[None]

    def _observed_reference(self, reference_maps):
        value = _real_values(reference_maps, 'observed reference maps')
        if value.shape != (2, 2, len(self.mask)) or not np.isfinite(value).all():
            raise ValueError('finite observed reference split maps required')
        if not np.all(self.response_window == 1):
            raise ValueError('observed-reference profile requires unit extra response; input beam is already carried by the maps')
        return value

    def build_prepared_model(self, prepared, budget=None):
        """Compose strict map metadata/validity with the observing operator."""
        from .api import PreparedCounterfactualFamily
        if not isinstance(prepared, PreparedCounterfactualFamily):
            raise TypeError('a prepared, validated SplitPair family is required')
        pair = prepared.pair
        if pair.metadata.nside != self.profile.nside or pair.metadata.lmax != self.profile.lmax:
            raise ValueError('prepared-map and observing geometries disagree')
        invalid = ~(pair.mask_a & pair.mask_b)
        if np.any(self.mask[invalid] != 0):
            raise ValueError('the observing mask must exclude every invalid split pixel')
        return self.build_model(prepared.family, budget=budget, reference_maps=prepared.reference_maps)

    def build_model(self, family, budget=None, reference_maps=None):
        """Cache real quadratic forms, checking the budget between transforms.

        There is no assumption of invertible map analysis. NaMaster's actual
        linear masked/purified alms are computed for each numerical basis map.
        """
        if family.lmax != self.profile.lmax:
            raise ValueError('family and observing band limits differ')
        check = budget if budget is not None else lambda: None
        observed = None if reference_maps is None else self._observed_reference(reference_maps)
        transformed = []
        for basis in np.concatenate((family.cos_basis, family.sin_basis)):
            check()
            transformed.append(np.asarray(self._field(basis).get_alms()).copy())
        basis_alms = np.asarray(transformed)
        check()
        field_a = self._field(family.reference_alms[0]) if observed is None else self._field_from_map(observed[0])
        fa = np.concatenate((np.asarray(field_a.get_alms())[None], basis_alms))
        check()
        field_b = self._field(family.reference_alms[1]) if observed is None else self._field_from_map(observed[1])
        fb = np.concatenate((np.asarray(field_b.get_alms())[None], basis_alms))
        if fa.shape != (1+2*family.nplanes, 2, len(self.ell)) or not np.isfinite(fa).all() or not np.isfinite(fb).all():
            raise ValueError('unexpected NaMaster harmonic response layout')
        quadratics = []
        for left, right in ((fa, fa), (fb, fb), (fa, fb)):
            for ex, ey in ((0, 0), (0, 1), (1, 0), (1, 1)):
                for band_weights in self.bin_weights:
                    check()
                    weight = band_weights[self.ell]*np.where(self.m == 0, 1., 2.)/(2*self.ell+1.)
                    take = weight != 0
                    raw = ((left[:, ex][:, take]*weight[take]) @ right[:, ey][:, take].conj().T).real
                    quadratics.append((raw+raw.T)/2)
        model = SymmetricQuadraticModel(np.stack(quadratics))
        reference = self.direct_features(family.reference_alms) if observed is None else self.direct_features_from_maps(observed)
        # Relative power scales, including a fixed floor for small EB channels.
        power_floor = max(float(np.max(np.abs(reference)))*1e-3, np.finfo(float).tiny)
        scales = np.maximum(np.abs(reference), power_floor)
        error = float(np.max(np.abs(model.features(np.zeros(family.nplanes))-reference)/scales))
        return model, scales, dict(reference_features=reference, zero_model_scaled_error=error,
                                   scale_rule='max(abs(reference_feature),1e-3*max(abs(reference_features)))',
                                   transformed_basis_sha256=array_digest(basis_alms),
                                   reference_masked_A_sha256=array_digest(fa[0]),
                                   reference_masked_B_sha256=array_digest(fb[0]),
                                   reference_mode='synthesized_retained_alms' if observed is None else 'original_observed_maps_with_unchanged_residual',
                                   reference_maps_sha256=None if observed is None else array_digest(observed))

    def provenance(self):
        source = Path(__file__)
        versions = {name: importlib.metadata.version(name) for name in ('pymaster', 'healpy', 'ducc0', 'numpy')}
        description = dict(profile=self.profile.manifest(), mask_sha256=array_digest(self.mask),
                           response_sha256=array_digest(self.response_window), versions=versions,
                           source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                           supported_scope='common scalar mask/common response; no deprojection; coupled binned realized spectra')
        description['operator_sha256'] = hashlib.sha256(json.dumps(description, sort_keys=True, allow_nan=False).encode()).hexdigest()
        return description


def cap_mask(nside, fraction, apodization_deg, *, holes=True):
    """Deterministic simulated footprint with a cosine rim and optional holes."""
    import ducc0
    if (not isinstance(nside, int) or nside < 1 or nside & (nside-1)
            or not np.isfinite(fraction) or not 0 < fraction <= 1
            or not np.isfinite(apodization_deg) or apodization_deg <= 0
            or type(holes) is not bool):
        raise ValueError('valid HEALPix cap geometry and positive rim width required')
    grid = ducc0.healpix.Healpix_Base(nside, 'RING')
    loc = grid.pix2ang(np.arange(grid.npix()))
    theta = loc[:, 0]
    cut = np.arccos(1-2*fraction)
    distance = cut-theta
    if holes:
        vectors = grid.pix2vec(np.arange(grid.npix()))
        for phi in (.4, 2.0):
            t = .55*cut
            center = np.array([np.sin(t)*np.cos(phi), np.sin(t)*np.sin(phi), np.cos(t)])
            outside = np.arccos(np.clip(vectors@center, -1, 1))-np.radians(4.)
            distance = np.minimum(distance, outside)
    u = np.clip(distance/np.radians(apodization_deg), 0, 1)
    return .5*(1-np.cos(np.pi*u))
