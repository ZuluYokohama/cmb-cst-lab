# CMB integration v1: maps, retained harmonics and cut-sky responses

`cmbcst.integration` is the first supported HEALPix integration slice. It reads
explicit split-map products, constructs a stored harmonic counterfactual family,
retains the original pixel residual, and can cache selected realized NaMaster
responses as quadratic forms. This guide describes the implemented API; the
separate [synthetic feasibility report](CUTSKY_FEASIBILITY_RUN.md) records
48 independently verified finite candidates across 12 cases.

The [research plan](CMB_INTEGRATION_RESEARCH_PLAN.md) describes the wider program.
The [earlier Planck v2 study](CLOSURE_FIBERS_V2_RUN.md) supplies prior evidence
under its own frozen protocol. These new adapters do not replace those analyses
or extend their conclusions to another observing pipeline.

## Installation and execution environment

From the repository root, install the extra appropriate to the requested path:

```bash
python -m pip install -e '.[integration]'
# Include the NaMaster cut-sky adapter:
python -m pip install -e '.[cutsky]'
```

The pins in [pyproject.toml](../pyproject.toml) are:

| Extra | Additional packages |
|---|---|
| `integration` | `h5py==3.16.0`, `healpy==1.20.1` |
| `cutsky` | The same packages, plus `fitsio==1.4.2`, `pymaster==3.0.1` |

NumPy, SciPy, astropy and ducc0 are base dependencies. The isolated WSL target
has exercised NaMaster 3.0.1, healpy 1.20.1 and h5py 3.16.0. This is the qualified
optional-library environment for this slice; native Windows cut-sky support is
not claimed. Missing requested dependencies raise an explicit error or cause a
clearly reported optional test skip. An absent backend is never substituted with
a passing result from another backend.

NaMaster's Python installation may compile its C library. Its maintainers list
GSL 2, FFTW 3 with OpenMP support and CFITSIO greater than version 3 as build
dependencies, with HEALPix C support built or supplied separately. Follow the
[maintainer installation instructions](https://github.com/LSSTDESC/NaMaster#installation)
and [dependency list](https://github.com/LSSTDESC/NaMaster#dependencies) for the
target system.

Set process thread limits **before importing** numerical backends, for example
in a new Linux/WSL shell:

```bash
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
```

Ducc receives an explicit one-to-eight `nthreads` argument. Healpy's spin
functions have no per-call thread argument: their OpenMP configuration remains
the caller's responsibility, and provenance reports that distinction. Optional
Torch quadratic evaluation requires a separately installed, compatible Torch
environment; neither I/O nor CPU cut-sky model construction requires CUDA.

## Split-map contract

`SplitPair` holds `a` and `b` Q/U arrays of shape `[2, 12*nside**2]`, separate
Boolean `mask_a`/`mask_b` validity arrays, and `SplitMetadata`. Finite masked
samples are retained. `pair.operator_maps('A')` or `'B'` explicitly returns a
zero-filled analysis view without overwriting those observations.

| Field | Supported declaration |
|---|---|
| Geometry | Positive power-of-two `nside`; integer `2 <= lmax <= 3*nside-1` |
| Ordering | `RING` or `NESTED` |
| Coordinates | `G`, `C` or `E`; `C`/`E` require a finite explicit `coordinate_epoch` |
| Polarization | `COSMO` or `IAU`; spin 2 |
| Units | `K_CMB` or `uK_CMB` |
| Validity | Boolean arrays and matching content hashes; apodization is separate |
| Response | A separate `ObservingResponse` for each split |

Only explicitly requested unit, U-sign convention and ordering conversions are
performed. Coordinate-frame rotation is not implemented by `converted`.
Missing, conflicting or nonfinite metadata is rejected. Q/U observations must
also be finite; even a NaN at a Boolean-invalid pixel is rejected.

An `ObservingResponse` is a bound declaration, not an instrument inversion:

```python
from cmbcst.integration import ObservingResponse

identity = ObservingResponse('identity', {})
unknown = ObservingResponse('unknown', {'reason': 'response has not been qualified'})
```

`diagonal` and `linear` declarations require an `asset_sha256` in their parameter
object. The reader records that asset reference; it does not load and apply an
arbitrary response matrix. `metadata.same_observing_response` compares declaration
fingerprints. Equal fingerprints do not prove that two instruments have equal
responses. Unknown responses may be stored, but the map-facing family generator
rejects them and requires a common known declaration and common split validity.

### A small artificial pair

This example uses an explicitly artificial identity response. Do not label a
real map's unqualified beam or transfer function as identity to bypass validation.

```python
import numpy as np
from cmbcst.integration import SplitPair, ObservingResponse

nside, lmax = 32, 64
npix = 12*nside**2
rng = np.random.default_rng(741)
a, b = rng.normal(size=(2, 2, npix))
valid = np.ones(npix, dtype=bool)
response = ObservingResponse('identity', {})

pair = SplitPair.create(
    a, b, nside=nside, lmax=lmax, ordering='RING',
    coordinate_frame='G', units='uK_CMB', polarization_convention='COSMO',
    mask_a=valid, mask_b=valid, response_a=response, response_b=response,
)
```

## Lossless paired files and external FITS inputs

```python
from cmbcst.integration import (
    write_split_pair_fits, read_split_pair_fits,
    write_split_pair_hdf5, read_split_pair_hdf5,
)

write_split_pair_fits('example_pair.fits', pair)
write_split_pair_hdf5('example_pair.h5', pair)
fits_pair = read_split_pair_fits('example_pair.fits')
hdf_pair = read_split_pair_hdf5('example_pair.h5')
```

Writers refuse to overwrite an existing artifact. Lossless round trips preserve
the float64 Q/U payload, Boolean validity and scientific metadata. Reading also
records the actual input file hash, so the complete provenance mapping need not
equal that of an in-memory pair.

Both native formats use schema version `cmbcst.split_pair.v1`. Their JSON
manifest contains `schema_version`, `metadata`, `array_sha256` for all four
arrays, and `input_sha256` source references. Metadata includes geometry, units,
conventions, mask hashes and both response declarations. Duplicate JSON keys,
unsupported fields, nonfinite values and mismatched array hashes are rejected.

| Paired FITS | Explicit HDF5 schema |
|---|---|
| HDUs, in order: `PRIMARY`, `A`, `B`, `MASK_A`, `MASK_B`, `METADATA` | Fixed group `/cmbcst/split_pair` |
| `A`/`B`: float64 image arrays | Datasets `a`, `b`: float64 arrays |
| Masks: uint8 0/1 images tagged `MASKTYPE=BOOLEAN_VALIDITY` | Datasets `mask_a`, `mask_b`: Boolean arrays |
| `METADATA`: UTF-8 JSON bytes in a uint8 image | Attributes `schema_version`, `manifest_json` |
| Required FITS checksums and matching primary geometry/response headers | Lossless datasets with Fletcher32 checksums |

A generic HDF5 file with datasets named Q and U is not this schema. The adapter
does not guess its geometry from its extension or shape. File hashes, canonical
array hashes and operator fingerprints are different bindings; use the helper
defined by each schema rather than interchanging their hash rules.

### Two existing HEALPix FITS split files

Supply the original metadata explicitly. The following assumes known Boolean
validity arrays and a documented common response asset for the input maps:

```python
import hashlib
from pathlib import Path
from cmbcst.integration import SplitMetadata, ObservingResponse, read_split_pair_fits
from cmbcst.integration.models import array_sha256

valid_a = np.load('valid_A.npy', allow_pickle=False)  # Boolean, full pixel count
valid_b = np.load('valid_B.npy', allow_pickle=False)
asset_hash = hashlib.sha256(Path('common_response.npz').read_bytes()).hexdigest()
common_response = ObservingResponse('diagonal', {
    'asset_sha256': asset_hash,
    'description': 'documented common response already carried by these maps',
})
declared = SplitMetadata(
    nside=2048, lmax=512, ordering='NESTED', coordinate_frame='G',
    units='K_CMB', polarization_convention='COSMO',
    mask_a_sha256=array_sha256(valid_a), mask_b_sha256=array_sha256(valid_b),
    response_a=common_response, response_b=common_response,
)
pair = read_split_pair_fits(
    'split_A.fits', 'split_B.fits', metadata=declared,
    fields=('Q_STOKES', 'U_STOKES'), hdu=1,
    target_units='uK_CMB', target_polarization='COSMO', target_ordering='RING',
)
```

The original headers must agree with `declared` **before** target conversions.
The reader requires full-sky implicit HEALPix indexing, `NSIDE`, `ORDERING`,
`COORDSYS`, `POLCCONV`, and compatible Q/U column units or `BUNIT`. When present,
`LMAX`, coordinate epoch and response fingerprints are checked too. A `VALID`
column must be Boolean or 0/1 integer validity. `BAD_DATA`/HEALPix unseen values
contribute to the derived validity mask while finite stored values remain intact.
The declared mask hashes must match that derived validity. Explicit-index partial
files require a separately qualified expansion operator and are rejected here.

For paired-schema inputs, the same `target_units`, `target_polarization` and
`target_ordering` options are available. Passing `metadata=...` additionally
checks the paired file's complete original metadata against that declaration.

## Prepare a family while retaining the observed residual

```python
from cmbcst.integration import generate_counterfactual_family

prepared = generate_counterfactual_family(
    pair, n_planes=128, shells=np.arange(30, 62), planes_per_shell=4,
    seed=82024102027, backend='ducc0', nthreads=8,
)
theta = .05*np.sin(np.arange(prepared.family.nplanes))
candidate_alms = prepared.family.candidate_alms(theta)
candidate_maps = prepared.candidate_maps(theta)       # [split, Q/U, pixel]
candidate_pair = prepared.candidate_pair(theta)
provenance = prepared.provenance()
```

The supported backend names are `ducc0` and `healpy`. Preparation explicitly
canonicalizes to RING/COSMO/uK_CMB and applies the area-weighted synthesis adjoint
to the validity-masked maps. This adjoint is not an inverse.

`prepared.reference_maps` retains the canonical original observations.
`prepared.residual_maps` stores their difference from synthesized retained alms.
For a harmonic coadd increment `delta_alm`, map reconstruction is

```text
candidate_maps = reference_maps + synthesize(delta_alm)[None, :, :]
```

The same increment is added to both original splits. Their full observed A−B
difference and their original pixel residual are retained numerically. Full-sky
Gram invariants concern the stored canonical harmonic family. Reanalyzing the
candidate pixels with masks and an approximate adjoint need not reproduce those
alms or preserve a newly computed spectral Gram matrix. The empirical split
difference is not an identified physical noise realization.

Each shell must have enough real directions orthogonal to both difference rows:
`2*ell+1-rank(D_E,D_B) >= 2*planes_per_shell`. Shells must be distinct retained
integer multipoles. The requested plane budget must fit; it is never silently
reduced. Without explicit shells, the API chooses a supported shell manifest.
Angles must be finite real vectors within ±π/4. Creating a candidate does not
automatically impose cut-sky equality constraints or a perturbation RMS bound.

## NaMaster coupled-spectrum model

The cut-sky adapter supports one common scalar mask, fixed purification/binning,
a common nonnegative scalar harmonic response, and no template deprojection.
It returns **coupled binned realized Cℓ**, with flattening order:

```text
pair AA, BB, AB  ×  channel EE, EB, BE, BB  ×  band
```

Here pair `BB` means split B crossed with itself; channel `BB` means B-mode
power. `namaster.labels` makes both meanings explicit, for example
`AA:BB:2-17`. Upper band edges are exclusive. The adapter does not apply a
workspace decoupling matrix, subtract noise bias, deconvolve an instrument beam
or claim to recover ambiguous cut-sky modes.

Use the canonical geometry and a mask defined on those same RING pixels:

```python
from cmbcst.integration.namaster import CutSkyProfile, NaMasterCutSky, cap_mask

p = prepared.pair.metadata
profile = CutSkyProfile(
    nside=p.nside, lmax=p.lmax,
    band_edges=(2, 18, 34, 50, p.lmax+1),
    purify_e=False, purify_b=True,
    n_iter=3, n_iter_mask=3, mask_lmax=p.lmax, nthreads=8,
)
mask = cap_mask(p.nside, fraction=.42, apodization_deg=10.)
mask[~prepared.pair.mask_a] = 0.
mask[~prepared.pair.mask_b] = 0.
namaster = NaMasterCutSky(profile, mask, response_window=np.ones(p.lmax+1))
model, scales, audit = namaster.build_prepared_model(prepared)

candidate_maps = prepared.candidate_maps(theta)
direct = namaster.direct_features_from_maps(candidate_maps)
cached = model.features(theta)
scaled_replay_error = np.max(np.abs(cached-direct)/scales)
```

Choose increasing band edges appropriate to the actual `lmax`; the displayed
edges match the artificial `lmax=64` example. When either purification option is
enabled, this qualified profile requires `mask_lmax == lmax`.

`build_prepared_model` checks the prepared map geometry and rejects a mask that
is nonzero outside common validity. The lower-level raw-array `build_model`
cannot inspect `SplitPair` metadata. The caller still chooses suitable
apodization and a genuinely common response. A valid numerical range alone
does not qualify a mask physically.

With `reference_maps=prepared.reference_maps`, the adapter carries the original
pixel residual into the constant term of its quadratic model. This path requires
a **unit extra response**; a nonunit `response_window` is rejected because the
observed maps already carry their declared response. Omitting `reference_maps`
instead models maps synthesized from retained alms, which has a different scope.
`namaster.candidate_maps(prepared.family, theta,
reference_maps=prepared.reference_maps)` also retains the residual under this
unit-extra-response profile.

`model.features` and `model.jacobian` use stored symmetric quadratic tensors in
coordinates `[1, cos(theta)-1, sin(theta)]`. The model supports NumPy batches and
optional native float64 Torch evaluation through `model.torch(device)`. Building
the response tensors invokes NaMaster transforms; cached feature evaluation does
not. `audit` records the reference mode, scales, zero/reference discrepancy and
transformed-basis hashes.

The displayed `theta` is an engineering replay example, not a cut-sky-locked
candidate. Agreement of `cached` with `direct` checks the response cache; it does
not establish that either equals the reference spectrum. Raw mode-coupling
ensemble relations are not used as a substitute for realized candidate replay.

## Continuation, verification and supported conclusions

`cmbcst.integration.continuation.continue_fiber` supplies bounded CPU continuation
against a fixed quadratic model, frozen scales and explicit minimum/maximum
perturbation RMS. Its `candidate_found` result requires independent reconstruction
and direct observing-operator checks before being treated as verified.

`certify_candidate` checks supplied finite arrays, declared numerical/RMS bounds
and replay evidence. The caller must supply the actual independent reconstruction;
the helper cannot prove runtime independence or authenticate a named trust anchor.
An H1 feasible candidate alone is not a closure counterexample. A counterexample
also requires a declared representation, transition, precision and independently
verified different-future evidence. No result here establishes universal closure,
statistical sufficiency, detector superiority or a validated physical noise law.

Use `provenance()` and stored numerical family/tensor arrays for replay. Hashes
bind bytes and settings; they are not signatures or JoVin receipt authentication.
The evidence statuses distinguish verified finite feasibility, verified scoped
counterexamples, completed finite searches finding none, incomplete execution,
verification failure and unsupported input. Preserve failed artifacts and all
declared limits.

Focused adapter tests live in `tests/test_integration_*.py`. Tests requiring
unavailable optional libraries remain explicitly skipped; they do not establish
backend compatibility. Current pilot measurements and scientific outcomes belong
in a separately frozen run report, not this API guide.

Pixell CAR/WCS adapters, a demonstrated SHTns CMB spin-2/GPU bridge, full instrument
inversion, anisotropic cut-sky weights, template deprojection and SBI calibration
remain future work described in the research plan. They are not current backend
names or capabilities of this slice.
