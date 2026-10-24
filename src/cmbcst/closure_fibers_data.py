"""Source-bound spherical samples and spectrum-preserving closure families.

Counterfactual states are represented by a coadd and a fixed split difference.
Real harmonic coordinate rotations fix that difference and preserve the full
four-field spectral Gram matrix. They are model interventions, not new skies
observed by Planck or coordinate rotations of an unchanged sky.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import numpy as np

SEED = 82024102027
LMAX = 512
THREADS = 8
EPSILON = 1e-12
SHELLS = np.rint(np.linspace(30, 216, 32)).astype(int)
SELECTED_TILES = (24, 6, 30, 34, 21, 11, 152, 170, 189, 174, 186, 166)
BANDS = ((30, 64), (65, 128), (129, 256), (257, 512))
SIGMA = np.radians(1.) / np.sqrt(8*np.log(2))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def harmonic_layout(lmax=LMAX):
    m = np.concatenate([np.full(lmax+1-k, k, dtype=int) for k in range(lmax+1)])
    ell = np.concatenate([np.arange(k, lmax+1) for k in range(lmax+1)])
    return ell, m


def infer_lmax(alm):
    length = np.shape(alm)[-1]
    lmax = int(round((np.sqrt(8*length+1)-3)/2))
    if (lmax+1)*(lmax+2)//2 != length:
        raise ValueError('harmonic array is not complete m-major storage')
    return lmax


def real_shell(alm, shell, lmax=None):
    """[m0, sqrt(2) Re(m1..ell), sqrt(2) Im(m1..ell)] coordinates."""
    alm = np.asarray(alm, dtype=np.complex128)
    lmax = infer_lmax(alm) if lmax is None else lmax
    ell, _ = harmonic_layout(lmax)
    ids = np.flatnonzero(ell == shell)
    if len(ids) != shell+1:
        raise ValueError('shell outside band limit')
    value = alm[..., ids]
    return np.concatenate([value[..., :1].real, np.sqrt(2)*value[..., 1:].real,
                           np.sqrt(2)*value[..., 1:].imag], axis=-1)


def complex_shell(values, shell):
    values = np.asarray(values, dtype=np.float64)
    if values.shape[-1] != 2*shell+1:
        raise ValueError('incorrect real harmonic shell width')
    out = np.empty((*values.shape[:-1], shell+1), dtype=np.complex128)
    out[..., 0] = values[..., 0]
    out[..., 1:] = (values[..., 1:shell+1] + 1j*values[..., shell+1:])/np.sqrt(2)
    return out


def shell_alm(values, shell):
    """Make a complete lmax=ell array with only one nonzero shell."""
    ell, _ = harmonic_layout(shell)
    out = np.zeros((*np.shape(values)[:-1], len(ell)), dtype=np.complex128)
    out[..., ell == shell] = complex_shell(values, shell)
    return out


def global_rms(alm):
    alm = np.asarray(alm, dtype=np.complex128)
    _, m = harmonic_layout(infer_lmax(alm))
    return float(np.sqrt(np.sum(np.abs(alm)**2*np.where(m == 0, 1., 2.))/(4*np.pi)))


def load_source_alms(root):
    root = Path(root)
    directory = root/'results/real_splits_v1/harmonics'
    values = np.stack([np.load(directory/f'alm_143{split}.npy', allow_pickle=False)
                       for split in ('A', 'B')])
    if values.shape != (2, 2, (LMAX+1)*(LMAX+2)//2) or not np.isfinite(values).all():
        raise ValueError('invalid source 143 GHz harmonics')
    coadd, difference = (values[0]+values[1])/2, values[0]-values[1]
    # All families use the same explicit M,D state convention.
    return np.stack([coadd+difference/2, coadd-difference/2])


def input_hashes(root):
    root = Path(root)
    paths = ('results/real_splits_v1/harmonics/alm_143A.npy',
             'results/real_splits_v1/harmonics/alm_143B.npy',
             'results/real_splits_v1/geometry.npz', 'results/real_splits_v1/tiles.json')
    return {name: sha(root/name) for name in paths}


def build_planes(reference_alms, tile, seed=SEED, shells=None, planes_per_shell=4):
    """Disjoint rank-two orthogonal rotations in each split-difference kernel.

    Arrays u/v are stored numerically, padded to a common real-shell width.
    Subsequent replay uses those arrays rather than repeating QR or RNG calls.
    """
    reference = np.asarray(reference_alms, dtype=np.complex128)
    if reference.ndim != 3 or reference.shape[:2] != (2, 2) or not np.isfinite(reference).all():
        raise ValueError('need finite [2 splits,2 E/B,nalm] reference')
    lmax = infer_lmax(reference)
    shells = SHELLS if shells is None else np.asarray(shells, dtype=int)
    if len(shells) == 0 or np.any(shells < 30) or np.any(shells > lmax) or len(np.unique(shells)) != len(shells):
        raise ValueError('invalid distinct harmonic shells')
    if not isinstance(planes_per_shell, int) or planes_per_shell < 1:
        raise ValueError('invalid number of rotation planes')
    maximum = 2*int(np.max(shells))+1
    count = len(shells)*planes_per_shell
    u, v = np.zeros((count, maximum)), np.zeros((count, maximum))
    mu, mv = np.zeros((count, 2)), np.zeros((count, 2))
    shell_ids, ranks = np.repeat(shells, planes_per_shell), []
    noise_error, orthogonal_error = 0., 0.
    coadd = (reference[0]+reference[1])/2
    difference = reference[0]-reference[1]
    cursor = 0
    for shell in shells:
        d = real_shell(difference, int(shell), lmax)
        mean = real_shell(coadd, int(shell), lmax)
        _, singular, vh = np.linalg.svd(d, full_matrices=False)
        rank = int(np.sum(singular > (singular[0]*1e-12 if singular[0] else 0.)))
        if d.shape[1]-rank < 2*planes_per_shell:
            raise ValueError('not enough noise-orthogonal plane directions')
        row = vh[:rank]
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(tile), int(shell)]))
        normal = rng.normal(size=(2*shell+1, 2*planes_per_shell))
        # Reproject before QR to reduce cancellation in nearly aligned draws.
        for _ in range(2):
            normal -= row.T@(row@normal)
        q, upper = np.linalg.qr(normal, mode='reduced')
        q *= np.where(np.diag(upper) < 0, -1., 1.)
        orthogonal_error = max(orthogonal_error, float(np.max(np.abs(q.T@q-np.eye(q.shape[1])))))
        noise_error = max(noise_error, float(np.max(np.abs(d@q))/(1+np.max(np.abs(d)))))
        if min(np.abs(np.diag(upper))) <= 1e-12:
            raise ValueError('degenerate random plane basis')
        width = 2*shell+1
        for j in range(planes_per_shell):
            u[cursor, :width], v[cursor, :width] = q[:, 2*j], q[:, 2*j+1]
            mu[cursor], mv[cursor] = mean@q[:, 2*j], mean@q[:, 2*j+1]
            ranks.append(rank)
            cursor += 1
    if max(noise_error, orthogonal_error) > 1e-10:
        raise ValueError('rotation-plane numerical invariant failed')
    band_ids = np.array([next(i for i, (lo, hi) in enumerate(BANDS) if lo <= shell <= hi)
                         for shell in shell_ids], dtype=np.int64)
    return dict(u=u, v=v, coadd_u=mu, coadd_v=mv, shells=shell_ids.astype(np.int64),
                ranks=np.asarray(ranks), rates=.5*SIGMA**2*shell_ids*(shell_ids+1),
                band_id=band_ids, global_weights=np.sum(mu*mu+mv*mv, axis=1)/(4*np.pi),
                seed=np.array(seed, dtype=np.int64), tile=np.array(tile, dtype=np.int64),
                lmax=np.array(lmax, dtype=np.int64),
                noise_annihilation_error=np.array(noise_error),
                orthogonality_error=np.array(orthogonal_error))


def apply_angles(reference_alms, planes, theta):
    reference = np.asarray(reference_alms, dtype=np.complex128)
    theta = np.asarray(theta, dtype=np.float64)
    if theta.shape != np.shape(planes['shells']) or not np.isfinite(theta).all() or np.max(np.abs(theta), initial=0.) > np.pi/4+1e-14:
        raise ValueError('angles outside the declared rotation family')
    coadd, difference = (reference[0]+reference[1])/2, reference[0]-reference[1]
    mean = coadd.copy()
    ell, _ = harmonic_layout(infer_lmax(reference))
    for j, shell in enumerate(planes['shells']):
        shell = int(shell); width = 2*shell+1
        u, v = planes['u'][j, :width], planes['v'][j, :width]
        mu, mv = planes['coadd_u'][j], planes['coadd_v'][j]
        cos = mu[:, None]*u+mv[:, None]*v
        sin = mv[:, None]*u-mu[:, None]*v
        delta = (np.cos(theta[j])-1)*cos+np.sin(theta[j])*sin
        mean[:, ell == shell] += complex_shell(delta, shell)
    return np.stack([mean+difference/2, mean-difference/2])


def spectral_gram(alms):
    value = np.asarray(alms, dtype=np.complex128)
    lmax = infer_lmax(value)
    ell, m = harmonic_layout(lmax)
    rows = value.reshape(4, -1)
    gram = np.empty((lmax+1, 4, 4))
    weights = np.where(m == 0, 1., 2.)
    for i in range(4):
        for j in range(i, 4):
            gram[:, i, j] = gram[:, j, i] = np.bincount(ell,
                weights=weights*(rows[i]*rows[j].conj()).real,
                minlength=lmax+1)/(2*np.arange(lmax+1)+1)
    return gram


def invariants(reference_alms, candidate_alms, tolerance=1e-10):
    reference, candidate = np.asarray(reference_alms), np.asarray(candidate_alms)
    if reference.shape != candidate.shape or not np.isfinite(candidate).all():
        raise ValueError('nonfinite or mismatched candidate harmonic state')
    if infer_lmax(reference) < 30:
        raise ValueError('closure spectral invariants require a retained ell>=30 band')
    before, after = spectral_gram(reference), spectral_gram(candidate)
    diagonal = np.diagonal(before, axis1=1, axis2=2)
    scale = np.sqrt(np.maximum(diagonal[:, :, None]*diagonal[:, None, :], 0.))
    active = np.arange(len(before)) >= 30
    normalized = np.abs(after-before)/np.maximum(scale, 1e-30)
    power_error = float(np.max(normalized[active]))
    signed_delta = (after-before)/np.maximum(scale, 1e-30)
    spectral_bins = np.rint(signed_delta[active]/1e-6).astype(np.int64)
    spectral_bins_zero = bool(np.all(spectral_bins == 0))
    zero_auto = diagonal == 0.
    zero_auto_exact = bool(np.all(np.diagonal(after,axis1=1,axis2=2)[zero_auto] == 0.))
    delta_d = (candidate[0]-candidate[1])-(reference[0]-reference[1])
    difference_error = float(np.max(np.abs(delta_d)/(1+np.abs(reference[0]-reference[1]))))
    ell, m = harmonic_layout(infer_lmax(reference))
    zeros = bool(np.all(candidate[..., ell < 30] == reference[..., ell < 30]))
    mzero = bool(np.all(candidate[..., m == 0].imag == 0.))
    perturbation = (candidate[0]+candidate[1]-reference[0]-reference[1])/2
    return dict(passed=bool(power_error <= tolerance and difference_error <= tolerance and zeros and mzero and spectral_bins_zero and zero_auto_exact),
                normalized_spectral_gram_error=power_error,
                spectral_bins_zero=spectral_bins_zero,
                spectral_minimum_margin=float(.5e-6-power_error),
                zero_reference_auto_shells_exact=zero_auto_exact,
                split_difference_scaled_error=difference_error,
                unmodified_removed_shells=zeros, mzero_real=mzero,
                global_perturbation_rms=global_rms(perturbation),
                reference_global_rms=global_rms((reference[0]+reference[1])/2))


def _sample(alm, locations, lmax=LMAX):
    import ducc0
    values = ducc0.sht.synthesis_general(alm=np.ascontiguousarray(alm), spin=2,
        lmax=int(lmax), mmax=int(lmax), loc=np.ascontiguousarray(locations),
        epsilon=EPSILON, nthreads=THREADS)
    return values[0]+1j*values[1]


def _source_metadata(root):
    import ducc0
    return dict(input_sha256=input_hashes(root), epsilon=EPSILON,
                data_source_sha256=sha(Path(__file__)), lmax=LMAX, threads=THREADS,
                numpy=np.__version__, ducc0=ducc0.__version__)


def _write_json(path, record):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def _reference_from_cache(directory, geometry, accepted, metadata):
    output = {name: np.load(directory/f'{name}.npy', mmap_mode='r', allow_pickle=False)
              for name in ('base', 'base_derivative', 'base_band', 'reference_alms')}
    output.update(geometry)
    output['accepted_tiles'] = np.asarray(accepted, dtype=np.int64)
    output['reference_global_rms'] = np.array(global_rms((output['reference_alms'][0]+output['reference_alms'][1])/2))
    output['metadata'] = metadata
    return output


def sample_reference(root, cache_dir, budget_callback=None):
    """Sample the unchanged real reference once for all 79 tiles.

    Returns memory-mapped base/base_derivative/base_band arrays, source alms and
    original geometry. No random planes are generated for this shared reference.
    """
    root, cache_dir = Path(root), Path(cache_dir)
    directory = cache_dir/'reference'
    directory.mkdir(parents=True, exist_ok=True)
    metadata = _source_metadata(root)
    with np.load(root/'results/real_splits_v1/geometry.npz', allow_pickle=False) as source:
        geometry = {name: source[name] for name in source.files}
    records = json.loads((root/'results/real_splits_v1/tiles.json').read_text())['records']
    accepted = [r['tile'] for r in records if r['accepted']]
    manifest = directory/'metadata.json'
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old['source'] != metadata or any(sha(directory/name) != digest for name, digest in old['array_sha256'].items()):
            raise ValueError('stale/changed reference cache; preserve it and use another cache directory')
        return _reference_from_cache(directory, geometry, accepted, old)
    if any(directory.glob('*.npy')):
        raise FileExistsError('incomplete reference cache is preserved; select a new directory')
    if budget_callback:
        budget_callback()
    started = time.perf_counter()
    reference = load_source_alms(root)
    count = len(geometry['loc'])
    locations = np.concatenate([geometry['loc'], geometry['end'].reshape(-1, 2)])
    ell, _ = harmonic_layout()
    rate = .5*SIGMA**2*ell*(ell+1)
    base = np.empty((4, 2, len(locations)), dtype=np.complex128)
    derivative = np.empty_like(base)
    bands = np.empty((4, 2, 4, count), dtype=np.complex128)
    for time_index in range(4):
        for split in range(2):
            if budget_callback:
                budget_callback()
            current = reference[split]*np.exp(-time_index*rate)
            base[time_index, split] = _sample(current, locations)
            derivative[time_index, split] = _sample(-current*rate, locations)
            for band, (lo, hi) in enumerate(BANDS):
                window = (ell >= lo)&(ell <= hi)
                bands[time_index, split, band] = _sample(current*window, geometry['loc'])
    arrays = dict(base=base, base_derivative=derivative, base_band=bands, reference_alms=reference)
    for name, values in arrays.items():
        if not np.isfinite(values).all():
            raise ValueError('nonfinite closure reference samples')
        np.save(directory/f'{name}.npy', values)
    metadata = dict(source=metadata, seconds=time.perf_counter()-started,
                    array_sha256={f'{name}.npy': sha(directory/f'{name}.npy') for name in arrays})
    _write_json(manifest, metadata)
    return _reference_from_cache(directory, geometry, accepted, metadata)


def extract_tile_reference(shared, tile):
    """A complete P=0 family for reference feature normalization on any tile."""
    tile = int(tile)
    if tile not in shared['accepted_tiles']:
        raise ValueError('tile is not part of the frozen accepted geometry')
    take = shared['tile_ids'] == tile
    anchors = np.flatnonzero(take)
    all_count, count = len(shared['loc']), len(anchors)
    index = np.concatenate([anchors, (all_count+np.arange(16*all_count).reshape(4, 4, all_count)[:, :, take]).ravel()])
    base = np.asarray(shared['base'][..., index])
    mean = (base[0, 0, :count]+base[0, 1, :count])/2
    result = dict(base=base, base_derivative=np.asarray(shared['base_derivative'][..., index]),
        base_band=np.asarray(shared['base_band'][..., take]),
        cos_basis=np.empty((0, len(index)), dtype=np.complex128),
        sin_basis=np.empty((0, len(index)), dtype=np.complex128),
        rates=np.empty(0), band_id=np.empty(0, dtype=np.int64), global_weights=np.empty(0),
        phase0=shared['phase0'][:, :, take], phase1=shared['phase1'][:, :, take],
        valid=shared['valid'][:, :, take], reference_local_rms=np.array(np.sqrt(np.mean(np.abs(mean)**2))),
        reference_global_rms=np.array(shared['reference_global_rms']), tile=np.array(tile),
        anchor_pixel=shared['anchor_pixel'][take], locations=np.concatenate([shared['loc'][take],shared['end'][:, :, take].reshape(-1,2)]))
    return result


def load_planes(cache_dir, tile):
    directory = Path(cache_dir)
    path = directory/f'planes_{int(tile)}.npz'
    manifest = directory/f'family_{int(tile)}.json'
    if not manifest.exists() or sha(path) != json.loads(manifest.read_text())['planes_sha256']:
        raise ValueError('missing or mismatched numerical plane provenance')
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def build_family(root, tile, cache_dir, budget_callback=None, shared=None):
    root, cache_dir, tile = Path(root), Path(cache_dir), int(tile)
    cache_dir.mkdir(parents=True, exist_ok=True)
    expected = dict(source=_source_metadata(root), tile=tile, seed=SEED,
                    shells=SHELLS.tolist(), planes_per_shell=4)
    family_path, planes_path = cache_dir/f'family_{tile}.npz', cache_dir/f'planes_{tile}.npz'
    manifest = cache_dir/f'family_{tile}.json'
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old['configuration'] != expected or sha(family_path) != old['family_sha256'] or sha(planes_path) != old['planes_sha256']:
            raise ValueError('changed family cache; preserve it and use another cache directory')
        with np.load(family_path, allow_pickle=False) as source:
            return {name: source[name] for name in source.files}
    if family_path.exists() or planes_path.exists():
        raise FileExistsError('preserving incomplete closure family cache')
    shared = sample_reference(root, cache_dir, budget_callback) if shared is None else shared
    family = extract_tile_reference(shared, tile)
    reference = shared['reference_alms']
    started = time.perf_counter()
    planes = build_planes(reference, tile)
    count = len(planes['shells'])
    cosine = np.empty((count, len(family['locations'])), dtype=np.complex128)
    sine = np.empty_like(cosine)
    for j, shell in enumerate(planes['shells']):
        if budget_callback:
            budget_callback()
        shell = int(shell); width = 2*shell+1
        u, v = planes['u'][j, :width], planes['v'][j, :width]
        mu, mv = planes['coadd_u'][j], planes['coadd_v'][j]
        cosine[j] = _sample(shell_alm(mu[:, None]*u+mv[:, None]*v, shell), family['locations'], shell)
        sine[j] = _sample(shell_alm(mv[:, None]*u-mu[:, None]*v, shell), family['locations'], shell)
    family.update(cos_basis=cosine, sin_basis=sine, rates=planes['rates'],
                  band_id=planes['band_id'], global_weights=planes['global_weights'])
    if not all(np.isfinite(value).all() for value in family.values()):
        raise ValueError('nonfinite closure family preprocessing')
    np.savez_compressed(planes_path, **planes)
    np.savez_compressed(family_path, **family)
    _write_json(manifest, dict(configuration=expected, seconds=time.perf_counter()-started,
        family_sha256=sha(family_path), planes_sha256=sha(planes_path),
        source_reference=shared['metadata'],
        plane_checks={name:float(planes[name]) for name in ('noise_annihilation_error','orthogonality_error')}))
    return family


def replay_candidate(root, tile, cache_dir, theta, shared=None, budget_callback=None):
    """Reconstruct actual coefficients and directly resample every reported time.

    Returns direct_family (with zero planes: evaluate at empty theta), candidate
    harmonic state, invariants, and direct perturbation ratios. No output is
    published or scientific candidate accepted by this function itself.
    """
    shared = sample_reference(root, cache_dir, budget_callback) if shared is None else shared
    reference = shared['reference_alms']
    family = extract_tile_reference(shared, tile)
    planes = load_planes(cache_dir, tile)
    candidate = apply_angles(reference, planes, theta)
    check = invariants(reference, candidate)
    ell, _ = harmonic_layout()
    rate = .5*SIGMA**2*ell*(ell+1)
    count = len(family['anchor_pixel'])
    for time_index in range(4):
        for split in range(2):
            if budget_callback:
                budget_callback()
            current = candidate[split]*np.exp(-time_index*rate)
            family['base'][time_index, split] = _sample(current, family['locations'])
            family['base_derivative'][time_index, split] = _sample(-current*rate, family['locations'])
            for band, (lo, hi) in enumerate(BANDS):
                family['base_band'][time_index, split, band] = _sample(current*((ell>=lo)&(ell<=hi)),family['locations'][:count])
    if budget_callback:
        budget_callback()
    baseline = extract_tile_reference(shared, tile)
    perturbation = ((family['base'][0,0,:count]+family['base'][0,1,:count])-
                    (baseline['base'][0,0,:count]+baseline['base'][0,1,:count]))/2
    ratios = np.array([np.sqrt(np.mean(np.abs(perturbation)**2))/float(baseline['reference_local_rms']),
                       check['global_perturbation_rms']/float(baseline['reference_global_rms'])])
    # The reference RMS scales belong to the original tile, not the perturbed one.
    family['reference_local_rms'] = baseline['reference_local_rms']
    return dict(direct_family=family, candidate_alms=candidate, invariants=check,
                perturbation_ratios=ratios)
