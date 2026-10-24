"""Read-only Planck Q/U, diagonal QuickPol filtering and spherical sampling."""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time
import numpy as np
from . import data,spectra,theory,DETECTORS
from .real_splits import spherical_pairs,local_frame


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):
            digest.update(block)
    return digest.hexdigest()


def pair_geometry(root):
    import ducc0
    from scipy.spatial import cKDTree
    base = ducc0.healpix.Healpix_Base(64,'RING')
    all_loc = base.pix2ang(np.arange(base.npix()))
    source_mask = root/'data/work/rtx5070_real/mask_30.npy'
    mask = np.load(source_mask,mmap_mode='r')
    if mask.shape!=(12*2048**2,) or not np.all((mask==0)|(mask==1)):
        raise ValueError('expected prior audited binary RING mask')
    nested = ducc0.healpix.Healpix_Base(2048,'NESTED')
    order = nested.nest2ring(np.arange(12*2048**2,dtype=np.int64))
    low_nested = mask[order].reshape(-1,1024).min(axis=1)>0
    del order
    low = np.empty(base.npix(),dtype=bool)
    low[ducc0.healpix.Healpix_Base(64,'NESTED').nest2ring(np.arange(base.npix()))] = low_nested
    bad_tree = cKDTree(base.pix2vec(np.flatnonzero(~low)))
    margin = 2*np.sin(np.radians(3.)/2)
    def allowed(loc):
        v = local_frame(loc)[0]
        distance = bad_tree.query(v.reshape(-1,3))[0].reshape(v.shape[:-1])
        return (distance>margin)&(np.abs(90-np.degrees(loc[...,0]))>=40)
    initial = (np.abs(90-np.degrees(all_loc[:,0]))>=45)&allowed(all_loc)
    anchors = np.flatnonzero(initial)
    loc = all_loc[anchors]
    end,p0,p1 = spherical_pairs(loc)
    valid = np.ones(end.shape[:-1],dtype=bool)
    for fraction in (.25,.5,.75,1.):
        inside = spherical_pairs(loc,np.array([.5,1,2,4])*fraction)[0]
        valid &= allowed(inside)
    tile_ids = ducc0.healpix.Healpix_Base(4,'RING').ang2pix(loc)
    return dict(loc=loc,end=end,phase0=p0,phase1=p1,valid=valid,tile_ids=tile_ids,
        anchor_pixel=anchors,mask_sha256=sha(source_mask))


def beam_members(path):
    want = {f'Wl_npipe6v20_{key}x{key}.fits':key for key in DETECTORS}
    found = {}
    with tarfile.open(path,'r|gz') as archive:
        for member in archive:
            basename = member.name.rsplit('/',1)[-1]
            if '/AB/' not in member.name or basename not in want:
                continue
            if not member.isfile() or want[basename] in found:
                raise ValueError('invalid/duplicate diagonal QuickPol member')
            found[want[basename]] = (member.name,archive.extractfile(member).read())
    if set(found)!=set(DETECTORS):
        raise ValueError('missing actual QuickPol diagonal beams; no fallback')
    return found


def prepare_observations(root,work,geometry,progress,budget):
    import ducc0
    from astropy.io import fits
    inventory = json.loads((root/'data/evidence/planck_dataset.json').read_text())['products']
    reference = {r['key']:r for r in inventory}
    archive = data.path_of('rimo')
    if sha(archive)!=reference['rimo']['sha256']:
        raise ValueError('beam archive SHA mismatch')
    members = beam_members(archive)
    lmax = 512
    ell,_ = spectra.alm_ell_m(lmax)
    target = theory.gaussian_beam(60.,lmax)
    locations = np.ascontiguousarray(np.concatenate([geometry['loc'],geometry['end'].reshape(-1,2)]))
    samples,provenance = [],[]
    spectra.NTHREADS = 8
    for key in DETECTORS:
        budget()
        start = time.perf_counter()
        progress('read_and_transform',key)
        path = data.path_of(key)
        digest = sha(path)
        if digest!=reference[key]['sha256']:
            raise ValueError(f'{key} input SHA mismatch')
        q,u,good = spectra.load_qu(str(path))
        if not np.isfinite(q).all() or not np.isfinite(u).all():
            raise ValueError('nonfinite raw Q/U')
        invalid = int((~good).sum())
        del good
        alm = spectra.map2alm_pol(q,u,np.ones(q.size),lmax)
        del q,u
        member,payload = members[key]
        with fits.open(io.BytesIO(payload),memmap=False) as hdus:
            windows = np.array([np.asarray(hdus[name].data[column])[0,:lmax+1] for name,column in (('EE','EE_2_EE'),('BB','BB_2_BB'))],dtype=float)
        if windows.shape!=(2,lmax+1) or not np.isfinite(windows).all() or np.any(windows[:,30:]<=0):
            raise ValueError('invalid diagonal QuickPol windows')
        filters = np.zeros_like(windows)
        filters[:,30:] = target[None,30:]/np.sqrt(windows[:,30:])
        alm *= filters[:,ell]
        if not np.isfinite(alm).all():
            raise ValueError('nonfinite filtered harmonics')
        np.save(work/f'alm_{key}.npy',alm)
        result = ducc0.sht.synthesis_general(alm=alm,spin=2,lmax=lmax,mmax=lmax,
            loc=locations,epsilon=1e-11,nthreads=8)
        value = result[0]+1j*result[1]
        if not np.isfinite(value).all():
            raise ValueError('nonfinite real map samples')
        samples.append(value)
        provenance.append(dict(key=key,sha256=digest,bytes=Path(path).stat().st_size,
            beam_member=member,beam_payload_sha256=hashlib.sha256(payload).hexdigest(),
            diagonal_windows=windows.tolist(),filtered_alm_sha256=sha(work/f'alm_{key}.npy'),
            zero_filled_invalid_pixels=invalid,seconds=time.perf_counter()-start))
        budget()
    sample = np.asarray(samples)
    count = len(geometry['loc'])
    return sample[:,:count],sample[:,count:].reshape(8,*geometry['end'].shape[:-1]),provenance
