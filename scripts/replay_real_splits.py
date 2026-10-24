"""Separate sampling execution from saved, source-bound filtered coefficients."""
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import ducc0
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cmbcst import DETECTORS,spectra
from cmbcst.real_splits_data import sha
OUT=ROOT/'results/real_splits_v1'


def main():
    start=time.perf_counter()
    provenance=json.loads((OUT/'inputs.json').read_text())['maps']
    with np.load(OUT/'geometry.npz') as d:
        loc=np.concatenate([d['loc'],d['end'].reshape(-1,2)])
    with np.load(OUT/'observations.npz') as d:
        observed=np.concatenate([d['anchor'],d['endpoint'].reshape(8,-1)],axis=1)
    records=[]
    directory=OUT/'harmonics';directory.mkdir(exist_ok=True)
    for i,key in enumerate(DETECTORS):
        target=directory/f'alm_{key}.npy'
        if not target.exists():
            shutil.copyfile(ROOT/f'data/work/real_splits_v1/alm_{key}.npy',target)
        if sha(target)!=provenance[i]['filtered_alm_sha256']:
            raise ValueError('saved harmonic bytes differ')
        alm=np.load(target)
        value=ducc0.sht.synthesis_general(alm=alm,spin=2,lmax=512,loc=loc,epsilon=1e-11,nthreads=8)
        replay=value[0]+1j*value[1]
        error=float(np.max(np.abs(replay-observed[i])/(1+np.abs(observed[i]))))
        records.append(dict(map=key,scaled_error=error,sha256=sha(target)))
        if error>1e-10:
            raise ValueError('actual sampling replay failed')
    # One complete raw-map replay with inverse index gathering rather than the
    # production NESTED-to-RING scattering, and explicit diagonal filter algebra.
    from astropy.io import fits
    inventory={v['key']:v for v in json.loads((ROOT/'data/evidence/planck_dataset.json').read_text())['products']}
    item=next(v for v in provenance if v['key']=='143A')
    raw=ROOT/'data/raw'/inventory['143A']['filename']
    if sha(raw)!=item['sha256']:
        raise ValueError('raw replay input changed')
    inverse=ducc0.healpix.Healpix_Base(2048,'RING').ring2nest(np.arange(12*2048**2,dtype=np.int64))
    with fits.open(raw,memmap=True) as hdus:
        q=np.asarray(hdus[1].data['Q_STOKES'],dtype=float).ravel()[inverse]
        u=np.asarray(hdus[1].data['U_STOKES'],dtype=float).ravel()[inverse]
    good=(np.abs(q)<1e20)&(np.abs(u)<1e20)
    sky=np.stack([np.where(good,q*1e6,0.),np.where(good,u*1e6,0.)])
    del inverse,q,u,good
    raw_alm=ducc0.sht.adjoint_synthesis(map=sky,spin=2,lmax=512,nthreads=8,**ducc0.healpix.Healpix_Base(2048,'RING').sht_info())*(4*np.pi/sky.shape[-1])
    del sky
    ell,_=spectra.alm_ell_m(512)
    sigma=np.pi/180/np.sqrt(8*np.log(2))
    filt=np.zeros((2,513))
    ls=np.arange(30,513)
    filt[:,30:]=np.exp(-ls*(ls+1)*sigma*sigma/2)/np.sqrt(np.asarray(item['diagonal_windows'])[:,30:])
    raw_alm*=filt[:,ell]
    recorded=np.load(directory/'alm_143A.npy')
    raw_error=float(np.max(np.abs(raw_alm-recorded)/(1+np.abs(recorded))))
    if raw_error>1e-10:
        raise ValueError('independent raw-map index/filter replay failed')
    # Different public sampling operators at identical pixel centers.
    rng=np.random.default_rng(730014)
    ell,m=spectra.alm_ell_m(32)
    alm=rng.normal(size=(2,len(ell)))+1j*rng.normal(size=(2,len(ell)))*(m>0)
    alm[:,ell<2]=0
    base=ducc0.healpix.Healpix_Base(32,'RING')
    grid=ducc0.sht.synthesis(alm=alm,spin=2,lmax=32,nthreads=8,**base.sht_info())
    direct=ducc0.sht.synthesis_general(alm=alm,spin=2,lmax=32,loc=base.pix2ang(np.arange(base.npix())),epsilon=1e-11,nthreads=8)
    error=float(np.max(np.abs(direct-grid)/(1+np.abs(grid))))
    if error>1e-10:
        raise ValueError('spin-2 direct/grid operator check failed')
    record=dict(status='passed',records=records,spin2_grid_direct_error=error,
        raw_map_replay=dict(map='143A',sha256=item['sha256'],scaled_error=raw_error,index_method='RING-to-NESTED gather; production uses NESTED-to-RING scatter',filter='independent explicit diagonal filter algebra'),seconds=time.perf_counter()-start,
        scope='Separate sampling execution of all eight maps, one raw 143A reanalysis with inverse index mapping and explicit filters, and direct/grid spin-2 fixture; shared ducc backend, not independent instrument model',
        source_sha256=sha(Path(__file__)))
    (OUT/'sampling_replay.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(status='passed',maps=len(records),maximum_sample_error=max(r['scaled_error'] for r in records),raw_143A_error=raw_error,spin2_grid_direct_error=error),indent=2))


if __name__=='__main__':
    main()
