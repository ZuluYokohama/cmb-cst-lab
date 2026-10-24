"""Map-schema round trips, explicit conventions and rejected ambiguity."""
import importlib
from dataclasses import replace
import numpy as np
import pytest

from cmbcst.integration.models import (MissingOptionalDependency,ObservingResponse,SplitMetadata,
    SplitPair,array_sha256)
from cmbcst.integration import io


def pair_fixture():
    a=np.arange(96,dtype=np.float64).reshape(2,48)/8
    b=a[::-1].copy();a[0,0]=-0.;b[1,2]=np.nextafter(0.,1.)
    ma=np.ones(48,dtype=bool);ma[3]=False
    mb=np.ones(48,dtype=bool);mb[7]=False
    return SplitPair.create(a,b,nside=2,lmax=3,ordering='RING',coordinate_frame='G',
        units='K_CMB',polarization_convention='COSMO',mask_a=ma,mask_b=mb,
        response_a=ObservingResponse('unknown',{'reason':'test split A response unavailable'}),
        response_b=ObservingResponse('unknown',{'reason':'test split B response unavailable'}))


def assert_payload_equal(left,right):
    for name in ('a','b','mask_a','mask_b'):
        assert array_sha256(getattr(left,name))==array_sha256(getattr(right,name))
    assert left.metadata.to_dict()==right.metadata.to_dict()
    assert left.metadata.fingerprint==right.metadata.fingerprint


def test_map_contract_conversions_masks_and_response_fingerprints():
    pair=pair_fixture()
    assert not pair.metadata.same_observing_response
    assert pair.metadata==SplitMetadata.from_dict(pair.metadata.to_dict())
    converted=pair.converted(units='uK_CMB',polarization_convention='IAU')
    assert np.array_equal(converted.a[0],pair.a[0]*1e6)
    assert np.array_equal(converted.a[1],-pair.a[1]*1e6)
    assert np.array_equal(converted.mask_a,pair.mask_a)
    assert pair.a[0,3]!=0 and pair.operator_maps('A')[0,3]==0
    assert pair.a.flags.writeable is False
    assert converted.metadata.units=='uK_CMB' and converted.metadata.polarization_convention=='IAU'
    assert array_sha256(pair.a)!=array_sha256(converted.a)
    assert array_sha256(pair.a)!=array_sha256(pair.a.astype(np.float32))
    same=replace(pair.metadata,response_b=pair.metadata.response_a)
    assert same.same_observing_response


@pytest.mark.parametrize('changes',[
    {'nside':0},{'nside':3},{'nside':True},{'lmax':6},{'lmax':1},
    {'coordinate_frame':'unknown'},{'coordinate_epoch':float('nan')},
    {'coordinate_frame':'C'},{'polarization_convention':None},{'units':'K'},
    {'ordering':'nested'},{'spin':0},{'mask_a_sha256':'0'*63},{'response_a':None},
])
def test_reject_missing_or_incompatible_metadata(changes):
    with pytest.raises(ValueError):
        replace(pair_fixture().metadata,**changes)


def test_reject_nonfinite_mask_weights_and_corrupted_hashes():
    pair=pair_fixture();bad=pair.a.copy();bad[0,0]=np.nan
    with pytest.raises(ValueError,match='finite'):
        SplitPair(bad,pair.b,pair.mask_a,pair.mask_b,pair.metadata)
    with pytest.raises(ValueError,match='Boolean'):
        SplitPair(pair.a,pair.b,pair.mask_a.astype(float),pair.mask_b,pair.metadata)
    with pytest.raises(ValueError,match='hash'):
        SplitPair(pair.a,pair.b,~pair.mask_a,pair.mask_b,pair.metadata)
    with pytest.raises(ValueError,match='finite JSON'):
        ObservingResponse('linear',{'asset_sha256':'a'*64,'gain':np.inf})
    with pytest.raises(ValueError,match='reason'):
        ObservingResponse('unknown',{})
    with pytest.raises(ValueError,match='asset_sha256'):
        ObservingResponse('diagonal',{})
    missing=pair.metadata.to_dict();missing.pop('units')
    with pytest.raises(ValueError,match='missing'):
        SplitMetadata.from_dict(missing)
    with pytest.raises(ValueError,match='duplicate'):
        io._json_read('{"units":"K_CMB","units":"uK_CMB"}')
    with pytest.raises(ValueError,match='nonfinite'):
        io._json_read('{"gain":NaN}')


def test_optional_dependencies_fail_honestly_without_fallback(monkeypatch,tmp_path):
    original=importlib.import_module
    def unavailable(name,*args,**kwargs):
        if name in ('h5py','astropy.io.fits'):
            raise ImportError('intentionally unavailable')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(io.importlib,'import_module',unavailable)
    for writer,path in ((io.write_split_pair_fits,tmp_path/'maps.fits'),
                        (io.write_split_pair_hdf5,tmp_path/'maps.h5')):
        with pytest.raises(MissingOptionalDependency,match='no adapter ran'):
            writer(path,pair_fixture())
        assert not path.exists()
    with pytest.raises(ValueError,match='explicit SplitMetadata'):
        io.read_split_pair_fits('A.fits','B.fits')


def test_lossless_fits_schema_and_checksum_independent_hash_rejection(tmp_path):
    fits=pytest.importorskip('astropy.io.fits',reason='actual FITS adapter requires astropy')
    pair=pair_fixture();path=tmp_path/'pair.fits'
    io.write_split_pair_fits(path,pair)
    actual=io.read_split_pair_fits(path)
    assert_payload_equal(pair,actual)
    assert 'paired_fits' in actual.input_sha256
    with pytest.raises(OSError):
        io.write_split_pair_fits(path,pair)
    corrupted=tmp_path/'changed.fits'
    with fits.open(path) as hdus:
        hdus['A'].data[0,1]+=1.
        hdus.writeto(corrupted,checksum=True)
    with pytest.raises(ValueError,match='hash mismatch'):
        io.read_split_pair_fits(corrupted)


def test_lossless_hdf5_explicit_schema_and_mask_type_rejection(tmp_path):
    h5py=pytest.importorskip('h5py',reason='actual HDF5 adapter requires h5py')
    pair=pair_fixture();path=tmp_path/'pair.h5'
    io.write_split_pair_hdf5(path,pair)
    actual=io.read_split_pair_hdf5(path)
    assert_payload_equal(pair,actual)
    assert 'paired_hdf5' in actual.input_sha256
    with pytest.raises(OSError):
        io.write_split_pair_hdf5(path,pair)
    with h5py.File(path,'r+') as container:
        group=container[io.HDF5_GROUP];values=group['mask_a'][()].astype(float)
        del group['mask_a'];group.create_dataset('mask_a',data=values)
    with pytest.raises(ValueError,match='Boolean'):
        io.read_split_pair_hdf5(path)
    unknown=tmp_path/'unrelated.h5'
    with h5py.File(unknown,'x') as container:
        container.create_dataset('Q',data=pair.a)
    with pytest.raises(ValueError,match='schema group'):
        io.read_split_pair_hdf5(unknown)


def test_external_fits_validates_original_headers_before_target_conversion(tmp_path):
    fits=pytest.importorskip('astropy.io.fits',reason='actual FITS adapter requires astropy')
    pair=pair_fixture();paths=[]
    for label in ('a','b'):
        path=tmp_path/(label+'.fits');paths.append(path)
        data=getattr(pair,label)
        table=fits.BinTableHDU.from_columns([
            fits.Column(name='Q_STOKES',format='D',unit='K_CMB',array=data[0]),
            fits.Column(name='U_STOKES',format='D',unit='K_CMB',array=data[1]),
            fits.Column(name='VALID',format='L',array=getattr(pair,'mask_'+label))])
        for key,value in dict(PIXTYPE='HEALPIX',NSIDE=2,ORDERING='RING',COORDSYS='G',POLCCONV='COSMO',
                CSTRSP=getattr(pair.metadata,'response_'+label).fingerprint).items():
            table.header[key]=value
        fits.HDUList([fits.PrimaryHDU(),table]).writeto(path,checksum=True)
    actual=io.read_split_pair_fits(*paths,metadata=pair.metadata)
    assert_payload_equal(pair,actual)
    converted=io.read_split_pair_fits(*paths,metadata=pair.metadata,target_units='uK_CMB',target_polarization='IAU')
    assert_payload_equal(pair.converted(units='uK_CMB',polarization_convention='IAU'),converted)
    with fits.open(paths[1],mode='update') as hdus:
        hdus[1].header['POLCCONV']='IAU'
    with pytest.raises(ValueError,match='POLCCONV conflicts'):
        io.read_split_pair_fits(*paths,metadata=pair.metadata,target_polarization='COSMO')


def test_ring_nested_conversion_is_exact_when_geometry_backend_available():
    pytest.importorskip('ducc0',reason='actual HEALPix permutation requires ducc0')
    pair=pair_fixture()
    nested=pair.converted(ordering='NESTED')
    assert nested.metadata.ordering=='NESTED'
    assert_payload_equal(pair,nested.converted(ordering='RING'))
