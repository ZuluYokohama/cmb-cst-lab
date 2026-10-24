"""Lossless paired FITS and explicit HDF5 schema I/O for HEALPix Q/U splits.

No reader guesses units, coordinate frames or polarization conventions. Native
paired products retain finite masked values and their Boolean validity masks.
Two-file FITS ingestion additionally requires an explicit metadata declaration.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import numpy as np

from .models import (SCHEMA_VERSION,MissingOptionalDependency,SplitMetadata,SplitPair,
                     array_sha256,canonical_json)

HDF5_GROUP = '/cmbcst/split_pair'
_ARRAYS = ('a','b','mask_a','mask_b')


def _dependency(name,label):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise MissingOptionalDependency(f'{label} requires the optional dependency {name}; no adapter ran') from exc


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):
            digest.update(block)
    return digest.hexdigest()


def _manifest(pair):
    return dict(schema_version=SCHEMA_VERSION,metadata=pair.metadata.to_dict(),
        array_sha256={name:array_sha256(getattr(pair,name)) for name in _ARRAYS},
        input_sha256=dict(pair.input_sha256))


def _json_read(value):
    if isinstance(value,bytes):
        value=value.decode('utf-8',errors='strict')
    if not isinstance(value,str):
        raise ValueError('schema metadata must be a UTF-8 JSON string')
    def object_pairs(pairs):
        result={}
        for key,item in pairs:
            if key in result:
                raise ValueError('duplicate metadata keys are forbidden')
            result[key]=item
        return result
    def constant(value):
        raise ValueError('nonfinite JSON metadata is forbidden')
    try:
        return json.loads(value,object_pairs_hook=object_pairs,parse_constant=constant)
    except (TypeError,json.JSONDecodeError) as exc:
        raise ValueError('invalid schema metadata JSON') from exc


def _from_manifest(manifest,arrays,input_hashes,declared=None):
    if not isinstance(manifest,dict) or set(manifest)!={'schema_version','metadata','array_sha256','input_sha256'}:
        raise ValueError('missing or unsupported paired schema fields')
    if manifest['schema_version']!=SCHEMA_VERSION:
        raise ValueError('unsupported paired schema version')
    metadata=SplitMetadata.from_dict(manifest['metadata'])
    if declared is not None and (not isinstance(declared,SplitMetadata) or declared.fingerprint!=metadata.fingerprint):
        raise ValueError('declared metadata conflicts with the file metadata')
    hashes=manifest['array_sha256']
    if not isinstance(hashes,dict) or set(hashes)!=set(_ARRAYS):
        raise ValueError('all four paired array hashes are required')
    for name in _ARRAYS:
        if array_sha256(arrays[name])!=hashes[name]:
            raise ValueError(f'{name} content hash mismatch')
    provenance=manifest['input_sha256']
    if not isinstance(provenance,dict):
        raise ValueError('input source hashes must be an object')
    # Preserve original lineage and bind the actual container just read.
    merged={**provenance,**input_hashes}
    return SplitPair(metadata=metadata,input_sha256=merged,**arrays)


def _converted(pair,units,polarization,ordering):
    if units is None and polarization is None and ordering is None:
        return pair
    return pair.converted(units=units,polarization_convention=polarization,ordering=ordering)


def _metadata_header(header,metadata):
    for key,value in (('NSIDE',metadata.nside),('LMAX',metadata.lmax),('ORDERING',metadata.ordering),
            ('COORDSYS',metadata.coordinate_frame),('POLCCONV',metadata.polarization_convention),
            ('BUNIT',metadata.units)):
        header[key]=value
    header['PIXTYPE']='HEALPIX'
    header['CSTRSPA']=metadata.response_a.fingerprint
    header['CSTRSPB']=metadata.response_b.fingerprint
    if metadata.coordinate_epoch is not None:
        header['EQUINOX']=float(metadata.coordinate_epoch)


def _check_header(header,metadata,*,unit=None,response=None,require_lmax=False):
    required=dict(PIXTYPE='HEALPIX',NSIDE=metadata.nside,ORDERING=metadata.ordering,
        COORDSYS=metadata.coordinate_frame,POLCCONV=metadata.polarization_convention)
    for key,expected in required.items():
        if key not in header:
            raise ValueError(f'missing required FITS {key} metadata')
        actual=header[key]
        if key=='NSIDE':
            if isinstance(actual,bool) or not isinstance(actual,(int,np.integer)):
                raise ValueError('FITS NSIDE must be an integer')
        else:
            actual=str(actual).strip().upper()
        if actual!=expected:
            raise ValueError(f'FITS {key} conflicts with explicitly declared metadata')
    if 'LMAX' in header:
        if isinstance(header['LMAX'],bool) or not isinstance(header['LMAX'],(int,np.integer)) or header['LMAX']!=metadata.lmax:
            raise ValueError('FITS LMAX conflicts with explicitly declared metadata')
    elif require_lmax:
        raise ValueError('missing paired FITS LMAX')
    original_unit=header.get('BUNIT') if unit is None else unit
    if original_unit is None or str(original_unit).strip()!=metadata.units:
        raise ValueError('missing or conflicting explicitly declared CMB units')
    if metadata.coordinate_frame in ('C','E') and 'EQUINOX' in header:
        epoch=header['EQUINOX']
        if isinstance(epoch,bool) or not isinstance(epoch,(int,float)) or not np.isfinite(epoch) or epoch!=metadata.coordinate_epoch:
            raise ValueError('FITS coordinate epoch conflicts with metadata')
    if response is None:
        for key,expected in (('CSTRSPA',metadata.response_a.fingerprint),('CSTRSPB',metadata.response_b.fingerprint)):
            if key not in header or str(header[key]).strip()!=expected:
                raise ValueError('paired FITS observing response fingerprint mismatch')
    elif 'CSTRSP' in header and str(header['CSTRSP']).strip()!=response.fingerprint:
        raise ValueError('FITS observing response fingerprint conflicts with metadata')


def _checksums(hdus,required):
    for hdu in hdus:
        for key,method in (('CHECKSUM',hdu.verify_checksum),('DATASUM',hdu.verify_datasum)):
            if key not in hdu.header:
                if required:
                    raise ValueError('paired FITS checksums are required')
            elif method()!=1:
                raise ValueError('FITS checksum validation failed')


def write_split_pair_fits(path,pair):
    """Write one lossless paired FITS container; never overwrite an artifact."""
    if not isinstance(pair,SplitPair):
        raise ValueError('SplitPair required')
    fits=_dependency('astropy.io.fits','FITS support')
    primary=fits.PrimaryHDU()
    primary.header['CSTSCHEM']=SCHEMA_VERSION
    _metadata_header(primary.header,pair.metadata)
    hdus=[primary]
    for name in _ARRAYS:
        value=getattr(pair,name)
        hdu=fits.ImageHDU(data=value.astype(np.uint8) if name.startswith('mask_') else value,name=name.upper())
        if name.startswith('mask_'):
            hdu.header['MASKTYPE']='BOOLEAN_VALIDITY'
        hdus.append(hdu)
    manifest=canonical_json(_manifest(pair)).encode('utf-8')
    hdus.append(fits.ImageHDU(data=np.frombuffer(manifest,dtype=np.uint8),name='METADATA'))
    with fits.HDUList(hdus) as container:
        container.writeto(Path(path),overwrite=False,checksum=True,output_verify='exception')
    return Path(path)


def _read_external_fits(path,metadata,fields,hdu,response):
    fits=_dependency('astropy.io.fits','FITS support')
    with fits.open(path,memmap=False,checksum=True) as hdus:
        _checksums(hdus,required=False)
        table=hdus[hdu]
        if str(table.header.get('INDXSCHM','IMPLICIT')).strip().upper()!='IMPLICIT':
            raise ValueError('explicit/partial indexed FITS requires a separately declared expansion operator')
        if not isinstance(fields,(tuple,list)) or len(fields)!=2 or fields[0]==fields[1]:
            raise ValueError('two distinct Q/U FITS fields must be explicitly named')
        arrays=[]
        for name in fields:
            if name not in table.columns.names:
                raise ValueError(f'missing Q/U FITS field {name}')
            column=table.columns[name]
            unit=column.unit if column.unit is not None else table.header.get('BUNIT')
            _check_header(table.header,metadata,unit=unit,response=response)
            if 'BUNIT' in table.header and str(table.header['BUNIT']).strip()!=metadata.units:
                raise ValueError('FITS column and global units conflict')
            value=np.asarray(table.data[name])
            if value.dtype.kind!='f' or value.dtype.itemsize>8:
                raise ValueError('floating FITS Q/U columns up to float64 required')
            arrays.append(np.asarray(value,dtype=np.float64).ravel())
        values=np.stack(arrays)
        if not np.isfinite(values).all():
            raise ValueError('nonfinite FITS observations are rejected; declare a finite masked product explicitly')
        mask=np.ones(values.shape[-1],dtype=bool)
        if 'VALID' in table.columns.names:
            raw=np.asarray(table.data['VALID']).ravel()
            if raw.dtype.kind not in 'biu' or not np.isin(raw,[0,1]).all():
                raise ValueError('VALID must be Boolean validity, not apodization weights')
            mask=raw.astype(bool)
        bad=table.header.get('BAD_DATA',-1.6375e30)
        if not isinstance(bad,(int,float)) or isinstance(bad,bool) or not np.isfinite(bad):
            raise ValueError('BAD_DATA metadata must be finite')
        mask &= ~np.any(np.isclose(values,bad,rtol=1e-6,atol=0.),axis=0)
        return values,mask


def read_split_pair_fits(path_a,path_b=None,*,metadata=None,fields=('Q_STOKES','U_STOKES'),hdu=1,
                         target_units=None,target_polarization=None,target_ordering=None):
    """Read our paired schema or two declared full-sky HEALPix FITS tables.

    Two-file inputs must match the original declared metadata BEFORE explicit
    target conversions. Missing beam response is represented only by a caller's
    explicit ``ObservingResponse('unknown', {'reason': ...})`` declaration.
    """
    if path_b is not None:
        if not isinstance(metadata,SplitMetadata):
            raise ValueError('two-file FITS ingestion requires explicit SplitMetadata')
        a,ma=_read_external_fits(path_a,metadata,fields,hdu,metadata.response_a)
        b,mb=_read_external_fits(path_b,metadata,fields,hdu,metadata.response_b)
        pair=SplitPair(a,b,ma,mb,metadata,dict(A=file_sha256(path_a),B=file_sha256(path_b)))
    else:
        fits=_dependency('astropy.io.fits','FITS support')
        with fits.open(path_a,memmap=False,checksum=True) as hdus:
            if (len(hdus)!=6 or [item.name for item in hdus]!=['PRIMARY','A','B','MASK_A','MASK_B','METADATA']
                    or hdus[0].header.get('CSTSCHEM')!=SCHEMA_VERSION):
                raise ValueError('unsupported or incomplete paired FITS schema')
            _checksums(hdus,required=True)
            raw=hdus['METADATA'].data
            if raw.dtype!=np.uint8 or raw.ndim!=1:
                raise ValueError('paired FITS metadata must be UTF-8 bytes')
            manifest=_json_read(raw.tobytes())
            arrays={}
            for name in _ARRAYS:
                value=np.asarray(hdus[name.upper()].data)
                if name.startswith('mask_'):
                    if (hdus[name.upper()].header.get('MASKTYPE')!='BOOLEAN_VALIDITY'
                            or value.dtype!=np.uint8 or not np.isin(value,[0,1]).all()):
                        raise ValueError('paired FITS masks must be Boolean validity')
                    value=value.astype(bool)
                elif value.dtype.kind!='f' or value.dtype.itemsize!=8:
                    raise ValueError('paired FITS Q/U storage must be float64')
                arrays[name]=np.array(value,dtype=bool if name.startswith('mask_') else np.float64,copy=True)
            pair=_from_manifest(manifest,arrays,{},metadata)
            _check_header(hdus[0].header,pair.metadata,require_lmax=True)
        lineage=dict(pair.input_sha256);lineage['paired_fits']=file_sha256(path_a)
        pair=SplitPair(pair.a,pair.b,pair.mask_a,pair.mask_b,pair.metadata,lineage)
    return _converted(pair,target_units,target_polarization,target_ordering)


def write_split_pair_hdf5(path,pair,*,group=HDF5_GROUP):
    """Write explicit datasets/metadata with exact float64 and Boolean storage."""
    if not isinstance(pair,SplitPair):
        raise ValueError('SplitPair required')
    if group!=HDF5_GROUP:
        raise ValueError('the versioned HDF5 schema has a fixed group path')
    h5py=_dependency('h5py','HDF5 support')
    with h5py.File(path,'x') as container:
        node=container.create_group(group)
        node.attrs['schema_version']=SCHEMA_VERSION
        node.attrs['manifest_json']=canonical_json(_manifest(pair))
        for name in _ARRAYS:
            node.create_dataset(name,data=getattr(pair,name),dtype=np.bool_ if name.startswith('mask_') else '<f8',
                fletcher32=True,chunks=True)
    return Path(path)


def read_split_pair_hdf5(path,*,metadata=None,group=HDF5_GROUP,target_units=None,
                         target_polarization=None,target_ordering=None):
    if group!=HDF5_GROUP:
        raise ValueError('the versioned HDF5 schema has a fixed group path')
    h5py=_dependency('h5py','HDF5 support')
    with h5py.File(path,'r') as container:
        if group not in container:
            raise ValueError('missing explicit paired HDF5 schema group')
        node=container[group]
        if set(node.keys())!=set(_ARRAYS) or set(node.attrs.keys())!={'schema_version','manifest_json'}:
            raise ValueError('missing or unsupported paired HDF5 fields')
        if node.attrs['schema_version']!=SCHEMA_VERSION:
            raise ValueError('unsupported HDF5 schema version')
        manifest=_json_read(node.attrs['manifest_json'])
        arrays={}
        for name in _ARRAYS:
            dataset=node[name]
            if (name.startswith('mask_') and dataset.dtype!=bool) or (not name.startswith('mask_')
                    and (dataset.dtype.kind!='f' or dataset.dtype.itemsize!=8)):
                raise ValueError('paired HDF5 requires exact float64 Q/U and Boolean validity datasets')
            arrays[name]=dataset[()]
        pair=_from_manifest(manifest,arrays,dict(paired_hdf5=file_sha256(path)),metadata)
    return _converted(pair,target_units,target_polarization,target_ordering)
