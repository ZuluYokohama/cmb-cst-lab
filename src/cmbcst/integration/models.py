"""Strict HEALPix split-map contracts; no implicit units or sky conventions."""
from __future__ import annotations

from dataclasses import dataclass,field,replace
import hashlib
import json
from types import MappingProxyType
from collections.abc import Mapping
import numpy as np

SCHEMA_VERSION = 'cmbcst.split_pair.v1'
UNITS = ('K_CMB','uK_CMB')
ORDERINGS = ('RING','NESTED')
POLARIZATION_CONVENTIONS = ('COSMO','IAU')
COORDINATE_FRAMES = ('G','C','E')


class MissingOptionalDependency(ImportError):
    """A requested adapter is unavailable; no replacement calculation ran."""


def _integer(value,label):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)):
        raise ValueError(f'{label} must be an integer')
    return int(value)


def validate_geometry(nside,lmax):
    nside,lmax = _integer(nside,'nside'),_integer(lmax,'lmax')
    if nside<1 or nside&(nside-1):
        raise ValueError('nside must be a positive power of two')
    if not 2<=lmax<=3*nside-1:
        raise ValueError('spin-2 lmax must be between 2 and 3*nside-1')
    return nside,lmax


def _sha(value,label):
    if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(f'{label} must be a lowercase SHA-256 hexadecimal digest')
    return value


def _freeze(value):
    if isinstance(value,Mapping):
        if any(not isinstance(key,str) or not key for key in value):
            raise ValueError('metadata object keys must be nonempty strings')
        return MappingProxyType({key:_freeze(item) for key,item in value.items()})
    if isinstance(value,(list,tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value,(str,bool,int)):
        return value
    if isinstance(value,float) and np.isfinite(value):
        return value
    raise ValueError('metadata must contain finite JSON values')


def _plain(value):
    if isinstance(value,Mapping):
        return {key:_plain(item) for key,item in value.items()}
    if isinstance(value,tuple):
        return [_plain(item) for item in value]
    return value


def canonical_json(value):
    return json.dumps(_plain(_freeze(value)),sort_keys=True,separators=(',',':'),allow_nan=False)


def array_sha256(values):
    """Bind dtype, shape and exact canonical little-endian C-order array bytes."""
    array = np.asarray(values)
    if array.dtype.kind not in 'bfciu' or array.dtype.hasobject:
        raise ValueError('numeric or boolean array required for a content hash')
    array = np.ascontiguousarray(array,dtype=array.dtype.newbyteorder('<'))
    header = canonical_json(dict(dtype=array.dtype.str,shape=list(array.shape))).encode()
    digest = hashlib.sha256(b'cmbcst.array.v1\0'+header+b'\0')
    digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()


@dataclass(frozen=True)
class ObservingResponse:
    """Declared response model, not a claim that an instrument was inverted.

    ``parameters`` binds beam/filter/mask asset hashes and model settings.
    An unknown response must be explicitly described with a nonempty reason.
    The fingerprint binds this declaration, not an unprovided response matrix.
    """
    kind: str
    parameters: Mapping

    def __post_init__(self):
        if self.kind not in ('identity','diagonal','linear','unknown'):
            raise ValueError('response kind must be identity, diagonal, linear or unknown')
        if not isinstance(self.parameters,Mapping):
            raise ValueError('response parameters must be a JSON object')
        parameters = _freeze(self.parameters)
        if self.kind=='identity' and parameters:
            raise ValueError('identity response has no parameters')
        if self.kind=='unknown' and (not isinstance(parameters.get('reason'),str) or not parameters['reason'].strip()):
            raise ValueError('unknown response needs an explicit nonempty reason')
        if self.kind in ('diagonal','linear'):
            _sha(parameters.get('asset_sha256'),'response asset_sha256')
        object.__setattr__(self,'parameters',parameters)

    def to_dict(self):
        return dict(kind=self.kind,parameters=_plain(self.parameters))

    @classmethod
    def from_dict(cls,value):
        if not isinstance(value,Mapping) or set(value)!= {'kind','parameters'}:
            raise ValueError('response needs exactly kind and parameters')
        return cls(**value)

    @property
    def fingerprint(self):
        return hashlib.sha256(canonical_json(self.to_dict()).encode()).hexdigest()


@dataclass(frozen=True)
class SplitMetadata:
    nside: int
    lmax: int
    ordering: str
    coordinate_frame: str
    units: str
    polarization_convention: str
    mask_a_sha256: str
    mask_b_sha256: str
    response_a: ObservingResponse
    response_b: ObservingResponse
    coordinate_epoch: float|None = None
    spin: int = 2
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        nside,lmax = validate_geometry(self.nside,self.lmax)
        object.__setattr__(self,'nside',nside);object.__setattr__(self,'lmax',lmax)
        for value,allowed,label in ((self.ordering,ORDERINGS,'ordering'),
                (self.coordinate_frame,COORDINATE_FRAMES,'coordinate_frame'),
                (self.units,UNITS,'units'),(self.polarization_convention,POLARIZATION_CONVENTIONS,'polarization_convention')):
            if value not in allowed:
                raise ValueError(f'{label} must be explicitly declared as one of {allowed}')
        if _integer(self.spin,'spin')!=2 or self.schema_version!=SCHEMA_VERSION:
            raise ValueError('unsupported spin or split-map schema')
        if self.coordinate_epoch is not None and (isinstance(self.coordinate_epoch,bool)
                or not isinstance(self.coordinate_epoch,(int,float)) or not np.isfinite(self.coordinate_epoch)):
            raise ValueError('coordinate_epoch must be finite or explicitly null')
        if self.coordinate_frame in ('C','E') and self.coordinate_epoch is None:
            raise ValueError('celestial/ecliptic coordinates require an explicit coordinate_epoch')
        _sha(self.mask_a_sha256,'mask_a_sha256');_sha(self.mask_b_sha256,'mask_b_sha256')
        if not isinstance(self.response_a,ObservingResponse) or not isinstance(self.response_b,ObservingResponse):
            raise ValueError('both observing responses must be explicitly declared')

    @property
    def same_observing_response(self):
        return self.response_a.fingerprint==self.response_b.fingerprint

    def to_dict(self):
        return dict(schema_version=self.schema_version,nside=self.nside,lmax=self.lmax,spin=2,
            ordering=self.ordering,coordinate_frame=self.coordinate_frame,coordinate_epoch=self.coordinate_epoch,
            units=self.units,polarization_convention=self.polarization_convention,
            mask_a_sha256=self.mask_a_sha256,mask_b_sha256=self.mask_b_sha256,
            response_a=self.response_a.to_dict(),response_b=self.response_b.to_dict())

    @classmethod
    def from_dict(cls,value):
        expected={'schema_version','nside','lmax','spin','ordering','coordinate_frame','coordinate_epoch',
            'units','polarization_convention','mask_a_sha256','mask_b_sha256','response_a','response_b'}
        if not isinstance(value,Mapping) or set(value)!=expected:
            raise ValueError('missing or unsupported split metadata fields')
        result = dict(value)
        for name in ('response_a','response_b'):
            result[name]=ObservingResponse.from_dict(result[name])
        return cls(**result)

    @property
    def fingerprint(self):
        return hashlib.sha256(canonical_json(self.to_dict()).encode()).hexdigest()


@dataclass(frozen=True)
class SplitPair:
    """Finite Q/U observations and Boolean validity, kept separate from weights.

    Masked finite samples are retained losslessly. ``operator_maps`` creates an
    explicit zero-filled analysis view; it does not overwrite observations.
    ``input_sha256`` records actual input files when read by an adapter.
    """
    a: np.ndarray
    b: np.ndarray
    mask_a: np.ndarray
    mask_b: np.ndarray
    metadata: SplitMetadata
    input_sha256: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.metadata,SplitMetadata):
            raise ValueError('validated SplitMetadata required')
        count=12*self.metadata.nside**2
        for name in ('a','b'):
            value=np.asarray(getattr(self,name))
            if value.dtype.kind!='f' or value.dtype.itemsize>8 or value.shape!=(2,count) or not np.isfinite(value).all():
                raise ValueError('finite floating Q/U arrays with shape [2,12*nside**2] required')
            value=np.array(value,dtype=np.float64,order='C',copy=True);value.flags.writeable=False
            object.__setattr__(self,name,value)
        for name in ('mask_a','mask_b'):
            value=np.asarray(getattr(self,name))
            if value.dtype!=bool or value.shape!=(count,):
                raise ValueError('Boolean validity masks required; apodization weights are separate')
            value=np.array(value,copy=True);value.flags.writeable=False
            if array_sha256(value)!=getattr(self.metadata,name+'_sha256'):
                raise ValueError(f'{name} content hash does not match metadata')
            object.__setattr__(self,name,value)
        if not isinstance(self.input_sha256,Mapping):
            raise ValueError('input_sha256 must be an object of source file hashes')
        for key,value in self.input_sha256.items():
            if not isinstance(key,str) or not key:
                raise ValueError('input hash names must be nonempty strings')
            _sha(value,'input SHA-256')
        object.__setattr__(self,'input_sha256',MappingProxyType(dict(self.input_sha256)))

    @classmethod
    def create(cls,a,b,*,nside,lmax,ordering,coordinate_frame,units,polarization_convention,
               response_a,response_b,mask_a=None,mask_b=None,coordinate_epoch=None,input_sha256=None):
        nside,lmax=validate_geometry(nside,lmax)
        ma=np.ones(12*nside**2,dtype=bool) if mask_a is None else np.asarray(mask_a)
        mb=np.ones(12*nside**2,dtype=bool) if mask_b is None else np.asarray(mask_b)
        metadata=SplitMetadata(nside,lmax,ordering,coordinate_frame,units,polarization_convention,
            array_sha256(ma),array_sha256(mb),response_a,response_b,coordinate_epoch)
        return cls(a,b,ma,mb,metadata,input_sha256 or {})

    def operator_maps(self,split):
        if split not in ('A','B'):
            raise ValueError('split must be A or B')
        return np.where(getattr(self,'mask_'+split.lower())[None],getattr(self,split.lower()),0.)

    def converted(self,*,units=None,polarization_convention=None,ordering=None):
        target_units=self.metadata.units if units is None else units
        target_pol=self.metadata.polarization_convention if polarization_convention is None else polarization_convention
        target_order=self.metadata.ordering if ordering is None else ordering
        if target_units not in UNITS or target_pol not in POLARIZATION_CONVENTIONS or target_order not in ORDERINGS:
            raise ValueError('unsupported explicit target convention')
        a,b=self.a.copy(),self.b.copy();ma,mb=self.mask_a.copy(),self.mask_b.copy()
        if target_units!=self.metadata.units:
            factor=1e6 if target_units=='uK_CMB' else 1e-6
            a*=factor;b*=factor
        if target_pol!=self.metadata.polarization_convention:
            a[1]*=-1;b[1]*=-1
        if target_order!=self.metadata.ordering:
            from .operators import reorder_healpix
            a=reorder_healpix(a,self.metadata.nside,self.metadata.ordering,target_order)
            b=reorder_healpix(b,self.metadata.nside,self.metadata.ordering,target_order)
            ma=reorder_healpix(ma,self.metadata.nside,self.metadata.ordering,target_order)
            mb=reorder_healpix(mb,self.metadata.nside,self.metadata.ordering,target_order)
        metadata=replace(self.metadata,units=target_units,polarization_convention=target_pol,ordering=target_order,
            mask_a_sha256=array_sha256(ma),mask_b_sha256=array_sha256(mb))
        return SplitPair(a,b,ma,mb,metadata,self.input_sha256)

    def provenance(self):
        return dict(metadata=self.metadata.to_dict(),metadata_sha256=self.metadata.fingerprint,
            a_sha256=array_sha256(self.a),b_sha256=array_sha256(self.b),input_sha256=dict(self.input_sha256),
            same_observing_response=self.metadata.same_observing_response)
