import os
import numpy as np
from scipy import constants
import fitsio
import logging
from .utils import utils

def distance(position,axis=-1):
    return np.sqrt((position**2).sum(axis=axis))

def cartesian_to_sky(position,wrap=True,degree=True):
    """Transform cartesian coordinates into distance, RA, Dec.

    Parameters
    ----------
    position : array of shape (N,3)
        position in cartesian coordinates.
    wrap : bool, optional
        whether to wrap ra into [0,2*pi]
    degree : bool, optional
        whether RA, Dec are in degree (True) or radian (False).

    Returns
    -------
    dist : array
        distance.
    ra : array
        RA.
    dec : array
        Dec.

    """
    dist = distance(position)
    ra = np.arctan2(position[:,1],position[:,0])
    if wrap: ra %= 2.*constants.pi
    dec = np.arcsin(position[:,2]/dist)
    if degree: return dist,ra/constants.degree,dec/constants.degree
    return dist,ra,dec

def sky_to_cartesian(dist,ra,dec,degree=True,dtype=None):
    """Transform distance, RA, Dec into cartesian coordinates.

    Parameters
    ----------
    dist : array
        distance.
    ra : array
        RA.
    dec : array
        Dec.
    degree : bool
        whether RA, Dec are in degree (True) or radian (False).
    dtype : dtype, optional
        return array dtype.

    Returns
    -------
    position : array
        position in cartesian coordinates; of shape (len(dist),3).

    """
    conversion = 1.
    if degree: conversion = constants.degree
    position = [None]*3
    cos_dec = np.cos(dec*conversion)
    position[0] = cos_dec*np.cos(ra*conversion)
    position[1] = cos_dec*np.sin(ra*conversion)
    position[2] = np.sin(dec*conversion)
    return (dist*np.asarray(position,dtype=dtype)).T

class Catalogue(object):

    logger = logging.getLogger('Catalogue')

    def __init__(self,columns={},fields=None,**attrs):

        self.columns = {}
        if fields is None: fields = columns.keys()
        for key in fields:
            self.columns[key] = np.asarray(columns[key])
        self.attrs = attrs

    def sort_fields(self,keep=None,remove=[]):
        if keep is None: keep = self.fields
        for rm in remove:
            keep.remove(rm)
        return keep

    def rename(self,field_old,field_new):
        self.columns[field_new] = self.columns[field_old]
        if field_new != field_old: del self.columns[field_old]

    def set_upper_case(self,fields=None):
        if fields is None: fields = self.fields
        for field in fields:
            self.rename(field,field.upper())

    def set_lower_case(self,fields=None):
        if fields is None: fields = self.fields
        for field in fields:
            self.rename(field,field.lower())

    def to_dict(self,keep=None,remove=[]):
        keep = self.sort_fields(keep=keep,remove=remove)
        return {field:self[field] for field in keep}

    def keep(self,keep=None,remove=[]):
        return self.__class__(self.to_dict(keep=keep,remove=remove))

    def getstate(self,keep=None,remove=[]):
        return {'columns':self.to_dict(keep=keep,remove=remove),'attrs':self.attrs}

    def setstate(self,state):
        self.__dict__.update(state)

    @classmethod
    def load(cls,path,**kwargs):
        if any(path.endswith(ext) for ext in ['.fits','.fits.gz','.fits.fz']):
            return cls.load_fits(path,**kwargs)
        return cls.load_npy(path,**kwargs)

    def save(self,path,**kwargs):
        if any(path.endswith(ext) for ext in ['.fits','.fits.gz','.fits.fz']):
            self.save_fits(path,**kwargs)
        else:
            self.save_npy(path,**kwargs)

    @classmethod
    def load_fits(cls,path,keep=None,**kwargs):
        self = cls()
        self.logger.info('Loading catalogue {}.'.format(path))
        array,header = fitsio.read(path,header=True,**kwargs)
        self = cls.from_array(array,keep=keep)
        self.header = header
        return self

    @classmethod
    def load_fits_header(cls,path,**kwargs):
        self = cls()
        self.logger.info('Loading header of catalogue {}.'.format(path))
        header = fitsio.read_header(path,**kwargs)
        return header

    @classmethod
    def load_npy(cls,path,fields=None):
        state = {}
        try:
            state = np.load(path,allow_pickle=True)[()]
        except IOError:
            raise IOError('Invalid path: {}.'.format(path))
            cls.logger.info('Loading {}: {}.'.format(cls.__name__,path))
        self = cls.loadstate(state).keep(keep=keep)

    @utils.set_mpi_comm
    def save_npy(self,save,keep=None,remove=[],comm=None,root=0):
        if comm.rank == root:
            self.logger.info('Saving {} to {}.'.format(self.__class__.__name__,save))
            utils.mkdir(save)
            np.save(save,self.getstate(keep=keep,remove=remove))

    @utils.set_mpi_comm
    def save_fits(self,save,keep=None,remove=[],comm=None,root=0,**kwargs):
        if comm.rank == root:
            array = self.to_array(keep=keep,remove=remove)
            self.logger.info('Saving catalogue to {}.'.format(save))
            utils.mkdir(save)
            fitsio.write(save,array,header=getattr(self,'header',kwargs.get('header',None)),clobber=True,**kwargs)

    @classmethod
    def from_array(cls,array,keep=None):
        return cls(array,fields=keep if keep is not None else array.dtype.names)

    def to_array(self,keep=None,remove=[]):
        keep = self.sort_fields(keep=keep,remove=remove)
        toret = np.empty(self.size,dtype=[(field,self[field].dtype,self[field].shape[1:]) for field in keep])
        for field in keep: toret[field] = self[field]
        return toret

    @classmethod
    def from_nbodykit(cls,catalogue,keep=None,allgather=True,**kwargs):
        if keep is None:
            columns = {key: catalogue[key].compute() for key in catalogue}
        else:
            columns = {key: catalogue[key].compute() for key in keep}
        if allgather:
            columns = {key: np.concatenate(catalogue.comm.allgather(columns[key]),axis=0) for key in columns}
        attrs = getattr(catalogue,'attrs',{})
        attrs.update(kwargs)
        return cls(columns=columns,**attrs)

    """
    def to_nbodykit(self,keep=None,remove=[]):

        from nbodykit.base.catalog import CatalogSource
        from nbodykit import CurrentMPIComm

        comm = CurrentMPIComm.get()
        if comm.rank == 0:
            source = self.keep(keep=keep,remove=remove)
        else:
            source = None
        source = comm.bcast(source)

        # compute the size
        #print('size',comm.size)
        start = comm.rank * source.size // comm.size
        end = (comm.rank + 1) * source.size // comm.size

        new = object.__new__(CatalogSource)
        new._size = end - start
        CatalogSource.__init__(new,comm=comm)
        for key in source.fields:
            new[key] = new.make_column(source[key])[start:end]
        new.attrs.update(source.attrs)

        return new
    """
    def to_nbodykit(self,keep=None,remove=[]):

        from nbodykit.base.catalog import CatalogSource
        from nbodykit import CurrentMPIComm

        new = object.__new__(CatalogSource)
        new._size = self.size
        CatalogSource.__init__(new,comm=CurrentMPIComm.get())
        for key in self.fields:
            new[key] = new.make_column(self[key])
        new.attrs.update(self.attrs)

        return new

    @classmethod
    @utils.set_mpi_comm
    def mpi_scatter(cls,self,root=0,comm=None,mask=None,counts=None):
        if comm.rank == root:
            if self is None:
                self,fields,size = None,None,None
                columns = {}
            else:
                fields,size,columns = self.fields,self.size,self.columns
                delattr(self,'columns')
                if mask is not None:
                    if isinstance(mask[0],np.bool_):
                        mask = np.flatnonzero(mask)
                    if not np.all(np.diff(mask) >= 0):
                        raise ValueError('You should pass a sorted mask array.')
                    indices = np.array_split(mask,comm.size)
                    counts = np.diff([0] + [index.max()+1 for index in indices[:-1]] + [size])
        else:
            self,fields,size,counts = None,None,None,None
            columns = {}
        self = comm.bcast(self)
        if self is None: return None
        fields = comm.bcast(fields)
        size = comm.bcast(size)
        counts = comm.bcast(counts)
        #print(counts)
        self.columns = {field:utils.mpi_scatter_array(columns.get(field,None),comm=comm,root=root,counts=counts) for field in fields}
        return self

    @utils.set_mpi_comm
    def mpi_gather(self,root=0,comm=None):
        for key in self:
            self[key] = utils.mpi_gather_array(self[key],comm,root=root)

    """
    @utils.mpi_comm
    def scatter_mpi(self,mask=None,comm=None):
        # compute the size
        if mask is None:
            mask = np.arange(self.size)
        if isinstance(mask[0],np.bool_):
            mask = np.flatnonzero(mask)
        if not np.all(np.diff(mask) >= 0):
            raise ValueError('You should pass a sorted mask array.')
        indices = np.array_split(mask,comm.size)
        ends = [index.max()+1 for index in indices[:-1]] + [-1]
        starts = [0] + ends[:-1]
        start = starts[comm.rank]
        end = ends[comm.rank]
        print(start,end)
        for key in self:
            self[key] = self[key][start:end]
        self.comm = comm
    """
    '''
    def scatter_mpi(self):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        # compute the size
        start = comm.rank * self.size // comm.size
        end = (comm.rank + 1) * self.size // comm.size
        for key in self:
            self[key] = self[key][start:end]
        self.comm = comm
        #return source
        """
        if comm.rank == 0:
            source = self.keep(keep=keep,remove=remove)
        else:
            source = None
        source = comm.bcast(source)

        new = object.__new__(self.__class__)
        new.__dict__.update(source.__dict__)
        #print(start,end,comm.size)
        for key in source.fields:
            new[key] = source[key][start:end]
        new.attrs.update(source.attrs)
        new.comm = comm
        return new
        """
    '''

    def shuffle(self,fields=None,seed=None):
        if fields is None: fields = self.fields
        rng = np.random.RandomState(seed=seed)
        indices = self.indices()
        rng.shuffle(indices)
        for key in fields: self[key] = self[key][indices]

    def indices(self):
        return np.arange(self.size)

    def slice(self,islice=0,nslices=1):
        size = len(self)
        return self[islice*size//nslices:(islice+1)*size//nslices]

    def downsample(self,factor=2.,rng=None,seed=None):
        if factor >= 1.: return self
        self.logger.info('Downsampling {} by factor {:.4f}.'.format(self.__class__.__name__,factor))
        if rng is None: rng = np.random.RandomState(seed=seed)
        mask = factor >= rng.uniform(0.,1.,len(self))
        return self[mask]

    def distance(self,position='Position',axis=-1):
        return distance(self[position],axis=axis)

    def cartesian_to_sky(self,position='Position',wrap=True,degree=True):
        return cartesian_to_sky(position=self[position],wrap=wrap,degree=degree)

    def sky_to_cartesian(self,distance='distance',ra='RA',dec='DEC',degree=True,dtype=None):
        return sky_to_cartesian(distance=self[distance],ra=self[ra],dec=self[dec],degree=degree,dtype=dtype)

    def footprintsize(self,ra='RA',dec='DEC',position=None,degree=True):
        # WARNING: assums footprint does not cross RA = 0
        if position is not None:
            degree = False
            _,ra,dec = cartesian_to_sky(position=self[position],wrap=True,degree=degree)
        else:
            ra,dec = self[ra],self[dec]
        conversion = 1.
        if degree: conversion = constants.degree
        ra = (ra*conversion) % (2.*constants.pi)
        dec = dec*conversion
        ra,dec = np.array([ra.min(),ra.max()]),np.array([dec.min(),dec.max()])
        ra_degree,dec_degree = ra/constants.degree,dec/constants.degree
        self.logger.info('RA x DEC: [{:.1f}, {:.1f}] x [{:.1f}, {:.1f}].'.format(ra_degree.min(),ra_degree.max(),dec_degree.min(),dec_degree.max()))
        position = sky_to_cartesian([1.,1.],ra,dec,degree=False,dtype=np.float64)
        return distance(position[0]-position[1])

    def box(self,position='Position',axis=-1):
        axis = 0 if axis == -1 else -1
        return (self[position].min(axis=axis),self[position].max(axis=axis))

    def boxsize(self,position='Position',axis=-1):
        lbox = np.diff(self.box(position=position,axis=axis),axis=0)[0]
        return np.sqrt((lbox**2).sum(axis=0))

    def mask_in_box(self,ramin,ramax,decmin,decmax):
        if decmin < -90. or decmax > 90. or decmax <= decmin or ramax <= ramin:
                raise ValueError('Strange input: [ramin, ramax, decmin, decmax] = {}'.format(radecbox))
        return ((self['RA'] >= ramin) & (self['RA'] < ramax) & (self['DEC'] >= decmin) & (self['DEC'] < decmax))

    def __getitem__(self,name):
        if isinstance(name,list) and isinstance(name[0],str):
            return [self[name_] for name_ in name]
        if isinstance(name,str):
            if name in self.fields:
                return self.columns[name]
            else:
                raise KeyError('There is no field {} in the data.'.format(name))
        else:
            import copy
            new = self.__class__({field:self.columns[field][name] for field in self.fields},**copy.deepcopy(self.attrs))
            return new

    def __setitem__(self,name,item):
        if isinstance(name,list) and isinstance(name[0],str):
            for name_,item_ in zip(name,item):
                self.data[name_] = item_
        if isinstance(name,str):
            self.columns[name] = item
        else:
            for key in self.fields:
                self.columns[key][name] = item

    def __delitem__(self,name):
        del self.columns[name]

    def    __contains__(self,name):
        return name in self.columns

    def __iter__(self):
        for field in self.columns:
            yield field

    def __str__(self):
        return str(self.columns)

    def __len__(self):
        return len(self[self.fields[0]])

    @property
    def size(self):
        return len(self)

    def zeros(self,dtype=np.float64):
        return np.zeros(len(self),dtype=dtype)

    def ones(self,dtype=np.float64):
        return np.ones(len(self),dtype=dtype)

    def falses(self):
        return self.zeros(dtype=np.bool_)

    def trues(self):
        return self.ones(dtype=np.bool_)

    def nans(self):
        return self.ones()*np.nan

    @property
    def fields(self):
        return list(self.columns.keys())

    def __delitem__(self,name):
        del self.columns[name]

    def __radd__(self,other):
        if other == 0: return self
        else: return self.__add__(other)

    def __add__(self,other):
        new = {}
        fields = [field for field in self.fields if field in other.fields]
        for field in fields:
            new[field] = np.concatenate([self[field],other[field]],axis=0)
        import copy
        attrs = copy.deepcopy(self.attrs)
        attrs.update(copy.deepcopy(other.attrs))
        return self.__class__(new,fields=fields,**attrs)

    @classmethod
    def loadstate(cls,state):
        self = cls()
        self.setstate(state)
        return self

    def copy(self):
        return self.__class__.loadstate(self.getstate())

    def deepcopy(self):
        import copy
        return copy.deepcopy(self)
