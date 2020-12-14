import logging
import os
import time
import json
import functools
import numpy as np
from scipy import constants,stats
import matplotlib

###############################################################################
# Physical quantities
###############################################################################

def depth_to_flux(x):
    return 5./np.sqrt(x)

def flux_to_depth(x):
    return (5./x)**2

def flux_to_mag(x):
    return -2.5*(np.log10(x)-9.)

def mag_to_flux(x):
    return 10.**(x/(-2.5)+9.)

def depth_to_mag(x):
    return flux_to_mag(depth_to_flux(x))

def mag_to_depth(x):
    return flux_to_depth(mag_to_flux(x))

def radec_to_thphi(ra,dec):
    return (-dec+90.)*constants.pi/180.,ra*constants.pi/180.

def thphi_to_radec(theta,phi):
    return 180./constants.pi*phi,-(180./constants.pi*theta-90)

###############################################################################
# Convenient functions
###############################################################################

def fill_hpmap(pix,map,nside,fill=0.):
    import healpy
    toret = np.full(healpy.nside2npix(nside),fill,dtype=map.dtype)
    toret[pix] = map
    return toret

def ud_hpmap(map,nside_out,pix=None,nest_out=False,nside_in=None,nest_in=False,**kwargs):
    import healpy
    if pix is not None:
        map = fill_hpmap(pix,map,nside_in if nside_in is not None else healpy.npix2nside(len(pix)))
    return healpy.ud_grade(map,nside_out,order_in='NESTED' if nest_in else 'RING',order_out='NESTED' if nest_out else 'RING',**kwargs)

def nside2resol(nside,degree=True):
    import healpy
    toret = healpy.nside2resol(nside,arcmin=False)
    if degree: toret /= constants.degree
    return toret

def nside2pixarea(nside,degree=True):
    import healpy
    return healpy.nside2resol(nside,degrees=degree)

def bin_ndarray(ndarray, new_shape, weights=None, operation=np.sum):
    """Bin an ndarray in all axes based on the target shape, by summing or
    averaging. Number of output dimensions must match number of input dimensions and
    new axes must divide old ones.

    Taken from https://stackoverflow.com/questions/8090229/resize-with-averaging-or-rebin-a-numpy-2d-array
    and https://nbodykit.readthedocs.io/en/latest/_modules/nbodykit/binned_statistic.html#BinnedStatistic.reindex.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    if ndarray.ndim != len(new_shape):
        raise ValueError('Shape mismatch: {} -> {}'.format(ndarray.shape,new_shape))
    if any([c % d != 0 for d,c in zip(new_shape,ndarray.shape)]):
        raise ValueError('New shape must be a divider of the original shape'.format(ndarray.shape,new_shape))
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    if weights is not None: weights = weights.reshape(flattened)

    for i in range(len(new_shape)):
        if weights is not None:
            ndarray = operation(ndarray, weights=weights, axis=-1*(i+1))
        else:
            ndarray = operation(ndarray, axis=-1*(i+1))

    return ndarray

def overlap(a,b):
    """Returns the indices for which a and b overlap.
    Warning: makes sense if and only if a and b elements are unique.
    Taken from https://www.followthesheep.com/?p=1366.
    """
    a1 = np.argsort(a)
    b1 = np.argsort(b)
    # use searchsorted:
    sort_left_a = a[a1].searchsorted(b[b1], side='left')
    sort_right_a = a[a1].searchsorted(b[b1], side='right')
    #
    sort_left_b = b[b1].searchsorted(a[a1], side='left')
    sort_right_b = b[b1].searchsorted(a[a1], side='right')

    # # which values are in b but not in a?
    # inds_b=(sort_right_a-sort_left_a==0).nonzero()[0]
    # # which values are in b but not in a?
    # inds_a=(sort_right_b-sort_left_b==0).nonzero()[0]

    # which values of b are also in a?
    inds_b = (sort_right_a-sort_left_a > 0).nonzero()[0]
    # which values of a are also in b?
    inds_a = (sort_right_b-sort_left_b > 0).nonzero()[0]

    return a1[inds_a], b1[inds_b]

def isnaninf(array):
    """Is nan or inf."""
    return np.isnan(array) | np.isinf(array)

def isnotnaninf(array):
    """Is not nan nor inf."""
    return ~isnaninf(array)

def digitized_statistics(indices,values=None,statistic='sum'):
    """Return the array of same shape as indices, filled with the required statistics."""
    if not isinstance(indices[0],np.integer):
        uniques,inverse = np.unique(indices,return_inverse=True)
        uniques = np.arange(len(uniques))
        indices = uniques[inverse]
    else:
        uniques = np.unique(indices)
    edges = np.concatenate([uniques,[uniques[-1]+1]])
    if values is None: values = np.ones(len(indices),dtype='f8')
    statistics,_,binnumber = stats.binned_statistic(indices,values,statistic=statistic,bins=edges)
    return statistics[binnumber-1]

def digitized_interp(ind1,ind2,val2,fill):
    """Return the array such that values of indices ind1 match val2 if ind1 in ind2, fill with fill otherwise."""
    val2 = np.asarray(val2)
    unique1,indices1,inverse1 = np.unique(ind1,return_index=True,return_inverse=True)
    unique2,indices2 = np.unique(ind2,return_index=True) #reduce ind2, val2 to uniqueness
    inter1,inter2 = overlap(unique1,unique2)
    tmp2 = val2[indices2]
    tmp1 = np.full(unique1.shape,fill_value=fill,dtype=type(fill))
    tmp1[inter1] = tmp2[inter2] #fill with val2 corresponding to matching ind1 and ind2
    return tmp1[inverse1]

'''
def interp_digitized_statistics(new,indices,fill,values=None,statistic='sum'):
    """Return the array of same shape as new, filled with the required statistics."""
    stats = digitized_statistics(indices,values=values,statistic=statistic)
    return digitized_interp(new,indices,stats,fill)
'''

def interp_digitized_statistics(new,indices,values=None,statistic='sum'):
    """Return the array of same shape as new, filled with the required statistics."""
    new,inverse = np.unique(new,return_inverse=True)
    if not isinstance(indices[0],np.integer):
        dnew = np.arange(len(new))
        indices = digitized_interp(indices,new,dnew,fill=-1)
        new = dnew
    edges = np.concatenate([new,[new[-1]+1]])
    if values is None: values = np.ones(len(indices),dtype='f8')
    toret = stats.binned_statistic(indices,values,statistic=statistic,bins=edges)[0]
    return toret[inverse]

def mkdir(path):
    try:
        os.makedirs(path) #MPI...
    except OSError:
        return

def saveplot(giveax=True):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self,ax=None,path=None,fig_kwargs={},**kwargs):
            isax = True
            if giveax:
                if ax is None:
                    isax = False
                    ax = matplotlib.pyplot.gca()
                func(self,ax,**kwargs)
            else:
                isax = False
                func(self,**kwargs)
                if ax is None:
                    ax = matplotlib.pyplot.gca()
            if path is not None:
                savefig(path,**fig_kwargs)
            elif not isax:
                matplotlib.pyplot.show()
            return ax
        return wrapper
    return decorator

def savefig(path,bbox_inches='tight',pad_inches=0.1,dpi=200,**kwargs):
    """Save matplotlib figure."""
    mkdir(path)
    logger.info('Saving figure to {}.'.format(path))
    matplotlib.pyplot.savefig(path,bbox_inches=bbox_inches,pad_inches=pad_inches,dpi=dpi,**kwargs)
    matplotlib.pyplot.close(matplotlib.pyplot.gcf())

def allnone(li):
    for el in li:
        if el is not None: return False
    return True

def get_mpi_comm():
    from mpi4py import MPI
    return MPI.COMM_WORLD

def set_mpi_comm(func):
    @functools.wraps(func)
    def wrapper(*args,**kwargs):
        comm = kwargs.get('comm',None)
        if comm is None: comm = get_mpi_comm()
        kwargs['comm'] = comm
        return func(*args,**kwargs)
    return wrapper

def mpi_gather_array(data, comm, root=0):
    """
    Taken from https://github.com/bccp/nbodykit/blob/master/nbodykit/utils.py
    Gather the input data array from all ranks to the specified ``root``.
    This uses `Gatherv`, which avoids mpi4py pickling, and also
    avoids the 2 GB mpi4py limit for bytes using a custom datatype
    Parameters
    ----------
    data : array_like
        the data on each rank to gather
    comm : MPI communicator
        the MPI communicator
    root : int, or Ellipsis
        the rank number to gather the data to. If root is Ellipsis or None,
        broadcast the result to all ranks.
    Returns
    -------
    recvbuffer : array_like, None
        the gathered data on root, and `None` otherwise
    """
    from mpi4py import MPI
    if root is None: root = Ellipsis
    if not isinstance(data, np.ndarray):
        raise ValueError("`data` must by numpy array in gather_array")

    # need C-contiguous order
    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data)
    local_length = data.shape[0]

    # check dtypes and shapes
    shapes = comm.allgather(data.shape)
    dtypes = comm.allgather(data.dtype)

    # check for structured data
    if dtypes[0].char == 'V':

        # check for structured data mismatch
        names = set(dtypes[0].names)
        if any(set(dt.names) != names for dt in dtypes[1:]):
            raise ValueError("mismatch between data type fields in structured data")

        # check for 'O' data types
        if any(dtypes[0][name] == 'O' for name in dtypes[0].names):
            raise ValueError("object data types ('O') not allowed in structured data in GatherArray")

        # compute the new shape for each rank
        newlength = comm.allreduce(local_length)
        newshape = list(data.shape)
        newshape[0] = newlength

        # the return array
        if root is Ellipsis or comm.rank == root:
            recvbuffer = np.empty(newshape, dtype=dtypes[0], order='C')
        else:
            recvbuffer = None

        for name in dtypes[0].names:
            d = GatherArray(data[name], comm, root=root)
            if root is Ellipsis or comm.rank == root:
                recvbuffer[name] = d

        return recvbuffer

    # check for 'O' data types
    if dtypes[0] == 'O':
        raise ValueError("object data types ('O') not allowed in structured data in GatherArray")

    # check for bad dtypes and bad shapes
    if root is Ellipsis or comm.rank == root:
        bad_shape = any(s[1:] != shapes[0][1:] for s in shapes[1:])
        bad_dtype = any(dt != dtypes[0] for dt in dtypes[1:])
    else:
        bad_shape = None; bad_dtype = None

    bad_shape, bad_dtype = comm.bcast((bad_shape, bad_dtype))

    if bad_shape:
        raise ValueError("mismatch between shape[1:] across ranks in GatherArray")
    if bad_dtype:
        raise ValueError("mismatch between dtypes across ranks in GatherArray")

    shape = data.shape
    dtype = data.dtype

    # setup the custom dtype
    duplicity = np.product(np.array(shape[1:], 'intp'))
    itemsize = duplicity * dtype.itemsize
    dt = MPI.BYTE.Create_contiguous(itemsize)
    dt.Commit()

    # compute the new shape for each rank
    newlength = comm.allreduce(local_length)
    newshape = list(shape)
    newshape[0] = newlength

    # the return array
    if root is Ellipsis or comm.rank == root:
        recvbuffer = np.empty(newshape, dtype=dtype, order='C')
    else:
        recvbuffer = None

    # the recv counts
    counts = comm.allgather(local_length)
    counts = np.array(counts, order='C')

    # the recv offsets
    offsets = np.zeros_like(counts, order='C')
    offsets[1:] = counts.cumsum()[:-1]

    # gather to root
    if root is Ellipsis:
        comm.Allgatherv([data, dt], [recvbuffer, (counts, offsets), dt])
    else:
        comm.Gatherv([data, dt], [recvbuffer, (counts, offsets), dt], root=root)

    dt.Free()

    return recvbuffer

def mpi_scatter_array(data, comm, root=0, counts=None):
    """
    Taken from https://github.com/bccp/nbodykit/blob/master/nbodykit/utils.py
    Scatter the input data array across all ranks, assuming `data` is
    initially only on `root` (and `None` on other ranks).
    This uses ``Scatterv``, which avoids mpi4py pickling, and also
    avoids the 2 GB mpi4py limit for bytes using a custom datatype
    Parameters
    ----------
    data : array_like or None
        on `root`, this gives the data to split and scatter
    comm : MPI communicator
        the MPI communicator
    root : int
        the rank number that initially has the data
    counts : list of int
        list of the lengths of data to send to each rank
    Returns
    -------
    recvbuffer : array_like
        the chunk of `data` that each rank gets
    """
    import logging
    from mpi4py import MPI
    if counts is not None:
        counts = np.asarray(counts, order='C')
        if len(counts) != comm.size:
            raise ValueError("counts array has wrong length!")

    # check for bad input
    if comm.rank == root:
        bad_input = not isinstance(data, np.ndarray)
    else:
        bad_input = None
    bad_input = comm.bcast(bad_input)
    if bad_input:
        raise ValueError("`data` must by numpy array on root in ScatterArray")

    if comm.rank == 0:
        # need C-contiguous order
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        shape_and_dtype = (data.shape, data.dtype)
    else:
        shape_and_dtype = None

    # each rank needs shape/dtype of input data
    shape, dtype = comm.bcast(shape_and_dtype)

    # object dtype is not supported
    fail = False
    if dtype.char == 'V':
         fail = any(dtype[name] == 'O' for name in dtype.names)
    else:
        fail = dtype == 'O'
    if fail:
        raise ValueError("'object' data type not supported in ScatterArray; please specify specific data type")

    # initialize empty data on non-root ranks
    if comm.rank != root:
        np_dtype = np.dtype((dtype, shape[1:]))
        data = np.empty(0, dtype=np_dtype)

    # setup the custom dtype
    duplicity = np.product(np.array(shape[1:], 'intp'))
    itemsize = duplicity * dtype.itemsize
    dt = MPI.BYTE.Create_contiguous(itemsize)
    dt.Commit()

    # compute the new shape for each rank
    newshape = list(shape)

    if counts is None:
        newlength = shape[0] // comm.size
        if comm.rank < shape[0] % comm.size:
            newlength += 1
        newshape[0] = newlength
    else:
        if counts.sum() != shape[0]:
            raise ValueError("the sum of the `counts` array needs to be equal to data length")
        newshape[0] = counts[comm.rank]

    # the return array
    recvbuffer = np.empty(newshape, dtype=dtype, order='C')

    # the send counts, if not provided
    if counts is None:
        counts = comm.allgather(newlength)
        counts = np.array(counts, order='C')

    # the send offsets
    offsets = np.zeros_like(counts, order='C')
    offsets[1:] = counts.cumsum()[:-1]

    # do the scatter
    comm.Barrier()
    comm.Scatterv([data, (counts, offsets), dt], [recvbuffer, dt])
    dt.Free()
    return recvbuffer

_logging_handler = None

def setup_logging(log_level="info"):
    """
    Turn on logging, with the specified level.
    Taken from nbodykit: https://github.com/bccp/nbodykit/blob/master/nbodykit/__init__.py.

    Parameters
    ----------
    log_level : 'info', 'debug', 'warning'
        the logging level to set; logging below this level is ignored.

    """

    # This gives:
    #
    # [ 000000.43 ]   0: 06-28 14:49  measurestats    INFO     Nproc = [2, 1, 1]
    # [ 000000.43 ]   0: 06-28 14:49  measurestats    INFO     Rmax = 120

    levels = {
            "info" : logging.INFO,
            "debug" : logging.DEBUG,
            "warning" : logging.WARNING,
            }

    logger = logging.getLogger();
    t0 = time.time()


    class Formatter(logging.Formatter):
        def format(self, record):
            s1 = ('[ %09.2f ]: ' % (time.time() - t0))
            return s1 + logging.Formatter.format(self, record)

    fmt = Formatter(fmt='%(asctime)s %(name)-15s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M ')

    global _logging_handler
    if _logging_handler is None:
        _logging_handler = logging.StreamHandler()
        logger.addHandler(_logging_handler)

    _logging_handler.setFormatter(fmt)
    logger.setLevel(levels[log_level])

#setup_logging()
logger = logging.getLogger('Utils')
