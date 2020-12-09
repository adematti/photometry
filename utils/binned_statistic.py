import copy
import logging
import numpy as np
from scipy import stats

class Binning(object):

    def __init__(self,samples=None,weights=None,edges=None,nbins=10,range=None,quantiles=None,scale='linear'):
        self.edges = edges
        if edges is None:
            if range is None:
                if quantiles is None:
                    range = [samples.min(axis=-1),samples.max(axis=-1)]
                    range[-1] = range[-1]+(range[-1]-range[0])*1e-5
                else:
                    range = np.percentile(samples,q=np.array(quantiles)*100.,axis=-1).T
            if range[0] is None: range[0] = samples.min(axis=-1)
            if range[-1] is None:
                range[-1] = samples.max(axis=-1)
                range[-1] = range[-1]+(range[-1]-range[0])*1e-5
            if isinstance(nbins,np.integer):
                if scale == 'linear':
                    self.edges = np.linspace(range[0],range[-1],nbins+1)
                elif scale == 'log':
                    self.edges = np.logspace(np.log10(range[0]),np.log10(range[-1]),nbins+1,base=10)
                else:
                    raise ValueError('Scale {} is unkown.'.format(scale))
            else:
                self.edges = np.histogram_bin_edges(samples,bins=nbins,range=range,weights=weights)

    @property
    def range(self):
        return (self.edges[0],self.edges[-1])

    @property
    def nbins(self):
        return len(self.edges)-1

    @property
    def centers(self):
        return (self.edges[:-1]+self.edges[1:])/2.

def digitize_dd(samples,weights=None,edges=None,nbins=10,ranges=None,quantiles=None,scales='linear',bounds=[-2,-1]):

    nsamples = len(samples)
    if not isinstance(weights,list) or np.isscalar(weights[0]): weights = [weights]*nsamples
    if not isinstance(edges,list) or np.isscalar(edges[0]): edges = [edges]*nsamples
    if not isinstance(nbins,list): nbins = [nbins]*nsamples
    if not isinstance(ranges,list) or np.isscalar(ranges[0]): ranges = [ranges]*nsamples
    if not isinstance(quantiles,list) or np.isscalar(quantiles[0]): quantiles = [quantiles]*nsamples
    if not isinstance(scales,list): scales = [scales]*nsamples
    edges = [Binning(samples=args[0],weights=args[1],edges=args[2],nbins=args[3],range=args[4],quantiles=args[5],scale=args[6]).edges for args in zip(samples,weights,edges,nbins,ranges,quantiles,scales)]
    binnumbers = []
    for sample,edge in zip(samples,edges):
        dig = np.digitize(sample,bins=edge,right=False)-1
        dig[dig<0] = bounds[0]
        dig[dig>=len(edge)-1] = bounds[-1]
        binnumbers.append(dig)

    return np.array(binnumbers),edges

def prodprod(arr,start=[]):
    if len(arr) == 0: return np.array(start)
    toret = [arr[0]]
    for a in arr[1:]: toret.append(a*toret[-1])
    return np.array(start+toret)

def ravel(arr,ndim):
    ravel = 0
    ndims = 1
    for arr_,ndim_ in zip(arr,ndim):
        ravel += arr_*ndims
        ndims *= ndim_
    return ravel

class BinnedStatistic(object):

    logger = logging.getLogger('BinnedStatistic')

    def __init__(self,samples,weights=None,**kwargs):

        samples = np.asarray(samples)
        ibins,self.edges = digitize_dd(samples,weights=None,bounds=[-1,-2],**kwargs)
        #if values is None: values = np.ones_like(samples[0])

        #print(self.edges,len(self.edges[0]),ibins.min(),samples[0][ibins[0]<=0])
        mask_good = (ibins>=0).all(axis=0)
        ibins = ibins[:,mask_good]
        #samples = samples[:,mask_good]
        #values = values[mask_good]

        ibins = ravel(ibins,self.nbins)
        uniques,inverse = np.unique(ibins,return_inverse=True)
        self.ibin_edges = np.concatenate([uniques,[uniques[-1]+1]])
        #print(np.diff(uniques))

        self.logger.info('Using {:d} bins.'.format(len(uniques)))
        #self.samples = np.array([stats.binned_statistic(ibins,values=sample,statistic='mean',bins=self.ibin_edges)[0] for sample in samples])
        #self.values = stats.binned_statistic(ibins,values=values,statistic=statistic,bins=self.ibin_edges)[0]

    @property
    def size(self):
        return len(self.values)

    @property
    def nbins(self):
        return [len(edge)-1 for edge in self.edges]

    def get_ibin(self,samples):
        ibins = digitize_dd(samples,edges=self.edges)[0]
        mask_bad = ~(ibins>=0).all(axis=0)
        ibins[:,mask_bad] = -np.prod(self.nbins)
        ibins = ravel(ibins,self.nbins)
        ibins[mask_bad] = -1
        return ibins

    def __call__(self,samples,values=None,statistic='sum'):
        ibins = self.get_ibin(samples)
        if values is None: values = np.ones_like(samples[0])
        return stats.binned_statistic(ibins,values=values,statistic=statistic,bins=self.ibin_edges)[0]

    def deepcopy(self):
        new = object.__new__(self.__class__)
        for key in ['edges','ibin_edges']:
            if hasattr(self,key): setattr(new,key,getattr(self,key).copy())
        return new
