import os
import logging
import numpy as np
from matplotlib import pyplot as plt
from .catalogue import Catalogue
from .utils import utils

def wrap_around(mask):
    mask = mask.copy()
    index = np.flatnonzero(mask)
    mask[max(index[0]-1,0)] = True
    mask[min(index[-1]+1,mask.size-1)] = True
    return mask

class Angular2PCF(object):

    logger = logging.getLogger('Angular2PCF')

    def __init__(self,data,randoms,data2=None,randoms2=None,root=0,**attrs):
        self.attrs = attrs
        if root is not None:
            data = Catalogue.mpi_scatter(data,root=root)
            randoms = Catalogue.mpi_scatter(randoms,root=root)
            data2 = Catalogue.mpi_scatter(data2,root=root)
            randoms2 = Catalogue.mpi_scatter(randoms2,root=root)
        self.data1 = data.to_nbodykit()
        self.randoms1 = randoms.to_nbodykit()
        self.data2 = data2.to_nbodykit() if data2 is not None else None
        self.randoms2 = randoms2.to_nbodykit() if randoms2 is not None else None
        self.edges = self.attrs.get('edges',None)
        self.load_RR(path_R1R2=self.attrs.get('path_R1R2',None))

    def set_keys(self):
        keys = {'ra':self.attrs.get('key_ra','RA'),'dec':self.attrs.get('key_dec','DEC'),'weight':self.attrs.get('key_weight','WEIGHT')}
        self.attrs.update(keys)
        return keys

    def load_RR(self,path_R1R2=None):
        from nbodykit.lab import SurveyDataPairCount
        self.R1R2 = None
        if path_R1R2 is not None and os.path.isfile(path_R1R2):
            try:
                self.R1R2 = self.__class__.load(path_R1R2).R1R2
                self.logger.info('Loading {}: {}.'.format(self.__class__.__name__,path_R1R2))
            except:
                self.R1R2 = SurveyDataPairCount.load(self.path_R1R2)
                self.logger.info('Loading {}: {}.'.format(self.R1R2.__class__.__name__,path_R1R2))
            edges = self.R1R2.attrs['edges']
            if self.edges is None:
                self.edges = edges
            else:
                assert np.allclose(self.edges,edges), 'Requested and loaded RR edges do not match'
        else:
            self.logger.info('File R1R2 {} not found. It will be recomputed.'.format(path_R1R2))

    def run(self,**kwargs):
        from nbodykit.lab import SurveyDataPairCount
        self.attrs.update(kwargs)
        kwargs.update(self.set_keys())
        self.edges = np.array(self.edges)
        self.edges[0] = max(self.edges[0],1e-12) #to avoid self-pairs
        if self.R1R2 is None:
            self.R1R2 = SurveyDataPairCount('angular',self.randoms1,self.edges,second=self.randoms2,**kwargs)
        #if self.path_R1R2 is not None:
        #    utils.mkdir(self.path_R1R2)
        #    self.R1R2.save(self.path_R1R2)
        self.D1D2 = SurveyDataPairCount('angular',self.data1,self.edges,second=self.data2,**kwargs)
        if (self.data2 is not None) and (self.randoms2 is not None):
            self.D1R2 = SurveyDataPairCount('angular',self.data1,self.edges,second=self.randoms2,**kwargs)
            self.D2R1 = SurveyDataPairCount('angular',self.data2,self.edges,second=self.randoms1,**kwargs)
        else:
            self.D1R2 = SurveyDataPairCount('angular',self.data1,self.edges,second=self.randoms1,**kwargs)
            self.D2R1 = self.D1R2
        self.set_corr()

    def set_corr(self):

        self.set_sep()

        fDD = self.R1R2.attrs['total_wnpairs']/self.D1D2.attrs['total_wnpairs']
        fDR = self.R1R2.attrs['total_wnpairs']/self.D1R2.attrs['total_wnpairs']
        fRD = self.R1R2.attrs['total_wnpairs']/self.D2R1.attrs['total_wnpairs']
        nonzero = self.R1R2.pairs['npairs'] > 0
        # init
        self.corr = np.zeros(self.D1D2.pairs.shape)
        self.corr[:] = np.nan

        # the Landy - Szalay estimator
        # (DD - DR - RD + RR) / RR
        DD = (self.D1D2.pairs['wnpairs'])[nonzero]
        DR = (self.D1R2.pairs['wnpairs'])[nonzero]
        RD = (self.D2R1.pairs['wnpairs'])[nonzero]
        RR = (self.R1R2.pairs['wnpairs'])[nonzero]
        corr = (fDD * DD - fDR * DR - fRD * RD)/RR + 1
        self.corr[nonzero] = corr[:]

    def set_sep(self):
        if self.R1R2.attrs['config'].get('output_thetaavg',True):
            self.logger.info('Setting sep from R1R2.')
            self.sep = self.R1R2.pairs['theta']
        elif self.D1D2.attrs['config'].get('output_thetaavg',True):
            self.logger.info('Setting sep from D1D2.')
            self.sep = self.D1D2.pairs['theta']
        else:
            self.logger.info('Setting sep from analytics.')
            thetamin,thetamax = np.deg2rad(self.edges[:-1]),np.deg2rad(self.edges[1:])
            self.sep = np.rad2deg((thetamin*np.cos(thetamin) - thetamax*np.cos(thetamax) + np.sin(thetamax) - np.sin(thetamin))/(np.cos(thetamin)-np.cos(thetamax)))

    def rebin(self,factor=1):
        ns = len(self.sep)//factor
        self.edges = self.edges[::factor]
        self.sep = utils.bin_ndarray(self.sep,(ns,),weights=self.R1R2.pairs['wnpairs'],operation=np.average)
        self.corr = utils.bin_ndarray(self.corr,(ns,),weights=self.R1R2.pairs['wnpairs'],operation=np.average)

    def __call__(self,sep):
        return np.interp(sep,self.sep,self.corr,left=0.,right=0.)

    def getstate(self):
        state = {}
        for key in ['D1D2','D1R2','D2R1','R1R2']:
            if hasattr(self,key): state[key] = getattr(self,key).__getstate__()
        for key in ['attrs','edges']:
            if hasattr(self,key): state[key] = getattr(self,key)
        return state

    def setstate(self,state):
        from nbodykit.lab import SurveyDataPairCount
        self.__dict__.update(state)
        for key in ['D1D2','D1R2','D2R1','R1R2']:
            if key in state:
                tmp = object.__new__(SurveyDataPairCount)
                tmp.__setstate__(state[key])
                setattr(self,key,tmp)
        self.set_corr()

    @utils.set_mpi_comm
    def save(self,path,comm=None):
        if comm.rank == 0:
            utils.mkdir(path)
            self.logger.info('Saving to: {}.'.format(path))
            np.save(path,self.getstate())

    @classmethod
    def load(cls,path):
        cls.logger.info('Loading: {}.'.format(path))
        self = object.__new__(cls)
        self.setstate(np.load(path,allow_pickle=True)[()])
        return self

    @utils.saveplot()
    def plot(self,ax,others=[],covariances=[],xlim=None,ylim=None,xscale='linear',yscale='linear',xlabel='\\theta',labels=None,markers=None,linestyles='-',colors=None):
        ntot = len(others)+1
        if colors is None: colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        if not isinstance(colors,list): colors = [colors]*ntot
        if not isinstance(labels,list): labels = [labels]*ntot
        if not isinstance(markers,list): markers = [markers]*ntot
        if not isinstance(linestyles,list): linestyles = [linestyles]*ntot
        covariances = covariances + [None]*ntot
        ylabel = 'w({})'.format(xlabel)
        def y(x,y):
            return y
        if yscale == 'xlinear':
            yscale = 'linear'
            ylabel = xlabel + ylabel
            def y(x,y):
                return x*y
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        for i,inst in enumerate([self]+others):
            mask = np.ones_like(inst.sep,dtype=np.bool_)
            if xlim is not None:
                if xlim[0] is not None: mask &= inst.sep>=xlim[0]
                if xlim[-1] is not None: mask &= inst.sep<=xlim[-1]
                mask = wrap_around(mask)
            if covariances[i] is not None:
                low = y(inst.sep,covariances[i].mean(inst.sep))-y(inst.sep,covariances[i].std(inst.sep))
                up = y(inst.sep,covariances[i].mean(inst.sep))+y(inst.sep,covariances[i].std(inst.sep))
                ax.fill_between(inst.sep[mask],low[mask],up[mask],facecolor=colors[i],alpha=0.2,linewidth=0)
            ax.plot(inst.sep[mask],y(inst.sep,inst.corr)[mask],color=colors[i],label=labels[i],marker=markers[i],linestyle=linestyles[i])
        if xlim is not None: ax.set_xlim(*xlim)
        if ylim is not None: ax.set_ylim(*ylim)
        ax.set_xlabel('${}$ [deg]'.format(xlabel))
        ax.set_ylabel('${}$'.format(ylabel))
        if not utils.allnone(labels): ax.legend()

    @classmethod
    def reader(cls,path):
        tmp = cls.load(path)
        return tmp.sep,tmp.corr

class AngularHP2PCF(Angular2PCF):

    logger = logging.getLogger('AngularHP2PCF')

    def __init__(self,density,density2=None,root=0,**attrs):
        self.attrs = attrs
        data1,randoms1,data2,randoms2 = None,None,None,None
        self.set_keys()
        props = [self.attrs['ra'],self.attrs['dec']]
        key_weight = self.attrs['weight']
        def get_data_randoms(dens):
            mask = dens.brickrandoms>0
            randoms = dens.to_catalogue(key_randoms=key_weight,props=props)[mask]
            data = dens.to_catalogue(key_data=key_weight,props=props)[mask]
            if self.use_inverse_weights:
                #data[key_weight] /= alpha*randoms[key_weight]
                data[key_weight] /= randoms[key_weight]
                data[key_weight] /= np.mean(data[key_weight])
            data.attrs['total_w1'] = np.sum(data[key_weight])
            data.attrs['total_w2'] = np.sum(data[key_weight]**2)
            if self.use_inverse_weights:
                randoms[key_weight][:] = 1.
                data[key_weight] = data[key_weight] - randoms[key_weight]
            else:
                alpha = np.sum(data[key_weight])/np.sum(randoms[key_weight])
                data[key_weight] = data[key_weight] - alpha*randoms[key_weight]
            #print(data[key_weight].mean()
            #print('after',data[key_weight].mean())
            return data,randoms

        if density is not None:
            self.attrs.update(getattr(density,'attrs',{}))
            data1,randoms1 = get_data_randoms(density)
        if density2 is not None:
            data2,randoms2 = get_data_randoms(density2)
        super(AngularHP2PCF,self).__init__(data=data1,randoms=randoms1,data2=data2,randoms2=randoms2,root=root,**self.attrs)

    @property
    def use_inverse_weights(self):
        return self.attrs.get('use_inverse_weights',False)

    def run(self,**kwargs):
        from nbodykit.lab import SurveyDataPairCount
        self.attrs.update(kwargs)
        kwargs.update(self.set_keys())
        self.edges = np.array(self.edges)
        self.edges[0] = max(self.edges[0],1e-12) #to avoid self-pairs
        if self.R1R2 is None:
            self.R1R2 = SurveyDataPairCount('angular',self.randoms1,self.edges,second=self.randoms2,**kwargs)
        self.D1D2 = SurveyDataPairCount('angular',self.data1,self.edges,second=self.data2,**kwargs)
        if self.D1D2.attrs['is_cross']:
            self.D1D2.attrs['total_wnpairs'] = 0.5*self.data1.attrs['total_w1']*self.data1.attrs['total_w2']
        else:
            self.D1D2.attrs['total_wnpairs'] = 0.5*(self.data1.attrs['total_w1']**2-self.data1.attrs['total_w2'])
        self.set_corr()

    def set_corr(self):
        self.set_sep()
        fDD = self.R1R2.attrs['total_wnpairs']/self.D1D2.attrs['total_wnpairs']
        nonzero = self.R1R2.pairs['npairs'] > 0
        self.corr = np.zeros(self.D1D2.pairs.shape)
        self.corr[:] = np.nan
        DD = (self.D1D2.pairs['wnpairs'])[nonzero]
        RR = (self.R1R2.pairs['wnpairs'])[nonzero]
        corr = fDD*DD/RR
        self.corr[nonzero] = corr[:]
        #print(DD,RR,fDD,corr)
        #print(corr[:],DD)

import healpy as hp

class AngularHPPS(object):

    logger = logging.getLogger('AngularHPPS')

    def __init__(self,density,density2=None,**attrs):

        self.attrs = attrs
        self.attrs.update(getattr(density,'attrs',{}))
        def get_map(dens):
            mask = dens.brickrandoms>0
            shotnoise = hp.nside2pixarea(self.nside,degrees=False)*(1./np.mean(dens.brickdata[mask]) + 1./np.mean(dens.brickrandoms[mask]))
            if self.use_inverse_weights:
                d = dens.brickdata/dens.brickrandoms
                d[mask] /= np.mean(d[mask])
                d[mask] -= 1.
                d[~mask] = 0.
            else:
                alpha = dens.brickdata[mask].sum()/dens.brickrandoms[mask].sum()
                d = dens.brickdata - alpha*dens.brickrandoms
                d[~mask] = 0.
                mask = alpha*dens.brickrandoms
            map = utils.fill_hpmap(dens.brickid,d,self.nside,fill=0)
            mask = utils.fill_hpmap(dens.brickid,mask,self.nside,fill=0)
            if dens.nest:
                map = hp.reorder(map,n2r=True)
                mask = hp.reorder(mask,n2r=True)
            return map,mask,shotnoise
        shotnoise = 0
        self.map1,mask1,shotnoise = get_map(density)
        self.map2 = None
        if density2 is not None:
            assert density2.nside == self.nside, 'Density map 2 nside = {:d}, different than density map 1 nside = {:d}'.format(self.nside,density2.nside)
            self.map2,mask2 = get_map(density2)[:2]
            norm = np.sum(mask1*mask2)*1./len(self.map1)
        else:
            norm = np.sum(mask1**2)*1./len(self.map1)
        self.attrs['norm'] = norm
        self.attrs['shotnoise'] = shotnoise
        self.logger.info('Norm is {:.4f}.'.format(norm))
        self.logger.info('Shotnoise is {:.4g}.'.format(shotnoise))

    @property
    def use_inverse_weights(self):
        return self.attrs.get('use_inverse_weights',False)

    def run(self,nthreads=None,**kwargs):
        self.attrs.update(kwargs)
        _nthreads = None
        if nthreads is not None:
            _nthreads = os.environ.get('OMP_NUM_THREADS',None)
            os.environ['OMP_NUM_THREADS'] = str(nthreads)
            self.logger.info('Using {:d} threads.'.format(nthreads))

        self.cells = hp.anafast(self.map1,map2=self.map2,alm=False,**kwargs)
        #self.power,alm = hp.anafast(self.map1,map2=self.map2,alm=True,**kwargs)
        #print(self.power[1],(np.abs(alm[1])**2+2*np.abs(alm[3072])**2)/3.)

        if _nthreads is not None:
            os.environ['OMP_NUM_THREADS'] = _nthreads

        self.set_power()

    def set_power(self):
        self.ells = np.arange(self.cells.size)
        pixwin = hp.pixwin(self.nside,pol=False,lmax=self.ellmax)
        #print(pixwin)
        #print(self.cells/self.norm,self.shotnoise)
        self.power = (self.cells/self.attrs['norm']-self.attrs['shotnoise'])/pixwin

    @property
    def ellmax(self):
        return self.ells.max()

    @property
    def nside(self):
        return self.attrs['nside']

    def rebin(self,factor=1):
        ns = len(self.ells)//factor
        self.power = utils.bin_ndarray(self.power,(ns,),weights=(2.*self.ells+1),operation=np.average)
        self.ells = utils.bin_ndarray(self.ells,(ns,),weights=(2.*self.ells+1),operation=np.average)

    def __call__(self,ell):
        return np.interp(ell,self.ells,self.power,left=0.,right=0.)

    def getstate(self):
        state = {}
        for key in ['ells','cells','attrs']:
            if hasattr(self,key): state[key] = getattr(self,key)
        return state

    def setstate(self,state):
        self.__dict__.update(state)
        self.set_power()

    def save(self,path):
        utils.mkdir(path)
        self.logger.info('Saving to: {}.'.format(path))
        np.save(path,self.getstate())

    @classmethod
    def load(cls,path):
        cls.logger.info('Loading: {}.'.format(path))
        self = object.__new__(cls)
        self.setstate(np.load(path,allow_pickle=True)[()])
        return self

    @utils.saveplot()
    def plot(self,ax,others=[],covariances=[],xlim=None,ylim=None,xscale='linear',yscale='linear',xlabel='\\ell',labels=None,markers=None,linestyles='-',colors=None):
        ntot = len(others)+1
        if colors is None: colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        if not isinstance(colors,list): colors = [colors]*ntot
        if not isinstance(labels,list): labels = [labels]*ntot
        if not isinstance(markers,list): markers = [markers]*ntot
        if not isinstance(linestyles,list): linestyles = [linestyles]*ntot
        covariances = covariances + [None]*ntot
        ylabel = 'C_{\\ell}'
        def y(x,y):
            return y
        if yscale == 'xlinear':
            yscale = 'linear'
            ylabel = xlabel + ylabel
            def y(x,y):
                return x*y
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        for i,inst in enumerate([self]+others):
            mask = np.ones_like(inst.ells,dtype=np.bool_)
            if yscale == 'log': mask &= inst.ells>0
            if xlim is not None: mask &= wrap_around((inst.ells>=xlim[0]) & (inst.ells<=xlim[-1]))
            if covariances[i] is not None:
                low = y(inst.ells,covariances[i].mean(inst.ells))-y(inst.ells,covariances[i].std(inst.ells))
                up = y(inst.ells,covariances[i].mean(inst.ells))+y(inst.ells,covariances[i].std(inst.ells))
                ax.fill_between(inst.ells[mask],low[mask],up[mask],facecolor=colors[i],alpha=0.2,linewidth=0)
            ax.plot(inst.ells[mask],y(inst.ells,inst.power)[mask],color=colors[i],label=labels[i],marker=markers[i],linestyle=linestyles[i])
        if xlim is not None: ax.set_xlim(*xlim)
        if ylim is not None: ax.set_ylim(*ylim)
        ax.set_xlabel('${}$'.format(xlabel))
        ax.set_ylabel('${}$'.format(ylabel))
        if not utils.allnone(labels): ax.legend()

    @classmethod
    def reader(cls,path):
        tmp = cls.load(path)
        return tmp.ells,tmp.power
