import os
import logging

import numpy as np
import healpy as hp
from matplotlib import pyplot as plt

from . import utils


class HealpixAngularPower(object):

    logger = logging.getLogger('HealpixAngularPower')

    def __init__(self,density, density2=None, **attrs):

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

    def run(self, nthreads=None, **kwargs):
        self.attrs.update(kwargs)
        _nthreads = None
        if nthreads is not None:
            _nthreads = os.environ.get('OMP_NUM_THREADS', None)
            os.environ['OMP_NUM_THREADS'] = str(nthreads)
            self.logger.info('Using {:d} threads.'.format(nthreads))

        self.cells = hp.anafast(self.map1, map2=self.map2, alm=False, **kwargs)
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

    def __getstate__(self):
        state = {}
        for key in ['ells','cells','attrs']:
            if hasattr(self,key): state[key] = getattr(self,key)
        return state

    def __setstate__(self,state):
        self.__dict__.update(state)
        self.set_power()

    def save(self,path):
        self.logger.info('Saving to: {}.'.format(path))
        utils.mkdir(os.path.dirname(path))
        np.save(path,self.__getstate__())

    @classmethod
    def load(cls,path):
        cls.logger.info('Loading: {}.'.format(path))
        self = object.__new__(cls)
        self.__setstate__(np.load(path,allow_pickle=True)[()])
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
