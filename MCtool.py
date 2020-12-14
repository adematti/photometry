import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
import fitsio
import healpy as hp
from .catalogue import Catalogue
from .target_selection import TargetSelection
from .utils import utils, Binning

class MCTool(object):

    logger = logging.getLogger('MCTool')

    def __init__(self,truth,rng=None,seed=None):
        self.truth = truth
        self.set_rng(rng=rng,seed=seed)
        self.sim = self.truth.deepcopy()
        self.truth.mask = self.truth.mask_ts(key_flux='FLUX')

    def set_rng(self,rng,seed=None):
        self.rng = rng
        if self.rng is None:
            self.rng = np.random.RandomState(seed=seed)

    def set_sim_params(self,flux_covariance=0,flux_adbias=0.,flux_mulbias=1.):
        self.flux_covariance = np.array(flux_covariance)
        if self.flux_covariance.size == 1:
            self.flux_covariance = np.ones(self.nbands)*self.flux_covariance
        if self.flux_covariance.ndim == 1:
            self.flux_covariance = np.diag(self.flux_covariance)
        #print(self.flux_covariance.shape)
        self.flux_adbias = np.array(flux_adbias)
        if self.flux_adbias.size == 1:
            self.flux_adbias = np.ones(self.nbands)*self.flux_adbias
        self.flux_mulbias = np.array(flux_mulbias)
        if self.flux_mulbias.size == 1:
            self.flux_mulbias = np.ones(self.nbands)*self.flux_mulbias

    def set_sel_params(self,sn_band_min=6,sn_flat_min=None,sn_red_min=None):
        self.sn_band_min = sn_band_min
        self.sn_flat_min = sn_flat_min
        self.sn_red_min = sn_red_min

    @property
    def size(self):
        return self.truth.size

    @property
    def bands(self):
        return self.truth.bands

    @property
    def nbands(self):
        return len(self.bands)

    def __call__(self):
        #print(self.truth.size,len(self.truth))
        flux_shifts = self.rng.multivariate_normal(mean=np.zeros(self.flux_covariance.shape[0],dtype='f8'),cov=self.flux_covariance,size=self.size).T
        #print(flux_shifts.shape)
        for ib,b in enumerate(self.bands):
            #print(flux_shifts[ib].shape,self.truth['FLUX_{}'.format(b)].shape,self.sim['MW_TRANSMISSION_{}'.format(b)].shape,self.flux_mulbias[ib].shape)
            self.sim['FLUX_{}'.format(b)] = self.truth['FLUX_{}'.format(b)]*self.sim['MW_TRANSMISSION_{}'.format(b)]*self.flux_mulbias[...,ib] + self.flux_adbias[...,ib] + flux_shifts[ib]
        #self.sim.set_estimated_transmission(ebvfac=self.ebvfac,Rv=self.Rv,key='EMW_TRANSMISSION')
        self.sim.set_estimated_flux(key='EFLUX',key_transmission='EMW_TRANSMISSION',key_flux='FLUX')
        #for b in self.bands:
        #    self.sim['EFLUX_{}'.format(b)] = self.truth['FLUX_{}'.format(b)]
        self.sim.mask = self.sim.mask_ts(key_flux='EFLUX')
        #tmp = self.truth.mask_ts(key_flux='FLUX')
        #print(np.sum(tmp!=self.sim.mask),self.truth.region,self.sim.region)
        self.mask_sn()

    def mask_sn(self):
        mask = self.sim.trues()
        for ib,b in enumerate(self.bands):
            mask |= self.sim['FLUX_{}'.format(b)] >= self.sn_band_min*self.flux_covariance[ib,ib]**0.5
        def combined_snr2(coeffs={b:1. for b in self.bands}):
            comb = np.sum([self.sim['FLUX_{}'.format(b)]/self.flux_covariance[ib,ib]/coeffs[b] for ib,b in enumerate(self.bands)],axis=0)
            weights = np.sum([1./self.flux_covariance[ib,ib]/coeffs[b] for ib,b in enumerate(self.bands)],axis=0)
            comb /= weights
            return comb**2 * weights
        if self.sn_flat_min is not None:
            mask |= combined_snr2(coeffs={b:1. for b in self.bands}) >= self.sn_flat_min**2
        if self.sn_red_min is not None:
            mask |= combined_snr2(coeffs={'G':2.5,'R':1.,'Z':0.4}) >= self.sn_red_min**2
        self.sim.mask &= mask
        return mask

    def get_efficiency(self):
        return self.sim.mask.sum()*1./self.truth.mask.sum()

    def copy(self):
        new = object.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        return new

    def map(self,catalogue,key_depth='PSFDEPTH',key_efficiency='MCEFF',key_redshift=None,set_transmission=False,ebvfac=1,Rv=None):
        mask = np.all([catalogue['{}_{}'.format(key_depth,b)]>0. for b in self.bands],axis=0)
        catalogue[key_efficiency] = catalogue.zeros()
        if key_redshift: catalogue[key_redshift] = -catalogue.ones()
        ntot = mask.sum()
        #print(ntot,catalogue.size)
        for ii,i in enumerate(np.flatnonzero(mask)):
            if ii % 1000 == 0:
                self.logger.info('{:d}/{:d} = {:.4f} objects treated.'.format(ii,ntot,ii/ntot))
            flux_covariance = [1/catalogue['{}_{}'.format(key_depth,b)][i] for b in self.bands]
            #print([catalogue['{}_{}'.format(key_depth,b)][i] for b in self.bands])
            self.set_sim_params(flux_covariance=flux_covariance,flux_adbias=0.,flux_mulbias=1.)
            if set_transmission:
                self.sim['EBV'] = catalogue['EBV'][i]
                self.sim.set_estimated_transmission(key='MW_TRANSMISSION')
                self.sim.set_estimated_transmission(ebvfac=ebvfac,Rv=Rv,key='EMW_TRANSMISSION')
            else:
                for b in self.sim.bands:
                    self.sim['MW_TRANSMISSION_{}'.format(b)] = catalogue['MW_TRANSMISSION_{}'.format(b)][i]
                    self.sim['EMW_TRANSMISSION_{}'.format(b)] = catalogue['EMW_TRANSMISSION_{}'.format(b)][i]
            self()
            catalogue[key_efficiency][i] = self.get_efficiency()
            #print(catalogue[key_efficiency][i])
            if key_redshift: catalogue[key_redshift][i] = np.mean(self.sim['REDSHIFT'][self.sim.mask])

    @utils.saveplot(giveax=False)
    def plot_histo(self,colors=None,xedges={}):
        nrows = 1
        if colors is None: colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        if not isinstance(colors,list): colors = [colors]*3
        toplot = ['FLUX_{}'.format(b) for b in self.bands] + (['REDSHIFT'] if 'REDSHIFT' in self.truth else [])
        ncols = len(toplot)
        fig, lax = plt.subplots(nrows,ncols,sharex=False,sharey=False,figsize=(4*ncols,4*nrows),squeeze=False)
        fig.subplots_adjust(hspace=0.1,wspace=0.2)
        lax = lax.flatten()
        edges = Binning(samples=self.truth[toplot[0]][self.truth.mask],**xedges).edges
        for iax,ax in enumerate(lax):
            hist = ax.hist(self.truth[toplot[iax]][self.truth.mask],bins=edges,label='true {}'.format(toplot[iax]),color=colors[0],histtype='stepfilled',alpha=0.2)
            hist = ax.hist(self.truth[toplot[iax]][self.sim.mask],bins=edges,label='selected {}'.format(toplot[iax]),color=colors[1],histtype='step')[0]
            if 'FLUX' in toplot[iax]:
                hist = ax.hist(self.sim['E'+toplot[iax]][self.sim.mask],bins=edges,label='selected E{}'.format(toplot[iax]),color=colors[2],histtype='step')[0]
            ax.set_xlabel(toplot[iax])
            if iax == 0: ax.legend()
