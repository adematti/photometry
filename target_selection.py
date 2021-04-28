import os
import logging
import glob
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import fitsio
from .catalogue import Catalogue
from .utils import utils, Binning

class TargetSelection(Catalogue):

        logger = logging.getLogger('TargetSelection')
        # http://legacysurvey.org/dr8/catalogs/#galactic-extinction-coefficients
        EXT_COEFFS = {'G':3.214,'R':2.165,'Z':1.211,'W1':0.184,'W2':0.113,'W3':0.0241,'W4':0.00910}

        def __init__(self,columns={},fields=None,radecbox=[0., 360., -90., 90.],tracer='ELG',region='N',case_sensitive=True,**attrs):
            super(TargetSelection,self).__init__(columns=columns,fields=fields,attrs=attrs)
            self.tracer = tracer
            self.radecbox = radecbox
            if self.tracer == 'ELG':
                self.bands = ['G','R','Z']
            self.region = region
            self.case_sensitive = case_sensitive

        """
        @mask.setter
        def mask(self,x):
            self['mask'] = x

        @property
        def mask(self):
            return self['mask']
        """
        @property
        def south(self):
            return self.region == 'S'

        @classmethod
        def load_targets(cls,path_dir,quick=False,downsample=None,keep=None,**kwargs):
            from desitarget import targetmask
            from desitarget.io import read_targets_in_box
            self = cls(**kwargs)
            self.columns = Catalogue.from_array(read_targets_in_box(path_dir,radecbox=self.radecbox,columns=keep,quick=quick,downsample=downsample)).columns
            if 'SV1_DESI_TARGET' in self:
                self.logger.info('Selecting {} in SV1.'.format(self.tracer))
                from desitarget.sv1 import sv1_targetmask
                mask = (self['DESI_TARGET_SV1'] & sv1_targetmask.desi_mask[self.tracer] > 0)
            else:
                self.logger.info('Selecting {} in nominal.'.format(self.tracer))
                mask = (self['DESI_TARGET'] & targetmask.desi_mask[self.tracer] > 0)
            if self.region: mask &= self.mask_region()
            self.logger.info('Selecting {:d}/{:d} targets.'.format(mask.sum(),mask.size))
            return self[mask]

        @classmethod
        def load_objects(cls,path_objects,downsample=None,**kwargs):
            self = cls(**kwargs)
            if isinstance(path_objects,list):
                for path in path_objects: self += cls.load_objects(path,**kwargs)
            else:
                path_objects =  glob.glob(path_objects)
                if len(path_objects) == 1:
                    rows = None
                    self.columns = Catalogue.load_fits(path_objects[0],rows=rows).columns
                    if downsample is not None:
                        # same as in https://github.com/desihub/desitarget/blob/master/py/desitarget/io.py
                        np.random.seed(616)
                        rows = np.random.choice(self.size,self.size//downsample,replace=False)
                        self = self[rows]
                    mask = self.trues()
                    if self.radecbox: mask = self.mask_in_box(*self.radecbox)
                    if self.region: mask &= self.mask_region()
                    self.logger.info('Selecting {:d}/{:d} targets.'.format(mask.sum(),mask.size))
                    self = self[mask]
                else:
                    self = cls.load_objects(path_objects,**kwargs)
            return self

        def flux_from_mag(self,b):
            return utils.mag_to_flux(self[b])

        def mag_from_flux(self,b,key_flux='FLUX'):
            return utils.flux_to_mag(self['{}_{}'.format(key_flux,b)])

        def estimated_transmission(self,b,ebvfac=1.,Rv=3.1,key_ebv='EBV'):
            coeffs = {b:self.EXT_COEFFS[b]*ebvfac for b in self.EXT_COEFFS}
            if Rv is not None:
                if Rv < 3.1:
                    #linear interpolation from Schlafly 2011 table
                    coeffs['G'] += (3.739-3.273)*(3.1-Rv)*ebvfac
                    coeffs['R'] += (2.113-2.176)*(3.1-Rv)*ebvfac
                    coeffs['Z'] += (1.175-1.217)*(3.1-Rv)*ebvfac
                    coeffs['W1'] += (-.1)*(Rv-3.1)*ebvfac
                else:
                    #linear interpolation from Schlafly 2011 table
                    coeffs['G'] += (3.006-3.273)*(Rv-3.1)*ebvfac
                    coeffs['R'] += (2.205-2.176)*(Rv-3.1)*ebvfac
                    coeffs['Z'] += (1.236-1.217)*(Rv-3.1)*ebvfac
                    coeffs['W1'] += (-.05)*(Rv-3.1)*ebvfac
            return 10**(-0.4*coeffs[b]*self[key_ebv])

        def estimated_flux(self,b,key_transmission='EMW_TRANSMISSION',key_flux='FLUX'):
            return self['{}_{}'.format(key_flux,b)]/self['{}_{}'.format(key_transmission,b)]

        def set_ebv(self,key='EBV',key_ra='RA',key_dec='DEC'):
            from desiutil import dust
            self[key] = dust.ebv(self[key_ra],self[key_dec])

        def set_flux_from_mag(self,key='FLUX'):
            for b in self.bands:
                self['{}_{}'.format(key,b)] = self.flux_from_mag(b)

        def set_mag_from_flux(self,**kwargs):
            for b in self.bands:
                self[b] = self.mag_from_flux(b,**kwargs)

        def set_estimated_transmission(self,key='EMW_TRANSMISSION',**kwargs):
            for b in self.bands:
                self['{}_{}'.format(key,b)] = self.estimated_transmission(b,**kwargs)

        def set_estimated_flux(self,key='EFLUX',**kwargs):
            for b in self.bands:
                self['{}_{}'.format(key,b)] = self.estimated_flux(b,**kwargs)

        def __getitem__(self,name):
            try:
                return super(TargetSelection,self).__getitem__(name)
            except KeyError:
                if self.case_sensitive:
                    raise KeyError('There is no field {} in the data. You may try case_sensitive = False.'.format(name))
                try:
                    return super(TargetSelection,self).__getitem__(name.upper())
                except KeyError:
                    return super(TargetSelection,self).__getitem__(name.lower())

        def __getattribute__(self,name):
            if name.startswith('apply_'):
                fun = object.__getattribute__(self,name.replace('apply_','mask_'))
                def wrapper(*args,**kwargs):
                    mask = fun(*args,**kwargs)
                    return self[mask]
                return wrapper
            return super(TargetSelection,self).__getattribute__(name)

        def mask_maskbit(self,key_nobs='NOBS',key_maskbits='MASKBITS',nobs=True,bits=None):
            mask = self.trues()
            if nobs:
                for b in self.bands: mask &= (self['{}_{}'.format(key_nobs,b)]>0)
            if bits is None:
                from desitarget.geomask import get_imaging_maskbits,get_imaging_maskbits
                bits = get_imaging_maskbits(get_imaging_maskbits())
            for bit in bits:
                mask &= (self[key_maskbits] & 2**bit) == 0
            return mask

        def mask_region(self):
            return self['PHOTSYS'].astype(type(self.region)) == self.region

        def mask_ts(self,star_gflux=0,key_flux='EFLUX',region=None):
            if region is not None: self.region = region
            from desitarget import cuts
            if self.tracer == 'ELG':
                return cuts.isELG_colors(gflux=self['{}_G'.format(key_flux)],rflux=self['{}_R'.format(key_flux)],zflux=self['{}_Z'.format(key_flux)],south=self.south)
            if self.tracer == 'STAR':
                mask = self['TYPE'] == 'PSF '
                mask &= self['{}_G'.format(key_flux)] > star_gflux
            return mask

        def mask_morphtype(self,morphtype):
            return self['MORPHTYPE'].astype(type(morphtype)) == morphtype

        @utils.saveplot()
        def plot_scatter(self,ax,prop1=None,prop2=None,propc=None,s=.2,vmin=None,vmax=None,color=None,xedges={},yedges={},title=None,clabel=None,**kwargs):
            if propc is not None: c = self[propc]
            else: c = color
            sc = ax.scatter(self[prop1],self[prop2],c=c,s=s,vmin=vmin,vmax=vmax,**kwargs)
            if propc is not None:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right',size='5%',pad=0.05)
                cbar = plt.colorbar(sc,cax=cax)
                if clabel: cbar.set_label(prop if not isinstance(clabel,str) else clabel,rotation=90)
            ax.set_xlabel(prop1)
            ax.set_ylabel(prop2)
            if xedges is not None:
                xlim = Binning(samples=self[prop1],**xedges).range
                ax.set_xlim(xlim)
            if yedges is not None:
                ylim = Binning(samples=self[prop2],**yedges).range
                ax.set_ylim(ylim)
            ax.set_title(title)

        @utils.saveplot()
        def plot_histo(self,ax,prop=None,xedges={},title=None,**kwargs):
            edges = Binning(samples=self[prop],**xedges).edges
            ax.hist(self[prop],bins=edges,**kwargs)
            ax.set_xlabel(prop)
            ax.set_title(title)
