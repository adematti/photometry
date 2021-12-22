import logging
import functools
import fitsio
import healpy as hp
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .catalogue import Catalogue
from .target_selection import TargetSelection
from .utils import utils, Binning, BinnedStatistic

class Properties(object):

    logger = logging.getLogger('Properties')

    def __init__(self,ids,catalogue,ids_catalogue=None,key_ids=None,weights=None):
        self.catalogue = catalogue
        self.ids = ids
        self.ids_catalogue = ids_catalogue if ids_catalogue is not None else self.catalogue[key_ids]
        self.statistics = TargetSelection()
        self.weights = weights
        if self.weights is None:
            self.logger.info('No weights provided, setting them to 1.')
            self.weights = self.statistics.ones()

    def deepcopy(self):
        new = object.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        new.statistics = self.statistics.deepcopy()
        new.weights = self.weights.copy()
        for key in ['catalogue','ids','ids_catalogue']:
            setattr(new,key,getattr(self,key))
        return new

    def __setitem__(self,name,item):
        self.statistics[name] = item

    def __contains__(self,key):
        return key in self.statistics

    @property
    def size(self):
        return self.statistics.size

    def __getitem__(self,name):
        if isinstance(name,str):
            if name not in self:
                self[name] = self._get_statistics(name)
            return self.statistics[name]
        new = self.deepcopy()
        new.ids = self.ids[name]
        new.statistics = self.statistics[name]
        new.weights = self.weights[name]
        return new

    def _get_statistics(self,name):
        #print(self.catalogue.fields)
        if name in self.catalogue:
            return self.get_statistics(name,statistic='mean')
        tmp = self.get_derived(name)
        if tmp is not None: return tmp
        for stat in ['mean_','var_','std_']:
            if name.startswith(stat):
                return self.get_statistics(name=name.replace(stat,''),statistic=stat.replace('_',''))
        raise ValueError('Field {} is unknown nor can be computed.'.format(name))

    def get_statistics(self,name=None,statistic='mean',ids=None,ids_catalogue=None,values_catalogue=None):
        if ids is None: ids = self.ids
        if ids_catalogue is None: ids_catalogue = self.ids_catalogue
        if values_catalogue is None: values_catalogue = self.catalogue[name]
        if statistic == 'std':
            statistic = lambda tab: np.std(tab,ddof=1)
        if statistic == 'var':
            statistic = lambda tab: np.var(tab,ddof=1)
        if statistic == 'median':
            statistic = lambda tab: np.median(tab)
        if statistic == 'frac':
            return utils.interp_digitized_statistics(ids,ids_catalogue,values=values_catalogue,statistic='sum')/utils.interp_digitized_statistics(ids,ids_catalogue,statistic='sum')
        self.logger.info('Calculating {} statistics.'.format(name))
        return utils.interp_digitized_statistics(ids,ids_catalogue,values=values_catalogue,statistic=statistic)

    def get_derived(self,name='SN2TOT_FLAT',bands=['G','R','Z']):
        if bands is None: bands = self.catalogue.bands
        if name == 'SN2TOT_FLAT':
            return np.sum([self.statistics.estimated_transmission(b=b)**2*self['PSFDEPTH_{}'.format(b)] for b in bands],axis=0)
        if name == 'SN2TOT_G':
            return np.sum([self.statistics.estimated_transmission(b=b)**2*self['PSFDEPTH_{}'.format(b)] for b in ['G']],axis=0)
        return None

class TargetDensity(object):

    logger = logging.getLogger('TargetDensity')

    def __init__(self,brickid=None,**attrs):
        self.brickid = brickid
        self.isbrickidprovided = self.brickid is not None
        self.attrs = attrs

    def get_bricks(self,catalogue):
        raise NotImplementedError('Must implement get_bricks().')

    def set_properties(self):
        raise NotImplementedError('Must implement set_properties().')

    def set_randoms(self,randoms,key_weight=None):
        brickid = self.get_bricks(randoms)
        #print(np.unique(brickid))
        weights = randoms[key_weight][brickid>=0] if key_weight else np.ones(np.sum(brickid>=0),dtype='f8')
        brickrandoms = np.bincount(brickid[brickid>=0],weights=weights,minlength=None)
        if self.isbrickidprovided:
            self.logger.info('Using provided brickid.')
            self.brickrandoms = utils.digitized_interp(self.brickid,np.arange(brickrandoms.size),brickrandoms,fill=0)
        else:
            self.logger.info('Inferring brickid from randoms.')
            mask = brickrandoms>0
            self.brickrandoms = brickrandoms[mask]
            self.brickid = np.flatnonzero(mask)
        self.logger.info('Found {:d} bricks with randoms.'.format(len(self.brickid)))

    def set_data(self,data,key_weight=None):
        self.brickdata = utils.interp_digitized_statistics(self.brickid,self.get_bricks(data),values=data[key_weight] if key_weight else None,statistic='sum')

    @property
    def brickdensity(self):
        return self.brickdata/self.brickrandoms

    @utils.saveplot()
    def plot_property_map(self,ax,prop='EBV',s=.2,vmin=None,vmax=None,xlim=None,ylim=None,title=None,clabel=None,**kwargs):
        sc = ax.scatter(self.properties['RA'],np.sin(self.properties['DEC']*np.pi/180),c=self.properties[prop],s=s,vmin=vmin,vmax=vmax,**kwargs)
        ax.set_xlabel('RA [deg]')
        ax.set_ylabel('sin(DEC)')
        if xlim is not None: ax.set_xlim(xlim)
        if ylim is not None: ax.set_ylim(ylim)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right',size='5%',pad=0.05)
        cbar = plt.colorbar(sc,cax=cax)
        if clabel: cbar.set_label(prop if not isinstance(clabel,str) else clabel,rotation=90)
        ax.set_title(title)

    @utils.saveplot()
    def plot_density_map(self,ax,s=.2,vmin=0.,vmax=2.,xlim=None,ylim=None,title=None,clabel=None,**kwargs):
        density = self.brickdensity*self.brickrandoms.sum()/self.brickdata.sum()
        sc = ax.scatter(self.properties['RA'],np.sin(self.properties['DEC']*np.pi/180),c=density,s=s,vmin=vmin,vmax=vmax,**kwargs)
        ax.set_xlabel('RA [deg]')
        ax.set_ylabel('sin(DEC)')
        if xlim is not None: ax.set_xlim(xlim)
        if ylim is not None: ax.set_ylim(ylim)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right',size='5%',pad=0.05)
        cbar = plt.colorbar(sc,cax=cax)
        if clabel: cbar.set_label('$n_{\\mathrm{gal}}/\\langle n_{\\mathrm{gal}} \\rangle - 1$' if not isinstance(clabel,str) else clabel,rotation=90)
        ax.set_title(title)

    @utils.saveplot()
    def plot_density_variations(self,ax,prop='EBV',xedges={},ylim=(-.3,.3),title=None,others=[],histos=[],var_kwargs={},histo_kwargs={},leg_kwargs={}):
        ntot = len(others)+1
        plterrors = var_kwargs.get('errors',True)
        labels = var_kwargs.get('labels',None)
        markers = var_kwargs.get('markers',None)
        linestyles = var_kwargs.get('linestyles',None)
        colors = var_kwargs.get('colors',plt.rcParams['axes.prop_cycle'].by_key()['color'])
        if not isinstance(plterrors,list): plterrors = [plterrors]*ntot
        if not isinstance(colors,list): colors = [colors]*ntot
        if not isinstance(labels,list): labels = [labels]*ntot
        if not isinstance(markers,list): markers = [markers]*ntot
        if not isinstance(linestyles,list): linestyles = [linestyles]*ntot
        histo_colors = histo_kwargs.get('colors',colors)
        histo_types = histo_kwargs.get('types','stepfilled')
        if not isinstance(histo_colors,list): histo_colors = [histo_colors]*len(histos)
        if not isinstance(histo_types,list): histo_types = [histo_types]*len(histos)
        binning = Binning(samples=self.properties[prop],weights=self.brickrandoms,**xedges)
        edges = binning.edges
        centers = binning.centers
        for ihisto,histo in enumerate(histos):
            #print(histo.properties.weights)
            p = np.histogram(histo.properties[prop],weights=histo.properties.weights,bins=edges)[0]
            p = max(ylim)/2.*p/p.max()
            ax.hist(centers,bins=edges,weights=p,color=histo_colors[ihisto],histtype=histo_types[ihisto],alpha=0.2)
        def bin(self):
            randoms = np.histogram(self.properties[prop],weights=self.brickrandoms,bins=edges)[0]
            data = np.histogram(self.properties[prop],weights=self.brickdata,bins=edges)[0]
            norm = randoms.sum()/data.sum()
            density = data/randoms*norm-1.
            errors = np.sqrt(data)/randoms*norm
            self.logger.info('Fraction of randoms not included in {} plot: {:.4f}.'.format(prop,1.-randoms.sum()/self.brickrandoms.sum()))
            return density,errors
        for iother,other in enumerate([self]+others):
            density,errors = bin(other)
            ax.errorbar(centers,density,errors*plterrors[iother],label=labels[iother],marker=markers[iother],color=colors[iother])
        ax.set_ylim(ylim)
        ax.grid(True)
        ax.set_xlabel(prop)
        ax.set_ylabel('$n_{\\mathrm{gal}}/\\langle n_{\\mathrm{gal}} \\rangle - 1$')
        if not utils.allnone(labels): ax.legend(**leg_kwargs)
        ax.set_title(title)

    def deepcopy(self):
        new = object.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        for key in ['brickdata','brickrandoms','brickid']:
            if hasattr(self,key):
                setattr(new,key,getattr(self,key).copy())
        for key in ['properties']:
            if hasattr(self,key):
                setattr(new,key,getattr(self,key).deepcopy())
        return new

    def __getitem__(self,name):
        new = self.deepcopy()
        for key in ['brickdata','brickrandoms','brickid','properties']:
            if hasattr(self,key): setattr(new,key,getattr(self,key)[name])
        return new

    def __mul__(self,other):
        new = self.deepcopy()
        new.brickdata *= other
        return new

    @property
    def size(self):
        return self.brickid.size

    def __truediv__(self,other):
        if not isinstance(other,self.__class__):
            new = self.deepcopy()
            #print(new.brickdata.dtype,new.brickrandoms.dtype)
            new.brickdata = new.brickdata / other
            #new.brickrandoms = new.brickrandoms*other
            return new
        index1,index2 = utils.overlap(self.brickid,other.brickid)
        if index1.size < self.size: self.logger.warning('Some numerator bricks cannot be found in denominator.')
        if index2.size < other.size: self.logger.info('Some denominator bricks cannot be found in numerator.')
        new = self[index1]; other = other[index2]
        new.brickdata = new.brickdata/other.brickdata
        new.brickrandoms = new.brickrandoms/other.brickrandoms
        return new

    def to_catalogue(self,key_data='data',key_randoms='randoms',props=[]):
        toret = {key_data:self.brickdata, key_randoms:self.brickrandoms}
        for prop in props: toret[prop] = self.properties[prop]
        return Catalogue(toret)


class BrickDensity(TargetDensity):

    logger = logging.getLogger('BrickDensity')

    def __init__(self,path=None,ref=None,**kwargs):
        if path is not None:
            self.bricks = Catalogue.load(path)
        else:
            self.bricks = None
            self.ref = ref
        super(BrickDensity,self).__init__(**kwargs)
        if not self.isbrickidprovided: self.brickid = self.get_bricks(self.bricks if self.bricks is not None else self.ref)

    def get_bricks(self,catalogue):
        return catalogue['BRICKID']

    def set_properties(self,weights='randoms'):
        if weights in ['data','randoms']: weights = getattr(self,'brick'+weights,None)
        self.properties = Properties(self.brickid,self.ref,ids_catalogue=self.get_bricks(self.ref),weights=weights)


class HealpixDensity(TargetDensity):

    logger = logging.getLogger('HealpixDensity')

    def __init__(self,map=None,ref=None,nside=None,nest=None,**kwargs):
        super(HealpixDensity,self).__init__(**kwargs)
        if map is not None:
            self.map = map
            self.attrs['nside'],self.attrs['nest'] = self.map.header['HPXNSIDE'],self.map.header['HPXNEST']
            self.logger.info('Found in header (nside,nest) = ({:d},{}).'.format(self.nside,self.nest))
            if nside is not None:
                assert self.nside == nside, 'nside = {:d} in input HEALPix does not match asked nside = {:d}'.format(self.nside,nside)
            if nest is not None:
                assert self.nest == nest, 'nest = {:d} in input HEALPix does not match asked nest = {:d}'.format(self.nest,nest)
        else:
            self.map = None
            self.attrs['nside'],self.attrs['nest'] = nside,nest
            self.ref = ref
            assert self.ref is not None, 'You must provide either healpix map (map) or property catalogue (ref)'
        npix = hp.nside2npix(self.nside)
        if not self.isbrickidprovided:
            self.brickid = np.arange(npix)
        else:
            assert (self.brickid>=0).all() and (self.brickid<npix).all(), 'Healpix id must be between 0 and {:d}'.format(npix)

    @property
    def nside(self):
        return self.attrs['nside']

    @property
    def nest(self):
        return self.attrs['nest']

    def get_bricks(self,catalogue):
        ra,dec = catalogue['RA'],catalogue['DEC']
        theta,phi = utils.radec_to_thphi(ra,dec)
        return hp.ang2pix(self.nside,theta,phi,nest=self.nest)

    def set_properties(self,weights='randoms'):
        if weights in ['data','randoms']: weights = getattr(self,'brick'+weights,None)
        if self.map is not None:
            self.properties = Properties(self.brickid,self.map,ids_catalogue=self.map['HPXPIXEL'],weights=weights)
        else:
            self.properties = Properties(self.brickid,self.ref,ids_catalogue=self.get_bricks(self.ref),weights=weights)
        theta,phi = hp.pix2ang(self.nside,self.properties.ids,nest=self.nest,lonlat=False)
        self.properties['RA'],self.properties['DEC'] = utils.thphi_to_radec(theta,phi)


class BinnedDensity(TargetDensity):

    logger = logging.getLogger('BinnedDensity')

    def __init__(self,ref=None,fields=[],**kwargs):
        self.fields = fields
        self.ref = ref
        self.binned_statistic = BinnedStatistic(samples=[self.ref[field] for field in self.fields],**kwargs)
        super(BinnedDensity,self).__init__(**kwargs)
        if not self.isbrickidprovided: self.brickid = self.binned_statistic.ibin_edges[-1]

    @property
    def edges(self):
        return self.binned_statistic.edges

    def get_bricks(self,catalogue):
        return self.binned_statistic.get_ibin([catalogue[field] for field in self.fields])

    def set_properties(self,weights='randoms'):
        if weights in ['data','randoms']: weights = getattr(self,'brick'+weights,None)
        self.properties = Properties(self.brickid,self.ref,ids_catalogue=self.get_bricks(self.ref),weights=weights)

    def deepcopy(self):
        new = super(BinnedDensity,self).deepcopy()
        for key in ['fields']:
            if hasattr(self,key):
                setattr(new,key,getattr(self,key).copy())
        for key in ['binned_statistic']:
            if hasattr(self,key):
                setattr(new,key,getattr(self,key).deepcopy())
        return new

    @utils.saveplot()
    def plot_density_variations(self,ax,prop='EBV',xedges={},**kwargs):
        if prop in self.fields and not xedges:
            xedges = {'edges':self.edges[self.fields.index(prop)]}
        return super(BinnedDensity,self).plot_density_variations(ax,prop=prop,xedges=xedges,**kwargs)
