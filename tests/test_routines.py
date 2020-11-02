import logging
import argparse
import numpy as np
from matplotlib import pyplot as plt
from photometry import *
from paths import *

setup_logging()

logger = logging.getLogger('PlotCheck')

try:
    parser = argparse.ArgumentParser(description='Plot checks')
    parser.add_argument('--save',type=int,help='save type',default=0,required=False)
    parser.add_argument('--plot',type=int,help='plot type',default=0,required=False)
    parser.add_argument('--region',type=str,help='region',default='N',required=False)
    args = parser.parse_args()
    type_save = args.save
    type_plot = args.plot
    region = args.region
except:
    logger.warning('Python mode.')


if type_save == 1:

    comm = utils.get_mpi_comm()
    if comm.rank == 0:
        truth = TargetSelection.load_objects(path_truth,tracer='ELG',region=None,case_sensitive=False)
        truth.set_flux_from_mag(key='FLUX')
        truth['REDSHIFT'] = truth['hsc_mizuki_photoz_best']
        truth.region = 'S'
        mask = np.all([~np.isnan(truth[b]) for b in truth.bands],axis=0)
        truth = truth[mask]
        map = Catalogue.load(path_hp)
        randoms = TargetSelection.load_objects(path_randoms,region=region)
        dens = HealpixDensity(map=map)
        dens.set_randoms(randoms=randoms)
        map = map[dens.brickid]
        mask = np.all([map['PSFDEPTH_{}'.format(b)]>0. for b in truth.bands],axis=0)
    else:
        truth = None
        map = None
        mask = None
    truth = Catalogue.mpi_scatter(truth,root=0)
    truth.mpi_gather(root=None)
    map = Catalogue.mpi_scatter(map,root=0,mask=mask)
    mc = MCTool(truth=truth,seed=42)
    mc.set_sel_params(ebvfac=1,Rv=None,sn_band_min=6,sn_flat_min=None,sn_red_min=None)
    mc.predict(map,key_depth='PSFDEPTH',key_efficiency='MCEFF',key_redshift='Z')
    map.mpi_gather(root=0)
    if comm.rank == 0: map.save(path_mctool)

def get_weighted_density(data,randoms,weighted=False,divide=True,return_weights=False):
    props = ['EBV'] + ['GALDEPTH_{}'.format(b) for b in data.bands] + ['STARDENS'] +  ['PSFSIZE_{}'.format(b) for b in data.bands]
    weights = None
    if weighted == 'lin' or not weighted:
        map = Catalogue.load(path_hp)
        dens = HealpixDensity(map=map)
        dens.set_randoms(randoms=randoms)
        dens.set_properties()
        dens.set_data(data=data)
    if weighted == 'lin':
        m = LinearModel(density=dens)
        m.fit(props=props)
        weights = m.predict()
    if weighted == 'mc':
        map = Catalogue.load(path_mctool)
        dens = HealpixDensity(map=map)
        dens.set_randoms(randoms=randoms)
        dens.set_properties()
        dens.set_data(data=data)
        weights = dens.properties['MCEFF']
    if weighted == 'mclin':
        map = Catalogue.load(path_mctool)
        dens = HealpixDensity(map=map)
        dens.set_randoms(randoms=randoms)
        dens.set_properties()
        dens.set_data(data=data)
        m = LinearModel(density=dens/dens.properties['MCEFF'])
        m.fit(props=props)
        weights = m.predict()*dens.properties['MCEFF']
    if weighted and divide:
        dens /= weights
    if return_weights: return dens,weights
    return dens

if type_plot == 1:

    toplot = ['lin','mclin']

    data = TargetSelection.load_objects(path_data,region=region)
    randoms = TargetSelection.load_objects(path_randoms,region=region)

    props = ['EBV'] + ['GALDEPTH_{}'.format(b) for b in data.bands] + ['STARDENS'] +  ['PSFSIZE_{}'.format(b) for b in data.bands]

    dens = get_weighted_density(data,randoms,weighted=False)
    others = []
    labels = None

    if 'lin' in toplot:
        others += [get_weighted_density(data,randoms,weighted='lin')]
        labels = ['raw','linear weights']

    if 'mc' in toplot:
        others += [get_weighted_density(data,randoms,weighted='mc')]
        if labels is not None:
            labels += ['MC weights']
        else:
            labels = ['raw','MC weights']

    if 'mclin' in toplot:
        others += [get_weighted_density(data,randoms,weighted='mclin')]
        if labels is not None:
            labels += ['MC+lin weights']
        else:
            labels = ['raw','MC+lin weights']

    fig,lax = plt.subplots(ncols=4,nrows=2,sharex=True,sharey=True,figsize=(16,6))
    fig.subplots_adjust(hspace=0.4,wspace=0.4)
    lax = lax.flatten()
    for iax,prop in enumerate(props):
        dens.plot_property_map(ax=lax[iax],prop=prop,s=0.1,title=prop,clabel=False)
    utils.savefig(path=dir_plot+'healpix_properties_{}GC.png'.format(region))

    fig,lax = plt.subplots(ncols=4,nrows=2,sharex=False,sharey=True,figsize=(16,6))
    fig.subplots_adjust(hspace=0.4,wspace=0.2)
    lax = lax.flatten()
    for iax,prop in enumerate(props):
        dens.plot_density_variations(ax=lax[iax],others=others,prop=prop,histos=[dens],var_kwargs={'labels':labels if iax==0 else None},leg_kwargs={},xedges={'quantiles':[0.01,0.99]})
    utils.savefig(path=dir_plot+'healpix_density_variations_{}GC.png'.format(region))

    fig,lax = plt.subplots(ncols=len([dens] + others),nrows=1,sharex=True,sharey=True,figsize=(12,4))
    fig.subplots_adjust(hspace=0.4,wspace=0.4)
    lax = lax.flatten()
    for iax,dens in enumerate([dens] + others):
        dens.plot_density_map(ax=lax[iax],s=1,title=labels[iax],vmin=0,vmax=2)
    utils.savefig(path=dir_plot+'healpix_density_map_{}GC.png'.format(region))
