import numpy as np
from photometry import *
from paths import *

setup_logging()

data = TargetSelection.load_objects(path_data,region=None)
randoms = TargetSelection.load_objects(path_randoms,region=None)

def test_brick():
    dens = BrickDensity(ref=randoms)
    dens.set_randoms(randoms=randoms)
    dens.set_properties()
    dens.set_data(data=data)
    dens.plot_property_map(path=dir_plot+'brick_map_EBV.png',prop='EBV',title='EBV',s=5.)
    dens.plot_density_map(path=dir_plot+'brick_density_map.png',title='target density',vmin=0,vmax=2)
    dens.plot_density_variations(path=dir_plot+'brick_density_variations_EBV.png',prop='EBV',histos=[dens],xedges={'range':[None,0.04]})

def test_healpix():
    dens = HealpixDensity(ref=randoms,nside=256)
    dens.set_randoms(randoms=randoms)
    dens.set_properties()
    dens.set_data(data=data)
    dens.plot_property_map(path=dir_plot+'healpix_map_EBV.png',prop='EBV',title='EBV',s=5.)
    dens.plot_density_map(path=dir_plot+'healpix_density_map.png',title='target density',vmin=0,vmax=2)
    dens.plot_density_variations(path=dir_plot+'healpix_density_variations_EBV.png',prop='EBV',histos=[dens],xedges={'range':[None,0.04]})

def test_binned():
    dens = BinnedDensity(ref=randoms,fields=['EBV'],nbins=100,ranges=[None,0.04])
    dens.set_randoms(randoms=randoms)
    dens.set_properties()
    dens.set_data(data=data)
    dens.plot_density_variations(path=dir_plot+'binned_density_variations_EBV_all.png',prop='EBV',histos=[dens])
    dens.plot_density_variations(path=dir_plot+'binned_density_variations_EBV.png',prop='EBV',xedges={'nbins':21,'range':[None,0.03]},histos=[dens])


if __name__ == '__main__':

    test_brick()
    test_healpix()
    test_binned()
