import numpy as np
from photometry import *
from paths import *

setup_logging()

nside = 1024
edges = np.linspace(0.,4.,161)

use_inverse_weights = False

def test_corr():
    comm = utils.get_mpi_comm()
    if comm.rank == 0:
        data = TargetSelection.load_objects(path_data,region=None)
        randoms = TargetSelection.load_objects(path_randoms,region=None)
        data['WEIGHT'] = data.ones()
        randoms['WEIGHT'] = randoms.ones()
    else:
        data = None
        randoms = None
    #data = Catalogue.mpi_scatter(data)
    #randoms = Catalogue.mpi_scatter(randoms)
    corr = Angular2PCF(data=data,randoms=randoms,root=0,edges=edges)
    corr.run(show_progress=True)
    corr.save(path_corr)

def test_corr_hp():
    comm = utils.get_mpi_comm()
    if comm.rank == 0:
        data = TargetSelection.load_objects(path_data,region=None)
        randoms = TargetSelection.load_objects(path_randoms,region=None)
        dens = HealpixDensity(nside=nside,nest=True,ref=randoms)
        dens.set_randoms(randoms=randoms)
        dens.set_properties()
        dens.set_data(data=data)
    else:
        dens = None
    #dens = comm.bcast(dens)
    corr = AngularHP2PCF(density=dens,root=0,edges=edges,use_inverse_weights=use_inverse_weights)
    corr.run(show_progress=True,output_thetaavg=False)
    corr.save(path_corr_hp)

def test_RR():
    comm = utils.get_mpi_comm()
    if comm.rank == 0:
        data = TargetSelection.load_objects(path_data,region=None)
        randoms = TargetSelection.load_objects(path_randoms,region=None)
        dens = HealpixDensity(nside=nside,nest=True,ref=randoms)
        dens.set_randoms(randoms=randoms)
        dens.set_properties()
        dens.set_data(data=data)
    else:
        dens = None
    #dens = comm.bcast(dens)
    corr = AngularHP2PCF(density=dens,root=0,edges=edges,path_R1R2=path_corr_hp,use_inverse_weights=use_inverse_weights)
    corr.run(show_progress=True)
    corr.save(path_corr_hp)

def test_power():
    data = TargetSelection.load_objects(path_data,region=None)
    randoms = TargetSelection.load_objects(path_randoms,region=None)
    dens = HealpixDensity(nside=nside,nest=True,ref=randoms)
    dens.set_randoms(randoms=randoms)
    dens.set_properties()
    dens.set_data(data=data)
    corr = AngularHPPS(density=dens,use_inverse_weights=use_inverse_weights)
    corr.run(nthreads=8)
    corr.save(path_power)

def plot_corr():
    corr = Angular2PCF.load(path_corr)
    others = []
    others += [AngularHP2PCF.load(path_corr_hp)]
    corr.plot(path=dir_plot+'correlation.png',others=others,xscale='linear',yscale='linear',xlim=(0.02,None),ylim=(0.,0.02))

def plot_power():
    corr = AngularHPPS.load(path_power)
    corr.rebin(32)
    corr.plot(path=dir_plot+'power.png',xscale='log',yscale='log')


if __name__ == '__main__':
    #print(utils.nside2resol(1024))
    #test_corr()
    test_corr_hp()
    test_RR()
    plot_corr()
    #test_power()
    #plot_power()
