import os
import numpy as np

from photometry import *
from paths import *


nside = 1024
edges = np.linspace(0.,4.,161)

use_inverse_weights = False

def test_power():

    data = TargetSelection.load_objects(path_data,region=None)
    randoms = TargetSelection.load_objects(path_randoms,region=None)
    dens = HealpixDensity(nside=nside,nest=True,ref=randoms)
    dens.set_randoms(randoms=randoms)
    dens.set_properties()
    dens.set_data(data=data)
    corr = HealpixAngularPower(density=dens,use_inverse_weights=use_inverse_weights)
    corr.run(nthreads=8)
    corr.save(path_power)


def plot_power():

    corr = HealpixAngularPower.load(path_power)
    corr.rebin(32)
    corr.plot(path=os.path.join(dir_plot, 'power.png'),xscale='log',yscale='log')


if __name__ == '__main__':

    setup_logging()
    test_power()
    plot_power()
