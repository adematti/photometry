import numpy as np
from photometry import *
from paths import *

setup_logging()

data = TargetSelection.load_objects(path_data,region=None)
randoms = TargetSelection.load_objects(path_randoms,region=None)

def test_simple():
    dens = HealpixDensity(ref=randoms,nside=256)
    dens.set_randoms(randoms=randoms)
    dens.set_properties()
    dens.set_data(data=data)
    dens.brickrandoms[:] = 1.
    dens.brickdata = dens.brickrandoms*dens.properties['EBV']
    #dens.brickdata *= dens.brickrandoms.sum()/dens.brickdata.sum()
    m = LinearModel(density=dens)
    m.fit(props=['EBV'])
    weights = m.predict()
    assert np.allclose(dens.properties['EBV']/np.mean(dens.properties['EBV']),weights)

test_simple()
