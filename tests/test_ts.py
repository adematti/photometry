import numpy as np
from photometry import *
from paths import *

setup_logging()

def test_mpi():
    comm = utils.get_mpi_comm()
    if comm.rank == 0:
        data = TargetSelection.load_objects(path_data,region=None)
        data['index'] = np.arange(data.size)
        mask = [0,1,3,10,12]
    else:
        data = None
        mask = None

    data = TargetSelection.mpi_scatter(data,mask=mask)
    print(comm.rank,data['index'])
    data['RA'][:] = 0.
    data.mpi_gather(root=None)
    print(comm.rank,data.size,data['RA'].sum())

def test_extinction():
    data = TargetSelection.load_objects(path_data,region=None)
    data.set_estimated_transmission(key='EMW_TRANSMISSION')
    for b in data.bands:
        assert np.allclose(data['MW_TRANSMISSION_{}'.format(b)],data['EMW_TRANSMISSION_{}'.format(b)])

def test_plot():
    data = TargetSelection.load_objects(path_data,region=None)
    data.plot_map(path=dir_plot+'map_data.png',prop1='RA',prop2='DEC',s=.2,title=None)
    data.set_estimated_flux(key='EFLUX',key_flux='FLUX',key_transmission='MW_TRANSMISSION')
    data.set_mag_from_flux(key_flux='EFLUX')
    data['G-R'] = data['G']-data['R']
    data['R-Z'] = data['R']-data['Z']
    data.plot_map(path=dir_plot+'color_data_ts.png',prop1='R-Z',prop2='G-R',s=2,title=None)
    data.plot_histo(path=dir_plot+'histo_EBV_data.png',prop='EBV',title=None)

def test_plot_hsc():
    truth = TargetSelection.load_objects(path_truth,tracer='ELG',region=None,case_sensitive=False)
    truth.set_flux_from_mag(key='FLUX')
    truth = truth[truth.mask_ts(key_flux='FLUX',region='S')]
    print(truth.fields)
    truth['G-R'] = truth['G']-truth['R']
    truth['R-Z'] = truth['R']-truth['Z']
    truth.plot_map(path=dir_plot+'color_hsc_ts.png',prop1='R-Z',prop2='G-R',propc='hsc_mizuki_photoz_best',xlim=[-1.,2.],ylim=[-0.5,2.],s=2,title=None)
    truth.plot_histo(path=dir_plot+'histo_redshift_hsc.png',prop='hsc_mizuki_photoz_best',xedges={'range':[0.,3.]},title=None)


test_mpi()
test_extinction()
test_plot()
test_plot_hsc()
