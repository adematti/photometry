import numpy as np
from photometry import *
from paths import *

setup_logging()

def get_truth():
    truth = TargetSelection.load_objects(path_truth,tracer='ELG',region=None,case_sensitive=False)
    truth.set_flux_from_mag(key='FLUX')
    truth['REDSHIFT'] = truth['hsc_mizuki_photoz_best']
    truth.region = 'S'
    mask = np.all([~np.isnan(truth[b]) for b in truth.bands],axis=0)
    m = (truth['REDSHIFT'] > -1) & (truth['REDSHIFT'] < 20)
    #print(truth['REDSHIFT'].min(),truth['REDSHIFT'].max())
    #print((~mask).sum(),(m & ~mask).sum())
    truth = truth[mask]
    return truth

def test_map():
    #print(mc.truth.mask.sum())
    comm = utils.get_mpi_comm()
    if comm.rank == 0:
        truth = get_truth()
        map = Catalogue.load(path_hp)
        mask = np.all([map['PSFDEPTH_{}'.format(b)]>0. for b in truth.bands],axis=0)
        mask[mask.cumsum()>500] = False
        map = map[mask]
        mask = mask[mask]
    else:
        truth = None
        map = None
        mask = None
    truth = Catalogue.mpi_scatter(truth,root=0)
    truth.mpi_gather(root=None)
    map = Catalogue.mpi_scatter(map,root=0,mask=mask)
    #hp.to_nbodykit()
    mc = MCTool(truth=truth,seed=42)
    mc.set_sel_params(sn_band_min=6,sn_flat_min=None,sn_red_min=None)
    mc.map(map,key_depth='PSFDEPTH',key_efficiency='MCEFF',key_redshift='Z',set_transmission=True)
    map.mpi_gather(root=0)
    #if comm.rank == 0: map.save(path_mctool)

def test_check():
    truth = get_truth()
    mc = MCTool(truth=truth,seed=42)
    mc.set_sim_params(flux_covariance=0.,flux_adbias=0.,flux_mulbias=1.)
    mc.sim['EBV'] = mc.sim.zeros()
    mc.sim.set_estimated_transmission(key='MW_TRANSMISSION')
    mc.sim.set_estimated_transmission(key='EMW_TRANSMISSION')
    #print(mc.sim['MW_TRANSMISSION_G'])
    mc.set_sel_params(sn_band_min=0,sn_flat_min=None,sn_red_min=None)
    mc()
    for b in mc.bands:
        assert np.all(mc.sim['EFLUX_{}'.format(b)] == mc.sim['FLUX_{}'.format(b)])
        assert np.all(mc.sim['EFLUX_{}'.format(b)] == mc.truth['FLUX_{}'.format(b)])
    print(mc.get_efficiency()) # round-off errors

def test_plot():
    truth = get_truth()
    mc = MCTool(truth=truth,seed=42)
    mc.set_sim_params(flux_covariance=1./np.array([1487.998, 515.4879, 197.86513]),flux_adbias=0.,flux_mulbias=1.)
    mc.sim['EBV'] = mc.sim.zeros()
    mc.sim.set_estimated_transmission(key='MW_TRANSMISSION')
    mc.sim.set_estimated_transmission(key='EMW_TRANSMISSION')
    #print(mc.sim['MW_TRANSMISSION_G'])
    mc.set_sel_params(sn_band_min=6,sn_flat_min=None,sn_red_min=None)
    mc()
    mc.plot_histo(path=dir_plot+'mctool.png')

test_map()
test_check()
test_plot()
