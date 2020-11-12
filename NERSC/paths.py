import os

path_dir = '/global/cscratch1/sd/adematti/'
path_input_truth = '/project/projectdirs/desi/users/ajross/MCdata/desi_mcsyst_truth.dr7.34ra38.-7dec-3.fits'

dr = 'dr9m'
path_input_hp = '/global/cfs/cdirs/desi/target/catalogs/dr9m/0.42.0/pixweight/main/resolve/dark/pixweight-dark.fits'
path_input_targets = '/global/cfs/cdirs/desi/target/catalogs/dr9m/0.42.0/targets/main/resolve/dark/'
path_input_randoms = '/global/cfs/cdirs/desi/target/catalogs/dr9m/0.42.0/randoms/resolve/randoms-randomized-1.fits'

# data randoms
dir_input = os.path.join(path_dir,'targets_ELG_{}/'.format(dr))
path_data = lambda region: dir_input + 'data_{}GC.fits'.format(region)
path_randoms = lambda region: dir_input + 'randoms_{}GC.fits'.format(region)
path_hp = path_input_hp
path_truth = path_input_truth

#mc tool
path_mctool = lambda region: os.path.join(path_dir,'mctool_ELG_{}/pixweight_HSC_{}GC.fits'.format(dr,region))

# correlations
path_mocks_data = lambda num: '/global/project/projectdirs/desi/users/shadaba/EZmock/FA_LSS/FA_EZmock_desi_ELG_v0_{:d}.fits'.format(num)
path_mocks_randoms = '/global/project/projectdirs/desi/users/shadaba/EZmock/FA_LSS/FA_EZmock_desi_ELG_v0_rand_00.fits'

dir_corr_data = os.path.join(path_dir,'targets_ELG_{}_angular_correlations/'.format(dr))
dir_corr_mocks = os.path.join(path_dir,'mocks_ELG_angular_correlations/')
def path_corr(estimator,region,mock=False,nside=None,weights=None,num=None):
    if mock: toret = dir_corr_mocks
    else: toret = dir_corr_data
    toret += '{}_{}GC'.format(estimator,region)
    if nside: toret += '_nside{:d}'.format(nside)
    if weights: toret += '_{}'.format(weights)
    if num is not None: toret += '_{}'.format(num)
    return toret + '.npy'

def path_covariance(estimator,*args,**kwargs):
    return path_corr('covariance_'+estimator,*args,**kwargs)

# plots
dir_plot = 'plots/'