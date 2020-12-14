from paths import *
from photometry import *
"""
for region in ['N','S']:
    path = path_data(region)
    cat = Catalogue.load(path)
    cat = cat.downsample(factor=1./4.,rng=None,seed=42)
    cat.save(path.replace('.fits','_cut.fits'))

    path = path_randoms(region)
    cat = Catalogue.load(path)
    cat = cat.downsample(factor=1./4.,rng=None,seed=42)
    cat.save(path.replace('.fits','_cut.fits'))
"""
"""
for camera in ['decam','90prime','mosaic']:

    path = os.path.join(path_dir,'Obiwan','dr9','data','survey-ccds-{}-dr9.kd.fits'.format(camera))
    cat = Catalogue.load(path)
    cat = cat.downsample(factor=1./4.,rng=None,seed=42)
    cat.save(path.replace('.fits','_cut.fits'))

    path = os.path.join(path_dir,'Obiwan','dr9','data','ccds-annotated-{}-dr9.kd.fits'.format(camera))
    cat = Catalogue.load(path)
    cat = cat.downsample(factor=1./4.,rng=None,seed=42)
    cat.save(path.replace('.fits','_cut.fits'))
"""
"""
for run in ['north','south']:

    path = os.path.join(path_dir,'Obiwan','dr9','ebv1000',run,'merged','matched_legacypipe_input.fits')
    import fitsio
    columns = ['ra','dec','brickname','ebv','dchisq'] + ['flux_g','mw_transmission_g'] + ['type'] + ['rchisq_{}'.format(b) for b in ['g','r','z']]
    cat = fitsio.read(path,columns=columns)
    fitsio.write(path.replace('.fits','_cut.fits'),cat,clobber=True)
"""
for run in ['north','south']:

    path = os.path.join(path_dir,'Obiwan','dr9','ebv1000',run,'merged','matched_legacypipe_input.fits')
    import fitsio
    bands = ['g','r','z']
    columns = ['injected','brick_primary','maskbits','ra','dec','brickname','ebv'] + ['nobs_{}'.format(b) for b in bands] + ['flux_{}'.format(b) for b in bands] + ['mw_transmission_{}'.format(b) for b in bands]
    cat = fitsio.read(path,columns=columns)
    cat = cat[(~cat['injected']) & cat['brick_primary']]
    fitsio.write(path.replace('.fits','_cut.fits'),cat,clobber=True)
