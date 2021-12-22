from photometry import *
from paths import *

setup_logging()

region = 'N'
radecbox = [170.,200.,35.,50.]

data = TargetSelection.load_targets(path_input_targets,tracer='ELG',region=region,radecbox=radecbox)
data.downsample(factor=0.1,seed=42).save(path_data)
randoms = TargetSelection.load_objects(path_input_randoms,tracer='ELG',region=region,radecbox=radecbox)
randoms = randoms.apply_maskbit()
randoms.save(path_randoms)

map = Catalogue.load(path_input_hp)
map.save(path_hp,keep=['HPXPIXEL', 'FRACAREA', 'STARDENS', 'EBV', 'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z',
                       'GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z', 'PSFDEPTH_W1', 'PSFDEPTH_W2', 'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z'])

import shutil
shutil.copyfile(path_input_truth,path_truth)
