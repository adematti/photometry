__version__ = '0.1'
__author__ = 'Arnaud de Mattia'
__all__ = ['utils','setup_logging']
__all__ += ['Catalogue','TargetSelection','MCTool','Properties','TargetDensity','BrickDensity','HealpixDensity','BinnedDensity','BaseModel','LinearModel']
__all__ += ['Angular2PCF','AngularHP2PCF','AngularHPPS']

from .utils import *
from .catalogue import Catalogue
from .target_selection import TargetSelection
from .MCtool import MCTool
from .density_variations import Properties,TargetDensity,BrickDensity,HealpixDensity,BinnedDensity
from .models import BaseModel,LinearModel
from .correlation_function import Angular2PCF,AngularHP2PCF,AngularHPPS
