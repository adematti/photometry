__version__ = '0.1'
__author__ = 'Arnaud de Mattia'

from .utils import setup_logging
from .catalogue import Catalogue
from .target_selection import TargetSelection
from .mc_tool import MCTool
from .density_variations import Properties, TargetDensity, BrickDensity, HealpixDensity, BinnedDensity
from .models import BaseModel, LinearModel
from .angular_power import HealpixAngularPower
