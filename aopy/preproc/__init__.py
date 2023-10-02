# For legacy code, include these modules in the top-level analysis module
# since they existed in base before becoming sub-modules
from .base import *
from .neuropixel import *
from .quality import *
from .wrappers import *

# For new submodules, import just the namespace
from . import optitrack
from . import bmi3d