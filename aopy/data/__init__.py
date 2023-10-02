# For legacy code, include these modules in the top-level analysis module
# since they existed in base before becoming sub-modules
from .base import *
from .bmi3d import *
from .eye import *
from .neuropixel import *

# For new submodules, import just the namespace
from . import optitrack
from . import peslab
# from . import db - don't import this by default.
