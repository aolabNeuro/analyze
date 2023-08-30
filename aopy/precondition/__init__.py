# For legacy code, include these modules in the top-level analysis module
# since they existed in base before becoming sub-modules
from .base import *
from .eye import *

# For new submodules, import just the namespace
