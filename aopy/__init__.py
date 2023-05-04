# __init__.py
import socket
from . import analysis, data, postproc, precondition, preproc, visualization, tutorial_functions, utils

# Set the default memory limit
if socket.gethostname() == 'crab-eating':
    utils.set_memory_limit_gb(100)
