# optitrack.py
#
# parser for optitrack data

import numpy as np

from .. import utils
from .. import data as aodata
from ..data import optitrack as ot

def parse_optitrack(data_dir, files):
    '''
    Parser for optitrack data

    Args:
        data_dir (str): where to look for the data
        files (dict): dictionary of files for this experiment
    
    Returns:
        tuple: tuple containing:
            | **data (dict):** optitrack data
            | **metadata (dict):** optitrack metadata
    '''
    # Check that there is optitrack data in files
    if not 'optitrack' in files:
        raise ValueError('Cannot parse nonexistent optitrack data!')

    # Load frame data
    optitrack_filename = files['optitrack']
    optitrack_metadata = ot.load_optitrack_metadata(data_dir, optitrack_filename)
    optitrack_pos, optitrack_rot = ot.load_optitrack_data(data_dir, optitrack_filename)

    # Load timing data from the ecube if present
    if 'ecube' in files:

        # Get the appropriate analog channel from bmi3d metadata
        try:
            _, bmi3d_event_metadata = aodata.load_bmi3d_hdf_table(data_dir, files['hdf'], 'sync_events')
            optitrack_strobe_channel = bmi3d_event_metadata['optitrack_sync_dch']
        except:
            optitrack_strobe_channel = 0

        # Load and parse the optitrack strobe signal
        digital_data, metadata = aodata.load_ecube_digital(data_dir, files['ecube'])
        samplerate = metadata['samplerate']
        optitrack_bit_mask = 1 << optitrack_strobe_channel
        optitrack_strobe = utils.extract_bits(digital_data, optitrack_bit_mask)
        optitrack_strobe_timestamps, _ = utils.detect_edges(optitrack_strobe, samplerate, rising=True, falling=False)
        # - check that eCube captured the same number of timestamps from esync as there are positions/rotations in the file
        if len(optitrack_pos) == len(optitrack_strobe_timestamps):
            optitrack_timestamps = optitrack_strobe_timestamps
            print("Optitrack strobes match exactly")
        # - otherwise assume they started at the same point, throw away or add zeros on the end if needed (throw a warning!)
        elif len(optitrack_pos) > len(optitrack_strobe_timestamps):
            n_extra = len(optitrack_pos) - len(optitrack_strobe_timestamps)
            print("{} too many optitrack positions recorded, truncating. Less than 50 is normal".format(n_extra))
            optitrack_pos = optitrack_pos[:len(optitrack_strobe_timestamps)]
            optitrack_rot = optitrack_rot[:len(optitrack_strobe_timestamps)]
            optitrack_timestamps = optitrack_strobe_timestamps
        # - optitrack has said they have issues getting the end of the recording to line up perfectly and to not worry about it :/
        else:
            n_extra = len(optitrack_strobe_timestamps) - len(optitrack_pos)
            print("{} too many optitrack strobe timestamps recorded, truncating. Less than 50 is normal".format(n_extra))
            optitrack_timestamps = optitrack_strobe_timestamps[:len(optitrack_pos)]
    
    # Otherwise just use the frame timing from optitrack
    else:
        print("Warning: using optitrack's internal timing")
        optitrack_timestamps = ot.load_optitrack_time(data_dir, optitrack_filename)

    # Organize everything into dictionaries
    optitrack = np.empty((len(optitrack_timestamps),), dtype=[('timestamp', 'f8'), ('position', 'f8', (3,)), ('rotation', 'f8', (4,))])
    optitrack['timestamp'] = optitrack_timestamps
    optitrack['position'] = optitrack_pos
    optitrack['rotation'] = optitrack_rot
    data_dict = {
        'data': optitrack,
    }
    optitrack_metadata.update({
        'source_dir': data_dir,
        'source_files': files,
    }) 
    # TODO: add metadata about where the timestamps came from
    return data_dict, optitrack_metadata

