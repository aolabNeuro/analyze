# data.py
# code for accessing neural data collected by the aoLab
import numpy as np
from .whitematter import ChunkedStream, Dataset
import h5py
import csv
import pandas as pd
import os

def load_optitrack_metadata(data_dir, filename, metadata_row=0, mocap_data_col_idx=np.arange(2,9)):
    '''
    This function loads optitrack metadata from .csv file that has 1 rigid body
    exported with the following settings:
        - Markers: Off
        - Unlabeled markers: Off
        - Quality Statistics: Off
        - Rigid Bodies: On
        - Rigid Body Markers: Off
        - Bones: Off
        - Bone Markers: Off
        - Header Information: On
        - Optitrack format Version(s): 1.23

    Required packages: csv, pandas

    Inputs:
        data_dir [string]: Directory to load data from
        filename [string]: File name to load within the data directory

    Outputs
        mocap_metadata [dict]: Dictionary of metadata for for an optitrack datafile
    '''

    # Constants for indexing into mocap data
    mocap_data_column_idx = 2 # Column index where data begins
    rigid_body_name_idx_csvrow = 3 #csv row that contains the rigid body name
    column_type_idx_csvrow = 5 #csv row that contains the column type 
    column_names_idx_csvrow = 6  #csv row that contains the column names

    # Initialize empty dict to add to
    mocap_metadata = {}

    filepath = os.path.join(data_dir, filename)
    # Create dict with .csv meta data
    with open(filepath) as csvfile:
        mocap_reader = csv.reader(csvfile, delimiter= ',')  # csv.reader object for optitrack data loading

        for idx_csvrow, row in enumerate(mocap_reader):
            # Create a dictionary of metadata for the first line
            if idx_csvrow == metadata_row:
                # For every pair of values in the list add to dict
                for idx_dictval in range(0, len(row), 2):
                    mocap_metadata[row[idx_dictval]]= row[idx_dictval + 1]

            # Get Rigid body name for each column
            elif idx_csvrow == rigid_body_name_idx_csvrow:
                mocap_metadata['Rigid Body Name'] = row[mocap_data_column_idx:][0]

            # Load whether column is rotation/position
            elif idx_csvrow == column_type_idx_csvrow:
                mocap_metadata['Data Column Motion Types'] = row[mocap_data_column_idx:]

            # Get column names for data
            elif idx_csvrow == column_names_idx_csvrow:
                mocap_metadata['Data Column Motion Names'] = row[mocap_data_column_idx:]  

    '''
    Check that data was loaded correctly.
    '''
    # Check that capture frame rate is the same as export frame rate
    assert mocap_metadata['Capture Frame Rate'] == mocap_metadata['Export Frame Rate'], 'Export and capture frame rate should be equal'
    mocap_metadata['samplerate'] = float(mocap_metadata['Capture Frame Rate'])
    # del mocap_metadata['Capture Frame Rate']
    # del mocap_metadata['Export Frame Rate']

    # Check that rotational coordinates in are in quaternion 
    assert mocap_metadata['Rotation Type'] == 'Quaternion', 'Rotation type must be Quaternion'

    return mocap_metadata

def load_optitrack_data(data_dir, filename):
    '''
    This function loads a series of x, y, z positional data from the optitrack
    .csv file that has 1 rigid body exported with the following settings:
        - Markers: Off
        - Unlabeled markers: Off
        - Quality Statistics: Off
        - Rigid Bodies: On
        - Rigid Body Markers: Off
        - Bones: Off
        - Bone Markers: Off
        - Header Information: On
        - Optitrack format Version(s): 1.23

    Required packages: pandas, numpy

    Inputs:
        data_dir [string]: Directory to load data from
        filename [string]: File name to load within the data directory

    Outputs
        mocap_data_pos [nt, 3]: Positional mocap data
        mocap_data_rot [nt, 4]: Rotational mocap data
    '''

    column_names_idx_csvrow = 5 # Header row index
    mocap_data_rot_column_idx = range(2,6) # Column index for rotation data
    mocap_data_pos_column_idx = range(6,9) # Column indices for position data
    filepath = os.path.join(data_dir, filename)
    # Load .csv file as a pandas data frame, convert to a numpy array, and remove
    # the 'Frame' and 'Time (Seconds)' columns.
    mocap_data_rot = pd.read_csv(filepath, header=column_names_idx_csvrow).to_numpy()[:,mocap_data_rot_column_idx]
    mocap_data_pos = pd.read_csv(filepath, header=column_names_idx_csvrow).to_numpy()[:,mocap_data_pos_column_idx]

    return mocap_data_pos, mocap_data_rot

def load_ecube_metadata(data_dir, data_source):
    '''
    Sums the number of channels and samples across all files in the data_dir

    Inputs: 
        data_dir [str]: eCube data directory
        source [str]: selects the source (AnalogPanel, Headstages, etc.)

    Output:
        metadata [dict]: Dictionary of metadata with fields
        - samplerate [float]: sampling rate of data for this source
        - data_source [str]: copied from the function argument
        - n_channels [int]: number of channels
        - n_samples [int]: number of samples for one channel
    '''

    # For now just load the metadata provieded by pyECubeSig
    # TODO: Really need the channel names and voltage per bit from the xml file
    n_channels = 0
    n_samples = 0
    dat = Dataset(data_dir)
    recordings = dat.listrecordings()
    for r in recordings: # r: (data_source, n_channels, n_samples)
        if data_source in r[0]:
            n_samples += r[2]  
            n_channels = r[1]
    samplerate = dat.samplerate
    metadata = dict(
        samplerate = samplerate,
        data_source = data_source,
        n_channels = n_channels,
        n_samples = n_samples,
    )
    return metadata

def load_ecube_data(data_dir, data_source, channels=None):
    '''
    Loads data from eCube for a given directory and datasource

    Requires load_ecube_metadata(), process_channels()

    Inputs:
        data_dir [str]: folder containing the data you want to load
        data_source [str]: type of data ("Headstage", "AnalogPanel", "DigitalPanel")
        channels [int array or None]: list of channel numbers (0-indexed) to load. If None, will load all channels by default

    Output:
        timeseries_data [nCh x nt]: all the data for the given source
    '''

    # Read metadata, check inputs
    metadata = load_ecube_metadata(data_dir, data_source)
    if channels is None:
        channels = range(metadata['n_channels'])
    elif len(channels) > metadata['n_channels']:
        raise ValueError("Supplied channel numbers are invalid")

    # Datatype is currently fixed for each data_source
    if data_source == 'DigitalPanel':
        dtype = np.uint64
    else:
        dtype = np.uint16

    # Fetch all the data for all the channels
    timeseries_data = process_channels(data_dir, data_source, channels, metadata['n_samples'], dtype=dtype)
    return timeseries_data

def proc_ecube_data(data_dir, data_source, hdf_filename):
    '''
    Loads and saves eCube data into an HDF file

    Requires load_ecube_metadata()

    Inputs:
        data_dir [str]: folder containing the data you want to load
        data_source [str]: type of data ("Headstage", "AnalogPanel", "DigitalPanel")
        hdf_filename [str]: name of hdf file to be written (or appended??)

    Output:
        None
    '''

    metadata = load_ecube_metadata(data_dir, data_source)
    n_samples = metadata['n_samples']
    n_channels = metadata['n_channels']
    if data_source == 'DigitalPanel':
        dtype = np.uint64
    else:
        dtype = np.uint16

    # Create an hdf dataset
    hdf = h5py.File(hdf_filename, 'a') # should append existing or write new?
    dset = hdf.create_dataset(data_source, (n_channels, n_samples), dtype=dtype)

    # Open and read the eCube data into the new hdf dataset
    process_channels(data_dir, data_source, range(n_channels), n_samples, data_out=dset)
    dat = Dataset(data_dir)
    dset.attrs['samplerate'] = dat.samplerate
    dset.attrs['data_source'] = data_source
    dset.attrs['channels'] = range(n_channels)

def get_eCube_data_sources(data_dir):
    '''
    Lists the available data sources in a given data directory

    Inputs: 
        data_dir [str]: eCube data directory

    Output:
        sources [str array]: available sources (AnalogPanel, Headstages, etc.)
    '''
    dat = Dataset(data_dir)
    return dat.listsources()

def process_channels(data_dir, data_source, channels, n_samples, dtype=None, data_out=None):
    '''
    Reads data from an ecube data source by channel until the number of samples requested 
    has been loaded. If a processing function is supplied, it will be applied to 
    each batch of data. If not, the data will be appended 

    Inputs:
        data_dir [str]: folder containing the data you want to load
        data_source [str]: type of data ("Headstage", "AnalogPanel", "DigitalPanel")
        channels [int array]: list of channels to process
        n_samples [int]: number of samples to read. Must be geq than a single chunk
        dtype [numpy dtype]: format for data_out if none supplied
        data_out [nCh, nt]: array of data to be written to. If None, it will be created

    Output:
        data_out [nchannels, n_samples]: Requested samples for requested channels
    '''
    if data_out == None:
        data_out = np.zeros((len(channels), n_samples), dtype=dtype)

    dat = Dataset(data_dir)
    dat.selectsource(data_source)
    chunk = dat.emitchunk(startat=0, debug=True)
    datastream = ChunkedStream(chunkemitter=chunk)

    idx_samples = 0 # keeps track of the number of samples already read/written
    while idx_samples < n_samples:
        try:
            data_chunk = next(datastream)
            data_len = np.shape(data_chunk)[1]
            data_out[:,idx_samples:idx_samples+data_len] = data_chunk[channels,:] # this might be where you filter data
            idx_samples += data_len
        except StopIteration:
            break
    return data_out