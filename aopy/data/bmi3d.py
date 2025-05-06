import sys
import traceback
import warnings
import os
import json

from functools import lru_cache
import numpy as np
import h5py
import tables
import pandas as pd
from tqdm.auto import tqdm
if sys.version_info >= (3,9):
    from importlib.resources import files, as_file
else:
    from importlib_resources import files, as_file

from .. import precondition
from .. import preproc
from .. import postproc
from .. import utils
from ..preproc.base import get_data_segment, get_data_segments, get_trial_segments, get_trial_segments_and_times, interp_timestamps2timeseries, sample_timestamped_data, trial_align_data
from ..preproc.bmi3d import get_target_events, get_ref_dis_frequencies
from ..whitematter import ChunkedStream, Dataset
from ..utils import derivative, get_pulse_edge_times, compute_pulse_duty_cycles, convert_digital_to_channels, detect_edges
from . import base 

############
# Raw data #
############
def get_ecube_data_sources(data_dir):
    '''
    Lists the available data sources in a given data directory

    Args: 
        data_dir (str): eCube data directory

    Returns:
        str array: available sources (AnalogPanel, Headstages, etc.)
    '''
    dat = Dataset(data_dir)
    return dat.listsources()
    
def load_ecube_metadata(data_dir, data_source):
    '''
    Sums the number of channels and samples across all files in the data_dir

    Args: 
        data_dir (str): eCube data directory
        source (str): selects the source (AnalogPanel, Headstages, etc.)

    Returns:
        dict: Dictionary of metadata with fields:
            | **samplerate (float):** sampling rate of data for this source
            | **data_source (str):** copied from the function argument
            | **n_channels (int):** number of channels
            | **n_samples (int):** number of samples for one channel
    '''

    # For now just load the metadata provieded by pyECubeSig
    # TODO: Really need the channel names and voltage per bit from the xml file
    # For now I am hard coding these. Please change!
    HEADSTAGE_VOLTSPERBIT = 1.907348633e-7
    ANALOG_VOLTSPERBIT = 3.0517578125e-4
    if data_source == 'Headstages':
        voltsperbit = HEADSTAGE_VOLTSPERBIT
    elif data_source == 'AnalogPanel':
        voltsperbit = ANALOG_VOLTSPERBIT
    else:
        voltsperbit = None

    n_channels = 0
    n_samples = 0
    dat = Dataset(data_dir)
    recordings = dat.listrecordings()
    for r in recordings: # r: (data_source, n_channels, n_samples)
        if data_source in r[0]:
            n_samples += r[2]  
            n_channels = r[1]
    if n_channels == 0:
        raise Exception('No data found for data source: {}'.format(data_source))
    samplerate = dat.samplerate
    metadata = dict(
        samplerate = samplerate,
        data_source = data_source,
        n_channels = n_channels,
        n_samples = n_samples,
        voltsperbit = voltsperbit,
    )
    return metadata

def load_ecube_data(data_dir, data_source, channels=None):
    '''
    Loads data from eCube for a given directory and datasource

    Requires load_ecube_metadata(), process_channels()

    Args:
        data_dir (str): folder containing the data you want to load
        data_source (str): type of data ("Headstages", "AnalogPanel", "DigitalPanel")
        channels (int array or None): list of channel numbers (0-indexed) to load. If None, will load all channels by default

    Returns:
        (nt, nch): all the data for the given source
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
        dtype = np.int16

    # Fetch all the data for all the channels
    n_samples = metadata['n_samples']
    timeseries_data = np.zeros((n_samples, len(channels)), dtype=dtype)
    n_read = 0
    for chunk in _process_channels(data_dir, data_source, channels, metadata['n_samples'], dtype=dtype):
        chunk_len = min(chunk.shape[0], n_samples-n_read)
        timeseries_data[n_read:n_read+chunk_len,:] = chunk[:chunk_len,:] # Deal with potentially corrupted data
        n_read += chunk_len
    return timeseries_data

def load_ecube_data_chunked(data_dir, data_source, channels=None, chunksize=728):
    '''
    Loads a data file one "chunk" at a time. Useful for replaying files as if they were online data.

    Args:
        data_dir (str): folder containing the data you want to load
        data_source (str): type of data ("Headstages", "AnalogPanel", "DigitalPanel")
        channels (int array or None): list of channel numbers (0-indexed) to load. If None, will load all channels by default
        chunksize (int): how many samples to include in each chunk

    Yields:
        (chunksize, nch): one chunk of data for the given source
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
        dtype = np.int16
    
    # Fetch all the channels but just return the generator
    n_samples = metadata['n_samples']
    n_channels = metadata['n_channels']
    kwargs = dict(maxchunksize=chunksize*n_channels*np.dtype(dtype).itemsize)
    return _process_channels(data_dir, data_source, channels, n_samples, **kwargs)

def proc_ecube_data(data_path, data_source, result_filepath, result_name='broadband_data', max_memory_gb=1.):
    '''
    Loads and saves eCube data into an HDF file

    Requires load_ecube_metadata()

    Args:
        data_path (str): path to folder containing the ecube data you want to load
        data_source (str): type of data ("Headstages", "AnalogPanel", "DigitalPanel")
        result_filepath (str): path to hdf file to be written (or appended)
        max_memory_gb (float, optional): max memory used to load binary data at one time

    Returns:
        tuple: tuple containing:
            | **dset (h5py.Dataset):** the new hdf dataset
            | **metadata (dict):** the ecube metadata
    '''

    # Load the metadata to figure out the datatypes
    metadata = load_ecube_metadata(data_path, data_source)
    if data_source == 'DigitalPanel':
        dtype = np.uint64
    else:
        dtype = np.int16
    chunksize = int(max_memory_gb * 1e9 / np.dtype(dtype).itemsize / metadata['n_channels'])

    # Create an hdf dataset
    hdf = h5py.File(result_filepath, 'a')
    dset = hdf.create_dataset(result_name, (metadata['n_samples'], metadata['n_channels']), dtype=dtype)

    # Write broadband data directly into the hdf file
    n_samples = 0
    for broadband_chunk in load_ecube_data_chunked(data_path, 'Headstages', chunksize=chunksize):
        chunk_len = broadband_chunk.shape[0]
        dset[n_samples:n_samples+chunk_len,:] = broadband_chunk
        n_samples += chunk_len

    return dset, metadata

def filter_lfp_from_broadband(broadband_filepath, result_filepath, mean_subtract=True, dtype='int16', max_memory_gb=1., **filter_kwargs):
    '''
    Filters local field potential (LFP) data from a given broadband signal file into an hdf file.

    Args:
        broadband_filepath (str): Path to the input broadband signal file.
        result_filepath (str): Path to save the filtered LFP data.
        mean_subtract (bool, optional): Whether to subtract the mean from the filtered LFP signal.
                                        Default is True.
        dtype (str, optional): Data type for the filtered LFP signal. Default is 'int16'.
        max_memory_gb (float, optional): Maximum memory (in gigabytes) to use for filtering. Default is 1.0 GB.
        **filter_kwargs: Additional keyword arguments to customize the filtering process.
                        These arguments will be passed to the filtering function.

    Raises:
        IOError: If the input broadband file is not found.
        MemoryError: If the specified max_memory_gb is insufficient for the filtering process.

    Note:
        This function is used in the :func:`~aopy.preproc.warppers.proc_lfp` wrapper.
    '''
    lfp_samplerate = filter_kwargs.pop('lfp_samplerate', 1000)

    metadata = base.load_hdf_group('', broadband_filepath, 'broadband_metadata')
    samplerate = metadata['samplerate']
    n_channels = int(metadata['n_channels'])
    n_samples = int(metadata['n_samples'])
    downsample_factor = int(samplerate/lfp_samplerate)
    lfp_samples = int(np.ceil(n_samples/downsample_factor))
    if 'low_cut' not in filter_kwargs:
        filter_kwargs['low_cut'] = 500
    if 'buttord' not in filter_kwargs:
        filter_kwargs['buttord'] = 4

    # Create an hdf dataset
    lfp_hdf = h5py.File(result_filepath, 'a') # should append existing or write new?
    dset = lfp_hdf.create_dataset('lfp_data', (lfp_samples, n_channels), dtype=dtype)

    # Figure out how much data we can load at once
    max_samples = int(max_memory_gb * 1e9 / np.dtype(dtype).itemsize)
    channel_chunksize = max(min(n_channels, max_samples // n_samples), 1)
    time_chunksize = min(n_samples, max_samples // channel_chunksize)
    print(f"{channel_chunksize} channels and {time_chunksize} samples in each chunk")
    
    # Load the broadband dataset
    bb_hdf = h5py.File(broadband_filepath, 'r')
    if 'broadband_data' not in bb_hdf:
        raise ValueError(f'broadband_data not found in file {broadband_filepath}')
    bb_data = bb_hdf['broadband_data']
    
    # Filter broadband data into LFP directly into the hdf file
    n_bb_samples = 0
    n_lfp_samples = 0
    while n_bb_samples < n_samples:
        n_ch = 0
        while n_ch < n_channels:
            broadband_chunk = bb_data[n_bb_samples:n_bb_samples+time_chunksize, n_ch:n_ch+channel_chunksize]
            lfp_chunk, _ = precondition.filter_lfp(broadband_chunk, samplerate, **filter_kwargs)
            chunk_len = lfp_chunk.shape[0]
            dset[n_lfp_samples:n_lfp_samples+chunk_len,n_ch:n_ch+channel_chunksize] = lfp_chunk
            n_ch += channel_chunksize
        n_bb_samples += time_chunksize
        n_lfp_samples += chunk_len
        
    if mean_subtract:
        dset -= np.mean(dset, axis=0, dtype=dtype) # hopefully this isn't constrained by memory
    lfp_hdf.close()
    bb_hdf.close()

    # Append the lfp metadata to the file
    lfp_metadata = metadata
    lfp_metadata['lfp_samplerate'] = lfp_samplerate # for backwards compatibility
    lfp_metadata['samplerate'] = lfp_samplerate
    lfp_metadata['n_samples'] = lfp_samples
    lfp_metadata.update(filter_kwargs)
    
    return dset, lfp_metadata

def filter_lfp_from_ecube(ecube_filepath, result_filepath, mean_subtract=True, dtype='int16', max_memory_gb=1., **filter_kwargs):
    '''
    Filters local field potential (LFP) data from an eCube recording file.

    Args:
        ecube_filepath (str): Path to the input eCube recording file.
        result_filepath (str): Path to save the filtered LFP data.
        mean_subtract (bool, optional): Whether to subtract the mean from the filtered LFP signal.
                                        Default is True.
        dtype (str, optional): Data type for the filtered LFP signal. Default is 'int16'.
        max_memory_gb (float, optional): Maximum memory (in gigabytes) to use for filtering. Default is 1.0 GB.
        **filter_kwargs: Additional keyword arguments to customize the filtering process.
                        These arguments will be passed to the filtering function.

    Raises:
        IOError: If the input eCube recording file is not found.
        MemoryError: If the specified max_memory_gb is insufficient for the filtering process.

    Note:
        This function is used in the :func:`~aopy.preproc.warppers.proc_lfp` wrapper.
    '''
    lfp_samplerate = filter_kwargs.pop('lfp_samplerate', 1000)

    metadata = load_ecube_metadata(ecube_filepath, 'Headstages')
    samplerate = metadata['samplerate']
    downsample_factor = int(samplerate/lfp_samplerate)
    n_channels = int(metadata['n_channels'])
    n_samples = int(metadata['n_samples'])
    if 'low_cut' not in filter_kwargs:
        filter_kwargs['low_cut'] = 500
    if 'buttord' not in filter_kwargs:
        filter_kwargs['buttord'] = 4

    # Figure out how much data we can load at once
    max_samples = int(max_memory_gb * 1e9 / np.dtype(dtype).itemsize)
    chunksize = int(max_samples / metadata['n_channels'])
    n_whole_chunks = int(n_samples / chunksize)
    n_remaining = n_samples - (chunksize * n_whole_chunks)
    lfp_samples = np.ceil(chunksize/downsample_factor) * n_whole_chunks + np.ceil(n_remaining/downsample_factor)
    print("lfp chunks should be ", np.ceil(chunksize/downsample_factor))
    print("last chunk should be ", np.ceil(n_remaining/downsample_factor))
    print("total samples: ", lfp_samples)

    # Create an hdf dataset
    hdf = h5py.File(result_filepath, 'a') # should append existing or write new?
    dset = hdf.create_dataset('lfp_data', (lfp_samples, n_channels), dtype=dtype)

    # Filter broadband data into LFP directly into the hdf file
    n_lfp_samples = 0
    for broadband_chunk in load_ecube_data_chunked(ecube_filepath, 'Headstages', chunksize=chunksize):
        lfp_chunk, _ = precondition.filter_lfp(broadband_chunk, samplerate, **filter_kwargs)
        chunk_len = lfp_chunk.shape[0]
        dset[n_lfp_samples:n_lfp_samples+chunk_len,:] = lfp_chunk
        n_lfp_samples += chunk_len
        
    if mean_subtract:
        dset -= np.mean(dset, axis=0, dtype=dtype) # hopefully this isn't constrained by memory
    hdf.close()

    # Append the lfp metadata to the file
    lfp_metadata = metadata
    lfp_metadata['lfp_samplerate'] = lfp_samplerate # for backwards compatibility
    lfp_metadata['samplerate'] = lfp_samplerate
    lfp_metadata['n_samples'] = lfp_samples
    lfp_metadata.update(filter_kwargs)
    
    return dset, lfp_metadata

def _process_channels(data_dir, data_source, channels, n_samples, dtype=None, debug=False, **dataset_kwargs):
    '''
    Reads data from an ecube data source by channel until the number of samples requested 
    has been loaded. If a processing function is supplied, it will be applied to 
    each batch of data. If not, the data will be appended 

    Args:
        data_dir (str): folder containing the data you want to load
        data_source (str): type of data ("Headstages", "AnalogPanel", "DigitalPanel")
        channels (int array): list of channels to process
        n_samples (int): number of samples to read. Must be geq than a single chunk
        dtype (np.dtype): format for data_out if none supplied
        data_out (nt, nch): array of data to be written to. If None, it will be created
        debug (bool): whether the data is read in debug mode
        dataset_kwargs (kwargs): list of key value pairs to pass to the ecube dataset
        
    Yields:
        (nt, nch): Chunks of the requested samples for requested channels
    '''

    dat = Dataset(data_dir, **dataset_kwargs)
    dat.selectsource(data_source)
    chunk = dat.emitchunk(startat=0, debug=debug)
    datastream = ChunkedStream(chunkemitter=chunk)

    idx_samples = 0 # keeps track of the number of samples already read/written
    while idx_samples < n_samples:
        try:
            data_chunk = next(datastream)
            data_len = np.shape(data_chunk)[1]
            if len(channels) == 1:
                yield data_chunk[channels,:].T 
            else:
                yield np.squeeze(data_chunk[channels,:]).T # this might be where you filter data
            idx_samples += data_len
        except StopIteration:
            break

def load_ecube_digital(path, data_dir):
    '''
    Just a wrapper around load_ecube_data() and load_ecube_metadata()

    Args:
        path (str): base directory where ecube data is stored
        data_dir (str): folder you want to load

    Returns:
        tuple: Tuple containing:
            | **data (nt):** digital data, arranged as 64-bit numbers representing the 64 channels
            | **metadata (dict):** metadata (see load_ecube_metadata() for details)
    '''
    data = load_ecube_data(os.path.join(path, data_dir), 'DigitalPanel')
    metadata = load_ecube_metadata(os.path.join(path, data_dir), 'DigitalPanel')
    return data, metadata

def load_emg_data(data_dir, emg_filename):
    '''
    Loads emg data

    Args:
        data_dir (str): base directory where emg data is stored
        emg_filename (str): hdf file you want to load

    Returns:
        tuple: Tuple containing:
            | **data (nt):** emg data
            | **metadata (dict):** metadata from the emg file containing samplerate
    '''
    emg_data, emg_metadata = load_bmi3d_hdf_table(data_dir, emg_filename, 'data')

    # Reshape the data
    if 'dtype' in emg_metadata:
        dtype = emg_metadata['dtype']
    else:
        dtype = 'f8'
        emg_metadata['dtype'] = dtype
    if 'n_ararys' in emg_metadata:
        nch = emg_metadata['n_arrays']*64
    else:
        nch = 64
        emg_metadata['n_arrays'] = 1
    emg_metadata['n_channels'] = nch # the `channels` in emg_metadata includes AUX channels
    emg_metadata['data_source'] = os.path.join(data_dir, emg_filename)
    emg_data_reshape = emg_data.view((dtype, (len(emg_data.dtype),)))
    emg_data = emg_data_reshape[:,:nch]
    return emg_data, emg_metadata

def load_emg_analog(data_dir, emg_filename):
    '''
    Loads emg analog data

    Args:
        data_dir (str): base directory where emg data is stored
        emg_filename (str): hdf file you want to load

    Returns:
        tuple: Tuple containing:
            | **data (nt):** analog data
            | **metadata (dict):** metadata from the emg file containing samplerate
    '''

    emg_data, emg_metadata = load_bmi3d_hdf_table(data_dir, emg_filename, 'data')
    if 'dtype' in emg_metadata:
        dtype = emg_metadata['dtype']
    else:
        dtype = 'f8'
        emg_metadata['dtype'] = dtype
    emg_metadata['n_channels'] = 16
    emg_metadata['data_source'] = os.path.join(data_dir, emg_filename)
    emg_data_reshape = emg_data.view((dtype, (len(emg_data.dtype),)))
    analog_data = emg_data_reshape[:,-24:-8] # AUX channels
    return analog_data, emg_metadata
    
def load_emg_digital(data_dir, emg_filename):
    '''
    Loads and converts emg analog data to 64-bit digital data.

    Args:
        data_dir (str): base directory where emg data is stored
        emg_filename (str): hdf file you want to load

    Returns:
        tuple: Tuple containing:
            | **data (nt):** digital data, arranged as 64-bit numbers
            | **metadata (dict):** metadata from the emg file containing samplerate
    '''
    analog_data, emg_metadata = load_emg_analog(data_dir, emg_filename)
    digital_data = np.zeros((len(analog_data),64), dtype='bool')
    digital_data[:,:analog_data.shape[1]] = utils.base.convert_analog_to_digital(analog_data, thresh=0.5)
    digital_data = utils.base.convert_channels_to_digital(digital_data)
        
    emg_metadata['n_channels'] = 16
    emg_metadata['data_source'] = os.path.join(data_dir, emg_filename)
    return digital_data, emg_metadata

def load_ecube_analog(path, data_dir, channels=None):
    '''
    Just a wrapper around load_ecube_data() and load_ecube_metadata()

    Args:
        path (str): base directory where ecube data is stored
        data_dir (str): folder you want to load
        channels (int array, optional): which channels to load

    Returns:
        tuple: Tuple containing:
            | **data (nt, nch):** analog data for the requested channels
            | **metadata (dict):** metadata (see load_ecube_metadata() for details)
    '''
    data = load_ecube_data(os.path.join(path, data_dir), 'AnalogPanel', channels)
    metadata = load_ecube_metadata(os.path.join(path, data_dir), 'AnalogPanel')
    return data, metadata

def load_ecube_headstages(path, data_dir, channels=None):
    '''
    Just a wrapper around load_ecube_data() and load_ecube_metadata()

    Args:
        path (str): base directory where ecube data is stored
        data_dir (str): folder you want to load
        channels (int array, optional): which channels to load

    Returns:
        tuple: Tuple containing:
            | **data (nt, nch):** analog data for the requested channels
            | **metadata (dict):** metadata (see load_ecube_metadata() for details)
    '''
    data = load_ecube_data(os.path.join(path, data_dir), 'Headstages', channels)
    metadata = load_ecube_metadata(os.path.join(path, data_dir), 'Headstages')
    return data, metadata


def get_e3v_video_frame_data( digital_data, sync_channel_idx, trigger_channel_idx, samplerate ):
    
    """get_e3v_video_frame_data

    Compute pulse times and duty cycles from e3vision video data frames collected on an ecube digital panel.

    Args:
        digital_data (nt, nch): array of data read from ecube digital panel
        sync_channel_idx (int): sync channel to read from digital_data. Indicates each video frame.
        trigger_channel_idx (int): trigger channel to read from digital_data. Indicates start/end video triggers.
        sample_rate (numeric): data sampling rate (Hz)

    Returns:
        pulse_times (np.array): array of floats indicating pulse start times
        duty_cycle (np.array): array of floats indicating pulse duty cycle (quotient of pulse width and pulse period)
    """

    trig_pulse_edges = get_pulse_edge_times(digital_data[:,trigger_channel_idx],samplerate)
    # watchtower triggers (start, end) are a triplet of pulses within a ~33ms window.
    trig_pulse_times = trig_pulse_edges[:,0]
    start_trig_time = trig_pulse_times[0]
    end_trig_time = trig_pulse_times[3]
    sync_pulse_edges = get_pulse_edge_times(digital_data[:,sync_channel_idx],samplerate)
    sync_pulse_times = sync_pulse_edges[:,0]
    start_pulse_idx = np.where(np.abs(sync_pulse_times-start_trig_time) == np.abs(sync_pulse_times-start_trig_time).min())[0][0]
    end_pulse_idx = np.where(np.abs(sync_pulse_times-end_trig_time) == np.abs(sync_pulse_times-end_trig_time).min())[0][0] + 1
    sync_pulse_times = sync_pulse_times[start_pulse_idx:end_pulse_idx]
    sync_duty_cycles = compute_pulse_duty_cycles(sync_pulse_edges[start_pulse_idx:end_pulse_idx,:])

    return sync_pulse_times, sync_duty_cycles

def load_bmi3d_hdf_table(data_dir, filename, table_name):
    '''
    Loads data and metadata from a table in an hdf file generated by BMI3D

    Args:
        data_dir (str): path to the data
        filename (str): name of the file to load from
        table_name (str): name of the table you want to load

    Returns:
        tuple: Tuple containing:
            | **data (ndarray):** data from bmi3d
            | **metadata (dict):** attributes associated with the table
    '''
    filepath = os.path.join(data_dir, filename)
    with tables.open_file(filepath, 'r') as f:
        if table_name not in f.root:
            raise ValueError(f"{table_name} not found in {filename}")
        table = getattr(f.root, table_name)
        param_keys = table.attrs._f_list("user")
        metadata = {k : getattr(table.attrs, k) for k in param_keys}
        return table.read(), metadata

def load_bmi3d_root_metadata(data_dir, filename):
    '''
    Root metadata not accessible using pytables, instead use h5py

    Args:
        data_dir (str): path to the data
        filename (str): name of the file to load from

    Returns:
        dict: key-value attributes
    '''
    with h5py.File(os.path.join(data_dir, filename), 'r') as f:
        return dict(f['/'].attrs.items())

def get_ecube_digital_input_times(path, data_dir, ch):
    '''
    Computes the times when digital input turns on or off in ecube
    For synchronizing openephys with ecube, use ch=-1.
    
    Args:
    path (str): base directory where ecube data is stored
    data_dir (str): folder you want to load
    ch (str): digital channel
        
    Returns:
        tuple: Tuple containing:
            | **on_times (n_times):** times at which sync line turned on
            | **off_times (n_times):** times at which sync line turned off    
    '''
    
    # Load ecube digital data
    digital_data, metadata = load_ecube_digital(path, data_dir)
    FS = metadata['samplerate']
    
    # Convert 64bit information into each channel data
    digital_data_ch = convert_digital_to_channels(digital_data)
        
    # Get on_times and off_times in digital data
    on_times,_ = detect_edges(digital_data_ch[:,ch], FS, rising=True, falling=False)
    off_times,_ = detect_edges(digital_data_ch[:,ch], FS, rising=False, falling=True)

    return on_times, off_times

#####################
# Preprocessed data #
#####################
def get_interp_task_data(exp_data, exp_metadata, datatype='cursor', samplerate=1000, step=1, **kwargs):
    '''
    Gets interpolated data from preprocessed experiment task cycles to the desired 
    sampling rate. Cursor kinematics are returned in screen coordinates, while user input 
    kinematics are returned either in their original raw coordinate system with datatype='user_raw' (e.g. 
    optitrack coordinates), in world coordinates with datatype='user_world', or in screen coordinates 
    with datatype='user_screen' (similar to cursor kinematics but without any bounding under position
    control).

    Args:
        exp_data (dict): A dictionary containing the experiment data.
        exp_metadata (dict): A dictionary containing the experiment metadata.
        datatype (str, optional): The type of kinematic data to interpolate. 
            - 'cursor' for cursor kinematics
            - 'user_raw' for raw input coordinates
            - 'user_world' for user input in world coordinates
            - 'user_screen' for user input in screen coordinates
            - 'reference' for reference kinematics
            - 'disturbance' for disturbance kinematics
            - 'targets' for target positions
            - other datatypes if they exist as exp_data['task'][<datatype>]
        samplerate (float, optional): The desired output sampling rate in Hz. 
            Defaults to 1000.
        step (int, optional): task data will be decimated with steps this big. Default 1.
        **kwargs: Additional keyword arguments to pass to sample_timestamped_data()

    Returns:
        data_time (ns, ...): Kinematic data interpolated and filtered 
            to the desired sampling rate.

    Examples:
        
        Cursor kinematics in screen coordinates (datatype 'cursor')

        .. code-block:: python
        
            exp_data, exp_metadata = load_preproc_exp_data(preproc_dir, 'test',  3498, '2021-12-13')
            cursor_interp = get_interp_task_data(exp_data, exp_metadata, datatype='cursor', samplerate=100)

            plt.figure()
            visualization.plot_trajectories([cursor_interp], [-10, 10, -10, 10])
        
        .. image:: _images/get_interp_cursor_centerout.png

        Raw input kinematics (datatype 'user_raw', 'hand', or 'manual_input')
       
        .. code-block:: python

            hand_interp = get_interp_task_data(exp_data, exp_metadata, datatype='hand', samplerate=100)
            ax = plt.axes(projection='3d')
            visualization.plot_trajectories([hand_interp], [-10, 10, -10, 10, -10, 10])

        .. image:: _images/get_interp_hand_centerout.png

        User input kinematics in world coordinates (datatype 'user_world')
       
        .. code-block:: python

            user_world = get_interp_task_data(exp_data, exp_metadata, datatype='user_world', samplerate=100)
            ax = plt.axes(projection='3d')
            visualization.plot_trajectories([user_world], [-10, 10, -10, 10, -10, 10])

        .. image:: _images/get_user_world.png

        User input kinematics in screen coordinates (datatype 'user_screen')
       
        .. code-block:: python

            user_screen = get_interp_task_data(exp_data, exp_metadata, datatype='user_screen', samplerate=100)
            ax = plt.axes(projection='3d')
            visualization.plot_trajectories([user_screen], [-10, 10, -10, 10, -10, 10])

        .. image:: _images/get_user_screen.png

        Target positions (datatype 'target')

        .. code-block:: python
            
            targets_interp = get_interp_task_data(exp_data, exp_metadata, datatype='targets', samplerate=100)
            time = np.arange(len(targets_interp))/100
            plt.plot(time, targets_interp[:,:,0]) # plot just the x coordinate
            plt.xlim(10, 20)
            plt.xlabel('time (s)')
            plt.ylabel('x position (cm)')

        .. image:: _images/get_interp_targets_centerout.png

        Cursor and target (datatype 'reference') kinematics

        .. code-block:: python
            
            exp_data, exp_metadata = load_preproc_exp_data(data_dir, 'test', 8461, '2023-02-25')
            cursor_interp = get_interp_task_data(exp_data, exp_metadata, datatype='cursor', samplerate=exp_metadata['fps'])
            ref_interp = get_interp_task_data(exp_data, exp_metadata, datatype='reference', samplerate=exp_metadata['fps'])
            time = np.arange(exp_metadata['fps']*120)/exp_metadata['fps']
            plt.plot(time, cursor_interp[:int(exp_metadata['fps']*120),1], color='blueviolet', label='cursor') # plot just the y coordinate
            plt.plot(time, ref_interp[:int(exp_metadata['fps']*120),1], color='darkorange', label='ref')
            plt.xlabel('time (s)')
            plt.ylabel('y position (cm)'); plt.ylim(-10,10)
            plt.legend()

        .. image:: _images/get_interp_cursor_tracking.png

        User, reference, and disturbance kinematics

        .. code-block:: python
            
            user_interp = get_interp_task_data(exp_data, exp_metadata, datatype='user', samplerate=exp_metadata['fps'])
            ref_interp = get_interp_task_data(exp_data, exp_metadata, datatype='reference', samplerate=exp_metadata['fps'])
            dis_interp = get_interp_task_data(exp_data, exp_metadata, datatype='disturbance', samplerate=exp_metadata['fps'])
            time = np.arange(exp_metadata['fps']*120)/exp_metadata['fps']
            plt.plot(time, user_interp[:int(exp_metadata['fps']*120),1], color='darkturquoise', label='user')
            plt.plot(time, ref_interp[:int(exp_metadata['fps']*120),1], color='darkorange', label='ref')
            plt.plot(time, dis_interp[:int(exp_metadata['fps']*120),1], color='tab:red', linestyle='--', label='dis')
            plt.xlabel('time (s)')
            plt.ylabel('y position (cm)'); plt.ylim(-10,10)
            plt.legend()

        .. image:: _images/get_interp_user_tracking.png

    Changes:
        2023-10-20: Added support for 'targets' datatype
        2024-01-29: Removed kinematic filtering below 15 Hz. See :func:`~aopy.precondition.filter_kinematics`.
    '''

    # Fetch the available timestamps
    try:
        clock = exp_data['clock']['timestamp_sync']
    except:
        clock = exp_data['clock']['timestamp_bmi3d']

    # Fetch the relevant BMI3D data
    if datatype in ['hand', 'user_raw', 'manual_input']:
        warnings.warn("Raw hand position is not recommended for analysis. Use 'user_world' instead for 3D world coordinate inputs.")
        data_cycles = exp_data['clean_hand_position'] # 3d hand position (e.g. raw optitrack coords: x,y,z) on each bmi3d cycle
    elif datatype == 'user_world':
        # 3d user input converted to world coordinates
        if 'exp_gain' in exp_metadata:
            scale = exp_metadata['scale']
        else:
            scale = np.sign(exp_metadata['scale'])
        data_cycles = postproc.bmi3d.convert_raw_to_world_coords(exp_data['clean_hand_position'], exp_metadata['rotation'], 
                                                  exp_metadata['offset'], scale)
    elif datatype == 'cursor':
        data_cycles = exp_data['task']['cursor'][:,[0,2,1]] # cursor position (from bmi3d coords: x,z,y) on each bmi3d cycle
    elif datatype == 'user_screen':
        # 3d user input converted to screen coordinates. Only works for singular mappings, not incremental mappings.
        if 'incremental_rotation' in exp_metadata['features']:
            warnings.warn("User input in screen coordinates is not recommended for incremental mappings. Use 'intended_cursor' instead.")
        if 'exp_gain' in exp_metadata:
            scale = exp_metadata['scale']
            exp_gain = exp_metadata['exp_gain']
        else:
            scale = np.sign(exp_metadata['scale'])
            exp_gain = np.abs(exp_metadata['scale'])
        user_world_cycles = postproc.bmi3d.convert_raw_to_world_coords(exp_data['clean_hand_position'], exp_metadata['rotation'], 
                                                  exp_metadata['offset'], scale)
        if 'exp_rotation' in exp_metadata:
            exp_rotation = exp_metadata['exp_rotation']
        else:
            exp_rotation = 'none'
        if 'perturbation_rotation_x' in exp_metadata:
            x_rot = exp_metadata['perturbation_rotation_x']
            z_rot = exp_metadata['perturbation_rotation_z']
        else:
            x_rot = 0
            z_rot = 0
        if 'pertubation_rotation' in exp_metadata:
            y_rot = exp_metadata['pertubation_rotation']
        else:
            y_rot = 0
        exp_mapping = postproc.bmi3d.get_world_to_screen_mapping(exp_rotation, x_rot, y_rot, z_rot, exp_gain)
        data_cycles = np.dot(user_world_cycles, exp_mapping)
    elif datatype in ['user', 'intended_cursor']:
        if datatype == 'user':
            warnings.warn("User input is not recommended. Use 'intended_cursor' instead for clarity.")
        dis_on = int(json.loads(exp_metadata['sequence_params'])['disturbance']) # whether disturbance was turned on (0 or 1)
        data_cycles = exp_data['task']['cursor'][:,[0,2,1]] - exp_data['task']['current_disturbance'][:,[0,2,1]]*dis_on # cursor position before disturbance added (bmi3d coords: x,z,y)
    elif datatype == 'reference':
        data_cycles =  exp_data['task']['current_target'][:,[0,2,1]] # target position (bmi3d coords: x,z,y)
    elif datatype == 'disturbance':
        dis_on = int(json.loads(exp_metadata['sequence_params'])['disturbance']) # whether disturbance was turned on (0 or 1)
        data_cycles = exp_data['task']['current_disturbance'][:,[0,2,1]]*dis_on # disturbance value (bmi3d coords: x,z,y)
    elif datatype == 'targets':
        data_cycles = get_target_events(exp_data, exp_metadata)
        clock = exp_data['events']['timestamp']
        kwargs['remove_nan'] = False # In this case we need to keep NaN values.
    elif datatype == 'cycle':
        data_cycles = np.arange(len(exp_data['task'])) # cycle number
    elif datatype in exp_data['task'].dtype.names:
        data_cycles = exp_data['task'][datatype]
    else:
        raise ValueError(f"Unknown datatype {datatype}")
    
    # Interpolate
    data_time = sample_timestamped_data(data_cycles[::step], clock[::step], samplerate, append_time=10, **kwargs)

    return data_time

def get_velocity_segments(*args, norm=True, **kwargs):
    '''
    Estimates velocity from cursor position, then finds the trial segments for velocity using 
    :func:`~aopy.postproc.get_kinematic_segments()`.
    
    Args:
        *args: arguments for :func:`~aopy.postproc.get_kinematic_segments`
        norm (bool): if the output segments should be normalized. Set to false to output component velocities.
        **kwargs: parameters for :func:`~aopy.postproc.get_kinematic_segments`
        
    Returns:
        tuple: tuple containing:
            | **velocities (ntrial):** array of velocity estimates for each trial
            | **trial_segments (ntrial):** array of numeric code segments for each trial
    '''
    def preproc(pos, fs):
        time = np.arange(pos.shape[0])/fs
        return derivative(time, pos, norm=norm), fs
    return get_kinematic_segments(*args, **kwargs, preproc=preproc)

@lru_cache(maxsize=1)
def get_task_data(preproc_dir, subject, te_id, date, datatype, samplerate=None, step=1, preproc=None, **kwargs):
    '''
    Return interpolated task data. Wraps :func:`~aopy.data.bmi3d.get_interp_task_data` but 
    caches the data for faster loading.

    Note: 
        You can avoid the phase shift in downsampled data when using get_interp_task_data by setting 
        upsamplerate=samplerate, so that it doesn't do any up/down sampling, only interpolation at the 
        same samplerate.

    Args:
        preproc_dir (str): base directory where the files live
        subject (str): Subject name
        te_id (int): Block number of Task entry object 
        date (str): Date of recording
        datatype (str): column of task data to load. 
        samplerate (float): choose the samplerate of the data in Hz. Default None,
            which uses the sampling rate of the experiment.
        step (int, optional): task data will be decimated with steps this big. Default 1.
        preproc (fn, optional): function mapping (position, fs) data to (kinematics, fs_new). For example,
            a smoothing function or an estimate of velocity from position
        kwargs: additional keyword arguments to pass to get_interp_task_data 

    Raises:
        ValueError: if the datatype is invalid

    Returns:
        tuple: tuple containing:
            | **kinematics (nt, nch):** kinematics from the given experiment after preprocessing
            | **samplerate (float):** the sampling rate of the kinematics after preprocessing

    Examples:

        .. code-block:: python

            subject = 'beignet'
            te_id = 4301
            date = '2021-01-01'
            ts_data, samplerate = get_task_data(preproc_dir, subject, te_id, date, 'cycle')
            time = np.arange(len(ts_data))/samplerate
            plt.figure()
            plt.plot(time[1:], 1/np.diff(ts_data), 'ko')
            plt.xlabel('time (s)')
            plt.ylabel('cycle step')
            plt.ylim(0, 2)
            
        .. image:: _images/get_cycle_data.png
    '''
    exp_data, exp_metadata = base.load_preproc_exp_data(preproc_dir, subject, te_id, date)
    if samplerate is None:
        samplerate = exp_metadata['fps']

    raw_data = get_interp_task_data(exp_data, exp_metadata, datatype, samplerate, step=step, **kwargs)
    if preproc is not None:
        data, samplerate = preproc(raw_data, samplerate)
    else:
        data = raw_data

    return data, samplerate

@lru_cache(maxsize=1)
def get_kinematics(preproc_dir, subject, te_id, date, samplerate, preproc=None, datatype='cursor', **kwargs):
    '''
    Return all kinds of kinematics from preprocessed data. Caches the data for faster loading. 

    Note: 
        You can avoid the phase shift in downsampled data when using get_interp_task_data by setting 
        upsamplerate=samplerate, so that it doesn't do any up/down sampling, only interpolation at the 
        same samplerate.

    Args:
        preproc_dir (str): base directory where the files live
        subject (str): Subject name
        te_id (int): Block number of Task entry object 
        date (str): Date of recording
        samplerate (float, optional): optionally choose the samplerate of the data in Hz. Default 1000.
        preproc (fn, optional): function mapping (position, fs) data to (kinematics, fs_new). For example,
            a smoothing function or an estimate of velocity from position
        datatype (str, optional): type of kinematics to load. Defaults to 'cursor'.   
        kwargs: additional keyword arguments to pass to get_interp_task_data 

    Raises:
        ValueError: if the datatype is invalid

    Returns:
        tuple: tuple containing:
            | **kinematics (nt, nch):** kinematics from the given experiment after preprocessing
            | **samplerate (float):** the sampling rate of the kinematics after preprocessing
    '''
    if 'eye' in datatype:
        eye_data, eye_metadata = base.load_preproc_eye_data(preproc_dir, subject, te_id, date)
        if datatype == 'eye_raw':
            eye_data = eye_data['raw_data']
        elif datatype == 'eye_closed_mask':
            eye_data = eye_data['eye_closed_mask']
        elif 'calibrated_data' in eye_data.keys():
            eye_data = eye_data['calibrated_data']
        else:
            raise ValueError(f"No calibrated eye data for {te_id}")
        
        time = np.arange(len(eye_data))/eye_metadata['samplerate']
        raw_kinematics, _ = interp_timestamps2timeseries(time, eye_data, samplerate)

        time = np.arange(len(raw_kinematics))/samplerate
        if preproc is not None:
            kinematics, samplerate = preproc(raw_kinematics, samplerate)
        else:
            kinematics = raw_kinematics
    else:
        kinematics, samplerate = get_task_data(preproc_dir, subject, te_id, date, datatype, 
                                               samplerate, preproc=preproc, **kwargs)

    return kinematics, samplerate

def _get_kinematic_segment(preproc_dir, subject, te_id, date, start_time, end_time, samplerate, 
                          preproc=None, datatype='cursor', **kwargs):
    '''
    Helper function to return one segment of kinematics
    '''
    kinematics, samplerate = get_kinematics(preproc_dir, subject, te_id, date, samplerate, preproc, datatype, **kwargs)
    assert kinematics is not None

    return get_data_segment(kinematics, start_time, end_time, samplerate), samplerate

def get_extracted_features(preproc_dir, subject, te_id, date, decoder, samplerate=None, start_time=None, 
                           end_time=None, datatype='lfp_power', preproc=None, **kwargs):
    '''
    Fetches online extracted features from readouts of a BCI experiment. Wrapper around get_task_data.
    
    Args:
        preproc_dir (str): base directory where the files live
        subject (str): Subject name
        te_id (int): Block number of Task entry object 
        date (str): Date of recording
        decoder (riglib.bmi.Decoder): decoder object with binlen and call_rate attributes
        samplerate (float, optional): optionally choose the samplerate of the data in Hz. Default None,
            uses the sampling rate of the experiment.
        start_time (float, optional): start time of the segment to load (in seconds). Default None,
            which loads from the beginning of the data.
        end_time (float, optional): end time of the segment to load (in seconds). Default None,
            which loads until the end of the data.
        datatype (str, optional): type of features to load. Defaults to 'lfp_power'.
        preproc (fn, optional): function mapping (state, fs) data to (state_new, fs_new). For example,
            a smoothing function.
        kwargs: additional keyword arguments to pass to sample_timestamped_data 

    Returns:
        tuple: tuple containing:
            | **state (nt, nfeats):** decoded states from the given experiment after preprocessing
            | **samplerate (float):** the sampling rate of the states after preprocessing

    '''
    step = int(decoder.call_rate*decoder.binlen)
    state, samplerate = get_task_data(
        preproc_dir, subject, te_id, date, datatype, samplerate=samplerate,
        step=step, preproc=preproc, **kwargs)
    
    if start_time is not None or end_time is not None:
        if start_time is None:
            start_time = 0
        if end_time is None:
            end_time = len(state)/samplerate
        state = get_data_segment(state, start_time, end_time, samplerate)
    return state, samplerate

def get_decoded_states(preproc_dir, subject, te_id, date, decoder, samplerate=None, start_time=None,
                       end_time=None, preproc=None, **kwargs):
    '''
    Fetches online decoded states from readouts in a BCI experiment. Wrapper around get_task_data.
    
    Args:
        preproc_dir (str): base directory where the files live
        subject (str): Subject name
        te_id (int): Block number of Task entry object 
        date (str): Date of recording
        decoder (riglib.bmi.Decoder): decoder object with binlen and call_rate attributes
        samplerate (float, optional): optionally choose the samplerate of the data in Hz. Default None,
            uses the sampling rate of the experiment.
        start_time (float, optional): start time of the segment to load (in seconds). Default None,
            which loads from the beginning of the data.
        end_time (float, optional): end time of the segment to load (in seconds). Default None,
            which loads until the end of the data.
        preproc (fn, optional): function mapping (state, fs) data to (state_new, fs_new). For example,
            a smoothing function.
        kwargs: additional keyword arguments to pass to sample_timestamped_data 

    Returns:
        tuple: tuple containing:
            | **state (nt, nstate):** decoded states from the given experiment after preprocessing
            | **samplerate (float):** the sampling rate of the states after preprocessing
    '''
    return get_extracted_features(preproc_dir, subject, te_id, date, decoder, samplerate=samplerate, start_time=start_time, 
                           end_time=end_time, datatype='decoder_state', preproc=preproc, **kwargs)

def get_kinematic_segments(preproc_dir, subject, te_id, date, trial_start_codes, trial_end_codes, 
                           trial_filter=lambda x:True, preproc=None, datatype='cursor',
                           samplerate=1000, **kwargs):
    '''
    Loads x,y,z cursor, hand, or eye trajectories for each "trial" from a preprocessed HDF file. Trials can
    be specified by numeric start and end codes. Trials can also be filtered so that only successful
    trials are included, for example. The filter is applied to numeric code segments for each trial. 
    Finally, the cursor data can be preprocessed by a supplied function to, for example, convert 
    position to velocity estimates. The preprocessing function is applied to the (time, position)
    cursor or eye data.

    See also:
        :func:`~aopy.data.bmi3d.get_kinematic_segment`, :func:`~aopy.data.bmi3d.get_kinematics`
    
    Example:
        subject = 'beignet'
        te_id = 4301
        date = '2021-01-01'
        trial_filter = lambda t: TRIAL_END not in t
        trajectories, segments = get_kinematic_segments(preproc_dir, subject, te_id, date,
                                                       [CURSOR_ENTER_CENTER_TARGET], 
                                                       [REWARD, TRIAL_END], 
                                                       trial_filter=trial_filter) 
    
    Args:
        preproc_dir (str): base directory where the files live
        subject (str): Subject name
        te_id (int): Block number of Task entry object 
        date (str): Date of recording
        trial_start_codes (list): list of numeric codes representing the start of a trial
        trial_end_codes (list): list of numeric codes representing the end of a trial
        trial_filter (fn, optional): function mapping trial segments to boolean values. Any trials
            for which the filter returns False will not be included in the output
        preproc (fn, optional): function mapping (position, samplerate) data to kinematics. For example,
            a smoothing function or an estimate of velocity from position
        datatype (str, optional): type of kinematics to load. Defaults to 'cursor'.    
        samplerate (float, optional): optionally choose the samplerate of the data in Hz. Default 1000.
        kwargs: additional keyword arguments to pass to get_kinematics
    
    Returns:
        tuple: tuple containing:
            | **trajectories (ntrial):** array of filtered cursor trajectories for each trial
            | **trial_segments (ntrial):** array of numeric code segments for each trial   

    Note:
        The sampling rate of the returned data might be different from the requested sampling rate if the
        preprocessing function does any modification to the length of the data.

    Modified September 2023 to include optional sampling rate argument     
    '''
    data, metadata = base.load_preproc_exp_data(preproc_dir, subject, te_id, date)
    event_codes = data['events']['code']
    event_times = data['events']['timestamp']
    trial_segments, trial_times = get_trial_segments(event_codes, event_times, 
                                                                  trial_start_codes, trial_end_codes)
    segments = [
        _get_kinematic_segment(preproc_dir, subject, te_id, date, t[0], t[1], samplerate, preproc, datatype, **kwargs)[0] 
        for t in trial_times
    ]
    trajectories = np.array(segments, dtype='object')
    trial_segments = np.array(trial_segments, dtype='object')
    success_trials = [trial_filter(t) for t in trial_segments]
    
    return trajectories[success_trials], trial_segments[success_trials]

def get_lfp_segments(preproc_dir, subject, te_id, date, trial_start_codes, trial_end_codes, 
                           trial_filter=lambda x:True):
    '''
    Loads lfp segments (different length for each trial) from a preprocessed HDF file. Trials can
    be specified by numeric start and end codes. Trials can also be filtered so that only successful
    trials are included, for example. The filter is applied to numeric code segments for each trial. 
        
    Args:
        preproc_dir (str): path to the preprocessed directory
        preproc_dir (str): base directory where the files live
        subject (str): Subject name
        te_id (int): Block number of Task entry object 
        date (str): Date of recording
        trial_start_codes (list): list of numeric codes representing the start of a trial
        trial_end_codes (list): list of numeric codes representing the end of a trial
        trial_filter (fn, optional): function mapping trial segments to boolean values. Any trials
            for which the filter returns False will not be included in the output
    
    Returns:
        tuple: tuple containing:
            | **lfp_segments (ntrial):** array of filtered lfp segments for each trial
            | **trial_segments (ntrial):** array of numeric code segments for each trial

    '''
    data, metadata = base.load_preproc_exp_data(preproc_dir, subject, te_id, date)
    lfp_data, lfp_metadata = base.load_preproc_lfp_data(preproc_dir, subject, te_id, date)
    samplerate = lfp_metadata['lfp_samplerate']

    event_codes = data['events']['code']
    event_times = data['events']['timestamp']

    trial_segments, trial_times = get_trial_segments(event_codes, event_times, 
                                                                  trial_start_codes, trial_end_codes)
    lfp_segments = np.array(get_data_segments(lfp_data, trial_times, samplerate), dtype='object')
    trial_segments = np.array(trial_segments, dtype='object')
    success_trials = [trial_filter(t) for t in trial_segments]
    
    return lfp_segments[success_trials], trial_segments[success_trials]


def get_lfp_aligned(preproc_dir, subject, te_id, date, trial_start_codes, trial_end_codes, 
                           time_before, time_after, trial_filter=lambda x:True):
    '''
    Loads lfp data (same length for each trial) from a preprocessed HDF file. Trials can
    be specified by numeric start and end codes. Trials can also be filtered so that only successful
    trials are included, for example. The filter is applied to numeric code segments for each trial. 

    Args:
        preproc_dir (str): base directory where the files live
        subject (str): Subject name
        te_id (int): Block number of Task entry object 
        date (str): Date of recording
        trial_start_codes (list): list of numeric codes representing the start of a trial
        trial_end_codes (list): list of numeric codes representing the end of a trial
        time_before (float): time before the trial start to include in the aligned lfp (in seconds)
        time_after (float): time after the trial end to include in the aligned lfp (in seconds)
        trial_filter (fn, optional): function mapping trial segments to boolean values. Any trials
            for which the filter returns False will not be included in the output
    
    Returns:
        (ntrials, nt, nch): aligned lfp data output from `func:aopy.preproc.trial_align_data`


    '''
    data, metadata = base.load_preproc_exp_data(preproc_dir, subject, te_id, date)
    lfp_data, lfp_metadata = base.load_preproc_lfp_data(preproc_dir, subject, te_id, date)
    samplerate = lfp_metadata['lfp_samplerate']

    event_codes = data['events']['code']
    event_times = data['events']['timestamp']

    trial_segments, trial_times = get_trial_segments(event_codes, event_times, 
                                                     trial_start_codes, trial_end_codes)
    trial_start_times = [t[0] for t in trial_times]
    assert len(trial_start_times) > 0, "No trials found"
    trial_aligned_data = trial_align_data(lfp_data, trial_start_times, time_before, time_after, samplerate) # (nt, nch, ntrial)
    success_trials = [trial_filter(t) for t in trial_segments]
    
    return trial_aligned_data[:,:,success_trials]

def get_ts_data_trial(preproc_dir, subject, te_id, date, trigger_time, time_before, time_after,
                      channels=None, datatype='lfp'):
    '''
    Simple wrapper around load_hdf_ts_trial for lfp or broadband data.
    
    Args:
        preproc_dir (str): base directory where the files live
        subject (str): Subject name
        te_id (int): Block number of Task entry object 
        date (str): Date of recording
        trigger_time (float): time (in seconds) in the recording at which the desired segment starts
        time_before (float): time (in seconds) to include before the trigger times
        time_after (float): time (in seconds) to include after the trigger times
        channels (int array, optional): which channel indices to load
        datatype (str, optional): choice of 'lfp' or 'broadband' data to load. Defaults to 'lfp'.    

    Returns:
        tuple: tuple containing:
            | **segment (nt, nch):** data segment from the given preprocessed file
            | **samplerate (float):** sampling rate of the returned data
    '''
    data_group='/'
    data_name=f'{datatype}_data'
    metadata_group=f'{datatype}_metadata'
    samplerate_key='samplerate'
    if datatype == 'lfp':
        samplerate_key='lfp_samplerate'
    filename = base.get_preprocessed_filename(subject, te_id, date, datatype)
    preproc_dir = os.path.join(preproc_dir, subject)

    try:
        samplerate = base.load_hdf_data(preproc_dir, filename, samplerate_key, metadata_group, cached=True)
        data = base.load_hdf_ts_trial(preproc_dir, filename, data_group, data_name, 
                                 samplerate, trigger_time, time_before, time_after, channels=channels)
    except FileNotFoundError as e:
        print(f"No data found in {preproc_dir} for subject {subject} on {date} ({te_id})")
        raise e

    return data, samplerate

def get_ts_data_segment(preproc_dir, subject, te_id, date, start_time, end_time,
                        channels=None, datatype='lfp'):
    '''
    Simple wrapper around load_hdf_ts_segment for lfp or broadband data.
    
    Args:
        preproc_dir (str): base directory where the files live
        subject (str): Subject name
        te_id (int): Block number of Task entry object 
        date (str): Date of recording
        trigger_time (float): time (in seconds) in the recording at which the desired segment starts
        time_before (float): time (in seconds) to include before the trigger times
        time_after (float): time (in seconds) to include after the trigger times
        channels (int array, optional): which channel indices to load
        datatype (str, optional): choice of 'lfp' or 'broadband' data to load. Defaults to 'lfp'.    

    Returns:
        tuple: tuple containing:
            | **segment (nt, nch):** data segment from the given preprocessed file
            | **samplerate (float):** sampling rate of the returned data
    '''
    data_group='/'
    data_name=f'{datatype}_data'
    metadata_group=f'{datatype}_metadata'
    samplerate_key='samplerate'
    if datatype == 'lfp':
        samplerate_key='lfp_samplerate'
    filename = base.get_preprocessed_filename(subject, te_id, date, datatype)
    preproc_dir = os.path.join(preproc_dir, subject)

    try:
        samplerate = base.load_hdf_data(preproc_dir, filename, samplerate_key, metadata_group)
        data = base.load_hdf_ts_segment(preproc_dir, filename, data_group, data_name, 
                                   samplerate, start_time, end_time, channels=channels)
    except FileNotFoundError as e:
        print(f"No data found in {preproc_dir} for subject {subject} on {date} ({te_id})")
        raise e

    return data, samplerate

def get_spike_data_segment(preproc_dir, subject, te_id, date, start_time, end_time, drive=1, bin_width=.01):
    '''
    Loads and extracts a segment of spiking data for a given subject and experiment, optionally binning the spike times.

    Args:
        preproc_dir (str): Path to the preprocessed data directory.
        subject (str): Subject name.
        te_id (str): Task entry number.
        date (str): The date of the experiment.
        start_time (float): The start time [s] of the segment to extract.
        end_time (float): The end time [s] of the segment to extract.
        drive (int, optional): Which drive (port) to load data from.
        bin_width (float, optional): The width of the bins [s]. Default is 0.01 (10ms) seconds. If set to `None`, no binning is applied and spike times are returned.
        
    Returns:
        tuple: A tuple containing:
            - spike_segment (dict): A dictionary where keys are unit labels and values are arrays of spike times (or binned spike counts) for that unit.
            - bins (numpy.ndarray or None): An array of bin edges if binning was applied, otherwise `None`.
    
    '''

    # Load data
    filename_mc = base.get_preprocessed_filename(subject, te_id, date, 'spike')
    spike_data = base.load_hdf_group(os.path.join(preproc_dir, subject), filename_mc, f'drive{drive}/spikes', cached=True)
    
    # Parse segment and bin spikes if necessary.
    spike_segment = {}
    for unit_label in list(spike_data.keys()):
        temp_spike_segment = spike_data[unit_label][np.logical_and(spike_data[unit_label]>=start_time, spike_data[unit_label]<=end_time)] - start_time
        if bin_width is None:
            spike_segment[unit_label] = temp_spike_segment + start_time
            bins = None
        else:
            spike_segment[unit_label], bins = precondition.bin_spike_times(temp_spike_segment, 0, end_time-start_time, bin_width)

    return spike_segment, bins

def get_spike_data_aligned(preproc_dir, subject, te_id, date, trigger_times, time_before, time_after, drive=1, bin_width=0.01):
    """
    Loads spike data for a given subject and experiment, then aligns binned spike to trigger times.

    .. image:: _images/spike_align_example.png

    Args:
        preproc_dir (str): Path to the preprocessed data directory.
        subject (str): Subject name.
        te_id (str): Task entry number.
        date (str): The date of the experiment.
        trigger_times (numpy.ndarray): 1D Array of trigger times (in seconds) for each trial to which spike data should be aligned.
        time_before (float): The amount of time (in seconds) before each trigger time to include in the aligned spike data.
        time_after (float): The amount of time (in seconds) after each trigger time to include in the aligned spike data.
        drive (int): The drive number corresponding to the spike data.
        bin_width (float, optional): The width of the bins [s]. Default is 0.01 (10ms) seconds. 

    Returns:
        tuple: A tuple containing:
            - spike_aligned (numpy.ndarray): A 3D array of aligned spike data with shape (ntime, nunits, ntrials), where:
                - ntime is the number of time bins between `time_before` and `time_after` around each trigger.
                - nch is the number of units.
                - ntrials is the number of trials (trigger events).
            - unit_labels (list of str): A list of unit labels corresponding to the 'nunits' dimension in the aligned spike data.
            - bins (numpy.ndarray): The time bin centers relative to the trigger times.
    """
    # Load data
    filename_mc = base.get_preprocessed_filename(subject, te_id, date, 'spike')
    spike_data = base.load_hdf_group(os.path.join(preproc_dir, subject), filename_mc, f'drive{drive}/spikes', cached=True)
    
    # Define relevant variables
    samplerate = int(np.round(1/bin_width))
    bins = np.arange(-time_before, time_after, bin_width) + bin_width/2
    ntime = int(np.round((time_after+time_before)/bin_width))
    nch = len(spike_data)
    ntrials = len(trigger_times)
    
    # Parse segment and bin spikes if necessary.
    unit_labels = list(spike_data.keys())
    spike_aligned = np.zeros((ntime, nch, ntrials))*np.nan #(ntime, nch, ntrials)
    for iunit, unit_label in enumerate(unit_labels):
        binned_spikes, _ = precondition.bin_spike_times(spike_data[unit_label], 0, trigger_times[-1]+time_after, bin_width)
        spike_aligned[:,iunit,:] = np.squeeze(preproc.base.trial_align_data(binned_spikes, trigger_times-(bin_width/2), time_before, time_after, samplerate)) # Squeeze to remove singleton dimension from only doing one unit at a time. Adjust trigger times to center on the previously binned spikes. 
        
    return spike_aligned, unit_labels, bins



@lru_cache(maxsize=1)
def _extract_lfp_features(preproc_dir, subject, te_id, date, decoder, samplerate=None, channels=None,
                         start_time=None, end_time=None, latency=0.02, datatype='lfp', preproc=None, 
                         decode=False, **kwargs):
    '''
    Extracts features from a BMI3D experiment using data aligned to the timestamps of the experiment.
    Using this function, you can replicate closely the features that would have been extracted from 
    a real-time BMI3D experiment, even if the experiment did not include a decoder.

    This private function has an optional paramter `decode` which allows you to run the features through
    the decoder before resampling. This is useful only for verifying that the features extracted offline
    are similar to those extracted online. Not to be used for any other purpose.

    Args:
        preproc_dir (str): base directory where the files live
        subject (str): Subject name
        te_id (int): Block number of Task entry object 
        date (str): Date of recording
        decoder (riglib.bmi.Decoder): decoder object with binlen and call_rate attributes
        samplerate (float, optional): optionally choose the samplerate of the data in Hz. Default None,
            uses the sampling rate of the experiment.
        channels (int array, optional): which channel indices to load. If None (the default), 
            uses the channels specified in the decoder.
        start_time (float, optional): time (in seconds) in the recording at which the desired segment starts
        end_time (float, optional): time (in seconds) in the recording at which the desired segment ends
        latency (float, optional): time (in seconds) to include before the trigger times
        datatype (str, optional): choice of 'lfp' or 'broadband' data to load. Defaults to 'lfp'. If
            the sampling rate of the data is different from the decoder, the data will be downsampled
            by decimation.
        preproc (fn, optional): function mapping (state, fs) data to (state_new, fs_new). For example,
            a smoothing function.
        decode (bool, optional): whether to run the features through the decoder before resampling. Only
            works if `channels` is None or `len(channels) == len(decoder.channels)`. Default False.
        kwargs: additional keyword arguments to pass to sample_timestamped_data 

    Returns:
        tuple: tuple containing:
            | **feats (nt, nfeats):** lfp features for the given channels after preprocessing
            | **samplerate (float):** the sampling rate of the states after preprocessing

    '''
    if start_time is None:
        start_time = 0.
    if end_time is None:
        ts_filename = base.get_preprocessed_filename(subject, te_id, date, datatype)
        preproc_dir_subject = os.path.join(preproc_dir, subject)
        ts_metadata = base.load_hdf_group(preproc_dir_subject, ts_filename, 
                                               f'{datatype}_metadata', cached=True)
        end_time = ts_metadata['n_samples']/ts_metadata['samplerate']

    # Set up extractor
    f_extractor = decoder.extractor_cls(None, **decoder.extractor_kwargs)
    if channels is None:
        channels = [c-1 for c in f_extractor.channels]
    lfp_samplerate = f_extractor.fs

    # Find times to extract
    exp_data, exp_metadata = base.load_preproc_exp_data(preproc_dir, subject, te_id, date)    
    step = int(decoder.call_rate * decoder.binlen)
    ts = exp_data['clock']['timestamp_sync'][::step]
    ts = ts[ts > start_time]
    if end_time is not None:
        ts = ts[ts < end_time]
    if len(ts) == 0:
        raise ValueError(f"No timestamps found in the specified time range ({start_time} to {end_time})")

    # Load ts data
    ts_start_time = start_time - f_extractor.win_len - latency
    ts_data, ts_samplerate = get_ts_data_segment(
        preproc_dir, subject, te_id, date, ts_start_time, 
        end_time, channels=channels, datatype=datatype
    )
    if len(ts_data) == 0:
        raise ValueError(f"No data found in the specified time range ({start_time} to {end_time})")
    if ts_samplerate != lfp_samplerate:
        downsample_factor = ts_samplerate // lfp_samplerate
        print(f"Downsampling by a factor of {downsample_factor}")
        ts_data = ts_data[::downsample_factor]
        ts_samplerate = lfp_samplerate
    
    # Extract
    n_pts = int(f_extractor.win_len * ts_samplerate)
    n_ch = len(channels)
    n_freq = 1
    if hasattr(f_extractor, 'bands'):
        n_freq = len(f_extractor.bands)
    cycle_data = np.zeros((len(ts), n_freq, n_ch))
    for i, t in enumerate(ts):
        sample_num = int((t-ts_start_time-latency) * ts_samplerate)
        cont_samples = ts_data[max(0,sample_num-n_pts):min(ts_data.shape[0], sample_num)]
        if cont_samples.shape[0] < n_pts:
            cycle_data[i] *= np.nan
        else:
            cycle_data[i] = f_extractor.extract_features(cont_samples.T).T.reshape(n_freq, n_ch)
    cycle_data = np.reshape(cycle_data, (len(ts), -1)) # (nt, nfeats)

    # Run the features through the decoder before resampling if necessary
    if decode and len(channels) == len(decoder.channels):
        cycle_data = decoder.decode(cycle_data.T)
    
    # Interpolate and preprocess
    if samplerate is None:
        samplerate = exp_metadata['fps']
    raw_data = sample_timestamped_data(cycle_data, ts, samplerate, append_time=10, **kwargs)
    if preproc is not None:
        data, samplerate = preproc(raw_data, samplerate)
    else:
        data = raw_data
    data = get_data_segment(data, start_time, end_time, samplerate) # cut off any extra data
    return data, samplerate

def extract_lfp_features(preproc_dir, subject, te_id, date, decoder, samplerate=None, channels=None,
                         start_time=None, end_time=None, latency=0.02, datatype='lfp', preproc=None, 
                         **kwargs):
    '''
    Extracts features from a BMI3D experiment using data aligned to the timestamps of the experiment.
    Using this function, you can replicate closely the features that would have been extracted from 
    a real-time BMI3D experiment, even if the experiment did not include a decoder.

    Args:
        preproc_dir (str): base directory where the files live
        subject (str): Subject name
        te_id (int): Block number of Task entry object 
        date (str): Date of recording
        decoder (riglib.bmi.Decoder): decoder object with binlen and call_rate attributes
        samplerate (float, optional): optionally choose the samplerate of the data in Hz. Default None,
            uses the sampling rate of the experiment.
        channels (int array, optional): which channel indices to load. If None (the default), 
            uses the channels specified in the decoder.
        start_time (float, optional): time (in seconds) in the recording at which the desired segment starts
        end_time (float, optional): time (in seconds) in the recording at which the desired segment ends
        latency (float, optional): time (in seconds) to include before the trigger times
        datatype (str, optional): choice of 'lfp' or 'broadband' data to load. Defaults to 'lfp'. If
            the sampling rate of the data is different from the decoder, the data will be downsampled
            by decimation.
        preproc (fn, optional): function mapping (state, fs) data to (state_new, fs_new). For example,
            a smoothing function.
        kwargs: additional keyword arguments to pass to sample_timestamped_data 

    Returns:
        tuple: tuple containing:
            | **feats (nt, nfeats):** lfp features for the given channels after preprocessing
            | **samplerate (float):** the sampling rate of the states after preprocessing

    Note:
        For best accuracy, use 'broadband' or other datatype without any filtering. Using filtered 'lfp'
        results in DC shifted features.

    Examples:

        .. code-block:: python

            subject = 'affi'
            te_id = 17269
            date = '2024-05-03'
            preproc_dir = data_dir
            start_time = 10
            end_time = 30

        Extract features using :func:`~aopy.data.bmi3d.extract_lfp_features` and states using 
        :func:`~aopy.data.bmi3d.extract_lfp_features` with `decode=True`:
        
        .. code-block:: python
            features_offline, samplerate_offline = extract_lfp_features(
                preproc_dir, subject, te_id, date, decoder, 
                start_time=start_time, end_time=end_time)
             
        Get online extracted features from :func:`~aopy.data.bmi3d.get_extracted_features` and
        states from :func:`~aopy.data.bmi3d.get_decoded_states` for comparison:

        .. code-block:: python

            features_online, samplerate_online = get_extracted_features(
                preproc_dir, subject, te_id, date, decoder,
                start_time=start_time, end_time=end_time)

        Plot the online and offline features:

        .. code-block:: python

            time_offline = np.arange(len(features_offline))/samplerate_offline + start_time
            time_online = np.arange(len(features_online))/samplerate_online + start_time

            plt.figure(figsize=(8,3))
            plt.plot(time_offline, features_offline[:,1], alpha=0.8, label='offline')
            plt.plot(time_online, features_online[:,1], alpha=0.8, label='online')
            plt.xlabel('time (s)')
            plt.ylabel('power')
            plt.legend()
            plt.title('readout 1')
                        
        .. image:: _images/extract_decoder_features.png
    '''
    return _extract_lfp_features(preproc_dir, subject, te_id, date, decoder, samplerate=samplerate, 
                                 channels=channels, start_time=start_time, end_time=end_time, latency=latency, 
                                 datatype=datatype, preproc=preproc, **kwargs)

def get_target_locations(preproc_dir, subject, te_id, date, target_indices):
    '''
    Loads the x,y,z location of targets in a preprocessed HDF file given by their index. Requires
    that the preprocessed `exp_data` includes a `trials` structured array containing `index` and 
    `target` fields (the default behavior of `:func:~aopy.preproc.proc_exp`)
    
    Args:
        preproc_dir (str): base directory where the files live
        subject (str): Subject name
        te_id (int): Block number of Task entry object 
        date (str): Date of recording
        target_indices (ntarg): a list of which targets to fetch
        
    Returns:
        ndarray: (ntarg x 3) array of coordinates of the given targets
    '''
    data, metadata = base.load_preproc_exp_data(preproc_dir, subject, te_id, date)
    try:
        trials = data['trials']
    except:
        trials = data['bmi3d_trials']
    locations = np.nan*np.zeros((len(target_indices), 3))
    for i in range(len(target_indices)):
        trial_idx = np.where(trials['index'] == target_indices[i])[0]
        if len(trial_idx) > 0:
            locations[i,:] = trials['target'][trial_idx[0]][[0,2,1]] # use x,y,z format
        else:
            raise ValueError(f"Target index {target_indices[i]} not found")
    return np.round(locations,4)

def get_trajectory_frequencies(preproc_dir, subject, te_id, date):
    '''
    For continuous tracking tasks, get the set of frequencies (in Hz) used to 
    generate the trajectories that were preesented on each trial of the experiment, 
    using :func:`~aopy.preproc.bmi3d.get_ref_dis_frequencies`. 

    Args:
        preproc_dir (str): base directory where the files live
        subject (str): Subject name
        te_id (int): Block number of Task entry object 
        date (str): Date of recording

    Returns:
        tuple: Tuple containing:
            | **freq_r (list of arrays):** (ntrial) list of (nfreq,) frequencies used to generate reference trajectory
            | **freq_d (list of arrays):** (ntrial) list of (nfreq,) frequencies used to generate disturbance trajectory
    '''
    data, metadata = base.load_preproc_exp_data(preproc_dir, subject, te_id, date)
    freq_r, freq_d = get_ref_dis_frequencies(data, metadata)
    return freq_r, freq_d

def get_source_files(preproc_dir, subject, te_id, date):
    '''
    Retrieves the dictionary of source files from a preprocessed file

    Args:
        preproc_dir (str): base directory where the files live
        subject (str): Subject name
        te_id (int): Block number of Task entry object 
        date (str): Date of recording

    Returns:
        tuple: tuple containing:
            | ** files (dict):** dictionary of (source, filepath) files that are associated with the given experiment
            | ** data_dir (str):** directory where the source files were located
    '''
    exp_data, exp_metadata = base.load_preproc_exp_data(preproc_dir, subject, te_id, date)
    return exp_metadata['source_files'], exp_metadata['source_dir']

def tabulate_behavior_data(preproc_dir, subjects, ids, dates, start_events, end_events, 
                           reward_events, penalty_events, metadata=[],
                           df=None, event_code_type='code', return_bad_entries=False, repeating_start_codes=False):
    '''
    Concatenate trials from across experiments. Experiments are given as lists of 
    subjects, task entry ids, and dates. Each list must be the same length. Trials 
    are defined by intervals between the given trial start and end codes. 

    Args:
        preproc_dir (str): base directory where the files live
        subjects (list of str): Subject name for each recording
        ids (list of int): Block number of Task entry object for each recording
        dates (list of str): Date for each recording
        start_events (list): list of numeric codes representing the start of a trial
        end_events (list): list of numeric codes representing the end of a trial
        reward_events (list): list of numeric codes representing rewards
        penalty_events (list): list of numeric codes representing penalties
        metadata (list, optional): list of metadata keys that should be included in the df
        df (DataFrame, optional): pandas DataFrame object to append. Defaults to None.
        event_code_type (str, optional): type of event codes to use. Defaults to 'code'. Other
            choices include 'event' and 'data'.
        return_bad_entries (bool, optional): If True, returns the list of task entries that could 
            not be loaded. Defaults to False.
        repeating_start_codes (bool): whether the start codes might occur multiple times within one segment. 
            Otherwise always use the last start code within a segment. May lead to segments spanning multiple 
            trials if used improperly. Defaults to False.


    Returns:
        pd.DataFrame: pandas DataFrame containing the concatenated trial data with columns:
            | **subject (str):** subject name
            | **te_id (str):** task entry id
            | **date (str):** date of recording
            | **event_codes (ntrial):** numeric code segments for each trial (specified by `event_code_type`)
            | **event_times (ntrial):** time segments for each trial
            | **event_idx (ntrial):** index segments for each trial
            | **reward (ntrial):** boolean values indicating whether each trial was rewarded
            | **penalty (ntrial):** boolean values indicating whether each trial was penalized
            | **%metadata_key% (ntrial):** requested metadata values for each key requested
    '''
    bad_entries = []
    if df is None:
        df = pd.DataFrame()

    entries = list(zip(subjects, dates, ids))
    for subject, date, te in tqdm(entries): 

        # Load data from bmi3d hdf 
        try:
            exp_data, exp_metadata = base.load_preproc_exp_data(preproc_dir, subject, te, date)
        except:
            print(f"Entry {subject} {date} {te} could not be loaded.")
            traceback.print_exc()
            bad_entries.append([subject,date,te])
            continue
        event_codes = exp_data['events'][event_code_type]
        event_times = exp_data['events']['timestamp']

        # Trial aligned event codes and event times
        tr_seg, tr_t, tr_idx = get_trial_segments_and_times(event_codes, event_times, start_events, end_events, 
                                                 repeating_start_codes, return_idx=True)
        reward = [np.any(np.isin(reward_events, ec)) for ec in tr_seg]
        penalty = [np.any(np.isin(penalty_events, ec)) for ec in tr_seg]
        
        # Build a dataframe for this task entry
        exp = {
            'subject': subject,
            'te_id': te, 
            'date': date, 
            'event_codes': tr_seg,
            'event_times': tr_t, 
            'event_idx': tr_idx,
            'reward': reward,
            'penalty': penalty,
        }

        # Add requested metadata
        for key in metadata:
            if key in exp_metadata:
                exp[key] = [exp_metadata[key] for _ in range(len(tr_seg))]
            else:
                exp[key] = None
                print(f"Entry {subject} {date} {te} does not have metadata {key}.")

        # Concatenate with existing dataframes
        df = pd.concat([df,pd.DataFrame(exp)], ignore_index=True)
    
    if return_bad_entries:
        return df, bad_entries
    else:
        return df

def tabulate_behavior_data_flash(preproc_dir, subjects, ids, dates, metadata=[], 
                                      df=None):
    '''
    Wrapper around tabulate_behavior_data() specifically for flash experiments. 
    Uses the task event names (b'TARGET_ON', b'REWARD', and b'TRIAL_END', specifically)
    to find start and end times for flash experiments.

    Args:
        preproc_dir (str): base directory where the files live
        subjects (list of str): Subject name for each recording
        ids (list of int): Block number of Task entry object for each recording
        dates (list of str): Date for each recording
        metadata (list, optional): list of metadata keys that should be included in the df
        df (DataFrame, optional): pandas DataFrame object to append. Defaults to None.

    Returns:
        pd.DataFrame: pandas DataFrame containing the concatenated trial data with columns:
            | **subject (str):** subject name
            | **te_id (str):** task entry id
            | **date (str):** date of recording
            | **event_names (ntrial):** event name segments for each trial
            | **event_times (ntrial):** time segments for each trial
            | **%metadata_key% (ntrial):** requested metadata values for each key requested
            | **flash_start_time (ntrial):** time the flash started
            | **flash_end_time (ntrial):** time the flash ended
'''
    # Use default "trial" definition
    trial_end_codes = [b'TRIAL_END']
    trial_start_codes = [b'TARGET_ON']
    reward_codes = [b'REWARD']
    penalty_codes = []

    # Concatenate base trial data
    new_df = tabulate_behavior_data(
        preproc_dir, subjects, ids, dates, trial_start_codes, trial_end_codes, 
        reward_codes, penalty_codes, metadata, df=None, event_code_type='event')
    if len(new_df) == 0:
        warnings.warn("No trials found")
        return df
    
    # Remove any unrewarded trials and then get rid of the 'reward' column
    new_df.drop(new_df.index[~new_df['reward']], inplace=True)
    new_df.drop(['reward', 'penalty'], axis=1, inplace=True)
    new_df.reset_index()
    
    # Add trial segment timing
    new_df['flash_start_time'] = np.nan
    new_df['flash_end_time'] = np.nan
    new_df['prev_trial_end_time'] = np.nan
    new_df['trial_end_time'] = np.nan
    for i in range(len(new_df)):
        event_times = new_df.loc[i, 'event_times']
        
        # Trial end times
        if i > 0 and new_df.loc[i-1, 'event_times'][-1] < event_times[0]:
            new_df.loc[i, 'prev_trial_end_time'] = new_df.loc[i-1, 'event_times'][-1]
        else:
            new_df.loc[i, 'prev_trial_end_time'] = 0.
        if i < len(new_df)-1 and new_df.loc[i+1, 'event_times'][0] > event_times[-1]:
            new_df.loc[i, 'trial_end_time'] = new_df.loc[i+1, 'event_times'][0]
        else:
            new_df.loc[i, 'trial_end_time'] = event_times[-1]

        # Flash starts when trial starts
        new_df.loc[i, 'flash_start_time'] = event_times[0]
        new_df.loc[i, 'flash_end_time'] = event_times[2]
            
    df = pd.concat([df, new_df], ignore_index=True)
    return df
    
def tabulate_behavior_data_center_out(preproc_dir, subjects, ids, dates, metadata=[], 
                                      df=None):
    '''
    Wrapper around tabulate_behavior_data() specifically for center-out experiments. 
    Makes use of the task codes saved in `/config/task_codes.yaml` to automatically 
    assign event codes for trial start, trial end, reward, penalty, and targets. 

    Args:
        preproc_dir (str): base directory where the files live
        subjects (list of str): Subject name for each recording
        ids (list of int): Block number of Task entry object for each recording
        dates (list of str): Date for each recording
        metadata (list, optional): list of metadata keys that should be included in the df
        df (DataFrame, optional): pandas DataFrame object to append. Defaults to None.

    Returns:
        pd.DataFrame: pandas DataFrame containing the concatenated trial data with columns:
            | **subject (str):** subject name
            | **te_id (str):** task entry id
            | **date (str):** date of recording
            | **event_codes (ntrial):** numeric code segments for each trial
            | **event_times (ntrial):** time segments for each trial
            | **reward (ntrial):** boolean values indicating whether each trial was rewarded
            | **penalty (ntrial):** boolean values indicating whether each trial was penalized
            | **%metadata_key% (ntrial):** requested metadata values for each key requested
            | **target_idx (ntrial):** index of the target that was presented
            | **target_location (ntrial):** location of the target that was presented
            | **center_target_on_time (ntrial):** time at which the trial started
            | **prev_trial_end_time (ntrial):** time at which the previous trial ended
            | **trial_end_time (ntrial):** time at which the trial ended
            | **trial_initiated (ntrial):** boolean values indicating whether the trial was initiated
            | **hold_start_time (ntrial):** time at which the hold period started
            | **hold_completed (ntrial):** boolean values indicating whether the hold period was completed
            | **delay_start_time (ntrial):** time at which the delay period started
            | **delay_completed (ntrial):** boolean values indicating whether the delay period was completed
            | **go_cue_time (ntrial):** time at which the go cue was presented
            | **reach_completed (ntrial):** boolean values indicating whether the reach was completed
            | **reach_end_time (ntrial):** time at which the reach was completed
            | **reward_start_time (ntrial):** time at which the reward was presented
            | **reward_end_time (ntrial):** time at which the reward was completed
            | **penalty_start_time (ntrial):** time at which the penalty was presented
            | **penalty_end_time (ntrial):** time at which the penalty was completed
            | **penalty_event (ntrial):** numeric code for the penalty event
    '''
    # Use default "trial" definition
    task_codes = load_bmi3d_task_codes()
    trial_end_codes = [task_codes['TRIAL_END']]
    trial_start_codes = [task_codes['CENTER_TARGET_ON']]
    reward_codes = [task_codes['REWARD']]
    penalty_codes = [task_codes['DELAY_PENALTY'], task_codes['HOLD_PENALTY'], task_codes['TIMEOUT_PENALTY']]
    target_codes = task_codes['PERIPHERAL_TARGET_ON']

    # Concatenate base trial data
    new_df = tabulate_behavior_data(
        preproc_dir, subjects, ids, dates, trial_start_codes, trial_end_codes, 
        reward_codes, penalty_codes, metadata, df=None)
    if len(new_df) == 0:
        warnings.warn("No trials found")
        return df

    # Add target info
    target_idx = [
        code[np.isin(code, target_codes)][0] - target_codes[0] + 1 # add 1 for center target
        if np.sum(np.isin(code, target_codes)) > 0 else 0 
        for code 
        in new_df['event_codes']
    ]
    target_location = [
        np.squeeze(get_target_locations(preproc_dir, s, te, d, [t_idx]))
        for s, te, d, t_idx 
        in zip(new_df['subject'], new_df['te_id'], new_df['date'], target_idx)
    ]
    new_df['target_idx'] = target_idx
    new_df['target_location'] = target_location

    # Add trial segment timing
    new_df['prev_trial_end_time'] = np.nan
    new_df['trial_end_time'] = np.nan
    new_df['center_target_on_time'] = np.nan
    new_df['trial_initiated'] = False
    new_df['hold_start_time'] = np.nan
    new_df['hold_completed'] = False
    new_df['delay_start_time'] = np.nan
    new_df['delay_completed'] = False
    new_df['go_cue_time'] = np.nan
    new_df['reach_completed'] = False
    new_df['reach_end_time'] = np.nan
    new_df['reward_start_time'] = np.nan
    new_df['penalty_start_time'] = np.nan
    new_df['penalty_event'] = np.nan
    for i in range(len(new_df)):
        event_codes = new_df.loc[i, 'event_codes']
        event_times = new_df.loc[i, 'event_times']

        # Trial end times
        if i > 0 and new_df.loc[i-1, 'event_times'][-1] < event_times[0]:
            new_df.loc[i, 'prev_trial_end_time'] = new_df.loc[i-1, 'event_times'][-1]
        else:
            new_df.loc[i, 'prev_trial_end_time'] = 0.
        new_df.loc[i, 'trial_end_time'] = event_times[-1]

        # Center target appears
        new_df.loc[i, 'center_target_on_time'] = event_times[0]
        
        # Trial initiated if cursor enters the center target
        hold_times = event_times[np.isin(event_codes, [task_codes['CURSOR_ENTER_CENTER_TARGET']])]
        new_df.loc[i, 'trial_initiated'] = len(hold_times) > 0
        if new_df.loc[i, 'trial_initiated']:
            new_df.loc[i, 'hold_start_time'] = hold_times[0]

        # Hold completed if peripheral target turns on (start of delay)
        delay_times = event_times[np.isin(event_codes, task_codes['PERIPHERAL_TARGET_ON'])]
        new_df.loc[i, 'hold_completed'] = len(delay_times) > 0
        if new_df.loc[i, 'hold_completed']:
            new_df.loc[i, 'delay_start_time'] = delay_times[0]

        # Delay completed when center target turns off (go cue)
        go_cue_times = event_times[np.isin(event_codes, task_codes['CENTER_TARGET_OFF'])]
        new_df.loc[i, 'delay_completed'] = len(go_cue_times) > 0
        if new_df.loc[i, 'delay_completed']:
            new_df.loc[i, 'go_cue_time'] = go_cue_times[0]

        # Reach completed if cursor enters target (regardless of whether the trial was successful)
        reach_times = event_times[np.isin(event_codes, task_codes['CURSOR_ENTER_PERIPHERAL_TARGET'])]
        new_df.loc[i, 'reach_completed'] = len(reach_times) > 0
        if new_df.loc[i, 'reach_completed']:
            new_df.loc[i, 'reach_end_time'] = reach_times[0]

        # Reward start times
        reward_times = event_times[np.isin(event_codes, task_codes['REWARD'])]      
        if len(reward_times) > 0:
            new_df.loc[i, 'reward_start_time'] = reward_times[0]

        # Penalty start times
        penalty_idx = np.isin(event_codes, penalty_codes)
        penatly_codes = event_codes[penalty_idx]
        penalty_times = event_times[penalty_idx]
        if len(penalty_times) > 0:
            new_df.loc[i, 'penalty_start_time'] = penalty_times[0]
            new_df.loc[i, 'penalty_event'] = penatly_codes[0]

    df = pd.concat([df, new_df], ignore_index=True)
    return df

def tabulate_behavior_data_out(preproc_dir, subjects, ids, dates, metadata=[], 
                               df=None):
    '''
    Wrapper around tabulate_behavior_data() specifically for out experiments (similar to
    center-out but without a trial-initiating center target). Makes use of the task codes 
    saved in `/config/task_codes.yaml` to automatically assign event codes for trial start, 
    trial end, reward, penalty, and targets. 

    Args:
        preproc_dir (str): base directory where the files live
        subjects (list of str): Subject name for each recording
        ids (list of int): Block number of Task entry object for each recording
        dates (list of str): Date for each recording
        metadata (list, optional): list of metadata keys that should be included in the df
        df (DataFrame, optional): pandas DataFrame object to append. Defaults to None.

    Returns:
        pd.DataFrame: pandas DataFrame containing the concatenated trial data with columns:
            | **subject (str):** subject name
            | **te_id (str):** task entry id
            | **date (str):** date of recording
            | **event_codes (ntrial):** numeric code segments for each trial
            | **event_times (ntrial):** time segments for each trial
            | **reward (ntrial):** boolean values indicating whether each trial was rewarded
            | **penalty (ntrial):** boolean values indicating whether each trial was penalized
            | **%metadata_key% (ntrial):** requested metadata values for each key requested
            | **target_idx (ntrial):** index of the target that was presented
            | **target_location (ntrial):** location of the target that was presented
            | **trial_start_time (ntrial):** time at which the trial started
            | **trial_end_time (ntrial):** time at which the trial ended
            | **reach_completed (ntrial):** boolean values indicating whether the reach was completed
            | **reach_end_time (ntrial):** time at which the reach was completed
            | **reward_start_time (ntrial):** time at which the reward was presented
            | **reward_end_time (ntrial):** time at which the reward was completed
            | **penalty_start_time (ntrial):** time at which the penalty was presented
            | **penalty_end_time (ntrial):** time at which the penalty was completed
            | **penalty_event (ntrial):** numeric code for the penalty event
    '''
    # Use default "trial" definition
    task_codes = load_bmi3d_task_codes()
    trial_end_codes = [task_codes['TRIAL_END']]
    trial_start_codes = task_codes['PERIPHERAL_TARGET_ON']
    reward_codes = [task_codes['REWARD']]
    penalty_codes = [task_codes['HOLD_PENALTY'], task_codes['TIMEOUT_PENALTY']]
    target_codes = task_codes['PERIPHERAL_TARGET_ON']

    # Concatenate base trial data
    new_df = tabulate_behavior_data(
        preproc_dir, subjects, ids, dates, trial_start_codes, trial_end_codes, 
        reward_codes, penalty_codes, metadata, df=None)
    if len(new_df) == 0:
        warnings.warn("No trials found")
        return df

    # Add target info
    target_idx = [
        code[np.isin(code, target_codes)][0] - target_codes[0] + 1 # add 1 for center target
        if np.sum(np.isin(code, target_codes)) > 0 else 0 
        for code 
        in new_df['event_codes']
    ]
    target_location = [
        np.squeeze(get_target_locations(preproc_dir, s, te, d, [t_idx]))
        for s, te, d, t_idx 
        in zip(new_df['subject'], new_df['te_id'], new_df['date'], target_idx)
    ]
    new_df['target_idx'] = target_idx
    new_df['target_location'] = target_location

    # Add trial segment timing
    new_df['prev_trial_end_time'] = np.nan
    new_df['trial_end_time'] = np.nan
    new_df['target_on_time'] = np.nan
    new_df['reach_completed'] = False
    new_df['reach_end_time'] = np.nan
    new_df['reward_start_time'] = np.nan
    new_df['reward_end_time'] = np.nan
    new_df['penalty_start_time'] = np.nan
    new_df['penalty_end_time'] = np.nan
    new_df['penalty_event'] = np.nan
    for i in range(len(new_df)):
        event_codes = new_df.loc[i, 'event_codes']
        event_times = new_df.loc[i, 'event_times']

        # Trial end times
        if i > 0 and new_df.loc[i-1, 'event_times'][-1] < event_times[0]:
            new_df.loc[i, 'prev_trial_end_time'] = new_df.loc[i-1, 'event_times'][-1]
        else:
            new_df.loc[i, 'prev_trial_end_time'] = 0.
        new_df.loc[i, 'trial_end_time'] = event_times[-1]

        # Target appears
        new_df.loc[i, 'target_on_time'] = event_times[0]

        # Reach completed if cursor enters target (regardless of whether the trial was successful)
        reach_times = event_times[np.isin(event_codes, task_codes['CURSOR_ENTER_PERIPHERAL_TARGET'])]
        new_df.loc[i, 'reach_completed'] = len(reach_times) > 0
        if new_df.loc[i, 'reach_completed']:
            new_df.loc[i, 'reach_end_time'] = reach_times[0]

        # Reward start times
        reward_times = event_times[np.isin(event_codes, task_codes['REWARD'])]      
        if len(reward_times) > 0:
            new_df.loc[i, 'reward_start_time'] = reward_times[0]

        # Penalty start times
        penalty_idx = np.isin(event_codes, penalty_codes)
        penatly_codes = event_codes[penalty_idx]
        penalty_times = event_times[penalty_idx]
        if len(penalty_times) > 0:
            new_df.loc[i, 'penalty_start_time'] = penalty_times[0]
            new_df.loc[i, 'penalty_event'] = penatly_codes[0]

    df = pd.concat([df, new_df], ignore_index=True)
    return df

def tabulate_behavior_data_corners(preproc_dir, subjects, ids, dates, metadata=[], 
                               df=None):
    '''
    Wrapper around tabulate_behavior_data() specifically for corner reaching experiments. 
    Makes use of the task codes saved in `/config/task_codes.yaml` to automatically 
    assign event codes for trial start, trial end, reward, penalty, and targets. 

    Args:
        preproc_dir (str): base directory where the files live
        subjects (list of str): Subject name for each recording
        ids (list of int): Block number of Task entry object for each recording
        dates (list of str): Date for each recording
        metadata (list, optional): list of metadata keys that should be included in the df
        df (DataFrame, optional): pandas DataFrame object to append. Defaults to None.

    Returns:
        pd.DataFrame: pandas DataFrame containing the concatenated trial data with columns:
            | **subject (str):** subject name
            | **te_id (str):** task entry id
            | **date (str):** date of recording
            | **event_codes (ntrial):** numeric code segments for each trial
            | **event_times (ntrial):** time segments for each trial
            | **event_idx (ntrial):** index segments for each trial
            | **reward (ntrial):** boolean values indicating whether each trial was rewarded
            | **penalty (ntrial):** boolean values indicating whether each trial was penalized
            | **%metadata_key% (ntrial):** requested metadata values for each key requested
            | **sequence_params (ntrial):** string of params used to generate all trajectories in the same task entry
            | **chain_length (ntrial):** number of targets presented in each trial
            | **target_idx (ntrial):** list of indices of the targets presented
            | **target_location (ntrial):** list of locations of the targets presented
            | **prev_trial_end_time (ntrial):** time at which the previous trial ended
            | **trial_end_time (ntrial):** time at which the trial ended
            | **first_target_on_time (ntrial):** time at which the trial started
            | **trial_initiated (ntrial):** boolean values indicating whether the trial was initiated
            | **hold_start_time (ntrial):** time at which the hold period started
            | **hold_completed (ntrial):** boolean values indicating whether the hold period was completed
            | **delay_start_time (ntrial):** time at which the delay period started
            | **delay_completed (ntrial):** boolean values indicating whether the delay period was completed
            | **go_cue_time (ntrial):** time at which the go cue was presented
            | **reach_completed (ntrial):** boolean values indicating whether the reach was completed
            | **reach_end_time (ntrial):** time at which the reach was completed
            | **reward_start_time (ntrial):** time at which the reward was presented
            | **penalty_start_time (ntrial):** time at which the penalty occurred
            | **penalty_event (ntrial):** numeric code for the penalty event
            | **pause_start_time (ntrial):** time at which the pause occurred
            | **pause_event (ntrial):** numeric code for the pause event

    Example:

        .. code-block:: python
        
            subject = 'churro'
            start_date = '2025-01-17'
            end_date = '2025-01-18'
            entries = db.lookup_mc_sessions(subject=subject, date=(date.fromisoformat(start_date), date.fromisoformat(end_date)))
            subjects, te_ids, te_dates = db.list_entry_details(entries)

            df = tabulate_behavior_data_corners(preproc_dir, subjects, te_ids, te_dates)
            display(df.head(8))

        .. image:: _images/tabulate_behavior_data_corners.png
    '''
    # Use default "trial" definition
    task_codes = load_bmi3d_task_codes()
    trial_end_codes = [task_codes['TRIAL_END'], task_codes['PAUSE_START'], task_codes['PAUSE']]
    trial_start_codes = task_codes['CORNER_TARGET_ON']
    reward_codes = [task_codes['REWARD']]
    penalty_codes = [task_codes['DELAY_PENALTY'], task_codes['HOLD_PENALTY'], task_codes['TIMEOUT_PENALTY']]
    target_codes = task_codes['CORNER_TARGET_ON']
    pause_codes = [task_codes['PAUSE_START'], task_codes['PAUSE_END'], task_codes['PAUSE']]

    # Concatenate base trial data
    if 'sequence_params' not in metadata:
        metadata.append('sequence_params')
    new_df = tabulate_behavior_data(
        preproc_dir, subjects, ids, dates, trial_start_codes, trial_end_codes, 
        reward_codes, penalty_codes, metadata, df=None, repeating_start_codes=True)
    if len(new_df) == 0:
        warnings.warn("No trials found")
        return df

    # Add target info
    chain_length = [
        json.loads(params)['chain_length']
        if 'chain_length' in json.loads(params) else 0
        for params
        in new_df['sequence_params']
    ]
    target_idx = [
        code[np.isin(code, target_codes)] - target_codes[0] + 1 # add 1 for center target, which doesn't exist in this task
        if np.sum(np.isin(code, target_codes)) > 0 else []
        for code 
        in new_df['event_codes']
    ]
    target_location = [
        np.squeeze(get_target_locations(preproc_dir, s, te, d, t_idx))
        for s, te, d, t_idx 
        in zip(new_df['subject'], new_df['te_id'], new_df['date'], target_idx)
    ]
    new_df['chain_length'] = chain_length
    new_df['target_idx'] = target_idx
    new_df['target_location'] = target_location

    # Add trial segment timing
    new_df['prev_trial_end_time'] = np.nan
    new_df['trial_end_time'] = np.nan
    new_df['first_target_on_time'] = np.nan
    new_df['trial_initiated'] = False
    new_df['hold_start_time'] = np.nan # aka first target enter time
    new_df['hold_completed'] = False
    new_df['delay_start_time'] = np.nan # aka second target on time
    new_df['delay_completed'] = False
    new_df['go_cue_time'] = np.nan # aka first target off time
    new_df['reach_completed'] = False
    new_df['reach_end_time'] = np.nan # aka second target enter time
    new_df['reward_start_time'] = np.nan
    new_df['penalty_start_time'] = np.nan
    new_df['penalty_event'] = np.nan
    new_df['pause_start_time'] = np.nan
    new_df['pause_event'] = np.nan
    for i in range(len(new_df)):
        event_codes = new_df.loc[i, 'event_codes']
        event_times = new_df.loc[i, 'event_times']

        # Trial end times
        if i > 0 and new_df.loc[i-1, 'event_times'][-1] < event_times[0]:
            new_df.loc[i, 'prev_trial_end_time'] = new_df.loc[i-1, 'event_times'][-1]
        else:
            new_df.loc[i, 'prev_trial_end_time'] = 0.
        new_df.loc[i, 'trial_end_time'] = event_times[-1]

        # First corner target appears
        new_df.loc[i, 'first_target_on_time'] = event_times[0]

        # Trial initiated if cursor enters the first corner target
        hold_times = event_times[np.isin(event_codes, [task_codes['CURSOR_ENTER_CORNER_TARGET']])] # this list may be as long as the number of corner targets in the chain
        new_df.loc[i, 'trial_initiated'] = len(hold_times) > 0
        if new_df.loc[i, 'trial_initiated']:
            new_df.loc[i, 'hold_start_time'] = hold_times[0] # entering the first corner target is the start of hold

        # Hold completed if second corner target turns on (start of delay)
        delay_times = event_times[np.isin(event_codes, task_codes['CORNER_TARGET_ON'])] # this list may be as long as the number of corner targets in the chain
        new_df.loc[i, 'hold_completed'] = len(delay_times) > 1
        if new_df.loc[i, 'hold_completed']:
            new_df.loc[i, 'delay_start_time'] = delay_times[1] # second corner target on is the start of delay

        # Delay completed when first corner target turns off (go cue)
        go_cue_times = event_times[np.isin(event_codes, task_codes['CORNER_TARGET_OFF'])] # this list may be as long as one less than the number of corner tagets in the chain
        new_df.loc[i, 'delay_completed'] = len(go_cue_times) > 0
        if new_df.loc[i, 'delay_completed']:
            new_df.loc[i, 'go_cue_time'] = go_cue_times[0]

        # Reach completed if cursor enters second corner target (regardless of whether the trial was successful)
        reach_times = event_times[np.isin(event_codes, task_codes['CURSOR_ENTER_CORNER_TARGET'])] # this list may be as long as the number of corner targets in the chain
        new_df.loc[i, 'reach_completed'] = len(reach_times) > 1
        if new_df.loc[i, 'reach_completed']:
            new_df.loc[i, 'reach_end_time'] = reach_times[1]

        # Reward start times
        reward_times = event_times[np.isin(event_codes, task_codes['REWARD'])]      
        if len(reward_times) > 0:
            new_df.loc[i, 'reward_start_time'] = reward_times[0]

        # Penalty start times
        penalty_idx = np.isin(event_codes, penalty_codes)
        penalty_events = event_codes[penalty_idx]
        penalty_times = event_times[penalty_idx]
        if len(penalty_times) > 0:
            new_df.loc[i, 'penalty_start_time'] = penalty_times[0]
            new_df.loc[i, 'penalty_event'] = penalty_events[0]

        # Pause events
        pause_idx = np.isin(event_codes, pause_codes)
        pause_events = event_codes[pause_idx]
        pause_times = event_times[pause_idx]
        if len(pause_times) > 0:
            new_df.loc[i, 'pause_start_time'] = pause_times[0]
            new_df.loc[i, 'pause_event'] = pause_events[0]

    df = pd.concat([df, new_df], ignore_index=True)
    return df

def tabulate_behavior_data_tracking_task(preproc_dir, subjects, ids, dates, metadata=[], df=None):
    '''
    Wrapper around tabulate_behavior_data() specifically for tracking task experiments. 
    Makes use of the task codes saved in `/config/task_codes.yaml` to automatically 
    assign event codes for trial start, trial end, reward, penalty.
    
    Args:
        preproc_dir (str): base directory where the files live
        subjects (list of str): Subject name for each recording
        ids (list of int): Block number of Task entry object for each recording
        dates (list of str): Date for each recording
        metadata (list, optional): list of metadata keys that should be included in the df
        df (DataFrame, optional): pandas DataFrame object to append. Defaults to None.

    Returns:
        pd.DataFrame: pandas DataFrame containing the concatenated trial data with columns:
            | **subject (str):** subject name
            | **te_id (str):** task entry id
            | **date (str):** date of recording
            | **event_codes (ntrial):** numeric code segments for each trial
            | **event_times (ntrial):** time segments for each trial
            | **event_idx (ntrial):** index segments for each trial            
            | **reward (ntrial):** boolean values indicating whether each trial was rewarded
            | **penalty (ntrial):** boolean values indicating whether each trial was penalized
            | **%metadata_key% (ntrial):** requested metadata values for each key requested
            | **sequence_params (ntrial):** string of params used to generate all trajectories in the same task entry
            | **ref_freqs (ntrial):** array of frequencies used to generate reference trajectory for each trial
            | **dis_freqs (ntrial):** array of frequencies used to generate disturbance trajectory for each trial
            | **prev_trial_end_time (ntrial):** time at which the previous trial ended
            | **target_on_time (ntrial):** time at which the trial started
            | **trial_initiated (ntrial):** boolean values indicating whether the trial was initiated (i.e. hold was attempted)
            | **hold_start_time (ntrial):** time at which the hold period started
            | **hold_completed (ntrial):** boolean values indicating whether the hold period was completed
            | **tracking_start_time (ntrial):** time at which the hold period ended and tracking started
            | **trajectory_start_time (ntrial):** time at which the ref & dis trajectories started (excluding the ramp up period)
            | **trajectory_end_time (ntrial):** time at which the ref & dis trajectories ended (excluding the ramp down period if the trial was rewarded)
            | **tracking_end_time (ntrial):** time at which tracking ended (whether with a reward or tracking out penalty)
            | **reward_start_time (ntrial):** time at which the reward was presented
            | **penalty_start_time (ntrial):** time at which the penalty occurred
            | **penalty_event (ntrial):** numeric code for the penalty event
            | **pause_start_time (ntrial):** time at which the pause occurred
            | **pause_event (ntrial):** numeric code for the pause event
            | **trial_end_time (ntrial):** time at which the trial ended

    Example:

        .. code-block:: python
            subject = 'churro'
            start_date = '2025-03-03'
            end_date = '2025-03-13'
            entries = db.lookup_tracking_sessions(subject=subject, date=(date.fromisoformat(start_date), date.fromisoformat(end_date)))
            subjects, te_ids, te_dates = db.list_entry_details(entries)

            df = tabulate_behavior_data_tracking_task(preproc_dir, subjects, te_ids, te_dates)
            display(df.head(8))

        .. image:: _images/tabulate_behavior_data_tracking_task.png
    '''
    # Use default "trial" definition
    task_codes = load_bmi3d_task_codes()
    trial_start_codes = [task_codes['CENTER_TARGET_ON']]
    trial_end_codes = [task_codes['TRIAL_END'], task_codes['PAUSE_START'], task_codes['PAUSE']]
    reward_codes = [task_codes['REWARD']]
    penalty_codes = [task_codes['TIMEOUT_PENALTY'], task_codes['HOLD_PENALTY'], task_codes['OTHER_PENALTY']]
    pause_codes = [task_codes['PAUSE_START'], task_codes['PAUSE_END'], task_codes['PAUSE']]
    
    # Concatenate base trial data
    if 'sequence_params' not in metadata:
        metadata.append('sequence_params')
    new_df, bad_entries = tabulate_behavior_data(
        preproc_dir, subjects, ids, dates, trial_start_codes, trial_end_codes, 
        reward_codes, penalty_codes, metadata, df=None, return_bad_entries=True)
    
    # Add frequency content of reference & disturbance trajectories
    ref_freqs = []
    dis_freqs = []
    for subject, te, date in zip(subjects, ids, dates):
        if [subject,date,te] in bad_entries:
            continue
        r, d = get_trajectory_frequencies(preproc_dir, subject, te, date)
        ref_freqs.extend(r)
        dis_freqs.extend(d)
    new_df['ref_freqs'] = ref_freqs
    new_df['dis_freqs'] = dis_freqs
    
    # Get ramp lengths
    ramp = [
        json.loads(params)['ramp']
        if 'ramp' in json.loads(params) else 0
        for params
        in new_df['sequence_params']
    ]
    ramp_down = [
        json.loads(params)['ramp_down']
        if 'ramp_down' in json.loads(params) else 0
        for params
        in new_df['sequence_params']
    ]

    # Add trial segment timing
    new_df['prev_trial_end_time'] = np.nan
    new_df['target_on_time'] = np.nan
    new_df['trial_initiated'] = False
    new_df['hold_start_time'] = np.nan
    new_df['hold_completed'] = False
    new_df['tracking_start_time'] = np.nan
    new_df['trajectory_start_time'] = np.nan
    new_df['trajectory_end_time'] = np.nan
    new_df['tracking_end_time'] = np.nan
    new_df['reward_start_time'] = np.nan
    new_df['penalty_start_time'] = np.nan
    new_df['penalty_event'] = np.nan
    new_df['pause_start_time'] = np.nan
    new_df['pause_event'] = np.nan
    new_df['trial_end_time'] = np.nan
    for i in range(len(new_df)):
        event_codes = new_df.iloc[i]['event_codes']
        event_times = new_df.iloc[i]['event_times']

        # Trial end times
        if i > 0 and new_df.loc[i-1, 'event_times'][-1] < event_times[0]:
            new_df.loc[i, 'prev_trial_end_time'] = new_df.loc[i-1, 'event_times'][-1]
        else:
            new_df.loc[i, 'prev_trial_end_time'] = 0.
        new_df.loc[i, 'trial_end_time'] = event_times[-1]

        # Trial start time (center target appears)
        new_df.loc[i, 'target_on_time'] = event_times[0]

        # Trial initiated when hold begins
        initiation_times = event_times[np.isin(event_codes, [task_codes['TRIAL_START']])]
        new_df.loc[i, 'trial_initiated'] = len(initiation_times) > 0 # if False, TIMEOUT_PENALTY
        if new_df.loc[i, 'trial_initiated']:
            new_df.loc[i, 'hold_start_time'] = initiation_times[0]

        # Hold completed when tracking begins (first time cursor enters target)
        tracking_start_times = event_times[np.isin(event_codes, [task_codes['CURSOR_ENTER_TARGET_RAMP_UP'],     # first occurrence is beginning of ramp up
                                                                 task_codes['CURSOR_ENTER_TARGET']])]           # if there's no ramp up, first occurrence is beginning of trajectory
        new_df.loc[i, 'hold_completed'] = len(tracking_start_times) > 0 # if False, HOLD_PENALTY
        if new_df.loc[i, 'hold_completed']:
            # Tracking begins
            new_df.loc[i, 'tracking_start_time'] = tracking_start_times[0]

            # Tracking ends in one of these ways: reward, tracking out penalty, experimenter pause
            tracking_end_times = event_times[np.isin(event_codes, [task_codes['REWARD'], task_codes['OTHER_PENALTY'], task_codes['PAUSE_START'], task_codes['PAUSE']])]
            new_df.loc[i, 'tracking_end_time'] = tracking_end_times[0]

            # Trajectory-tracking is a specific portion of tracking that excludes any ramp periods 
            # (earlier versions of the task did not use ramp-specific event codes, these were included in later versions for more accurate segmenting)
            ramp_up_events = event_codes[np.isin(event_codes, [task_codes['CURSOR_ENTER_TARGET_RAMP_UP']])]
            ramp_down_events = event_codes[np.isin(event_codes, [task_codes['CURSOR_ENTER_TARGET_RAMP_DOWN'], task_codes['CURSOR_LEAVE_TARGET_RAMP_DOWN']])]

            # EARLIER VERSION HANDLING: Trajectory begins after the ramp up period at the start of tracking (as long as no penalty or pause interrupted ramp up)
            if ramp[i] > 0 and len(ramp_up_events) < 1:
                if new_df.loc[i, 'tracking_start_time'] + ramp[i] < new_df.loc[i, 'tracking_end_time']:
                    new_df.loc[i, 'trajectory_start_time'] = new_df.loc[i, 'tracking_start_time'] + ramp[i]

            # LATER VERSION HANDLING: Trajectory begins the first time cursor interacts with target in a non-ramp state
            else:
                trajectory_start_times = event_times[np.isin(event_codes, [task_codes['CURSOR_ENTER_TARGET'],       # transition from ramp up to trajectory occurred while cursor was tracking in OR there's no ramp up
                                                                           task_codes['CURSOR_LEAVE_TARGET']])]     # transition from ramp up to trajectory occurred while cursor was tracking out
                if len(trajectory_start_times) > 0:
                    new_df.loc[i, 'trajectory_start_time'] = trajectory_start_times[0]

            # Trajectory ends in one of these ways: first occurrence of a ramp down event, reward, tracking out penalty, experimenter pause
            trajectory_end_times = event_times[np.isin(event_codes, [task_codes['CURSOR_ENTER_TARGET_RAMP_DOWN'],   # transition from trajectory to ramp down occurred while cursor was tracking in
                                                                     task_codes['CURSOR_LEAVE_TARGET_RAMP_DOWN'],   # transition from trajectory to ramp down occurred while cursor was tracking out
                                                                     task_codes['REWARD'],
                                                                     task_codes['OTHER_PENALTY'],
                                                                     task_codes['PAUSE_START'], task_codes['PAUSE']])]
            if ~np.isnan(new_df.loc[i, 'trajectory_start_time']) and len(trajectory_end_times) > 0:
                new_df.loc[i, 'trajectory_end_time'] = trajectory_end_times[0]

            # EARLIER VERSION HANDLING: Trajectory excludes ramp down period at end of tracking if trial was successful
            if new_df.loc[i, 'reward'] and ramp_down[i] > 0 and len(ramp_down_events) < 1:
                new_df.loc[i, 'trajectory_end_time'] = new_df.loc[i, 'tracking_end_time'] - ramp_down[i]

        # Reward start times
        reward_times = event_times[np.isin(event_codes, task_codes['REWARD'])]      
        if len(reward_times) > 0:
            new_df.loc[i, 'reward_start_time'] = reward_times[0]

        # Penalty start times
        penalty_idx = np.isin(event_codes, penalty_codes)
        penalty_events = event_codes[penalty_idx]
        penalty_times = event_times[penalty_idx]
        if len(penalty_times) > 0:
            new_df.loc[i, 'penalty_start_time'] = penalty_times[0]
            new_df.loc[i, 'penalty_event'] = penalty_events[0]

        # Pause events
        pause_idx = np.isin(event_codes, pause_codes)
        pause_events = event_codes[pause_idx]
        pause_times = event_times[pause_idx]
        if len(pause_times) > 0:
            new_df.loc[i, 'pause_start_time'] = pause_times[0]
            new_df.loc[i, 'pause_event'] = pause_events[0]

    df = pd.concat([df, new_df], ignore_index=True)
    return df

def tabulate_stim_data(preproc_dir, subjects, ids, dates, metadata=['stimulation_site'], 
                       debug=True, df=None, **kwargs):
    '''
    Concatenate stimulation data from across experiments. Experiments are given as lists of 
    subjects, task entry ids, and dates. Each list must be the same length. 
    
    Args:
        preproc_dir (str): base directory where the files live
        subjects (list of str): Subject name for each recording
        ids (list of int): Block number of Task entry object for each recording
        dates (list of str): Date for each recording
        metadata (list, optional): list of metadata keys that should be included in the df. By default,
            only `stimulation_site` is included.
        debug (bool, optional): Passed to :func:`~aopy.preproc.laser.find_stim_times`, if True prints
            an laser sensor alignment plot for each trial. Defaults to True.
        df (DataFrame, optional): pandas DataFrame object to append. Defaults to None.
        kwargs (dict, optional): optional keyword arguments to pass to :func:`~aopy.preproc.laser.find_stim_times`

    Returns:
        pd.DataFrame: pandas DataFrame containing the concatenated trial data
            | **subject (str):** subject name
            | **te_id (str):** task entry id
            | **date (str):** date of stimulation
            | **stimulation_site (int):** site of stimulation
            | **%metadata_key% (ntrial):** requested metadata values for each key requested
            | **trial_time (float):** time of stimulation within recording
            | **trial_width (float):** width of stimulation pulse
            | **trial_gain (float):** fraction of maximum laser power setting
            | **trial_power (float):** power (in mW) of stimulation pulse at the fiber output

    Note:
        Only supports single-site stimulation.
    '''
    if df is None:
        df = pd.DataFrame()

    entries = list(zip(subjects, dates, ids))
    for subject, date, te in tqdm(entries): 

        # Load data from bmi3d hdf 
        try:
            exp_data, exp_metadata = base.load_preproc_exp_data(preproc_dir, subject, te, date)
        except:
            print(f"Entry {subject} {date} {te} could not be loaded.")
            traceback.print_exc()
            continue

        # Find laser trial times 
        if 'laser_trigger' in kwargs and 'laser_sensor' in kwargs:
            laser_triggers = [kwargs.pop('laser_trigger')]
            laser_sensors = [kwargs.pop('laser_sensor')]
            stim_sites = ['stimulation_site']
        else:
            lasers = load_bmi3d_lasers()
            possible_stim_sites = [laser['stimulation_site'] for laser in lasers]
            possible_triggers = [laser['trigger'] for laser in lasers]
            possible_sensors = [laser['sensor'] for laser in lasers]
            idx = np.array([n in exp_metadata.keys() and exp_metadata[n] != '' for n in possible_stim_sites])
            laser_triggers = np.array(possible_triggers)[idx]
            laser_sensors = np.array(possible_sensors)[idx]
            stim_sites = np.array(possible_stim_sites)[idx]

        print('laser_triggers:', laser_triggers)
        print('laser_sensors:', laser_sensors)
            
        for stim_idx in range(len(laser_triggers)):
            try:
                print('laser_trigger:', laser_triggers[stim_idx])
                print('laser_sensor:', laser_sensors[stim_idx])

                trial_times, trial_widths, trial_gains, trial_powers = preproc.bmi3d.get_laser_trial_times(
                    preproc_dir, subject, te, date, debug=debug, laser_trigger=laser_triggers[stim_idx], 
                    laser_sensor=laser_sensors[stim_idx], **kwargs)
            except:
                print(f"Problem extracting stimulation trials from entry {subject} {date} {te}")
                traceback.print_exc()
                continue

            # Tabulate everything together
            exp = {
                'subject': subject,
                'te_id': te, 
                'date': date, 
                'trial_time': trial_times,
                'trial_width': trial_widths, 
                'trial_gain': trial_gains,
                'trial_power': trial_powers,
            }

            # Add requested metadata
            for key in metadata:
                if key == 'stimulation_site' and 'qwalor_switch_rdy_dch' in exp_metadata:

                    # Switched laser with multiple stim sites
                    exp['stimulation_site'] = preproc.bmi3d.get_switched_stimulation_sites(
                        preproc_dir, subject, te, date, trial_times, debug=debug
                    )

                elif key == 'stimulation_site' and len(laser_triggers) > 1:
                    exp['stimulation_site'] = exp_metadata[stim_sites[stim_idx]]

                elif key in exp_metadata:
                    exp[key] = [exp_metadata[key] for _ in range(len(trial_times))]
                else:
                    exp[key] = None
                    print(f"Entry {subject} {date} {te} does not have metadata {key}.")


            # Concatenate with existing dataframe
            df = pd.concat([df,pd.DataFrame(exp)], ignore_index=True)
    
    return df

def tabulate_poisson_trial_times(preproc_dir, subjects, ids, dates, metadata=[], 
                                 poisson_mu=0.25, refractory_period=0.1, df=None):
    '''
    Generate poisson-spaced trial times for the given recordings. Recordings are given as 
    lists of subjects, task entry ids, and dates. Each list must be the same length. See 
    :func:`~aopy.preproc.utils.generate_poisson_timestamps` for more information on the 
    poisson-spaced trial times that are generated.
    
    Args:
        preproc_dir (str): base directory where the files live
        subjects (list of str): Subject name for each recording
        ids (list of int): Block number of Task entry object for each recording
        dates (list of str): Date for each recording
        metadata (list, optional): list of metadata keys that should be included in the df.
            By default empty.
        poisson_mu (float, optional): mean of the inter-trial times in seconds. Default 0.25.
        refractory_period (float, optional): minimum time between trials in seconds. Default 0.1.
        df (DataFrame, optional): pandas DataFrame object to append. Defaults to None.

    Returns:
        pd.DataFrame: pandas DataFrame containing the concatenated trial data
            | **subject (str):** subject name
            | **te_id (str):** task entry id
            | **date (str):** date of each trial
            | **%metadata_key% (ntrial):** requested metadata values for each key requested
            | **trial_time (float):** time generated within recording
    '''
    if df is None:
        df = pd.DataFrame()

    entries = list(zip(subjects, dates, ids))
    for subject, date, te in tqdm(entries): 

        # Load data from bmi3d hdf 
        try:
            exp_data, exp_metadata = base.load_preproc_exp_data(preproc_dir, subject, te, date)
        except:
            print(f"Entry {subject} {date} {te} could not be loaded.")
            traceback.print_exc()
            continue

        # Generate trial times 
        try:
            min_time = exp_data['clock']['timestamp_sync'][0]
            max_time = exp_data['clock']['timestamp_sync'][-1]
            trial_times = utils.generate_poisson_timestamps(poisson_mu, max_time, min_time, refractory_period)
        except:
            print(f"Problem extracting stimulation trials from entry {subject} {date} {te}")
            traceback.print_exc()
            continue
                        
        # Tabulate everything together
        exp = {
            'subject': subject,
            'te_id': te, 
            'date': date, 
            'trial_time': trial_times,
        }

        # Add requested metadata
        for key in metadata:
            if key in exp_metadata:
                exp[key] = [exp_metadata[key] for _ in range(len(trial_times))]
            else:
                exp[key] = None
                print(f"Entry {subject} {date} {te} does not have metadata {key}.")

        # Concatenate with existing dataframe
        df = pd.concat([df,pd.DataFrame(exp)], ignore_index=True)
    
    return df

def tabulate_kinematic_data(preproc_dir, subjects, te_ids, dates, start_times, end_times, 
                            samplerate=1000, preproc=None, datatype='cursor', **kwargs):
    '''
    Grab kinematics data from trials across arbitrary preprocessed files.

    Args:
        preproc_dir (str): base directory where the files live
        subjects (list of str): Subject name for each recording
        ids (list of int): Block number of Task entry object for each recording
        dates (list of str): Date for each recording
        start_times (list of float): times in the recording at which the desired segments starts
        end_times (list of float): times in the recording at which the desired segments ends
        samplerate (float, optional): optionally choose the samplerate of the data in Hz. Default 1000.
        preproc (fn, optional): function mapping (position, fs) data to (kinematics, fs_new). For example,
            a smoothing function or an estimate of velocity from position
        datatype (str, optional): type of kinematics to tabulate. Defaults to 'cursor'.  
        kwargs (dict, optional): optional keyword arguments to pass to :func:`~aopy.preproc.get_kinematic_segment`  

    Returns:
        (ntrial,): list of tensors of (nt, nch) kinematics from each trial
    '''

    assert len(subjects) == len(te_ids) == len(dates) == len(start_times) == len(end_times)
    
    segments = [_get_kinematic_segment(preproc_dir, s, t, d, ts, te, samplerate, preproc, datatype, **kwargs)[0] 
                for s, t, d, ts, te in zip(subjects, te_ids, dates, start_times, end_times)]
    trajectories = np.array(segments, dtype='object')
    return trajectories

def tabulate_task_data(preproc_dir, subjects, te_ids, dates, start_times, end_times, 
                       datatype, samplerate=None, steps=1, preproc=None, **kwargs):
    '''
    Grab task data from trials across arbitrary preprocessed files.

    Args:
        preproc_dir (str): base directory where the files live
        subjects (list of str): Subject name for each recording
        ids (list of int): Block number of Task entry object for each recording
        dates (list of str): Date for each recording
        start_times (list of float): times in the recording at which the desired segments starts
        end_times (list of float): times in the recording at which the desired segments ends
        datatype (str): column of task data to load. 
        samplerate (float, optional): choose the samplerate of the data in Hz. Default None,
            which uses the sampling rate of the experiment.
        steps (list of int, optional): task data will be decimated with steps this big. If a single
            integer is given, it will be applied to all trials. Default 1.
        preproc (fn, optional): function mapping (position, fs) data to (kinematics, fs_new). For example,
            a smoothing function or an estimate of velocity from position
        kwargs: additional keyword arguments to pass to get_interp_task_data 
    Returns:
        tuple: tuple containing:
            | **segments (ntrial,):** list of tensors of (nt, nch) task data from each trial
            | **samplerate (float):** samplerate of the task data
    '''

    assert len(subjects) == len(te_ids) == len(dates) == len(start_times) == len(end_times)
    try:
        if len(steps) == len(subjects):
            pass
    except:
        if steps is None:
            steps = 1
        steps = [steps for _ in range(len(subjects))]

    segments = []
    for s, t, d, ts, te, step in zip(subjects, te_ids, dates, start_times, end_times, steps):
        task_data, samplerate = get_task_data(
            preproc_dir, s, t, d, datatype, samplerate=samplerate, step=step,
            preproc=preproc, **kwargs) # cached for each unique recording
        segments.append(get_data_segment(task_data, ts, te, samplerate))

    segments = np.array(segments, dtype='object')
    return segments, samplerate

def tabulate_lfp_features(preproc_dir, subjects, te_ids, dates, start_times, end_times,
                          decoders, samplerate=None, channels=None, datatype='lfp', 
                          preproc=None, **kwargs):
    '''
    Extract (new, offline) lfp feature segments across arbitrary preprocessed files. Uses
    a decoder object to extract features from either lfp or broadband timeseries data. Can
    be applied offline to arbitrary channels. If used on broadband data from a BCI experiment,
    the features extracted will be (nearly) the same as the online features if the same decoder
    is used.

    Args:
        preproc_dir (str): base directory where the files live
        subjects (list of str): Subject name for each recording
        ids (list of int): Block number of Task entry object for each recording
        dates (list of str): Date for each recording
        datatype (str, optional): column of task data to load. Default 'lfp_power'.
        start_times (list of float): times in the recording at which the desired segments starts
        end_times (list of float): times in the recording at which the desired segments ends
        decoders (list of riglib.bmi.Decoder): decoder objects for each recording. If only one decoder
            is supplied, it will be applied to all recordings.
        samplerate (float, optional): choose the samplerate of the data in Hz. Default None,
            which uses the sampling rate of the experiment.
        channels (list of int, optional): list of channel indices to extract. Default None, which
            extracts all channels.
        datatype (str, optional): type of data to load. Default 'lfp'.
        preproc (fn, optional): function mapping (position, fs) data to (kinematics, fs_new). For example,
            a smoothing function or an estimate of velocity from position
        decode (bool, optional): whether to decode the lfp features. Default False.
        kwargs: additional keyword arguments 

    Returns:
        tuple: tuple containing:
            | **segments (ntrial,):** list of tensors of (nt, nfeat) feature data from each trial
            | **samplerate (float):** samplerate of the feature data
    
    Examples:

        Plot online extracted lfp features and overlay offline extracted feature segments

        .. code-block:: python  

            subject = 'affi'
            te_id = 17269
            date = '2024-05-03'
            subjects = [subject, subject, subject]
            te_ids = [te_id, te_id, te_id]
            dates = [date, date, date]
            start_time = 10
            end_time = 30
            start_times = [10, 15, 20]
            end_times = [14, 18, 28]

        Load the decoder that was used in the experiment

        .. code-block:: python

            with open(os.path.join(data_dir, 'test_decoder.pkl'), 'rb') as file:
            decoder = pickle.load(file)

        Load the full features for comparison

        .. code-block:: python

            features_offline, samplerate_offline = extract_lfp_features(
                preproc_dir, subject, te_id, date, decoder, 
                start_time=start_time, end_time=end_time)
            features_online, samplerate_online = get_extracted_features(
                preproc_dir, subject, te_id, date, decoder,
                start_time=start_time, end_time=end_time)

            time_offline = np.arange(len(features_offline))/samplerate_offline + start_time
            time_online = np.arange(len(features_online))/samplerate_online + start_time

            plt.figure(figsize=(8,3))
            plt.plot(time_offline, features_offline[:,1], alpha=0.8, label='offline')
            plt.plot(time_online, features_online[:,1], alpha=0.8, label='online')
            plt.xlabel('time (s)')
            plt.ylabel('power')
            plt.title('readout 1')

        Tabulate the segments

        .. code-block:: python

            features_offline, samplerate_offline = tabulate_lfp_features(
                preproc_dir, subjects, te_ids, dates, start_times, end_times, decoder)
            features_online, samplerate_online = tabulate_feature_data(
                preproc_dir, subjects, te_ids, dates, start_times, end_times, decoder)

            for idx in range(len(start_times)):
                time_offline = np.arange(len(features_offline[idx]))/samplerate_offline + start_times[idx]
                time_online = np.arange(len(features_online[idx]))/samplerate_online + start_times[idx]
                plt.plot(time_offline, features_offline[idx][:,1], 'k--')
                plt.plot(time_online, features_online[idx][:,1], 'k--')
            
        Add legend

        .. code-block:: python

            plt.plot([], [], 'k--', label='segments')
            plt.legend()

        .. image:: _images/tabulate_lfp_features.png
            
    See also:
        :func:`~aopy.data.bmi3d.tabulate_feature_data`
        :func:`~aopy.data.bmi3d.tabulet_state_data`
    '''
    assert len(subjects) == len(te_ids) == len(dates) == len(start_times) == len(end_times)
    try:
        if len(decoders) == len(subjects):
            pass
    except:
        decoders = [decoders for _ in range(len(subjects))]

    segments = []
    for s, t, d, ts, te, dec in zip(subjects, te_ids, dates, start_times, end_times, decoders):
        lfp_features, samplerate = extract_lfp_features(
            preproc_dir, s, t, d, dec, samplerate=samplerate, channels=channels,
            start_time=ts, end_time=te, datatype=datatype, preproc=preproc, 
            **kwargs
        )
        segments.append(lfp_features)

    return segments, samplerate

def tabulate_feature_data(preproc_dir, subjects, te_ids, dates, start_times, end_times, decoders,
                          datatype='lfp_power', samplerate=None, preproc=None, **kwargs):
    '''
    Grab (online extracted) decoder feature segments across arbitrary preprocessed files. Wrapper 
    around tabulate_task_data.

    Args:
        preproc_dir (str): base directory where the files live
        subjects (list of str): Subject name for each recording
        ids (list of int): Block number of Task entry object for each recording
        dates (list of str): Date for each recording
        datatype (str, optional): column of task data to load. Default 'lfp_power'.
        samplerate (float, optional): choose the samplerate of the data in Hz. Default None,
            which uses the sampling rate of the experiment.
        start_times (list of float): times in the recording at which the desired segments starts
        end_times (list of float): times in the recording at which the desired segments ends
        decoders (list of riglib.bmi.Decoder): decoder object with binlen and call_rate attributes. If
            only one decoder is supplied, it will be applied to all recordings.
        preproc (fn, optional): function mapping (position, fs) data to (kinematics, fs_new). For example,
            a smoothing function or an estimate of velocity from position
        kwargs: additional keyword arguments to pass to get_interp_task_data 
    
    Returns:
        tuple: tuple containing:
            | **segments (ntrial,):** list of tensors of (nt, nfeat) feature data from each trial
            | **samplerate (float):** samplerate of the feature data
    '''
    try:
        if len(decoders) == len(subjects):
            steps = [int(decoder.call_rate*decoder.binlen) for decoder in decoders]
    except:
        steps = int(decoders.call_rate*decoders.binlen)
    return tabulate_task_data(preproc_dir, subjects, te_ids, dates, start_times, end_times, 
                              datatype, samplerate=samplerate, steps=steps, preproc=preproc, **kwargs)
    
def tabulate_state_data(preproc_dir, subjects, te_ids, dates, start_times, end_times, decoders,
                        datatype='decoder_state', samplerate=None, preproc=None, **kwargs):
    '''
    Grab (online decoded) state segments across arbitrary preprocessed files. Wrapper around 
    tabulate_task_data.

    Args:
        preproc_dir (str): base directory where the files live
        subjects (list of str): Subject name for each recording
        ids (list of int): Block number of Task entry object for each recording
        dates (list of str): Date for each recording
        datatype (str, optional): column of task data to load. Default 'decoder_state'.
        samplerate (float, optional): choose the samplerate of the data in Hz. Default None,
            which uses the sampling rate of the experiment.
        start_times (list of float): times in the recording at which the desired segments starts
        end_times (list of float): times in the recording at which the desired segments ends
        decoders (list of riglib.bmi.Decoder): decoder object with binlen and call_rate attributes. If
            only one decoder is supplied, it will be applied to all recordings.
        preproc (fn, optional): function mapping (position, fs) data to (kinematics, fs_new). For example,
            a smoothing function or an estimate of velocity from position
        kwargs: additional keyword arguments to pass to get_interp_task_data 

    Returns:
        tuple: tuple containing:
            | **segments (ntrial,):** list of tensors of (nt, nfeat) state data from each trial
            | **samplerate (float):** samplerate of the state data
    '''
    try:
        if len(decoders) == len(subjects):
            steps = [int(decoder.call_rate*decoder.binlen) for decoder in decoders]
    except:
        steps = int(decoders.call_rate*decoders.binlen)
    return tabulate_task_data(preproc_dir, subjects, te_ids, dates, start_times, end_times, 
                              datatype, samplerate=samplerate, steps=steps, preproc=preproc, **kwargs)

def tabulate_ts_data(preproc_dir, subjects, te_ids, dates, trigger_times, time_before, time_after, 
                     channels=None, datatype='lfp'):
    '''
    Grab rectangular timeseries data from trials across arbitrary preprocessed files.
    
    Args:
        preproc_dir (str): base directory where the files live
        subjects (list of str): Subject name for each recording
        ids (list of int): Block number of Task entry object for each recording
        dates (list of str): Date for each recording
        trigger_times (list of float): times in the recording at which the desired trials start
        time_before (float): time (in seconds) to include before the trigger times
        time_after (float): time (in seconds) to include after the trigger times
        channels (list of int, optional): list of channel indices to include. Defaults to None.
        datatype (str, optional): choice of 'lfp' or 'broadband' data to load. Defaults to 'lfp'.    
        
    Returns:
        tuple: tuple containing:
            | **data (nt, nch, ntr):** tensor of data from each channel and trial
            | **samplerate (float):** sampling rate of the data
    '''

    assert len(subjects) == len(te_ids) == len(dates) == len(trigger_times)
    
    # Get the first trial
    segment_1, samplerate = get_ts_data_trial(
        preproc_dir, subjects[0], te_ids[0], dates[0], trigger_times[0], 
        time_before, time_after, channels=channels, datatype=datatype
    )
        
    # Construct the tensor using the first trial as a template
    if segment_1.ndim == 1:
        segment_1 = np.expand_dims(segment_1, 1)
    nt, nch = segment_1.shape
    segments = np.zeros((nt, nch, len(trigger_times)), like=segment_1)
    segments[:,:,0] = segment_1
    
    # Add the remaining trial
    idx = 1
    for s, t, d, tr in list(zip(subjects, te_ids, dates, trigger_times))[1:]:
        segments[:,:,idx] = get_ts_data_trial(preproc_dir, s, t, d, tr, 
                                              time_before, time_after, 
                                              channels=channels, datatype=datatype)[0]
        idx += 1
        
    return segments, samplerate

# Also, tabulate_ts_data has some errors in the docstring!
def tabulate_ts_segments(preproc_dir, subjects, te_ids, dates, start_times, end_times,
                         channels=None, datatype='lfp'):
    '''
    Grab nonrectangular timeseries data from trials across arbitrary preprocessed files.
    
    Args:
        preproc_dir (str): base directory where the files live
        subjects (list of str): Subject name for each recording
        ids (list of int): Block number of Task entry object for each recording
        dates (list of str): Date for each recording
        start_times (list of float): times in the recording at which the desired segments start
        end_times (list of float): times in the recording at which the desired segments end
        channels (list of int, optional): list of channel indices to include. Defaults to None.
        datatype (str, optional): choice of 'lfp' or 'broadband' data to load. Defaults to 'lfp'.    
        
    Returns:
        tuple: tuple containing:
            | **data (list of (nt, nch)):** list of data segments
            | **samplerate (float):** sampling rate of the data
    '''

    assert len(subjects) == len(te_ids) == len(dates) == len(start_times) == len(end_times)
    
    # Fetch the segments
    segments = []
    for s, t, d, st, et in list(zip(subjects, te_ids, dates, start_times, end_times)):
        segment, samplerate = get_ts_data_segment(preproc_dir, s, t, d, st, et, 
                                                  channels=channels, datatype=datatype)
        segments.append(segment)
        
    return segments, samplerate

def tabulate_spike_data_segments(preproc_dir, subjects, te_ids, dates, start_times, end_times, drives, bin_width=0.01):
    '''
    Grab nonrectangular timeseries data from trials across arbitrary preprocessed files.

    Args:
        preproc_dir (str): base directory where the files live
        subjects (list of str): Subject name for each recording
        ids (list of int): Block number of Task entry object for each recording
        dates (list of str): Date for each recording
        start_times (list of float): times in the recording at which the desired segments start
        end_times (list of float): times in the recording at which the desired segments end
        drives (list): Defines which drive to load data from. For neuropixel data this is usually '1' or '2'
        bin_width (int): Bin width to bin spike times at. If None, the segments of spike times will be returned. 

    Returns:
            tuple: A tuple containing:
                - segments (list of dicts): A list where each element is a dictionary of spike data for a unit in a specific experiment.
                - bins (numpy.ndarray or None): An array of bin edges if binning was applied, otherwise `None`.
    '''
    assert len(subjects) == len(te_ids) == len(dates) == len(start_times) == len(end_times) == len(drives)
    
    # Fetch the segments
    segments = []
    for s, t, d, dr, st, et in list(zip(subjects, te_ids, dates, drives, start_times, end_times)):
        segment, bins = get_spike_data_segment(preproc_dir, s, t, d, st, et, dr, bin_width=bin_width)
        segments.append(segment)
        
    return segments, bins

def load_bmi3d_task_codes(filename='task_codes.yaml'):
    '''
    Load the default BMI3D task codes. File-specific codes can be found in exp_metadata['event_sync_dict']

    Args:
        filename (str, optional): filename of the task codes to load. Defaults to 'task_codes.yaml'.

    Returns:
        dict: (name, code) task code dictionary
    '''
    return base.load_yaml_config(filename)[0]

def load_bmi3d_lasers(filename='lasers.yaml'):
    '''
    Load the config metadata for BMI3D lasers.

    Args:
        filename (str, optional): filename of the laser names to load. Defaults to 'laser_names.yaml'.

    Returns:
        list: list of lasers available in the config. Each laser is a dictionary with keys
            - name: name of the laser
            - stimulation_site: name of the metadata key for the stimulation site
            - trigger: name of the metadata key for the trigger channel
            - trigger_dch: index of the trigger digital channel
            - sensor: name of the metadata key for the sensor channel
            - sensor_ach: index of the sensor analog channel
    '''
    return base.load_yaml_config(filename)