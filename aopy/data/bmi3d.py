import sys
from functools import lru_cache
import traceback
import warnings
from matplotlib import pyplot as plt
import os

import numpy as np
import h5py
import tables
import pandas as pd
import json
from tqdm.auto import tqdm
from scipy import interpolate
if sys.version_info >= (3,9):
    from importlib.resources import files, as_file
else:
    from importlib_resources import files, as_file

from .. import precondition
from .. import preproc
from ..preproc.base import get_data_segment, get_data_segments, get_trial_segments, get_trial_segments_and_times, interp_timestamps2timeseries, sample_timestamped_data, trial_align_data
from ..preproc.bmi3d import get_target_events
from ..whitematter import ChunkedStream, Dataset
from ..utils import derivative, get_pulse_edge_times, compute_pulse_duty_cycles, convert_digital_to_channels, detect_edges
from .base import load_preproc_exp_data, load_preproc_eye_data, load_preproc_lfp_data, yaml_read, get_preprocessed_filename, load_hdf_data, load_hdf_ts_segment

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
        chunk_len = chunk.shape[0]
        timeseries_data[n_read:n_read+chunk_len,:] = chunk
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
def get_interp_kinematics(exp_data, exp_metadata, datatype='cursor', samplerate=1000):
    '''
    Gets interpolated and filtered kinematic data from preprocessed experiment 
    data to the desired sampling rate. Cursor kinematics are returned in 
    screen coordinates, while other kinematics are returned in their original
    coordinate system (e.g. hand kinematics in optitrack coordinates).

    Examples:
        
        Cursor kinematics

        .. code-block:: python
        
            exp_data, exp_metadata = load_preproc_exp_data(preproc_dir, 'test',  3498, '2021-12-13')
            cursor_interp = get_interp_kinematics(exp_data, exp_metadata, datatype='cursor', samplerate=100)

            plt.figure()
            visualization.plot_trajectories([cursor_interp], [-10, 10, -10, 10])
        
        .. image:: _images/get_interp_cursor_centerout.png

        Hand kinematics
       
        .. code-block:: python

            hand_interp = get_interp_kinematics(exp_data, exp_metadata, datatype='hand', samplerate=100)
            ax = plt.axes(projection='3d')
            visualization.plot_trajectories([hand_interp], [-10, 10, -10, 10, -10, 10])

        .. image:: _images/get_interp_hand_centerout.png

        Target positions

        .. code-block:: python
            
            targets_interp = get_interp_kinematics(exp_data, exp_metadata, datatype='targets', samplerate=100)
            time = np.arange(len(targets_interp))/100
            plt.plot(time, targets_interp[:,:,0]) # plot just the x coordinate
            plt.xlim(10, 20)
            plt.xlabel('time (s)')
            plt.ylabel('x position (cm)')

        .. image:: _images/get_interp_targets_centerout.png

        Cursor and target (reference) kinematics

        .. code-block:: python
            
            exp_data, exp_metadata = load_preproc_exp_data(data_dir, 'test', 8461, '2023-02-25')
            cursor_interp = get_interp_kinematics(exp_data, exp_metadata, datatype='cursor', samplerate=exp_metadata['fps'])
            ref_interp = get_interp_kinematics(exp_data, exp_metadata, datatype='reference', samplerate=exp_metadata['fps'])
            time = np.arange(exp_metadata['fps']*120)/exp_metadata['fps']
            plt.plot(time, cursor_interp[:int(exp_metadata['fps']*120),1], color='blueviolet', label='cursor') # plot just the y coordinate
            plt.plot(time, ref_interp[:int(exp_metadata['fps']*120),1], color='darkorange', label='ref')
            plt.xlabel('time (s)')
            plt.ylabel('y position (cm)'); plt.ylim(-10,10)
            plt.legend()

        .. image:: _images/get_interp_cursor_tracking.png

        User, reference, and disturbance kinematics

        .. code-block:: python
            
            user_interp = get_interp_kinematics(exp_data, exp_metadata, datatype='user', samplerate=exp_metadata['fps'])
            ref_interp = get_interp_kinematics(exp_data, exp_metadata, datatype='reference', samplerate=exp_metadata['fps'])
            dis_interp = get_interp_kinematics(exp_data, exp_metadata, datatype='disturbance', samplerate=exp_metadata['fps'])
            time = np.arange(exp_metadata['fps']*120)/exp_metadata['fps']
            plt.plot(time, user_interp[:int(exp_metadata['fps']*120),1], color='darkturquoise', label='user')
            plt.plot(time, ref_interp[:int(exp_metadata['fps']*120),1], color='darkorange', label='ref')
            plt.plot(time, dis_interp[:int(exp_metadata['fps']*120),1], color='tab:red', linestyle='--', label='dis')
            plt.xlabel('time (s)')
            plt.ylabel('y position (cm)'); plt.ylim(-10,10)
            plt.legend()

        .. image:: _images/get_interp_user_tracking.png

    Args:
        exp_data (dict): A dictionary containing the experiment data.
        exp_metadata (dict): A dictionary containing the experiment metadata.
        datatype (str, optional): The type of kinematic data to interpolate. 
            For 'hand' kinematics, interp the 'clean_hand_position' experiment data
            For 'cursor' kinematics, interp the x and z position of the 'cursor' task data
            For other kinematics, try to interp exp_data['task'][datatype]
        samplerate (float, optional): The desired output sampling rate in Hz. 
            Defaults to 1000.

    Returns:
        data_time (ns, ...): Kinematic data interpolated and filtered 
            to the desired sampling rate.
    '''
    kwargs = {}

    # Fetch the available timestamps
    try:
        clock = exp_data['clock']['timestamp_sync']
    except:
        clock = exp_data['clock']['timestamp_bmi3d']

    # Fetch the relevant BMI3D data
    if datatype == 'hand':
        data_cycles = exp_data['clean_hand_position'] # 3d hand position (optitrack coords: x,y,z) on each bmi3d cycle
    elif datatype == 'cursor':
        data_cycles = exp_data['task']['cursor'][:,[0,2]] # 2d cursor position (bmi3d coords: x,z) on each bmi3d cycle
    elif datatype == 'user':
        dis_on = int(json.loads(exp_metadata['sequence_params'])['disturbance']) # whether disturbance was turned on (0 or 1)
        data_cycles = exp_data['task']['cursor'][:,[0,2]] - exp_data['task']['current_disturbance'][:,[0,2]]*dis_on # 1d cursor position before disturbance added (bmi3d coords: z)
    elif datatype == 'reference':
        data_cycles =  exp_data['task']['current_target'][:,[0,2]] # 1d target position (bmi3d coords: z)
    elif datatype == 'disturbance':
        dis_on = int(json.loads(exp_metadata['sequence_params'])['disturbance']) # whether disturbance was turned on (0 or 1)
        data_cycles = exp_data['task']['current_disturbance'][:,[0,2]]*dis_on # 1d disturbance value (bmi3d coords: z)
    elif datatype == 'targets':
        data_cycles = get_target_events(exp_data, exp_metadata)
        clock = exp_data['events']['timestamp']
        kwargs['remove_nan'] = False # In this case we need to keep NaN values.
    elif datatype in exp_data['task'].dtype.names:
        data_cycles = exp_data['task'][datatype]
    else:
        raise ValueError(f"Unknown datatype {datatype}")
    
    # Interpolate
    data_time = sample_timestamped_data(data_cycles, clock, samplerate, 
                                        upsamplerate=10000, append_time=10, **kwargs)
    if 'remove_nan' not in kwargs:
        data_time = precondition.filter_kinematics(data_time, samplerate)
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
def get_kinematics(preproc_dir, subject, te_id, date, samplerate, preproc=None, datatype='cursor'):
    '''
    Return all kinds of kinematics from preprocessed data

    Args:
        preproc_dir (str): base directory where the files live
        subject (str): Subject name
        te_id (int): Block number of Task entry object 
        date (str): Date of recording
        samplerate (float, optional): optionally choose the samplerate of the data in Hz. Default 1000.
        preproc (fn, optional): function mapping (position, fs) data to (kinematics, fs_new). For example,
            a smoothing function or an estimate of velocity from position
        datatype (str, optional): type of kinematics to load. Defaults to 'cursor'.    

    Raises:
        ValueError: if the datatype is invalid

    Returns:
        tuple: tuple containing:
            | **kinematics (nt, nch):** kinematics from the given experiment after preprocessing
            | **samplerate (float):** the sampling rate of the kinematics after preprocessing
    '''
    data, metadata = load_preproc_exp_data(preproc_dir, subject, te_id, date)

    if 'eye' in datatype:
        eye_data, eye_metadata = load_preproc_eye_data(preproc_dir, subject, te_id, date)
        if datatype == 'eye_raw':
            eye_data = eye_data['raw_data']
        elif 'calibrated_data' in eye_data.keys():
            eye_data = eye_data['calibrated_data']
        else:
            raise ValueError(f"No calibrated eye data for {te_id}")
        
        time = np.arange(len(eye_data))/eye_metadata['samplerate']
        raw_kinematics, _ = interp_timestamps2timeseries(time, eye_data, samplerate)
    else:
        raw_kinematics = get_interp_kinematics(
            data, metadata, datatype, samplerate=samplerate
        )

    time = np.arange(len(raw_kinematics))/samplerate
    if preproc is not None:
        kinematics, samplerate = preproc(raw_kinematics, samplerate)
    else:
        kinematics = raw_kinematics

    return kinematics, samplerate

def get_kinematic_segment(preproc_dir, subject, te_id, date, start_time, end_time, samplerate, preproc=None, datatype='cursor'):
    '''
    Return one segment of kinematics

    Args:
        preproc_dir (str): base directory where the files live
        subject (str): Subject name
        te_id (int): Block number of Task entry object 
        date (str): Date of recording
        start_time (float): time in the recording at which the desired segment starts
        end_time (float): time in the recording at which the desired segment ends
        samplerate (float, optional): optionally choose the samplerate of the data in Hz. Default 1000.
        preproc (fn, optional): function mapping (position, fs) data to (kinematics, fs_new). For example,
            a smoothing function or an estimate of velocity from position
        datatype (str, optional): type of kinematics to load. Defaults to 'cursor'.    

    Returns:
        tuple: tuple containing:
            | **segment (nt, nch):** single kinematic segment from the given experiment after preprocessing
            | **samplerate (float):** the sampling rate of the kinematics after preprocessing
    '''
    kinematics, samplerate = get_kinematics(preproc_dir, subject, te_id, date, samplerate, preproc, datatype)
    assert kinematics is not None

    return get_data_segment(kinematics, start_time, end_time, samplerate), samplerate

def get_kinematic_segments(preproc_dir, subject, te_id, date, trial_start_codes, trial_end_codes, 
                           trial_filter=lambda x:True, preproc=None, datatype='cursor',
                           samplerate=1000):
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
    
    Returns:
        tuple: tuple containing:
            | **trajectories (ntrial):** array of filtered cursor trajectories for each trial
            | **trial_segments (ntrial):** array of numeric code segments for each trial   

    Note:
        The sampling rate of the returned data might be different from the requested sampling rate if the
        preprocessing function does any modification to the length of the data.

    Modified September 2023 to include optional sampling rate argument     
    '''
    data, metadata = load_preproc_exp_data(preproc_dir, subject, te_id, date)
    event_codes = data['events']['code']
    event_times = data['events']['timestamp']
    trial_segments, trial_times = get_trial_segments(event_codes, event_times, 
                                                                  trial_start_codes, trial_end_codes)
    segments = [
        get_kinematic_segment(preproc_dir, subject, te_id, date, t[0], t[1], samplerate, preproc, datatype)[0] 
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
    data, metadata = load_preproc_exp_data(preproc_dir, subject, te_id, date)
    lfp_data, lfp_metadata = load_preproc_lfp_data(preproc_dir, subject, te_id, date)
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
    data, metadata = load_preproc_exp_data(preproc_dir, subject, te_id, date)
    lfp_data, lfp_metadata = load_preproc_lfp_data(preproc_dir, subject, te_id, date)
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

def get_ts_data_segment(preproc_dir, subject, te_id, date, trigger_time, time_before, time_after,
                       datatype='lfp'):
    '''
    Simple wrapper around get_tsdata_segment for lfp or broadband data.
    
    Args:
        preproc_dir (str): base directory where the files live
        subject (str): Subject name
        te_id (int): Block number of Task entry object 
        date (str): Date of recording
        trigger_time (float): time (in seconds) in the recording at which the desired segment starts
        time_before (float): time (in seconds) to include before the trigger times
        time_after (float): time (in seconds) to include after the trigger times
        datatype (str, optional): choice of 'lfp' or 'broadband' data to load. Defaults to 'lfp'.    

    Returns:
        tuple: tuple containing:
            | **segment (nt, nch):** data segment from the given preprocessed file
            | **samplerate (float):** sampling rate of the returned data
    '''
    if datatype == 'lfp':
        data_group='/'
        data_name='lfp_data'
        metadata_group='lfp_metadata'
        samplerate_key='lfp_samplerate'
    elif datatype == 'broadband':
        data_group='/'
        data_name='broadband_data'
        metadata_group='broadband_metadata'
        samplerate_key='samplerate'
    filename = get_preprocessed_filename(subject, te_id, date, datatype)
    preproc_dir = os.path.join(preproc_dir, subject)

    try:
        samplerate = load_hdf_data(preproc_dir, filename, samplerate_key, metadata_group)
        data = load_hdf_ts_segment(preproc_dir, filename, data_group, data_name, 
                                    samplerate, trigger_time, time_before, time_after)
    except FileNotFoundError as e:
        print(f"No data found in {preproc_dir} for subject {subject} on {date} ({te_id})")
        raise e

    return data, samplerate
    
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
    data, metadata = load_preproc_exp_data(preproc_dir, subject, te_id, date)
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
    exp_data, exp_metadata = load_preproc_exp_data(preproc_dir, subject, te_id, date)
    return exp_metadata['source_files'], exp_metadata['source_dir']

def tabulate_behavior_data(preproc_dir, subjects, ids, dates, trial_start_codes, 
                           trial_end_codes, reward_codes, penalty_codes, metadata=[],
                           df=None):
    '''
    Concatenate trials from across experiments. Experiments are given as lists of 
    subjects, task entry ids, and dates. Each list must be the same length. Trials 
    are defined by intervals between the given trial start and end codes. 

    Args:
        preproc_dir (str): base directory where the files live
        subjects (list of str): Subject name for each recording
        ids (list of int): Block number of Task entry object for each recording
        dates (list of str): Date for each recording
        trial_start_codes (list): list of numeric codes representing the start of a trial
        trial_end_codes (list): list of numeric codes representing the end of a trial
        reward_codes (list): list of numeric codes representing rewards
        penalty_codes (list): list of numeric codes representing penalties
        metadata (list, optional): list of metadata keys that should be included in the df
        df (DataFrame, optional): pandas DataFrame object to append. Defaults to None.

    Returns:
        pd.DataFrame: pandas DataFrame containing the concatenated trial data with columns:
            | **subject (str):** subject name
            | **te_id (str):** task entry id
            | **date (str):** date of stimulation
            | **event_codes (ntrial):** numeric code segments for each trial
            | **event_times (ntrial):** time segments for each trial
            | **reward (ntrial):** boolean values indicating whether each trial was rewarded
            | **penalty (ntrial):** boolean values indicating whether each trial was penalized
            | **%metadata_key% (ntrial):** requested metadata values for each key requested
    '''
    if df is None:
        df = pd.DataFrame()

    entries = list(zip(subjects, dates, ids))
    for subject, date, te in tqdm(entries): 

        # Load data from bmi3d hdf 
        try:
            exp_data, exp_metadata = load_preproc_exp_data(preproc_dir, subject, te, date)
        except:
            print(f"Entry {subject} {date} {te} could not be loaded.")
            traceback.print_exc()
            continue
        event_codes = exp_data['events']['code']
        event_times = exp_data['events']['timestamp']

        # Trial aligned event codes and event times
        tr_seg, tr_t = get_trial_segments_and_times(event_codes, event_times, trial_start_codes, trial_end_codes)
        reward = [np.any(np.isin(reward_codes, ec)) for ec in tr_seg]
        penalty = [np.any(np.isin(penalty_codes, ec)) for ec in tr_seg]
        
        # Build a dataframe for this task entry
        exp = {
            'subject': subject,
            'te_id': te, 
            'date': date, 
            'event_codes': tr_seg,
            'event_times': tr_t, 
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
    
    return df

def tabulate_behavior_data_center_out(preproc_dir, subjects, ids, dates, metadata=[], 
                                      df=None):
    '''
    Wrapper around tabulate_behavior_data() specifically for center-out experiments. 
    Makes use of the task codes saved in `/config/task_codes.yaml` to automatically 
    assign event codes for trial start, trial end, reward, penalty, and targets. 
    Trial start can optionally include the center target.

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
            | **date (str):** date of stimulation
            | **event_codes (ntrial):** numeric code segments for each trial
            | **event_times (ntrial):** time segments for each trial
            | **reward (ntrial):** boolean values indicating whether each trial was rewarded
            | **penalty (ntrial):** boolean values indicating whether each trial was penalized
            | **%metadata_key% (ntrial):** requested metadata values for each key requested
            | **target_idx (ntrial):** index of the target that was presented
            | **target_location (ntrial):** location of the target that was presented
            | **hold_start_time (ntrial):** time at which the hold period started
            | **hold_completed (ntrial):** boolean values indicating whether the hold period was completed
            | **delay_start_time (ntrial):** time at which the delay period started
            | **delay_completed (ntrial):** boolean values indicating whether the delay period was completed
            | **go_cue_time (ntrial):** time at which the go cue was presented
            | **reach_completed (ntrial):** boolean values indicating whether the reach was completed
            | **reach_end_time (ntrial):** time at which the reach was completed
    '''
    # Use default "trial" definition
    task_codes = load_bmi3d_task_codes()
    trial_end_codes = [task_codes['TRIAL_END']]
    trial_start_codes = [task_codes['CENTER_TARGET_ON']]
    reward_codes = [task_codes['REWARD']]
    penalty_codes = [task_codes['HOLD_PENALTY'], task_codes['TIMEOUT_PENALTY']]
    target_codes = task_codes['PERIPHERAL_TARGET_ON']

    # Concatenate base trial data
    new_df = tabulate_behavior_data(
        preproc_dir, subjects, ids, dates, trial_start_codes, trial_end_codes, 
        reward_codes, penalty_codes, metadata, df=None)
    
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
    new_df['trial_initiated'] = np.zeros(len(new_df), dtype='bool')
    new_df['hold_start_time'] = np.nan*np.zeros(len(new_df))
    new_df['hold_completed'] = np.zeros(len(new_df), dtype='bool')
    new_df['delay_start_time'] = np.nan*np.zeros(len(new_df))
    new_df['delay_completed'] = np.zeros(len(new_df), dtype='bool')
    new_df['go_cue_time'] = np.nan*np.zeros(len(new_df))
    new_df['reach_completed'] = np.zeros(len(new_df), dtype='bool')
    new_df['reach_end_time'] = np.nan*np.zeros(len(new_df))
    for i in range(len(new_df)):
        event_codes = new_df.loc[i, 'event_codes']
        event_times = new_df.loc[i, 'event_times']
        
        # Trial initiated if cursor enters the center target
        _, hold_times = get_trial_segments(event_codes, event_times,
                                            task_codes['CENTER_TARGET_ON'], [task_codes['CURSOR_ENTER_CENTER_TARGET']])
        new_df.loc[i, 'trial_initiated'] = len(hold_times) > 0
        if new_df.loc[i, 'trial_initiated']:
            new_df.loc[i, 'hold_start_time'] = hold_times[0][-1]

        # Hold completed if peripheral target turns on (start of delay)
        _, delay_times = get_trial_segments(event_codes, event_times,
                                            [task_codes['CURSOR_ENTER_CENTER_TARGET']], task_codes['PERIPHERAL_TARGET_ON'])
        new_df.loc[i, 'hold_completed'] = len(delay_times) > 0
        if new_df.loc[i, 'hold_completed']:
            new_df.loc[i, 'delay_start_time'] = delay_times[0][-1]

        # Delay completed when center target turns off (go cue)
        _, go_cue_times = get_trial_segments(event_codes, event_times,
                                            task_codes['PERIPHERAL_TARGET_ON'], [task_codes['CENTER_TARGET_OFF']])
        new_df.loc[i, 'delay_completed'] = len(go_cue_times) > 0
        if new_df.loc[i, 'delay_completed']:
            new_df.loc[i, 'go_cue_time'] = go_cue_times[0][-1]

        # Reach completed if cursor enters target (regardless of whether the trial was successful)
        _, reach_times = get_trial_segments(event_codes, event_times,
                                            [task_codes['CENTER_TARGET_OFF']], task_codes['CURSOR_ENTER_PERIPHERAL_TARGET'])
        new_df.loc[i, 'reach_completed'] = len(reach_times) > 0
        if new_df.loc[i, 'reach_completed']:
            new_df.loc[i, 'reach_end_time'] = reach_times[0][-1]

    df = pd.concat([df, new_df], ignore_index=True)
    return df

def tabulate_kinematic_data(preproc_dir, subjects, te_ids, dates, start_times, end_times, 
                            samplerate=1000, preproc=None, datatype='cursor'):
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

    Returns:
        (ntrial,): list of tensors of (nt, nch) kinematics from each trial
    '''

    assert len(subjects) == len(te_ids) == len(dates) == len(start_times) == len(end_times)

    segments = [get_kinematic_segment(preproc_dir, s, t, d, ts, te, samplerate, preproc, datatype)[0] 
                for s, t, d, ts, te in zip(subjects, te_ids, dates, start_times, end_times)]
    trajectories = np.array(segments, dtype='object')
    return trajectories

def tabulate_ts_data(preproc_dir, subjects, te_ids, dates, trigger_times, time_before, time_after, 
                     datatype='lfp'):
    '''
    Grab timeseries data from trials across arbitrary preprocessed files.
    
    Args:
        preproc_dir (str): base directory where the files live
        subjects (list of str): Subject name for each recording
        ids (list of int): Block number of Task entry object for each recording
        dates (list of str): Date for each recording
        trigger_times (list of float): times in the recording at which the desired segments starts
        time_before (float): time (in seconds) to include before the trigger times
        time_after (float): time (in seconds) to include after the trigger times
        samplerate (float, optional): optionally choose the samplerate of the data in Hz. Default 1000.
        preproc (fn, optional): function mapping (position, fs) data to (kinematics, fs_new). For example,
            a smoothing function or an estimate of velocity from position
        datatype (str, optional): choice of 'lfp' or 'broadband' data to load. Defaults to 'lfp'.    
        
    Returns:
        tuple: tuple containing:
            | **data (nt, nch, ntr):** tensor of data from each channel and trial
            | **samplerate (float):** sampling rate of the data
    '''

    assert len(subjects) == len(te_ids) == len(dates) == len(trigger_times)
    
    # Get the first segment
    segment_1, samplerate = get_ts_data_segment(
        preproc_dir, subjects[0], te_ids[0], dates[0], trigger_times[0], 
        time_before, time_after, datatype=datatype
    )
        
    # Construct the tensor using the first segment as a template
    if segment_1.ndim == 1:
        segment_1 = np.expand_dims(segment_1, 1)
    nt, nch = segment_1.shape
    segments = np.zeros((nt, nch, len(trigger_times)), like=segment_1)
    segments[:,:,0] = segment_1
    
    # Add the remaining segments
    idx = 1
    for s, t, d, tr in list(zip(subjects, te_ids, dates, trigger_times))[1:]:
        segments[:,:,idx] = get_ts_data_segment(preproc_dir, s, t, d, tr, 
                                      time_before, time_after, datatype=datatype)[0]
        idx += 1
        
    return segments, samplerate

def tabulate_stim_data(preproc_dir, subjects, ids, dates, debug=True, df=None, **kwargs):
    '''
    Concatenate stimulation data from across experiments. Experiments are given as lists of 
    subjects, task entry ids, and dates. Each list must be the same length. 
    
    Args:
        preproc_dir (str): base directory where the files live
        subjects (list of str): Subject name for each recording
        ids (list of int): Block number of Task entry object for each recording
        dates (list of str): Date for each recording
        debug (bool, optional): 
        df (DataFrame, optional): pandas DataFrame object to append. Defaults to None.
        kwargs (dict, optional): optional keyword arguments to pass to :func:`~aopy.preproc.bmi3d.find_laser_stim_times`

    Returns:
        pd.DataFrame: pandas DataFrame containing the concatenated trial data
            | **subject (str):** subject name
            | **te_id (str):** task entry id
            | **date (str):** date of stimulation
            | **stimulation_site (int):** site of stimulation
            | **trial_time (float):** time of stimulation within recording
            | **trial_width (float):** width of stimulation pulse
            | **trial_power (float):** gain of stimulation pulse
            | **trial_found (bool):** whether an analog laser signal was recorded on this trial
            | **width_above_thr (bool):** if the width of the analog signal was above the cutoff
            | **power_above_thr (bool):** if the gain of the analog signal was above the cutoff

    Note:
        Only supports single-site stimulation.
    '''
    if df is None:
        df = pd.DataFrame()

    entries = list(zip(subjects, dates, ids))
    for subject, date, te in tqdm(entries): 

        # Load data from bmi3d hdf 
        try:
            exp_data, exp_metadata = load_preproc_exp_data(preproc_dir, subject, te, date)
        except:
            print(f"Entry {subject} {date} {te} could not be loaded.")
            traceback.print_exc()
            continue

        # Find laser trial times 
        try:
            (trial_times, trial_widths, trial_powers, times_not_found, widths_above_thr, 
             powers_above_thr) = preproc.bmi3d.get_laser_trial_times(
                preproc_dir, subject, te, date, debug=debug, **kwargs)
        except:
            print(f"Problem extracting stimulation trials from entry {subject} {date} {te}")
            traceback.print_exc()
            continue
                        
        # Tabulate everything together
        exp = {
            'subject': subject,
            'te_id': te, 
            'date': date, 
            'stimulation_site': exp_metadata['stimulation_site'],
            'trial_time': trial_times,
            'trial_width': trial_widths, 
            'trial_power': trial_powers,
            'trial_found': ~times_not_found,
            'width_above_thr': widths_above_thr,
            'power_above_thr': powers_above_thr,
        }
        df = pd.concat([df,pd.DataFrame(exp)], ignore_index=True)
    
    return df
        
def load_bmi3d_task_codes(filename='task_codes.yaml'):
    '''
    Load the default BMI3D task codes. File-specific codes can be found in exp_metadata['event_sync_dict']

    Args:
        filename (str, optional): filename of the task codes to load. Defaults to 'task_codes.yaml'.

    Returns:
        dict: (name, code) task code dictionary
    '''
    config_dir = files('aopy').joinpath('config')
    params_file = as_file(config_dir.joinpath(filename))
    with params_file as f:
        task_codes = yaml_read(f)[0]
    return task_codes
