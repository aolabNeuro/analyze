from aopy.preproc.base import get_data_segments, get_trial_segments, interp_timestamps2timeseries, trial_align_data
from ..whitematter import ChunkedStream, Dataset
from ..utils import derivative, get_pulse_edge_times, compute_pulse_duty_cycles
from ..data import load_preproc_exp_data, load_preproc_eye_data, load_preproc_lfp_data
import os
import numpy as np
import h5py
import tables


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
    metadata = load_ecube_metadata(os.path.join(path, data_dir), 'AnalogPanel')
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

#####################
# Preprocessed data #
#####################
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
    return get_kinematic_segments(*args, **kwargs, preproc=lambda x, y: derivative(x, y, norm=norm))


def get_kinematic_segments(preproc_dir, subject, te_id, date, trial_start_codes, trial_end_codes, 
                           trial_filter=lambda x:True, preproc=lambda t, x : x, datatype='cursor'):
    '''
    Loads x,y,z cursor, hand, or eye trajectories for each "trial" from a preprocessed HDF file. Trials can
    be specified by numeric start and end codes. Trials can also be filtered so that only successful
    trials are included, for example. The filter is applied to numeric code segments for each trial. 
    Finally, the cursor data can be preprocessed by a supplied function to, for example, convert 
    position to velocity estimates. The preprocessing function is applied to the (time, position)
    cursor or eye data.
    
    Example:
        subject = 'beignet'
        te_id = 4301
        date = '2021-01-01'
        trial_filter = lambda t: TRIAL_END not in t
        trajectories, segments = get_trial_trajectories(preproc_dir, subject, te_id, date,
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
        data (str, optional): choice of 'cursor', 'hand', or 'eye' kinematics to load
    
    Returns:
        tuple: tuple containing:
            | **trajectories (ntrial):** array of filtered cursor trajectories for each trial
            | **trial_segments (ntrial):** array of numeric code segments for each trial
        
    '''
    data, metadata = load_preproc_exp_data(preproc_dir, subject, te_id, date)

    if datatype == 'cursor':
        raw_kinematics = data['cursor_interp']
        samplerate = metadata['cursor_interp_samplerate']
    elif datatype == 'hand':
        hand_data_cycles = data['bmi3d_task']['manual_input']
        clock = data['clock']['timestamp_sync']
        samplerate = metadata['analog_samplerate']
        time = np.arange(int((clock[-1] + 10)*samplerate))/samplerate
        hand_data_cycles = _correct_hand_traj(data['bmi3d_task'])
        raw_kinematics, _ = interp_timestamps2timeseries(clock, hand_data_cycles, sampling_points=time, interp_kind='linear')

        # print('hi', data['cursor_interp'].shape, hand_data_cycles.shape, raw_kinematics.shape, pts_to_remove)
    elif datatype == 'eye':
        eye_data, eye_metadata = load_preproc_eye_data(preproc_dir, subject, te_id, date)
        samplerate = eye_metadata['samplerate']
        raw_kinematics = eye_data['calibrated_data']
    else:
        raise ValueError(f"Unknown datatype {datatype}")

    time = np.arange(len(raw_kinematics))/samplerate
    kinematics = preproc(time, raw_kinematics)
    assert kinematics is not None

    event_codes = data['events']['code']
    event_times = data['events']['timestamp']

    trial_segments, trial_times = get_trial_segments(event_codes, event_times, 
                                                                  trial_start_codes, trial_end_codes)
    trajectories = np.array(get_data_segments(kinematics, trial_times, samplerate), dtype='object')
    trial_segments = np.array(trial_segments, dtype='object')
    success_trials = [trial_filter(t) for t in trial_segments]
    
    return trajectories[success_trials], trial_segments[success_trials]

def _correct_hand_traj(bmi3d_task_data):
    '''
    This function removes hand position data points when the cursor is simultaneously stationary in all directions.
    These hand position data points are artifacts. 
        
    Args:
        exp_data (dict): BMI3D task data
    
    Returns:
        hand_position (nt, 3): Corrected hand position
    '''

    hand_position = bmi3d_task_data['manual_input']

    # Set hand position to np.nan if the cursor position doesn't update. This indicates an optitrack error moved the hand outside the boundary.
    bad_pt_mask = np.zeros(bmi3d_task_data['cursor'].shape, dtype=bool) 
    bad_pt_mask[1:,0] = (np.diff(bmi3d_task_data['cursor'], axis=0)==0)[:,0] & (np.diff(bmi3d_task_data['cursor'], axis=0)==0)[:,1] & (np.diff(bmi3d_task_data['cursor'], axis=0)==0)[:,2]
    bad_pt_mask[1:,1] = (np.diff(bmi3d_task_data['cursor'], axis=0)==0)[:,0] & (np.diff(bmi3d_task_data['cursor'], axis=0)==0)[:,1] & (np.diff(bmi3d_task_data['cursor'], axis=0)==0)[:,2]
    bad_pt_mask[1:,2] = (np.diff(bmi3d_task_data['cursor'], axis=0)==0)[:,0] & (np.diff(bmi3d_task_data['cursor'], axis=0)==0)[:,1] & (np.diff(bmi3d_task_data['cursor'], axis=0)==0)[:,2]
    hand_position[bad_pt_mask] = np.nan

    return hand_position

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
    Loads lfp segments (different length for each trial) from a preprocessed HDF file. Trials can
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
    print(lfp_data.shape)
    trial_aligned_data = trial_align_data(lfp_data, trial_start_times, time_before, time_after, samplerate) #(ntrial, nt, nch)
    success_trials = [trial_filter(t) for t in trial_segments]
    
    return trial_aligned_data[success_trials]

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
    locations = []
    for i in range(len(target_indices)):
        trial_idx = np.where(trials['index'] == target_indices[i])[0][0]
        locations.append(trials['target'][trial_idx][[0,2,1]])
    return np.array(locations)

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
            |** files (dict):** dictionary of (source, filepath) files that are associated with the given experiment
            |** data_dir (str):** directory where the source files were located
    '''
    exp_data, exp_metadata = load_preproc_exp_data(preproc_dir, subject, te_id, date)
    return exp_metadata['source_files'], exp_metadata['source_dir']