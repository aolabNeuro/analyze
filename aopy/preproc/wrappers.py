from .base import *
from .bmi3d import parse_bmi3d
from .oculomatic import parse_oculomatic
from .optitrack import parse_optitrack
from .. import postproc
from .. import precondition
from ..precondition import eye
from ..data import load_ecube_data_chunked, load_ecube_metadata, proc_ecube_data, save_hdf, load_hdf_group, get_hdf_dictionary, get_preprocessed_filename
from ..data import load_preproc_lfp_data, load_preproc_broadband_data, load_preproc_eye_data
import os
import h5py

'''
proc_* wrappers
'''
def proc_single(data_dir, files, preproc_dir, subject, te_id, date, preproc_jobs, overwrite=False, **kwargs):
    '''
    Preprocess a single recording, given a list of raw data files, into a series of hdf records with the same prefix.
    Args:
        data_dir (str): File directory of collected session data
        files (dict): dict of file names to process in data_dir
        preproc_dir (str): Target directory for processed data
        subject (str): Subject name
        te_id (int): Block number of Task entry object 
        date (str): Date of recording
        preproc_jobs (list): list of proc_types to generate
        overwrite (bool, optional): Overwrite files in result_dir. Defaults to False.
    '''
    if os.path.basename(os.path.normpath(preproc_dir)) != subject:
        preproc_dir = os.path.join(preproc_dir, subject)
    if not os.path.exists(preproc_dir):
        os.mkdir(preproc_dir)
    preproc_dir_base = os.path.dirname(preproc_dir)

    if 'exp' in preproc_jobs:
        print('processing experiment data...')
        exp_filename = get_preprocessed_filename(subject, te_id, date, 'exp')
        proc_exp(
            data_dir,
            files,
            preproc_dir,
            exp_filename,
            overwrite=overwrite,
        )
    if 'eye' in preproc_jobs:
        print('processing eyetracking data...')
        exp_filename = get_preprocessed_filename(subject, te_id, date, 'exp')
        eye_filename = get_preprocessed_filename(subject, te_id, date, 'eye')
        proc_eyetracking(
            data_dir,
            files,
            preproc_dir,
            exp_filename,
            eye_filename,
            overwrite=overwrite,
        )
        eye_data, eye_metadata = load_preproc_eye_data(preproc_dir_base, subject, te_id, date)
        assert 'raw_data' in eye_data.keys(), "No eye data found"
        assert eye_data['raw_data'].shape == (eye_metadata['n_samples'], eye_metadata['n_channels'])
    if 'broadband' in preproc_jobs:
        print('processing broadband data...')
        broadband_filename = get_preprocessed_filename(subject, te_id, date, 'broadband')
        proc_broadband(
            data_dir,
            files,
            preproc_dir,
            broadband_filename,
            overwrite=overwrite
        )
        broadband_data, broadband_metadata = load_preproc_broadband_data(preproc_dir_base, subject, te_id, date)
        assert broadband_data.shape == (broadband_metadata['n_samples'], broadband_metadata['n_channels'])
    if 'lfp' in preproc_jobs:
        print('processing local field potential data...')
        lfp_filename = get_preprocessed_filename(subject, te_id, date, 'lfp')
        proc_lfp(
            data_dir,
            files,
            preproc_dir,
            lfp_filename,
            overwrite=overwrite,
            filter_kwargs=kwargs # pass any remaining kwargs to the filtering function
        )
        lfp_data, lfp_metadata = load_preproc_lfp_data(preproc_dir_base, subject, te_id, date)
        assert lfp_data.shape == (lfp_metadata['n_samples'], lfp_metadata['n_channels'])

def proc_exp(data_dir, files, result_dir, result_filename, overwrite=False, save_res=True):
    '''
    Process experiment data files. Loads 'hdf' and 'ecube' (if present) data, parses, and 
    prepares experiment data and metadata.

    Note:
        Currently supports BMI3D only. 
    
    The above data is prepared into structured arrays:
        exp_data:
            task ([('cursor', '<f8', (3,)), ('trial', 'u8', (1,)), ('time', 'u8', (1,)), ...])
            state ([('msg', 'S', (1,)), ('time', 'u8', (1,))])
            clock ([('timestamp', 'f8', (1,)), ('time', 'u8', (1,))])
            events ([('timestamp', 'f8', (1,)), ('time', 'u8', (1,)), ('event', 'S32', (1,)), 
                ('data', 'u2', (1,)), ('code', 'u2', (1,))])
            trials ([('trial', 'u8'), ('index', 'u8'), (condition_name, 'f8', (3,)))])
        exp_metadata:
            source_dir (str)
            source_files (dict)
            bmi3d_start_time (float)
            n_cycles (int)
            n_trials (int)
            <other metadata from bmi3d>
    
    Args:
        data_dir (str): where the data files are located
        files (dict): dictionary of filenames indexed by system
        result_filename (str): where to store the processed result
        overwrite (bool): whether to remove existing processed files if they exist

    Returns:
        None
    '''   
    # Check if a processed file already exists
    filepath = os.path.join(result_dir, result_filename)
    if not overwrite and os.path.exists(filepath):
        contents = get_hdf_dictionary(result_dir, result_filename)
        if "exp_data" in contents or "exp_metadata" in contents:
            print("File {} already preprocessed, doing nothing.".format(result_filename))
            return
    
    # Prepare the BMI3D data
    if 'hdf' not in files:
        print("No HDF data found.")
        return

    bmi3d_data, bmi3d_metadata = parse_bmi3d(data_dir, files)
    if save_res:
        save_hdf(result_dir, result_filename, bmi3d_data, "/exp_data", append=True)
        save_hdf(result_dir, result_filename, bmi3d_metadata, "/exp_metadata", append=True)
    return bmi3d_data, bmi3d_metadata

def proc_mocap(data_dir, files, result_dir, result_filename, overwrite=False):
    '''
    Process motion capture files:
        Loads metadata, position data, and rotation data from 'optitrack' files
        If present, reads 'hdf' metadata to find appropriate strobe channel
        If present, loads 'ecube' analog data representing optitrack camera strobe
    The data is prepared along with timestamps into HDF datasets:
        mocap_data:
            optitrack [('position', 'f8', (3,)), ('rotation', 'f8', (4,)), ('timestamp', 'f8', (1,)]
        mocap_metadata:
            source_dir (str)
            source_files (dict)
            samplerate (float)
            <other metadata from motive>
    
    Args:
        data_dir (str): where the data files are located
        files (dict): dictionary of filenames indexed by system
        result_filename (str): where to store the processed result
        overwrite (bool): whether to remove existing processed files if they exist

    Returns:
        None
    '''  
    # Check if a processed file already exists
    filepath = os.path.join(result_dir, result_filename)
    if not overwrite and os.path.exists(filepath):
        contents = get_hdf_dictionary(result_dir, result_filename)
        if "mocap_data" in contents or "mocap_metadata" in contents:
            print("File {} already preprocessed, doing nothing.".format(result_filename))
            return

    # Parse Optitrack data
    if 'optitrack' in files:
        optitrack_data, optitrack_metadata = parse_optitrack(data_dir, files)
        save_hdf(result_dir, result_filename, optitrack_data, "/mocap_data", append=True)
        save_hdf(result_dir, result_filename, optitrack_metadata, "/mocap_metadata", append=True)

def proc_eyetracking(data_dir, files, result_dir, exp_filename, result_filename, debug=True, overwrite=False, save_res=True, **kwargs):
    '''
    Loads eyedata from ecube analog signal and calculates calibration profile using least square fitting.
    Requires that experimental data has already been preprocessed in the same result hdf file.
    
    The data is prepared into HDF datasets:
    
    eye_data:
        raw_data (nt, nch): raw eye data
        calibrated_data (nt, nch): calibrated eye data
        coefficients (nch, 2): linear regression coefficients
        correlation_coeff (nch): best fit correlation coefficients from linear regression
        cursor_calibration_data (ntr, 2): cursor coordinates used for calibration
        eye_calibration_data (ntr, nch): eye coordinates used for calibration
    
    eye_metadata:
        samplerate (float): sampling rate of the calibrated eye data
        see :func:`aopy.preproc.parse_oculomatic` for oculomatic metadata

    Args:
        data_dir (str): where the data files are located
        files (dict): dictionary of filenames indexed by system
        result_dir (str): where to store the processed result 
        result_filename (str): what to call the preprocessed filename
        debug (bool, optional): if true, prints additional debug messages
        overwrite (bool, optional): whether to recalculated and overwrite existing preprocessed eyetracking data
        save_res (bool, optional): whether to save the calculated eyetracking data
        **kwargs (dict, optional): keyword arguments to pass to :func:`aopy.preproccalc_eye_calibration()`

    Returns:
        eye_dict (dict): all the data pertaining to eye tracking, calibration
        eye_metadata (dict): metadata for eye tracking

    Example:
        Uncalibrated raw data:

        .. image:: _images/eye_trajectories.png

        After calibration:
        
        .. image:: _images/eye_trajectories_calibrated.png
    '''
    # Check if data already exists
    filepath = os.path.join(result_dir, result_filename)
    if not overwrite and os.path.exists(filepath):
        contents = get_hdf_dictionary(result_dir, result_filename)
        if "eye_data" in contents and "eye_metadata" in contents:
            print("Eye data already preprocessed in {}, returning existing data.".format(result_filename))
            eye_data = load_hdf_group(result_dir, result_filename, 'eye_data')
            eye_metadata = load_hdf_group(result_dir, result_filename, 'eye_metadata')
            return eye_data, eye_metadata
    
    # Load the preprocessed experimental data
    try:
        exp_data = load_hdf_group(result_dir, exp_filename, 'exp_data')
        exp_metadata = load_hdf_group(result_dir, exp_filename, 'exp_metadata')
    except (FileNotFoundError, ValueError):
        raise ValueError(f"File {result_filename} does not include preprocessed experimental data. Please call proc_exp() first.")
    
    # Parse the raw eye data; this could be extended in the future to support other eyetracking hardware
    eye_data, eye_metadata = parse_oculomatic(data_dir, files, debug=debug)
    eye_mask = eye_data['mask']
    eye_data = eye_data['data']

    try:
        # Calibrate the eye data
        cursor_samplerate = exp_metadata['cursor_interp_samplerate']
        cursor_data = exp_data['cursor_interp']
        events = exp_data['events']
        event_codes = events['code']
        event_times = events['timestamp'] # time points in the ecube time frame
        coeff, correlation_coeff, cursor_calibration_data, eye_calibration_data = calc_eye_calibration(
            cursor_data, cursor_samplerate, eye_data, eye_metadata['samplerate'], 
            event_times, event_codes, return_datapoints=True, **kwargs)

        calibrated_eye_data = postproc.get_calibrated_eye_data(eye_data, coeff)
        eye_dict = {
            'eye_closed_mask': eye_mask,
            'raw_data': eye_data,
            'calibrated_data': calibrated_eye_data,
            'coefficients': coeff,
            'correlation_coeff': correlation_coeff,
            'cursor_calibration_data': cursor_calibration_data,
            'eye_calibration_data': eye_calibration_data
        }

    except (KeyError, ValueError):
        # If there is no cursor data or there aren't enough trials, this will fail. 
        # We should still save the eye data, just don't include the calibrated data
        eye_dict = {'raw_data': eye_data}

    # Save everything into the HDF file
    if save_res:
        save_hdf(result_dir, result_filename, eye_dict, "/eye_data", append=True)
        save_hdf(result_dir, result_filename, eye_metadata, "/eye_metadata", append=True)
    return eye_dict, eye_metadata

def proc_broadband(data_dir, files, result_dir, result_filename, overwrite=False, max_memory_gb=1.):
    '''
    Process broadband data:
        Loads 'ecube' headstage data and metadata
    Saves broadband data into the HDF datasets:
        broadband_data (nt, nch)
        broadband_metadata (dict)

    Args:
        data_dir (str): where the data files are located
        files (dict): dictionary of filenames indexed by system
        result_filename (str): where to store the processed result
        overwrite (bool, optional): whether to remove existing processed files if they exist
        max_memory_gb (float, optional): max memory used to load binary data at one time

    Returns:
        None
    '''

    # Check if a processed file already exists
    filepath = os.path.join(result_dir, result_filename)
    if not overwrite and os.path.exists(filepath):
        contents = get_hdf_dictionary(result_dir, result_filename)
        if "broadband_data" in contents:
            raise FileExistsError("File {} already preprocessed, doing nothing.".format(result_filename))
    elif os.path.exists(filepath):
        os.remove(filepath) # maybe bad, since it deletes everything, not just broadband data

    # Copy the broadband data into an HDF dataset
    if 'ecube' in files:
        
        # Process the binary data
        data_filepath = os.path.join(data_dir, files['ecube'])
        result_filepath = os.path.join(result_dir, result_filename)
        _, metadata = proc_ecube_data(data_filepath, 'Headstages', result_filepath, result_name='broadband_data', max_memory_gb=max_memory_gb)

        # Append the broadband metadata to the file
        save_hdf(result_dir, result_filename, metadata, "/broadband_metadata", append=True)

def proc_lfp(data_dir, files, result_dir, result_filename, overwrite=False, max_memory_gb=1., filter_kwargs={}):
    '''
    Process lfp data:
        Loads 'ecube' headstage data and metadata
    Saves broadband data into the HDF datasets:
        lfp_data (nt, nch)
        lfp_metadata (dict)
    
    Args:
        data_dir (str): where the data files are located
        files (dict): dictionary of filenames indexed by system
        result_filename (str): where to store the processed result
        overwrite (bool, optional): whether to remove existing processed files if they exist
        max_memory_gb (float, optional): max memory used to load binary data at one time
        filter_kwargs (dict, optional): keyword arguments to pass to :func:`aopy.precondition.filter_lfp`
    Returns:
        None
    '''  
    # Check if a processed file already exists
    filepath = os.path.join(result_dir, result_filename)
    if not overwrite and os.path.exists(filepath):
        contents = get_hdf_dictionary(result_dir, result_filename)
        if "lfp_data" in contents:
            print("File {} already preprocessed, doing nothing.".format(result_filename))
            return
    elif os.path.exists(filepath):
        os.remove(filepath) # maybe bad, since it deletes everything, not just lfp_data

    # Preprocess neural data into lfp   
    dtype = 'int16'
    if 'ecube' in files:
        data_path = os.path.join(data_dir, files['ecube'])
        metadata = load_ecube_metadata(data_path, 'Headstages')
        samplerate = metadata['samplerate']
        chunksize = int(max_memory_gb * 1e9 / np.dtype(dtype).itemsize / metadata['n_channels'])
        lfp_samplerate = filter_kwargs.pop('lfp_samplerate', 1000)
        downsample_factor = int(samplerate/lfp_samplerate)
        lfp_samples = int(np.ceil(metadata['n_samples']/downsample_factor))
        n_channels = int(metadata['n_channels'])

        # Create an hdf dataset
        result_filepath = os.path.join(result_dir, result_filename)
        hdf = h5py.File(result_filepath, 'a') # should append existing or write new?
        dset = hdf.create_dataset('lfp_data', (lfp_samples, n_channels), dtype=dtype)

        # Filter broadband data into LFP directly into the hdf file
        n_samples = 0
        for broadband_chunk in load_ecube_data_chunked(data_path, 'Headstages', chunksize=chunksize):
            lfp_chunk = precondition.filter_lfp(broadband_chunk, samplerate, **filter_kwargs)
            chunk_len = lfp_chunk.shape[0]
            dset[n_samples:n_samples+chunk_len,:] = lfp_chunk
            n_samples += chunk_len
        hdf.close()

        # Append the lfp metadata to the file
        lfp_metadata = metadata
        lfp_metadata['lfp_samplerate'] = lfp_samplerate # for backwards compatibility
        lfp_metadata['samplerate'] = lfp_samplerate
        lfp_metadata['n_samples'] = lfp_samples
        lfp_metadata['low_cut'] = 500
        lfp_metadata['buttord'] = 4
        lfp_metadata.update(filter_kwargs)

    save_hdf(result_dir, result_filename, lfp_metadata, "/lfp_metadata", append=True)
